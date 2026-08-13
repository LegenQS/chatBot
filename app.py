import os
import re
import gc
import json
import time
import atexit
import shutil
import webbrowser
import platform
from pathlib import Path

# Route Hugging Face downloads through a mirror when huggingface.co is blocked
# (e.g. mainland China): run with HF_MIRROR=1, or export your own HF_ENDPOINT.
# MUST run before huggingface_hub / sentence_transformers are imported below,
# because huggingface_hub reads HF_ENDPOINT once at import time.
if os.environ.get("HF_MIRROR") and not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
import numpy as np
import faiss
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

import build_index
from model_config import (
    MODEL_DIR,
    MODEL_TIERS,
    TIER_ORDER,
    TIER_SIZE,
    EMBED_DIR,
    model_path,
    tier_ready,
    download_tier,
    download_embed_model,
)

# ---- PATHS ----
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "manual.index"
DOC_STORE_PATH = BASE_DIR / "manual_docs.json"
IMAGE_DIR = BASE_DIR / "images"
I18N_PATH = BASE_DIR / "i18n.json"


def has_gpu():
    """Best-effort detector for an available GPU accelerator."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return True  # Apple Silicon (Metal)
    return shutil.which("nvidia-smi") is not None  # NVIDIA (CUDA)


def shutdown_app():
    """Terminate the whole Streamlit server process (no Ctrl+C needed)."""
    # Small pause so the "stopped" message is flushed to the browser before the
    # process ends. os._exit terminates immediately from this worker thread,
    # bypassing Streamlit's own SIGTERM handler; it's fully cross-platform
    # (unlike SIGKILL, which doesn't exist on Windows).
    time.sleep(0.8)
    os._exit(0)


def resolve_tier():
    # Preferred default: env override wins, else pick by hardware.
    tier = os.environ.get("MODEL_TIER", "auto").lower()
    if tier not in MODEL_TIERS:
        tier = "quality" if has_gpu() else "fast"
    return tier


def pick_initial_tier():
    # Never kick off a big download just to start up: prefer the hardware
    # default if it's already downloaded, else any downloaded tier, else the
    # smallest ("fast", ~2GB) which downloads on first launch.
    preferred = resolve_tier()
    if tier_ready(preferred):
        return preferred
    for t in TIER_ORDER:
        if tier_ready(t):
            return t
    return "fast"

# ---- Example questions shown on the empty screen ----
EXAMPLES = {
    "zh": [
        "炉丝不加热怎么排查？",
        "如何检测固态继电器是否损坏？",
        "温度测量异常如何检查？",
    ],
    "en": [
        "How do I troubleshoot a heating element that won't heat?",
        "How can I tell if a solid-state relay is damaged?",
        "How do I inspect abnormal temperature readings?",
    ],
}

st.set_page_config(page_title="Machine Assistant", page_icon="🔧", layout="centered")


@st.cache_resource
def _install_clean_exit():
    # llama.cpp's Metal backend calls ggml_abort() (SIGABRT) when its GPU device
    # is freed during normal Python teardown — on macOS that surfaces as a
    # "Python quit unexpectedly" crash on every Ctrl+C / shutdown. Hard-exit the
    # process at interpreter shutdown so those crashing native destructors never
    # run. cache_resource ensures this registers exactly once per process.
    atexit.register(os._exit, 0)
    return True


_install_clean_exit()

# ---- Load localization file ----
with open(I18N_PATH, "r", encoding="utf-8") as f:
    i18n = json.load(f)


def _build_llm(tier):
    # Download only if needed. This runs only for a deliberate load/switch
    # (see the sidebar), so a large pull is never triggered just by browsing
    # the dropdown.
    if not tier_ready(tier):
        with st.spinner(f"Downloading the {tier} model ({TIER_SIZE[tier]}) — one time only…"):
            download_tier(tier)
    # Point llama.cpp at the first shard; it auto-loads any remaining shards.
    # n_gpu_layers=-1 offloads everything to the GPU (CUDA on Windows/Linux,
    # Metal on Apple Silicon). Harmless on CPU-only builds — it just stays on CPU.
    return Llama(
        model_path=str(model_path(tier)),
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_batch=512,
        n_gpu_layers=-1,
        verbose=False,
    )


# A single shared slot holds the active model. This persists across reruns and
# sessions (cache_resource), so we manage the lifecycle by hand instead of
# caching per tier — that lets us fully release the old model BEFORE building
# the new one, so two models are never resident in memory/VRAM at once.
@st.cache_resource(show_spinner=False)
def _llm_slot():
    return {"tier": None, "llm": None}


def load_llm(tier):
    slot = _llm_slot()
    if slot["tier"] == tier and slot["llm"] is not None:
        return slot["llm"]

    # Release the previously-loaded model first.
    if slot["llm"] is not None:
        try:
            slot["llm"].close()
        except Exception:
            pass
        slot["llm"] = None
        slot["tier"] = None
        gc.collect()

    slot["llm"] = _build_llm(tier)
    slot["tier"] = tier
    return slot["llm"]


@st.cache_resource(show_spinner=False)
def load_embed_model():
    # Use the local copy if present and valid; otherwise (missing folder, or a
    # truncated pytorch_model.bin that makes torch.load raise EOFError) fetch it
    # once from Hugging Face. model/ is gitignored, so a fresh machine lands here.
    try:
        return SentenceTransformer(str(EMBED_DIR))
    except Exception:
        with st.spinner("Fetching the embedding model (~470 MB, one time only)…"):
            download_embed_model()
        return SentenceTransformer(str(EMBED_DIR))


@st.cache_resource(show_spinner=False)
def load_index(_embed_model):
    # Build the index on first run if it isn't there yet.
    if not INDEX_PATH.exists() or not DOC_STORE_PATH.exists():
        with st.spinner("Building the search index for the first time…"):
            build_index.build_index(embed_model=_embed_model, verbose=False)
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOC_STORE_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "active_tier" not in st.session_state:
    # The model that is actually loaded. Only changes on a deliberate click,
    # never just by browsing the dropdown — that's what keeps a big download
    # from starting by accident (and prevents overlapping loads).
    st.session_state.active_tier = pick_initial_tier()

# Auto-open browser on first run (when launched from a .command/.bat script).
# Check if running in headless mode (launched from start script).
if os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true":
    if "browser_opened_once" not in st.session_state:
        try:
            import time
            time.sleep(2)  # Give server time to fully start
            webbrowser.open("http://localhost:8520")
            st.session_state.browser_opened_once = True
        except Exception as e:
            pass  # Silently fail if browser can't open; user can navigate manually.

# ---- Sidebar (must run before models load, so the active tier is known) ----
with st.sidebar:
    st.header(i18n["en"]["settings_header"])
    lang_choice = st.radio(
        i18n["en"]["select_language"],
        ["中文", "English"],
        index=0,
        key="sidebar_lang",
    )
    st.session_state.lang = "zh" if lang_choice == "中文" else "en"
    localization = i18n[st.session_state.lang]

    st.divider()
    selected_tier = st.selectbox(
        localization["model_label"],
        TIER_ORDER,
        index=TIER_ORDER.index(st.session_state.active_tier),
        format_func=lambda t: localization[f"tier_{t}"],
        key="model_select",
        help=localization["model_help"],
    )

    # Selecting a tier does nothing on its own — switching is an explicit click.
    if selected_tier != st.session_state.active_tier:
        if tier_ready(selected_tier):
            if st.button("🔀 " + localization["switch_btn"], width="stretch", type="primary"):
                st.session_state.active_tier = selected_tier
                st.rerun()
        else:
            st.caption("⬇️ " + localization["needs_download"].format(size=TIER_SIZE[selected_tier]))
            st.caption(localization["predownload_hint"])
            st.code(f"python download_model.py {selected_tier}", language="bash")
            if st.button(
                localization["download_load_btn"].format(size=TIER_SIZE[selected_tier]),
                width="stretch",
            ):
                st.session_state.active_tier = selected_tier
                st.rerun()
    else:
        ready = tier_ready(selected_tier)
        st.caption(("✅ " + localization["model_ready"]) if ready
                   else ("⬇️ " + localization["needs_download"].format(size=TIER_SIZE[selected_tier])))

    st.divider()
    if st.button(localization["remove_cache"], width="stretch"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    if st.button(localization["quit_btn"], width="stretch"):
        st.session_state.quitting = True
        st.rerun()

    st.caption(f"⚙️ {st.session_state.active_tier} · {'GPU' if has_gpu() else 'CPU'}")

# ---- Shutdown screen: render a goodbye, then stop the server process ----
if st.session_state.get("quitting"):
    st.title(localization["title"])
    st.success("🛑 " + localization["quit_done"])
    st.caption(localization["quit_hint"])
    shutdown_app()
    st.stop()

# ---- Load models (after the active tier is known) ----
with st.spinner("Warming up the assistant…"):
    embed_model = load_embed_model()
    index, docs = load_index(embed_model)
    llm = load_llm(st.session_state.active_tier)


# ---- Embedding ----
def embed(text):
    emb = embed_model.encode(
        [f"query: {text}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    return emb.astype("float32")


def clean_chunk_text(text):
    text = re.sub(r"<</?SYS>>", "", text)
    text = re.sub(r"<\[/INST\]>", "", text)
    text = re.sub(r"\[INST\]", "", text)
    return text.strip()


# ---- Retrieval ----
# multilingual-e5 similarity scores sit in a compressed high range (relevant
# hits ~0.90+, unrelated text still ~0.85), so the floor is set fairly high to
# keep obvious noise out of the LLM context. The grounded system prompt is the
# final guard for anything borderline that slips through.
def retrieve(query, top_k=3, min_score=0.80):
    q_emb = np.array([embed(query)], dtype="float32")
    D, I = index.search(q_emb, top_k)
    return [(float(s), docs[i]) for s, i in zip(D[0], I[0]) if s > min_score]


# ---- Build chat messages for the LLM ----
def build_messages(user_question, retrieved_docs):
    context = ""
    for d in retrieved_docs:
        context += f"Section: {d['metadata']['path']}\n"
        if d.get("instructions"):
            context += "Instructions:\n" + "\n".join(d["instructions"]) + "\n"
        if d.get("warnings"):
            context += "Warnings:\n" + "\n".join(d["warnings"]) + "\n"
        context += "\n"

    return [
        {"role": "system", "content": localization["system_prompt"]},
        {
            "role": "user",
            "content": (
                localization["sources_label"] + ":\n" + context + "\n"
                + localization["question"] + ": " + user_question + "\n"
                + localization["question_prompt"]
            ),
        },
    ]


# ---- Render helpers ----
def source_excerpt(chunk, limit=280):
    """The actual manual text a source refers to, trimmed so users can locate it."""
    lines = list(chunk.get("instructions", []))
    lines += [f"⚠ {w}" for w in chunk.get("warnings", [])]
    text = " ".join(lines).strip()
    return text[:limit].rstrip() + "…" if len(text) > limit else text


def render_sources(sources):
    if not sources:
        return
    with st.expander(localization["view_sources"]):
        for i, s in enumerate(sources):
            if i:
                st.divider()
            # Numbered location, e.g. "第一章 炉丝加热 › 2. 检测固态继电器…"
            st.markdown(f"📖 **{s['path']}**")
            if s.get("excerpt"):
                st.markdown(f"> {s['excerpt']}")
            for img in s.get("images", []):
                img_path = IMAGE_DIR / img
                if img_path.exists():
                    st.image(str(img_path))


def generate_answer(question):
    """Return (answer_text, sources) for a question, streaming into the UI."""
    retrieved = retrieve(question)
    if not retrieved:
        return localization["not_found_prompt"], []

    top_score, top_chunk = retrieved[0]

    # Fast path: one strongly-matching section -> return the manual text
    # verbatim with no LLM call (fastest, and exact by construction).
    if top_score > 0.90 and len(retrieved) == 1:
        if top_chunk.get("instructions"):
            answer_text = "\n\n".join(top_chunk["instructions"])
        elif top_chunk.get("warnings"):
            answer_text = "\n\n".join(top_chunk["warnings"])
        else:
            answer_text = localization["not_found_prompt"]
        st.markdown(answer_text)
    else:
        messages = build_messages(question, [d for _, d in retrieved])
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=0.0,
            top_k=1,
            top_p=0.8,
            stream=True,
        )
        answer_text = ""
        placeholder = st.empty()
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                answer_text += delta["content"]
                placeholder.markdown(answer_text + " ▌")
        placeholder.markdown(answer_text)

    sources = [
        {
            "path": d["metadata"]["path"],
            "excerpt": source_excerpt(d),
            "images": d["metadata"].get("images", []),
        }
        for _, d in retrieved
    ]
    return clean_chunk_text(answer_text), sources


# ---- Header ----
st.title(localization["title"])
st.caption(localization["subtitle"])

# ---- Chat input ----
# Must be called at the app top level (not inside a tab/container) so Streamlit
# pins it to the bottom of the viewport. Inside a tab it renders inline, which
# makes it float in the middle while an answer streams, then jump on rerun.
typed = st.chat_input(localization["chat_input_placeholder"])

# ---- Tab selector: Chat or Document Editor ----
tab_chat, tab_editor = st.tabs(["💬 Chat", "📖 Document Editor"])

# ---- CHAT TAB ----
with tab_chat:
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(msg["question"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["answer"])
            render_sources(msg.get("sources", []))

    # ---- Empty state: welcome + suggested questions ----
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(localization["welcome"])
        st.caption(localization["suggestions_header"])
        cols = st.columns(len(EXAMPLES[st.session_state.lang]))
        for i, (col, ex) in enumerate(zip(cols, EXAMPLES[st.session_state.lang])):
            if col.button(ex, key=f"ex_{i}", width="stretch"):
                st.session_state.pending = ex
                st.rerun()

    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question:
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(localization["thinking_spinner"]):
                answer_text, sources = generate_answer(question)
            render_sources(sources)

        st.session_state.messages.append({
            "question": question,
            "answer": answer_text,
            "sources": sources,
        })
        st.rerun()

# ---- DOCUMENT EDITOR TAB ----
with tab_editor:
    from doc_editor import (
        load_current_docs,
        save_docs_to_json,
        rebuild_index,
        rebuild_chunks,
        list_images,
        render_doc_structure,
        add_new_chapter,
        add_new_section,
        add_new_paragraph,
        organize_docs_by_chapter,
    )

    st.header(localization["doc_editor_title"])
    st.markdown(localization["doc_editor_subtitle"])

    # ---- Load current document ----
    docs = load_current_docs()
    if not docs:
        st.error("No document loaded. Please run `parse_doc.py` first.")
    else:
        # ---- Add content section ----
        st.subheader(localization["add_content_header"])
        add_type = st.radio(
            localization["add_content_question"],
            [localization["add_chapter_label"], localization["add_section_label"], localization["add_paragraph_label"]],
            horizontal=True
        )

        if add_type == localization["add_chapter_label"]:
            new_chapter = st.text_input(
                localization["add_chapter_title"],
                placeholder=localization["add_chapter_placeholder"]
            )
            if st.button(localization["add_chapter_button"], width="stretch"):
                if new_chapter.strip():
                    docs = add_new_chapter(docs, new_chapter)
                    save_docs_to_json(docs)
                    st.success(f"{localization['add_success']} {new_chapter}")
                    st.rerun()
                else:
                    st.error(localization["add_error"])

        elif add_type == localization["add_section_label"]:
            chapters = organize_docs_by_chapter(docs)
            chapter_list = list(chapters.keys())
            selected_chapter = st.selectbox(localization["select_chapter"], chapter_list)
            new_section = st.text_input(
                localization["add_section_title"],
                placeholder=localization["add_section_placeholder"]
            )
            if st.button(localization["add_section_button"], width="stretch"):
                if new_section.strip():
                    docs = add_new_section(docs, selected_chapter, new_section)
                    save_docs_to_json(docs)
                    st.success(f"{localization['add_success']} {new_section}")
                    st.rerun()
                else:
                    st.error(localization["add_error"])

        elif add_type == localization["add_paragraph_label"]:
            chapters = organize_docs_by_chapter(docs)
            chapter_list = list(chapters.keys())
            selected_chapter = st.selectbox(localization["select_chapter"], chapter_list, key="para_chapter")
            section_list = list(chapters.get(selected_chapter, {}).keys())
            selected_section = st.selectbox(localization["select_section"], section_list, key="para_section")

            col1, col2 = st.columns([3, 2])
            with col1:
                para_type = st.radio(
                    localization["add_paragraph_type"],
                    [localization["add_paragraph_instruction"], localization["add_paragraph_warning"]],
                    horizontal=True,
                    key="para_type"
                )
            with col2:
                st.markdown("**" + localization["upload_images_button"] + "**")
                uploaded_files = st.file_uploader(
                    localization["upload_images_button"],
                    type=["png", "jpg", "jpeg", "gif"],
                    accept_multiple_files=True,
                    label_visibility="collapsed"
                )
                if uploaded_files:
                    from pathlib import Path
                    Path("images").mkdir(exist_ok=True)
                    for uploaded_file in uploaded_files:
                        save_path = Path("images") / uploaded_file.name
                        save_path.write_bytes(uploaded_file.getbuffer())
                    st.success(f"Uploaded {len(uploaded_files)} image(s)")
                    st.rerun()

            new_paragraph = st.text_area(
                localization["add_paragraph_content"],
                placeholder="Type the content here..."
            )
            if st.button(localization["add_paragraph_button"], width="stretch"):
                if new_paragraph.strip():
                    docs = add_new_paragraph(
                        docs, selected_chapter, selected_section, new_paragraph,
                        para_type.lower() if para_type == localization["add_paragraph_instruction"] else "warning"
                    )
                    save_docs_to_json(docs)
                    st.success(f"{localization['add_success']}")
                    st.rerun()
                else:
                    st.error(localization["add_error"])

        st.divider()

        # ---- Document structure preview ----
        st.subheader(localization["doc_structure"])
        render_doc_structure(docs)

        # ---- Image manager ----
        st.subheader("🖼️ " + localization["image_management"])
        images = list_images()
        if images:
            st.markdown(f"**{len(images)} images**")
            cols = st.columns(5)
            for idx, img in enumerate(images[:5]):
                with cols[idx % 5]:
                    st.markdown(f"- {img}")
            if len(images) > 5:
                st.markdown(f"- ... and {len(images) - 5} more")
        else:
            st.info("No images yet.")

        # ---- Rebuild buttons ----
        st.subheader("🔄 Rebuild Index")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Re-parse Document", width="stretch"):
                with st.spinner("Re-parsing document..."):
                    success, output = rebuild_chunks()
                    if success:
                        st.success("✅ Document re-parsed successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Parse failed:\n{output}")

        with col2:
            if st.button("🔄 Rebuild Search Index", width="stretch"):
                with st.spinner("Rebuilding search index..."):
                    success, output = rebuild_index()
                    if success:
                        st.success("✅ Search index rebuilt successfully!")
                        # Reload the models so new index is picked up
                        st.rerun()
                    else:
                        st.error(f"❌ Rebuild failed:\n{output}")

        # ---- Full rebuild ----
        st.markdown("---")
        if st.button("🔄 Full Rebuild (Parse + Index)", width="stretch", type="primary"):
            with st.spinner("Parsing document..."):
                parse_ok, parse_out = rebuild_chunks()
            if not parse_ok:
                st.error(f"Parse failed: {parse_out}")
            else:
                with st.spinner("Building search index..."):
                    build_ok, build_out = rebuild_index()
                if build_ok:
                    st.success("✅ Full rebuild complete! Restart the app to use the updated index.")
                    st.rerun()
                else:
                    st.error(f"Index build failed: {build_out}")
