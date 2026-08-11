import os
import re
import gc
import json
import shutil
import platform
from pathlib import Path

import streamlit as st
import numpy as np
import faiss
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

import build_index

# ---- PATHS ----
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "manual.index"
DOC_STORE_PATH = BASE_DIR / "manual_docs.json"
IMAGE_DIR = BASE_DIR / "images"
MODEL_DIR = BASE_DIR / "model"
EMBED_PATH = MODEL_DIR / "e5-small"
I18N_PATH = BASE_DIR / "i18n.json"

# ---- Local LLM (auto-downloaded once from Hugging Face, no API key needed) ----
# Two tiers. "quality" (7B) is the better answer quality and is fast on a GPU;
# "fast" (3B) keeps things snappy on CPU-only machines. Split GGUF files are
# listed in load order — llama.cpp auto-loads the remaining shards.
MODEL_TIERS = {
    "fast": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "files": ["qwen2.5-3b-instruct-q4_k_m.gguf"],
    },
    "quality": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "files": [
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        ],
    },
    "powerful": {
        "repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "files": [
            "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
            "qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf",
            "qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf",
        ],
    },
}

# Display order + short size hints for the sidebar model picker.
TIER_ORDER = ["fast", "quality", "powerful"]


def has_gpu():
    """Best-effort detector for an available GPU accelerator."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return True  # Apple Silicon (Metal)
    return shutil.which("nvidia-smi") is not None  # NVIDIA (CUDA)


def resolve_tier():
    # Default tier: env override wins, else pick by hardware. The sidebar
    # picker can still change it at runtime.
    tier = os.environ.get("MODEL_TIER", "auto").lower()
    if tier not in MODEL_TIERS:
        tier = "quality" if has_gpu() else "fast"
    return tier

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

# ---- Load localization file ----
with open(I18N_PATH, "r", encoding="utf-8") as f:
    i18n = json.load(f)


# ---- Ensure the LLM weights exist locally (download once from Hugging Face) ----
def ensure_model(cfg):
    from huggingface_hub import hf_hub_download
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fn in cfg["files"]:
        if not (MODEL_DIR / fn).exists():
            with st.spinner(f"First run: downloading local model file {fn} (one time only)…"):
                hf_hub_download(repo_id=cfg["repo"], filename=fn, local_dir=str(MODEL_DIR))
    # Point llama.cpp at the first shard; it auto-loads any remaining shards.
    return MODEL_DIR / cfg["files"][0]


def _build_llm(tier):
    model_path = ensure_model(MODEL_TIERS[tier])
    # n_gpu_layers=-1 offloads everything to the GPU (CUDA on Windows/Linux,
    # Metal on Apple Silicon). Harmless on CPU-only builds — it just stays on CPU.
    return Llama(
        model_path=str(model_path),
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
    return SentenceTransformer(str(EMBED_PATH))


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

# ---- Sidebar (must run before models load, so the chosen tier is known) ----
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

    selected_tier = st.selectbox(
        localization["model_label"],
        TIER_ORDER,
        index=TIER_ORDER.index(resolve_tier()),
        format_func=lambda t: localization[f"tier_{t}"],
        key="model_tier",
        help=localization["model_help"],
    )

    if st.button(localization["remove_cache"], use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.divider()
    st.caption(f"⚙️ {selected_tier} · {'GPU' if has_gpu() else 'CPU'}")

# ---- Load models (after the tier is known) ----
with st.spinner("Warming up the assistant…"):
    embed_model = load_embed_model()
    index, docs = load_index(embed_model)
    llm = load_llm(selected_tier)


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
def render_sources(sources):
    if not sources:
        return
    with st.expander(localization["view_sources"]):
        for s in sources:
            st.markdown(f"**{s['path']}**")
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
        {"path": d["metadata"]["path"], "images": d["metadata"].get("images", [])}
        for _, d in retrieved
    ]
    return clean_chunk_text(answer_text), sources


# ---- Header ----
st.title(localization["title"])
st.caption(localization["subtitle"])

# ---- Chat history ----
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
        if col.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

# ---- Input ----
typed = st.chat_input(localization["chat_input_placeholder"])
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
