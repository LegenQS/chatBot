import os

import streamlit as st
import numpy as np
import faiss
import json
from pathlib import Path
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ---- PATHS ----
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "manual.index"
DOCSTORE_PATH = BASE_DIR / "manual_docs.json"
IMAGE_DIR = BASE_DIR / "images"
EMBED_PATH = BASE_DIR / "model" / "e5-small"
MODEL_PATH = BASE_DIR / "model" / "mpt-7b-instruct.Q5_0.gguf"
I18N_PATH = BASE_DIR / "i18n.json"

# ---- Load FAISS index and docs ----
index = faiss.read_index(str(INDEX_PATH))
with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)


# ---- Load localization file ----
with open(I18N_PATH, "r", encoding="utf-8") as f:
    i18n = json.load(f)


@st.cache_resource
def load_llm(model_path):
    return Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=os.cpu_count() - 2
    )


@st.cache_resource
def load_embed_model():
    return SentenceTransformer(str(EMBED_PATH))


# ---- Local LLM ----
llm = load_llm(MODEL_PATH)
embed_model = load_embed_model()

# ---- Initialize conversation memory & busy flag ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # tuples: (role, message)
if "image_reference" not in st.session_state:
    st.session_state.image_reference = []
if "busy" not in st.session_state:
    st.session_state.busy = False

# ---- Language selection in sidebar ----
with st.sidebar:
    st.header(i18n["en"]["settings_header"])
    lang_choice = st.radio(
        i18n["en"]["settings_header"],
        ["中文", "English"],
        index=0,
        key="sidebar_lang"
    )
    st.session_state.lang = "zh" if lang_choice == "中文" else "en"
    localization = i18n[st.session_state.lang]
    remove_cache = st.button(localization["remove_cache"])


if remove_cache:
    st.session_state.chat_history = []
    st.session_state.image_reference = []


# ---- Embedding ----
def embed(text):
    text = f"query: {text}"
    emb = embed_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    return emb.astype("float32")


# ---- Retrieval ----
def retrieve(query, top_k=2):
    q_emb = np.array([embed(query)], dtype="float32")
    D, I = index.search(q_emb, top_k)
    return [docs[i] for i in I[0]]


# ---- Build GPT-style messages ----
def build_messages(user_question, retrieved_docs):
    # --- system instruction ---
    prompt_message = [{"role": "system", "content": localization["system_prompt"]}]

    # --- add retrieved docs ---
    context = "\n\n".join(d["content"] for d in retrieved_docs)
    if context:
        prompt_message.append({
            "role": "system",
            "content": f"Reference material:\n{context}"
        })

    # --- add current question ---
    prompt_message.append({"role": "user", "content": user_question})

    return prompt_message


# ---- Streamlit UI ----
st.title(localization["title"])


# Display chat
for i, (role, content) in enumerate(st.session_state.chat_history):
    if role in ["用户", "user"]:
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)

    # display images if any
    if i < len(st.session_state.image_reference) and st.session_state.image_reference[i]:
        st.markdown("**" + localization["sources_label"] + ":**")
        for img_path in st.session_state.image_reference[i]:
            st.image(str(img_path))

# Input and submit
submit_disabled = st.session_state.busy
question = st.chat_input(localization["chat_input_placeholder"], disabled=submit_disabled)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(localization["thinking_spinner"]):

            retrieved = retrieve(question)
            messages = build_messages(question, retrieved)

            stop_tokens = ["User:", "Assistant:"] if st.session_state.lang == "en" else ["用户:", "助手:"]

            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=300,
                temperature=0.2,
                stop=stop_tokens,
                stream=True
            )

            # Streaming response
            answer_text = ""
            placeholder = st.empty()

            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    answer_text += delta["content"]
                    placeholder.markdown(answer_text)

        # Save history AFTER streaming
        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", answer_text))

    # Show sources
    image_reference = []
    for d in retrieved:
        st.markdown(localization["sources_label"] + ":" + f"{d['metadata']['path']}")
        for img in d["metadata"]["images"]:
            img_path = IMAGE_DIR / img
            if img_path.exists():
                st.image(str(img_path))
                image_reference.append(img_path)
    if image_reference:
        st.session_state.image_reference.append(image_reference)
