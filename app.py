import os
import re

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
DOC_STORE_PATH = BASE_DIR / "manual_docs.json"
IMAGE_DIR = BASE_DIR / "images"
EMBED_PATH = BASE_DIR / "model" / "e5-small"
MODEL_PATH = BASE_DIR / "model" / "Qwen2.5-7B-Instruct-Q5_K_M.gguf"
I18N_PATH = BASE_DIR / "i18n.json"

# ---- Load FAISS index and docs ----
index = faiss.read_index(str(INDEX_PATH))
with open(DOC_STORE_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

# ---- Load localization file ----
with open(I18N_PATH, "r", encoding="utf-8") as f:
    i18n = json.load(f)


@st.cache_resource
def load_llm(model_path):
    return Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=max(8, os.cpu_count()),
        n_batch=512
    )


@st.cache_resource
def load_embed_model():
    return SentenceTransformer(str(EMBED_PATH))


# ---- Local LLM ----
llm = load_llm(MODEL_PATH)
embed_model = load_embed_model()

# ---- Initialize conversation memory & busy flag ----
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    st.session_state.messages = []


# ---- Embedding ----
def embed(text):
    text = f"query: {text}"
    emb = embed_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
    return emb.astype("float32")


def clean_chunk_text(text):
    # Remove system / inst tokens
    text = re.sub(r"<</?SYS>>", "", text)
    text = re.sub(r"<\[/INST\]>", "", text)
    text = re.sub(r"\[INST\]", "", text)
    # Optionally remove extra whitespace
    text = text.strip()
    return text


# ---- Retrieval ----
def retrieve(query, top_k=3):
    q_emb = np.array([embed(query)], dtype="float32")
    D, I = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if score > 0.4:
            results.append((score, docs[idx]))

    return results


def chunk_to_text(chunks):
    parts = []
    if "section_title" in chunks and chunks["section_title"]:
        parts.append(chunks["section_title"])
    if "instructions" in chunks:
        parts.extend(chunks["instructions"])
    if "warnings" in chunks:
        parts.extend([f"⚠ {w}" for w in chunks["warnings"]])
    return "\n".join(parts)


# ---- Build GPT-style messages ----
def build_messages(user_question, retrieved_docs):
    prompt_message = [{
        "role": "system",
        "content": localization["system_prompt"]
    }]

    context = ""
    for d in retrieved_docs:
        context += f"Section: {d['metadata']['path']}\n"
        if d.get("instructions"):
            context += "Instructions:\n" + "\n".join(d["instructions"]) + "\n"
        if d.get("warnings"):
            context += "Warnings:\n" + "\n".join(d["warnings"]) + "\n"
        context += "\n"

    prompt_message.append({
        "role": "user",
        "content": localization["sources_label"] + context +
                   localization["question"] + user_question + localization["question_prompt"]
    })
    return prompt_message


# ---- Streamlit UI ----
st.title(localization["title"])

# Display chat
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["question"])

    with st.chat_message("assistant"):
        st.markdown(msg["answer"])

        if msg["reference_images"]:
            st.markdown("**" + localization["sources_label"] + ":**")
            for img_path in msg["reference_images"]:
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

            if not retrieved:
                answer_text = localization["not_found_prompt"]

            else:
                top_score, top_chunk = retrieved[0]

                if top_score > 0.6 and len(retrieved) == 1:
                    if top_chunk.get("instructions"):
                        answer_text = "\n".join(top_chunk["instructions"])
                    elif top_chunk.get("warnings"):
                        answer_text = "\n".join(top_chunk["warnings"])
                    else:
                        answer_text = localization["not_found_prompt"]

                else:
                    # Use LLM extraction mode
                    chunks_only = [d for _, d in retrieved]
                    messages = build_messages(question, chunks_only)

                    stream = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=300,
                        temperature=0.0,
                        top_k=1,
                        top_p=0.8,
                        stream=True
                    )

                    answer_text = ""
                    placeholder = st.empty()
                    for chunk in stream:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            answer_text += delta["content"]
                            placeholder.markdown(answer_text)

    # Show sources
    image_reference = []
    for score, d in retrieved:
        st.markdown(localization["sources_label"] + ":" + f"{d['metadata']['path']}")
        for img in d["metadata"]["images"]:
            img_path = IMAGE_DIR / img
            if img_path.exists():
                st.image(str(img_path))
                image_reference.append(img_path)

    # Save history after Q & A
    st.session_state.messages.append({
        "question": question,
        "answer": clean_chunk_text(answer_text),
        "reference_images": image_reference
    })
    st.rerun()
