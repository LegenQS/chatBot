# app_offline_multi.py
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
MODEL_PATH = BASE_DIR / "model" / "mpt-7b-instruct.Q5_0.gguf"

# ---- Load FAISS index and docs ----
index = faiss.read_index(str(INDEX_PATH))
with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

# ---- Local LLM ----
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048
)

# ---- Local embedding model ----
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text):
    return embed_model.encode([text], convert_to_numpy=True)[0]


# ---- Retrieval ----
def retrieve(query, top_k=3):
    q_emb = np.array([embed(query)], dtype="float32")
    D, I = index.search(q_emb, top_k)
    return [docs[i] for i in I[0]]


# ---- Generate answer using local LLM ----
def answer_question(question):
    retrieved_question = retrieve(question)
    context = "\n\n".join(d["content"] for d in retrieved_question)

    prompt = f"""
        你是一个工业操作手册问答助手。
        请根据以下文档内容回答用户问题，并使用中文回答。
        文档内容:
        {context}
    
        用户问题:
        {question}
    
        请用中文回答：
        """

    res = llm(prompt=prompt, max_tokens=400, stop=["\n\n"])
    answer_text = res["choices"][0]["text"].strip()
    return answer_text, retrieved_question


# ---- Streamlit UI ----
st.title("🛠 Multi-Manual Assistant (Fully Offline)")

question = st.text_input("Ask a question about the machine:")

if st.button("Submit") and question:
    answer, retrieved = answer_question(question)

    st.markdown("### 📖 Answer")
    st.write(answer)

    st.markdown("### 🔍 Source Sections")
    for d in retrieved:
        st.write(d["metadata"]["path"])

    st.markdown("### 🖼 Related Images")
    for d in retrieved:
        for img in d["metadata"]["images"]:
            img_path = IMAGE_DIR / img
            if img_path.exists():
                st.image(str(img_path))
