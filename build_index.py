import json
import faiss
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
BASE_DIR = Path(__file__).resolve().parent
JSON_PATHS = [
    BASE_DIR / "chunk" / "error-diagnose-and-maintenance-instruction-chunk.json",
    BASE_DIR / "chunk" / "heating-system-checking-instruction-chunk.json"
]
INDEX_PATH = BASE_DIR / "manual.index"
DOCSTORE_PATH = BASE_DIR / "manual_docs.json"
VECTORS_PATH = BASE_DIR / "manual_vectors.npy"

# ---- Local embedding model ----
EMBED_PATH = BASE_DIR / "model" / "e5-small"

# load embeddings from local path
embed_model = SentenceTransformer(str(EMBED_PATH))


# ---- Flatten sections from JSON ----
def flatten_sections(data):
    docs = []

    def traverse(section, path):
        full_path = " > ".join(path + [section["section_title"]])

        content = "\n".join(section.get("instructions", []))
        warnings = "\n".join(section.get("warnings", []))

        full_text = f"""
            Section: {full_path}
            
            Instructions:
            {content}
            
            Warnings:
            {warnings}
            """

        docs.append({
            "content": full_text,
            "metadata": {
                "path": full_path,
                "images": section.get("visual_reference", {}).get("image_ids", [])
            }
        })

        for sub in section.get("subsections", []):
            traverse(sub, path + [section["section_title"]])

    for ch in data["chapters"]:
        for sec in ch.get("sections", []):
            traverse(sec, [ch["chapter_title"]])

    return docs


# ---- Embedding ----
def embed_texts(texts):
    texts = [f"passage: {t}" for t in texts]
    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def main():
    all_docs = []

    for json_path in JSON_PATHS:
        print(f"Loading {json_path.name} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = flatten_sections(data)
        all_docs.extend(docs)

    if VECTORS_PATH.exists():
        print("Override vectors...")

    texts = [d["content"] for d in all_docs]
    vectors = embed_texts(texts)
    np.save(VECTORS_PATH, vectors)
    print("✔ Vectors precomputed and saved.")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))

    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print("✔ Offline multi-manual index built successfully")


if __name__ == "__main__":
    main()
