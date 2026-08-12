import json
import faiss
from pathlib import Path

import numpy as np

# -------- CONFIG --------
BASE_DIR = Path(__file__).resolve().parent
JSON_PATHS = [
    BASE_DIR / "chunk" / "error-diagnose-and-maintenance-instruction-chunk.json",
    BASE_DIR / "chunk" / "heating-system-checking-instruction-chunk.json"
]
INDEX_PATH = BASE_DIR / "manual.index"
DOC_STORE_PATH = BASE_DIR / "manual_docs.json"
VECTORS_PATH = BASE_DIR / "manual_vectors.npy"

def load_embed_model():
    # Delegates to the shared loader, which downloads the embedding weights if
    # the local copy is missing or corrupt. Imported lazily so this module stays
    # cheap to import until a build is actually required.
    from model_config import load_embed_model as _load
    return _load()


# ---- Flatten sections from JSON ----
def flatten_sections(data):
    docs = []

    def traverse(section, path):
        full_path = " > ".join(path + [section["section_title"]])

        docs.append({
            "section_title": section.get("section_title", ""),
            "instructions": section.get("instructions", []),
            "warnings": section.get("warnings", []),
            "visual_reference": section.get("visual_reference", {}),
            "metadata": {
                "path": full_path,
                "images": section.get("visual_reference", {}).get("image_ids", [])
            },
            "subsections": section.get("subsections", [])
        })

        for sub in section.get("subsections", []):
            traverse(sub, path + [section["section_title"]])

    for ch in data.get("chapters", []):
        for sec in ch.get("sections", []):
            traverse(sec, [ch.get("chapter_title", "")])

    return docs


# ---- Convert chunk to text for embedding ----
def chunk_to_text(chunk):
    parts = []
    if chunk.get("section_title"):
        parts.append(chunk["section_title"])
    if chunk.get("instructions"):
        parts.extend(chunk["instructions"])
    if chunk.get("warnings"):
        parts.extend([f"⚠ {w}" for w in chunk["warnings"]])
    return "\n".join(parts)


# ---- Embedding ----
def embed_texts(embed_model, texts):
    texts = [f"passage: {t}" for t in texts]
    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def build_index(embed_model=None, verbose=True):
    """Build the FAISS index + doc store from the chunk JSON files.

    Can be called from app.py (passing an already-loaded embedding model)
    or run standalone via ``python build_index.py``. Returns the index path.
    """
    if embed_model is None:
        embed_model = load_embed_model()

    all_docs = []
    for json_path in JSON_PATHS:
        if verbose:
            print(f"Loading {json_path.name} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_docs.extend(flatten_sections(data))

    # Use chunk_to_text to generate embedding input
    texts = [chunk_to_text(d) for d in all_docs]
    vectors = embed_texts(embed_model, texts)
    np.save(VECTORS_PATH, vectors)
    if verbose:
        print("✔ Vectors precomputed and saved.")

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))

    # Save the structured docs (with instructions/warnings/images)
    with open(DOC_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    if verbose:
        print("✔ Offline multi-manual index built successfully")
    return INDEX_PATH


if __name__ == "__main__":
    build_index()
