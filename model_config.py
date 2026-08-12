"""Shared local-model catalog, used by both app.py and download_model.py.

All models are GGUF quantizations of Qwen2.5-Instruct, pulled once from
Hugging Face (public repos, no API key) and then run fully offline.
"""
import os
from pathlib import Path

# China / restricted-network convenience: HF_MIRROR=1 routes downloads through
# hf-mirror.com (a full Hugging Face mirror). Set before huggingface_hub loads.
if os.environ.get("HF_MIRROR") and not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# ---- Embedding model (local sentence-transformers) ----
# The folder is named "e5-small" but the weights are multilingual-e5-small
# (good for the Chinese manual). model/ is gitignored, so this may be missing or
# truncated on a fresh machine — hence the download-on-demand fallback below.
EMBED_DIR = MODEL_DIR / "e5-small"
EMBED_REPO = "intfloat/multilingual-e5-small"


def download_embed_model():
    """(Re)download the embedding weights into EMBED_DIR. Skips the heavy
    onnx/openvino variants. Resumable; overwrites a corrupt local copy."""
    from huggingface_hub import snapshot_download

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=EMBED_REPO,
        local_dir=str(EMBED_DIR),
        ignore_patterns=["onnx/*", "*.onnx", "openvino/*", "*.md", ".gitattributes"],
    )


def load_embed_model():
    """Load the local embedding model, downloading it first if the folder is
    missing or the weights are corrupt/truncated (torch raises EOFError)."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(str(EMBED_DIR))
    except Exception:
        download_embed_model()
        return SentenceTransformer(str(EMBED_DIR))

# Split GGUF files are listed in load order — llama.cpp auto-loads the rest
# once pointed at the first shard.
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

# Display order + approximate on-disk sizes for the UI.
TIER_ORDER = ["fast", "quality", "powerful"]
TIER_SIZE = {"fast": "~2GB", "quality": "~4.7GB", "powerful": "~9GB"}


def tier_ready(tier):
    """True if every shard for this tier is already on disk."""
    return all((MODEL_DIR / f).exists() for f in MODEL_TIERS[tier]["files"])


def model_path(tier):
    """Path to the first shard (what llama.cpp is pointed at)."""
    return MODEL_DIR / MODEL_TIERS[tier]["files"][0]


def download_tier(tier):
    """Download every shard for a tier into MODEL_DIR. Resumable; returns the
    first-shard path. Safe to re-run — existing files are reused."""
    from huggingface_hub import hf_hub_download

    cfg = MODEL_TIERS[tier]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fn in cfg["files"]:
        if not (MODEL_DIR / fn).exists():
            hf_hub_download(repo_id=cfg["repo"], filename=fn, local_dir=str(MODEL_DIR))
    return model_path(tier)
