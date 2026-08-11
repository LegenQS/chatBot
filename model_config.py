"""Shared local-model catalog, used by both app.py and download_model.py.

All models are GGUF quantizations of Qwen2.5-Instruct, pulled once from
Hugging Face (public repos, no API key) and then run fully offline.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

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
