"""Pre-download local models from a normal terminal.

Run this instead of downloading inside the app when you want a progress bar and
the ability to cancel with Ctrl+C (downloads resume where they left off).

Usage:
    python download_model.py            # smallest model (fast / 3B)
    python download_model.py quality    # 7B
    python download_model.py powerful   # 14B
    python download_model.py all        # every tier
"""
import os
import sys

# China / restricted-network convenience: HF_MIRROR=1 routes downloads through
# hf-mirror.com. Must be set before huggingface_hub is imported.
if os.environ.get("HF_MIRROR") and not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from model_config import MODEL_TIERS, TIER_ORDER, TIER_SIZE, download_tier, tier_ready


def main(argv):
    tiers = argv or ["fast"]
    if tiers == ["all"]:
        tiers = list(TIER_ORDER)

    unknown = [t for t in tiers if t not in MODEL_TIERS]
    if unknown:
        print(f"Unknown tier(s): {', '.join(unknown)}")
        print(f"Choose from: {', '.join(TIER_ORDER)} (or 'all')")
        return 1

    for tier in tiers:
        if tier_ready(tier):
            print(f"✔ {tier} ({TIER_SIZE[tier]}) already downloaded — skipping.")
            continue
        print(f"↓ Downloading {tier} model ({TIER_SIZE[tier]})… press Ctrl+C to cancel.")
        download_tier(tier)
        print(f"✔ {tier} ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nCancelled. Re-run the same command to resume where it stopped.")
        raise SystemExit(130)
