#!/usr/bin/env python3
"""
Pre-download IDM-VTON models to local cache for offline inference on Runpod.

Run once before starting endpoint_app.py:
  python download_models.py
"""
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

if __name__ == "__main__":
    print("Downloading IDM-VTON model from HuggingFace Hub...")
    print("This may take 5-10 minutes depending on connection speed.\n")

    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    print(f"Cache directory: {cache_dir}\n")

    try:
        snapshot_download(
            "yisol/IDM-VTON",
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
        print("\n✓ Model downloaded successfully!")
        print("You can now run: python endpoint_app.py")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)
