from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
CKPT_ROOT = ROOT / "ckpt"

REQUIRED = {
    "humanparsing/parsing_atr.onnx": {
        "repo_id": "yisol/IDM-VTON",
        "repo_type": "space",
        "filename": "ckpt/humanparsing/parsing_atr.onnx",
    },
    "humanparsing/parsing_lip.onnx": {
        "repo_id": "yisol/IDM-VTON",
        "repo_type": "space",
        "filename": "ckpt/humanparsing/parsing_lip.onnx",
    },
    "densepose/model_final_162be9.pkl": {
        "repo_id": "yisol/IDM-VTON",
        "repo_type": "space",
        "filename": "ckpt/densepose/model_final_162be9.pkl",
    },
    "openpose/ckpts/body_pose_model.pth": {
        "repo_id": "lllyasviel/Annotators",
        "repo_type": "model",
        "filename": "body_pose_model.pth",
    },
}


def ensure_ckpts() -> None:
    for rel, spec in REQUIRED.items():
        target = CKPT_ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"OK   {target}")
            continue

        print(f"DOWN {target}")
        downloaded = hf_hub_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            filename=spec["filename"],
        )
        shutil.copy2(downloaded, target)
        print(f"DONE {target}")


if __name__ == "__main__":
    ensure_ckpts()
    print("All required checkpoints are present.")
