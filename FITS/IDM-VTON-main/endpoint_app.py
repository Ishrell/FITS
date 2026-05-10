from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

from fits_single_infer import run_single_tryon


ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = Path(os.getenv("ENDPOINT_RUNTIME_ROOT", str(ROOT_DIR / "endpoint_runtime"))).resolve()
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

QUALITY_PRESETS = {
    "HQ (recommended)": {"steps": 30, "width": 768, "height": 1024},
    "Preview (faster)": {"steps": 12, "width": 576, "height": 768},
}


def _preset_values(profile: str) -> dict[str, int]:
    return QUALITY_PRESETS.get(profile, QUALITY_PRESETS["HQ (recommended)"])


def tryon(person_path: str, cloth_path: str, profile: str) -> Image.Image:
    if not person_path:
        raise gr.Error("Please upload a person image.")
    if not cloth_path:
        raise gr.Error("Please upload a garment image.")

    settings = _preset_values(profile)
    req_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = RUNTIME_ROOT / f"{req_id}.jpg"

    run_single_tryon(
        person_path=person_path,
        cloth_path=cloth_path,
        out_path=str(out_path),
        steps=settings["steps"],
        seed=42,
        garment_desc="upper body garment",
        width=settings["width"],
        height=settings["height"],
    )

    if not out_path.is_file():
        raise gr.Error("The endpoint finished without producing an output image.")

    return Image.open(out_path).convert("RGB")


with gr.Blocks(title="FITS IDM-VTON Endpoint") as demo:
    gr.Markdown("# FITS IDM-VTON Endpoint")
    gr.Markdown("Upload a person photo and a garment image. HQ is the default; Preview is available for quick checks.")

    with gr.Row():
        person = gr.Image(type="filepath", label="Person photo")
        cloth = gr.Image(type="filepath", label="Garment image")

    profile = gr.Radio(
        choices=list(QUALITY_PRESETS.keys()),
        value="HQ (recommended)",
        label="Quality profile",
    )

    run_btn = gr.Button("Generate try-on", variant="primary")
    output = gr.Image(type="pil", label="Result")

    run_btn.click(
        fn=tryon,
        inputs=[person, cloth, profile],
        outputs=output,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=port)