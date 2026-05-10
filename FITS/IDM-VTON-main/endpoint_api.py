from __future__ import annotations

import os
import traceback
from datetime import datetime
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from fits_single_infer import get_cached_runtime, run_single_tryon


ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = Path(os.getenv("ENDPOINT_RUNTIME_ROOT", str(ROOT_DIR / "endpoint_runtime"))).resolve()
INPUT_ROOT = RUNTIME_ROOT / "inputs"
OUTPUT_ROOT = RUNTIME_ROOT / "outputs"

INPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

APP = FastAPI(title="FITS IDM-VTON API", version="1.0.0")
INFER_LOCK = Lock()

QUALITY_PRESETS = {
    "hq": {"steps": 30, "width": 768, "height": 1024},
    "preview": {"steps": 12, "width": 576, "height": 768},
}

REQUIRED_CKPTS = [
    ROOT_DIR / "ckpt" / "humanparsing" / "parsing_atr.onnx",
    ROOT_DIR / "ckpt" / "humanparsing" / "parsing_lip.onnx",
    ROOT_DIR / "ckpt" / "densepose" / "model_final_162be9.pkl",
    ROOT_DIR / "ckpt" / "openpose" / "ckpts" / "body_pose_model.pth",
]


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _preset_values(profile: str) -> dict[str, int]:
    return QUALITY_PRESETS.get(profile.lower(), QUALITY_PRESETS["hq"])


def _missing_ckpts() -> list[str]:
    missing: list[str] = []
    for p in REQUIRED_CKPTS:
        if not p.is_file():
            missing.append(str(p))
    return missing


def _warm_runtime() -> None:
    has_cuda = torch.cuda.is_available()
    device = "cuda:0" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32
    get_cached_runtime(device, dtype)


@APP.on_event("startup")
def on_startup() -> None:
    if os.getenv("IDMVTON_WARMUP_ON_START", "0") == "1":
        try:
            _warm_runtime()
        except Exception:
            traceback.print_exc()


@APP.get("/healthz")
def healthz() -> dict[str, str]:
    missing = _missing_ckpts()
    return {
        "status": "ok" if not missing else "degraded",
        "missing_ckpts": "" if not missing else " | ".join(missing),
    }


@APP.post("/warmup")
def warmup() -> dict[str, str]:
    try:
        _warm_runtime()
        return {"status": "ok", "message": "runtime loaded"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {type(exc).__name__}: {exc}") from exc


@APP.post("/tryon")
async def tryon(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    profile: str = Form("hq"),
    garment_desc: str = Form("upper body garment"),
    seed: int = Form(42),
) -> FileResponse:
    missing = _missing_ckpts()
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Required checkpoints are missing. Run `python download_ckpts.py` first. Missing: "
                + ", ".join(missing)
            ),
        )

    req_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    person_ext = Path(person.filename or "person.jpg").suffix or ".jpg"
    garment_ext = Path(garment.filename or "garment.jpg").suffix or ".jpg"

    person_path = INPUT_ROOT / f"{req_id}_person{person_ext}"
    garment_path = INPUT_ROOT / f"{req_id}_garment{garment_ext}"
    output_path = OUTPUT_ROOT / f"{req_id}_result.jpg"

    try:
        person_bytes = await person.read()
        garment_bytes = await garment.read()

        if not person_bytes:
            raise HTTPException(status_code=400, detail="Person image is empty.")
        if not garment_bytes:
            raise HTTPException(status_code=400, detail="Garment image is empty.")

        person_path.write_bytes(person_bytes)
        garment_path.write_bytes(garment_bytes)

        settings = _preset_values(profile)

        with INFER_LOCK:
            run_single_tryon(
                person_path=str(person_path),
                cloth_path=str(garment_path),
                out_path=str(output_path),
                steps=settings["steps"],
                seed=seed,
                garment_desc=garment_desc,
                width=settings["width"],
                height=settings["height"],
            )

        if not output_path.is_file():
            raise HTTPException(status_code=500, detail="Inference finished without output image.")

        return FileResponse(path=str(output_path), media_type="image/jpeg", filename=output_path.name)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}: {exc}") from exc
    finally:
        _safe_unlink(person_path)
        _safe_unlink(garment_path)


@APP.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "FITS IDM-VTON API",
            "endpoints": {
                "health": "/healthz",
                "warmup": "/warmup",
                "tryon": "/tryon (POST multipart: person, garment, profile=hq|preview)",
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(APP, host="0.0.0.0", port=port, workers=1)
