#!/usr/bin/env python
"""Long-lived IDM-VTON worker for FITS.

Reads JSON requests from stdin, runs a single try-on, and writes JSON responses
on stdout. This keeps the diffusion model warm between requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fits_single_infer import run_single_tryon


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle_request(req: dict) -> dict:
    person = str(req.get("person", ""))
    cloth = str(req.get("cloth", ""))
    output = str(req.get("output", ""))
    steps = int(req.get("steps", 30))
    seed = int(req.get("seed", 42))
    garment_desc = str(req.get("garment_desc", "upper body garment"))
    width = int(req.get("width", 768))
    height = int(req.get("height", 1024))

    if not person or not Path(person).is_file():
        return {"ok": False, "error": f"Person image not found: {person}"}
    if not cloth or not Path(cloth).is_file():
        return {"ok": False, "error": f"Garment image not found: {cloth}"}
    if not output:
        return {"ok": False, "error": "Output path is missing."}

    try:
        run_single_tryon(
            person_path=person,
            cloth_path=cloth,
            out_path=output,
            steps=steps,
            seed=seed,
            garment_desc=garment_desc,
            width=width,
            height=height,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out_path = Path(output)
    if not out_path.is_file():
        return {"ok": False, "error": "Worker finished without producing an output image."}

    return {"ok": True, "output": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="FITS IDM-VTON worker")
    parser.add_argument("--serve", action="store_true", help="Run as a long-lived stdin/stdout worker")
    args = parser.parse_args()

    if not args.serve:
        parser.error("Use --serve to run the worker.")

    _emit({"ok": True, "ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line in {"quit", "exit"}:
            _emit({"ok": True, "bye": True})
            return 0
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"Invalid JSON: {exc}"})
            continue

        resp = _handle_request(req)
        _emit(resp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
