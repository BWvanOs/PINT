
    
  
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

MODEL_NAME = "fluorescence_nuclei_and_cells"


def emit(event: str, **fields) -> None:
    """Machine-readable progress; safe to parse while the worker runs."""
    print("PINT_EVENT:" + json.dumps({"event": event, **fields}), flush=True)


def segment_one(model, job: dict, settings: dict) -> dict:
    input_path = Path(job["input"])
    out_nuclei = Path(job["out_nuclei"])
    out_cells = Path(job["out_cells"])
    out_json = Path(job["out_json"])
    name = str(job.get("sample", input_path.stem))

    image = imread(input_path).astype(np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected [C,Y,X] TIFF, got {image.shape}")
    if not np.isfinite(image).all():
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    prediction = model.eval_medium_image(
        image,
        pixel_size=settings["pixel_size"],
        tile_size=settings["tile_size"],
        batch_size=settings["batch_size"],
        target="all_outputs",
        normalise=True,
        return_image_tensor=False,
        rescale_output=True,
    )
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    prediction = np.asarray(prediction)
    if prediction.ndim == 4 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 3 or prediction.shape[0] != 2:
        raise ValueError(
            "Expected nuclear and whole-cell outputs with shape "
            f"[2,Y,X], got {prediction.shape}"
        )

    nuclei = prediction[0].astype(np.uint32)
    cells = prediction[1].astype(np.uint32)
    if nuclei.shape != image.shape[1:] or cells.shape != image.shape[1:]:
        raise ValueError("Output mask dimensions do not match input image")

    for path in (out_nuclei, out_cells, out_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_nuclei, nuclei)
    imwrite(out_cells, cells)

    summary = {
        "status": "ok",
        "sample": name,
        "model": MODEL_NAME,
        "image_shape": list(image.shape),
        "pixel_size": settings["pixel_size"],
        "device": str(model.inference_device),
        "n_nuclei": int(np.count_nonzero(np.unique(nuclei))),
        "n_cells": int(np.count_nonzero(np.unique(cells))),
        "nuclei_mask": str(out_nuclei),
        "cells_mask": str(out_cells),
        "mask_dtype": "uint32",
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", help="JSON batch manifest; model loaded once")
    parser.add_argument("--input")
    parser.add_argument("--out-nuclei")
    parser.add_argument("--out-cells")
    parser.add_argument("--out-json")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.pixel_size <= 0:
        parser.error("Pixel size must be positive")
    if args.tile_size <= 0 or args.batch_size <= 0:
        parser.error("Tile size and batch size must be positive")

    if args.manifest:
        data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        jobs = data["jobs"]
        if not isinstance(jobs, list) or not jobs:
            parser.error("Manifest must have a non-empty 'jobs' list")
    else:
        for key in ("input", "out_nuclei", "out_cells", "out_json"):
            if getattr(args, key) is None:
                parser.error(f"--{key.replace('_', '-')} is required without --manifest")
        jobs = [{
            "input": args.input,
            "out_nuclei": args.out_nuclei,
            "out_cells": args.out_cells,
            "out_json": args.out_json,
            "sample": Path(args.input).stem,
        }]

    settings = {
        "pixel_size": args.pixel_size,
        "tile_size": args.tile_size,
        "batch_size": args.batch_size,
    }

    from instanseg import InstanSeg
    device = None if args.device == "auto" else args.device
    emit("model_loading", total=len(jobs))
    model = InstanSeg(MODEL_NAME, device=device, verbosity=1)
    emit("model_ready", device=str(model.inference_device), total=len(jobs))

    failed = 0
    for index, job in enumerate(jobs, 1):
        sample = str(job.get("sample", Path(job["input"]).stem))
        emit("roi_started", index=index, total=len(jobs), sample=sample)
        try:
            result = segment_one(model, job, settings)
            emit("roi_completed", index=index, total=len(jobs), **result)
        except Exception as exc:
            failed += 1
            emit("roi_failed", index=index, total=len(jobs), sample=sample, error=str(exc))
            traceback.print_exc(file=sys.stderr)

    emit("batch_completed", total=len(jobs), succeeded=len(jobs)-failed, failed=failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

