from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite


def _read_2d(path: str | Path) -> np.ndarray:
    arr = imread(str(path))

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Expected 2D image at {path}, got shape {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image at {path}, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / float(info.max)
    else:
        arr = arr.astype(np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mesmer on PINT nuclear/boundary inputs.")
    parser.add_argument("--nuclear", required=True, help="Path to 2D nuclear input TIFF.")
    parser.add_argument("--boundary", required=True, help="Path to 2D boundary/cytoplasm input TIFF.")
    parser.add_argument("--out-mask", required=True, help="Output label-mask TIFF.")
    parser.add_argument("--out-json", required=True, help="Output JSON summary.")
    parser.add_argument("--image-mpp", type=float, default=1.0, help="Microns per pixel. IMC often uses ~1.0.")
    parser.add_argument(
        "--compartment",
        default="whole-cell",
        choices=["whole-cell", "nuclear"],
        help="Mesmer compartment output.",
    )

    args = parser.parse_args()

    nuclear = _read_2d(args.nuclear)
    boundary = _read_2d(args.boundary)

    if nuclear.shape != boundary.shape:
        raise ValueError(
            f"Nuclear and boundary images must have same shape. "
            f"Got nuclear={nuclear.shape}, boundary={boundary.shape}"
        )

    from deepcell.applications import Mesmer

    app = Mesmer()

    # Mesmer expects batch, y, x, channels.
    x = np.stack([nuclear, boundary], axis=-1)
    x = x[None, ...].astype(np.float32)

    pred = app.predict(
        x,
        image_mpp=float(args.image_mpp),
        compartment=args.compartment,
    )

    mask = np.squeeze(pred)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D Mesmer mask after squeeze, got shape {mask.shape}")

    mask = mask.astype(np.uint32)

    out_mask = Path(args.out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)

    # Use uint32 to be safe. Later we can downcast to uint16 if max label <= 65535.
    imwrite(str(out_mask), mask, dtype=np.uint32)

    summary = {
        "status": "ok",
        "image_shape": list(mask.shape),
        "image_mpp": float(args.image_mpp),
        "compartment": args.compartment,
        "n_labels": int(mask.max()),
        "mask_dtype": str(mask.dtype),
        "out_mask": str(out_mask),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()