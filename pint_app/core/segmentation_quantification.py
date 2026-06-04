from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread


def _safe_channel_name(channel_name: str) -> str:
    out = "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
        for ch in str(channel_name)
    )
    out = out.strip("_")
    return out or "Channel"


def _safe_file_stem(name: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
        for ch in str(name)
    )


def quantify_mask_intensities(
    *,
    image_stack: np.ndarray,
    channel_names: list[str],
    mask: np.ndarray,
    sample_name: str,
    mask_name: str | None = None,
    include_median: bool = True,
    include_sum: bool = False,
) -> pd.DataFrame:
    """
    Quantify all image channels per Mesmer label.

    image_stack must be shaped (channels, y, x).
    mask must be shaped (y, x), with 0 as background.
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected image_stack shape (C, Y, X), got {image_stack.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")

    n_channels, img_h, img_w = image_stack.shape

    if mask.shape != (img_h, img_w):
        raise ValueError(
            f"Image and mask dimensions differ for {sample_name}: "
            f"image={(img_h, img_w)}, mask={mask.shape}"
        )

    if len(channel_names) != n_channels:
        raise ValueError(
            f"Number of channel names ({len(channel_names)}) does not match "
            f"image channels ({n_channels})"
        )

    mask = np.asarray(mask)
    labels = np.unique(mask)
    labels = labels[labels > 0]

    if labels.size == 0:
        return pd.DataFrame()

    flat_mask = mask.ravel()

    flat_channels = [
        np.asarray(image_stack[i], dtype=np.float32).ravel()
        for i in range(n_channels)
    ]

    rows = []

    for label in labels:
        pix = flat_mask == label
        area = int(pix.sum())

        if area == 0:
            continue

        ys, xs = np.nonzero(mask == label)

        row = {
            "SampleName": sample_name,
            "ROIName": sample_name,
            "CellMaskName": mask_name or sample_name,
            "ObjectNumber": int(label),
            "Location_Center_X": float(xs.mean()),
            "Location_Center_Y": float(ys.mean()),
            "Area": area,
            "Cluster": "Unclustered",
            "Condition": "Unassigned",
        }

        for channel_name, flat_img in zip(channel_names, flat_channels):
            prefix = _safe_channel_name(channel_name)
            vals = flat_img[pix]
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                row[f"{prefix}_mean"] = np.nan
                if include_median:
                    row[f"{prefix}_median"] = np.nan
                if include_sum:
                    row[f"{prefix}_sum"] = np.nan
                continue

            row[f"{prefix}_mean"] = float(np.mean(vals))

            if include_median:
                row[f"{prefix}_median"] = float(np.median(vals))

            if include_sum:
                row[f"{prefix}_sum"] = float(np.sum(vals))

        rows.append(row)

    return pd.DataFrame(rows)


def quantify_mesmer_masks_for_dataset(
    *,
    images: dict[str, np.ndarray],
    channels: dict[str, list[str]],
    mask_folder: str | Path,
    mask_suffix: str = "_mesmer_mask_uint32.tiff",
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quantify all saved Mesmer masks for a pushed PINT image dataset.

    Returns
    -------
    cell_table:
        One row per cell/object.

    mask_table:
        One row per ROI/mask file.
    """
    mask_folder = Path(mask_folder)

    all_cells = []
    mask_rows = []

    sample_names = list(images.keys())

    for i, sample_name in enumerate(sample_names, start=1):
        if progress is not None:
            progress(f"Quantifying {sample_name} ({i}/{len(sample_names)})")

        safe_stem = _safe_file_stem(sample_name)
        mask_path = mask_folder / f"{safe_stem}{mask_suffix}"

        mask_row = {
            "SampleName": sample_name,
            "ROIName": sample_name,
            "CellMaskName": sample_name,
            "MaskFile": mask_path.name,
            "MaskPath": str(mask_path),
            "MaskExists": mask_path.exists(),
            "NCells": 0,
            "Status": "Not started",
            "Error": "",
        }

        if not mask_path.exists():
            mask_row["Status"] = "Mask missing"
            mask_row["Error"] = f"Mask not found: {mask_path}"
            mask_rows.append(mask_row)
            continue

        try:
            mask = imread(str(mask_path))
            img_stack = images[sample_name]
            channel_names = channels.get(sample_name, [])

            cell_df = quantify_mask_intensities(
                image_stack=img_stack,
                channel_names=channel_names,
                mask=mask,
                sample_name=sample_name,
                mask_name=sample_name,
            )

            mask_row["NCells"] = int(len(cell_df))
            mask_row["Status"] = "OK"

            if not cell_df.empty:
                all_cells.append(cell_df)

        except Exception as e:
            mask_row["Status"] = "Failed"
            mask_row["Error"] = str(e)

        mask_rows.append(mask_row)

    cell_table = pd.concat(all_cells, ignore_index=True) if all_cells else pd.DataFrame()
    mask_table = pd.DataFrame(mask_rows)

    return cell_table, mask_table