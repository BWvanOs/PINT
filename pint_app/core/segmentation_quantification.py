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
        ch if ch.isalnum() or ch in ("_", "-", ".", "(", ")") else "_"
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
    Quantify all image channels per segmentation label.

    image_stack must be shaped (channels, y, x).
    mask must be shaped (y, x), with 0 as background.

    The mask is indexed once up front. Geometry and channel statistics
    are then calculated from that shared cell-to-pixel mapping rather
    than repeatedly scanning the complete mask for every cell.
    """

    if image_stack.ndim != 3:
        raise ValueError(
            f"Expected image_stack shape (C, Y, X), "
            f"got {image_stack.shape}"
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got {mask.shape}"
        )

    n_channels, img_h, img_w = image_stack.shape

    if mask.shape != (img_h, img_w):
        raise ValueError(
            f"Image and mask dimensions differ for {sample_name}: "
            f"image={(img_h, img_w)}, mask={mask.shape}"
        )

    if len(channel_names) != n_channels:
        raise ValueError(
            f"Number of channel names ({len(channel_names)}) "
            f"does not match image channels ({n_channels})"
        )

    # --------------------------------------------------------
    # Build the cell-to-pixel mapping once
    # --------------------------------------------------------

    flat_mask = np.asarray(
        mask
    ).ravel()

    foreground_positions = np.flatnonzero(
        flat_mask > 0
    )

    if foreground_positions.size == 0:
        return pd.DataFrame()

    foreground_labels = flat_mask[
        foreground_positions
    ]

    # Convert arbitrary mask labels to compact indices:
    #
    # original labels:
    #   1, 2, 7, 19
    #
    # compact indices:
    #   0, 1, 2, 3
    #
    # This means bincount does not allocate up to the largest
    # possible label number.
    labels, inverse = np.unique(
        foreground_labels,
        return_inverse=True,
    )

    n_objects = len(labels)

    # Number of mask pixels belonging to every cell.
    areas = np.bincount(
        inverse,
        minlength=n_objects,
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    # Convert flattened pixel positions back to X/Y coordinates.
    ys = foreground_positions // img_w
    xs = foreground_positions % img_w

    x_sums = np.bincount(
        inverse,
        weights=xs,
        minlength=n_objects,
    )

    y_sums = np.bincount(
        inverse,
        weights=ys,
        minlength=n_objects,
    )

    center_x = x_sums / areas
    center_y = y_sums / areas

    # --------------------------------------------------------
    # Prepare shared grouping for exact medians
    # --------------------------------------------------------

    # Sorting the compact cell indices groups all pixels belonging
    # to the same cell together.
    #
    # Importantly, this sort depends only on the mask, so it is
    # calculated once and reused for every image channel.
    if include_median:
        pixel_order = np.argsort(
            inverse,
            kind="stable",
        )

        group_ends = np.cumsum(
            areas
        )

        group_starts = (
            group_ends - areas
        )

    # --------------------------------------------------------
    # Create the cell table
    # --------------------------------------------------------

    result_columns = {
        "SampleName": np.repeat(
            sample_name,
            n_objects,
        ),
        "ROIName": np.repeat(
            sample_name,
            n_objects,
        ),
        "CellMaskName": np.repeat(
            mask_name or sample_name,
            n_objects,
        ),
        "ObjectNumber": labels.astype(
            np.int64
        ),
        "Location_Center_X": center_x,
        "Location_Center_Y": center_y,
        "Area": areas.astype(
            np.int64
        ),
    }

    # --------------------------------------------------------
    # Quantify every channel
    # --------------------------------------------------------

    for channel_index, channel_name in enumerate(
        channel_names
    ):
        prefix = _safe_channel_name(
            channel_name
        )

        flat_image = np.asarray(
            image_stack[channel_index],
            dtype=np.float32,
        ).ravel()

        # Only retrieve image pixels that actually belong to cells.
        values = flat_image[
            foreground_positions
        ]

        # Preserve the old behaviour:
        # NaN/inf pixels are ignored for intensity statistics.
        finite = np.isfinite(
            values
        )

        finite_counts = np.bincount(
            inverse[finite],
            minlength=n_objects,
        )

        sums = np.bincount(
            inverse[finite],
            weights=values[finite],
            minlength=n_objects,
        )

        means = np.full(
            n_objects,
            np.nan,
            dtype=np.float64,
        )

        np.divide(
            sums,
            finite_counts,
            out=means,
            where=finite_counts > 0,
        )

        result_columns[
            f"{prefix}_mean"
        ] = means

        if include_sum:
            channel_sums = sums.astype(
                np.float64,
                copy=True,
            )

            channel_sums[
                finite_counts == 0
            ] = np.nan

            result_columns[
                f"{prefix}_sum"
            ] = channel_sums

        # ----------------------------------------------------
        # Exact medians
        # ----------------------------------------------------

        if include_median:
            sorted_values = values[
                pixel_order
            ]

            medians = np.full(
                n_objects,
                np.nan,
                dtype=np.float64,
            )

            for object_index in range(
                n_objects
            ):
                start = int(
                    group_starts[
                        object_index
                    ]
                )

                end = int(
                    group_ends[
                        object_index
                    ]
                )

                object_values = (
                    sorted_values[
                        start:end
                    ]
                )

                object_values = (
                    object_values[
                        np.isfinite(
                            object_values
                        )
                    ]
                )

                if object_values.size:
                    medians[
                        object_index
                    ] = np.median(
                        object_values
                    )

            result_columns[
                f"{prefix}_median"
            ] = medians

    return pd.DataFrame(result_columns)


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
        mask_path = (mask_folder / f"{safe_stem}_mesmer_mask_uint32.tiff")

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