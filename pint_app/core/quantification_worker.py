from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread

try:
    import pyarrow  # noqa: F401
except ImportError as e:
    raise ImportError(
        "PINT quantification requires pyarrow for "
        "temporary Parquet checkpoint storage. "
        "Install it with 'conda install -c conda-forge pyarrow' "
    ) from e

from pint_app.core.load_tiffs import (
    load_tiffs_raw,
)

from pint_app.core.segmentation_quantification import (
    quantify_mask_intensities,
)


# ============================================================
# Small helpers
# ============================================================

def _write_json_atomic(
    path: Path,
    payload: dict,
) -> None:
    """
    Write worker status atomically so the Shiny process
    never reads a half-written JSON file.
    """

    path = Path(path)

    tmpPath = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmpPath.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        tmpPath,
        path,
    )


def _safe_file_stem(
    name: str,
) -> str:
    """
    Convert an ROI/sample name into a safe filename while
    preserving useful characters such as parentheses.
    """

    return "".join(
        ch
        if ch.isalnum()
        or ch in (
            "_",
            "-",
            ".",
            "(",
            ")",
        )
        else "_"
        for ch in str(name)
    )


def _load_image(
    record: dict,
) -> tuple[np.ndarray, list[str]]:
    """
    Load one quantification image.

    Supported worker inputs:
    - TIFF / OME-TIFF from disk
    - staged NumPy arrays created by PINT
    """

    imagePath = Path(
        record["ImagePath"]
    )

    imageFormat = str(
        record.get(
            "ImageFormat",
            "tiff",
        )
    ).lower()

    # --------------------------------------------------------
    # Staged PINT image
    # --------------------------------------------------------

    if imageFormat == "npy":
        imageStack = np.load(
            imagePath,
            mmap_mode="r",
            allow_pickle=False,
        )

        channelNames = [
            str(value)
            for value in record[
                "Channels"
            ]
        ]

        return (
            imageStack,
            channelNames,
        )

    # --------------------------------------------------------
    # TIFF / OME-TIFF
    # --------------------------------------------------------

    if imageFormat == "tiff":
        loadedImages, loadedChannels = (
            load_tiffs_raw(
                str(
                    imagePath.parent
                ),
                validate_consistent=False,
                standardize_channel_names=True,
                file_paths=[
                    imagePath
                ],
            )
        )

        if len(loadedImages) != 1:
            raise ValueError(
                f"Expected one image from "
                f"{imagePath.name}, but loaded "
                f"{len(loadedImages)}."
            )

        loadedName = next(
            iter(
                loadedImages
            )
        )

        return (
            loadedImages[
                loadedName
            ],
            loadedChannels[
                loadedName
            ],
        )

    raise ValueError(
        f"Unsupported image format: "
        f"{imageFormat!r}"
    )


# ============================================================
# Final output creation
# ============================================================

def _combine_roi_parquets_to_csv(
    parquetPaths: list[Path],
    outputPath: Path,
) -> None:
    """
    Convert per-ROI Parquet checkpoints into one final CSV.

    Files are processed one ROI at a time so the complete
    quantification table never needs to exist in RAM at once.
    """

    outputPath = Path(
        outputPath
    )

    partialPath = outputPath.with_suffix(
        ".partial.csv"
    )

    if partialPath.exists():
        partialPath.unlink()

    headerWritten = False

    for parquetPath in parquetPaths:
        roiDf = pd.read_parquet(
            parquetPath
        )

        if roiDf.empty:
            continue

        roiDf.to_csv(
            partialPath,
            mode=(
                "a"
                if headerWritten
                else "w"
            ),
            header=(
                not headerWritten
            ),
            index=False,
        )

        headerWritten = True

        del roiDf

    # It is possible, though unusual, that every mask
    # contains zero cells.
    if not headerWritten:
        pd.DataFrame(
            columns=[
                "ROIName",
                "ObjectNumber",
                "Location_Center_X",
                "Location_Center_Y",
                "Area",
            ]
        ).to_csv(
            partialPath,
            index=False,
        )

    os.replace(
        partialPath,
        outputPath,
    )


# ============================================================
# Quantification job
# ============================================================

def run_job(
    jobPath: str | Path,
) -> None:

    jobPath = Path(
        jobPath
    )

    job = json.loads(
        jobPath.read_text(
            encoding="utf-8"
        )
    )

    records = job[
        "records"
    ]

    outputDir = Path(
        job["output_dir"]
    )

    outputDir.mkdir(
        parents=True,
        exist_ok=True,
    )

    statusPath = Path(
        job["status_path"]
    )

    finalCellPath = Path(
        job["cell_table_path"]
    )

    finalMaskPath = Path(
        job["mask_table_path"]
    )

    # --------------------------------------------------------
    # Temporary/checkpoint storage
    # --------------------------------------------------------

    # The viewer already creates a unique job directory.
    # Put ROI checkpoints beside job.json and status.json.
    tempDir = (
        jobPath.parent
        / "roi_parquet"
    )

    tempDir.mkdir(
        parents=True,
        exist_ok=True,
    )

    partialMaskPath = (
        finalMaskPath.with_suffix(
            ".partial.csv"
        )
    )

    if partialMaskPath.exists():
        partialMaskPath.unlink()

    # This run has its own unique work directory, but clear
    # stale Parquet files anyway in case the job is restarted.
    for parquetPath in tempDir.glob(
        "*.parquet"
    ):
        parquetPath.unlink()

    total = len(
        records
    )

    totalCells = 0
    completed = 0
    failed = 0

    maskRows = []
    parquetPaths: list[Path] = []

    # --------------------------------------------------------
    # Initial worker status
    # --------------------------------------------------------

    _write_json_atomic(
        statusPath,
        {
            "state": "running",
            "current": 0,
            "total": total,
            "completed": 0,
            "failed": 0,
            "cells": 0,
            "sample_name": "",
            "message": (
                "Starting quantification."
            ),
        },
    )

    # ========================================================
    # Quantify ROI by ROI
    # ========================================================

    for i, record in enumerate(
        records,
        start=1,
    ):
        sampleName = str(
            record["SampleName"]
        )

        maskPath = Path(
            record["MaskPath"]
        )

        _write_json_atomic(
            statusPath,
            {
                "state": "running",
                "current": i - 1,
                "total": total,
                "completed": completed,
                "failed": failed,
                "cells": totalCells,
                "sample_name": sampleName,
                "message": (
                    f"Quantifying "
                    f"{sampleName} "
                    f"({i}/{total})"
                ),
            },
        )

        print(
            f"Quantifying "
            f"{sampleName} "
            f"({i}/{total})",
            flush=True,
        )

        maskRow = {
            "ROIName": sampleName,
            "MaskFile": (
                maskPath.name
            ),
            "MaskPath": (
                str(maskPath)
            ),
            "MaskExists": (
                maskPath.exists()
            ),
            "NCells": 0,
            "Status": (
                "Not started"
            ),
            "Error": "",
        }

        # Initialize these so cleanup is simple and explicit.
        imageStack = None
        channelNames = None
        mask = None
        cellDf = None

        try:
            # ------------------------------------------------
            # Check mask
            # ------------------------------------------------

            if not maskPath.exists():
                raise FileNotFoundError(
                    f"Mask not found: "
                    f"{maskPath}"
                )

            # ------------------------------------------------
            # Load one ROI
            # ------------------------------------------------

            (
                imageStack,
                channelNames,
            ) = _load_image(
                record
            )

            mask = imread(
                str(maskPath)
            )

            # ------------------------------------------------
            # Quantify
            # ------------------------------------------------

            cellDf = (
                quantify_mask_intensities(
                    image_stack=imageStack,
                    channel_names=channelNames,
                    mask=mask,
                    sample_name=sampleName,
                )
            )

            nCells = int(
                len(cellDf)
            )

            maskRow[
                "NCells"
            ] = nCells

            maskRow[
                "Status"
            ] = "OK"

            totalCells += (
                nCells
            )

            completed += 1

            # ------------------------------------------------
            # Save one independent ROI checkpoint
            # ------------------------------------------------

            if not cellDf.empty:
                roiPath = (
                    tempDir
                    / (
                        f"{i:04d}_"
                        f"{_safe_file_stem(sampleName)}"
                        ".parquet"
                    )
                )

                cellDf.to_parquet(
                    roiPath,
                    index=False,
                    engine="pyarrow",
                )

                parquetPaths.append(
                    roiPath
                )

                print(
                    f"   ↳ Saved checkpoint: "
                    f"{roiPath.name} "
                    f"({nCells:,} cells)",
                    flush=True,
                )

            else:
                print(
                    f"   ↳ No cells found in "
                    f"{sampleName}.",
                    flush=True,
                )

        except Exception as e:
            failed += 1

            maskRow[
                "Status"
            ] = "Failed"

            maskRow[
                "Error"
            ] = str(e)

            print(
                f"❌ {sampleName}: "
                f"{e}",
                flush=True,
            )

        maskRows.append(
            maskRow
        )

        # ----------------------------------------------------
        # Explicitly release large per-ROI objects
        # ----------------------------------------------------

        del imageStack
        del channelNames
        del mask
        del cellDf

        # ----------------------------------------------------
        # Update progress/status
        # ----------------------------------------------------

        _write_json_atomic(
            statusPath,
            {
                "state": "running",
                "current": i,
                "total": total,
                "completed": completed,
                "failed": failed,
                "cells": totalCells,
                "sample_name": sampleName,
                "message": (
                    f"Finished "
                    f"{sampleName} "
                    f"({i}/{total})"
                ),
            },
        )

    # ========================================================
    # Finalize mask table
    # ========================================================

    _write_json_atomic(
        statusPath,
        {
            "state": "running",
            "current": total,
            "total": total,
            "completed": completed,
            "failed": failed,
            "cells": totalCells,
            "sample_name": "",
            "message": (
                "Finalizing quantification tables..."
            ),
        },
    )

    maskDf = pd.DataFrame(
        maskRows
    )

    maskDf.to_csv(
        partialMaskPath,
        index=False,
    )

    os.replace(
        partialMaskPath,
        finalMaskPath,
    )

    # ========================================================
    # Combine ROI Parquet checkpoints into final CSV
    # ========================================================

    _combine_roi_parquets_to_csv(
        parquetPaths,
        finalCellPath,
    )

    # ========================================================
    # Finished
    # ========================================================

    _write_json_atomic(
        statusPath,
        {
            "state": "complete",
            "current": total,
            "total": total,
            "completed": completed,
            "failed": failed,
            "cells": totalCells,
            "sample_name": "",
            "message": (
                f"Quantification complete: "
                f"{totalCells:,} cells from "
                f"{completed:,}/{total:,} "
                f"matched mask(s)."
            ),
        },
    )

    print(
        f"✅ Quantification complete: "
        f"{totalCells:,} cells from "
        f"{completed:,}/{total:,} "
        f"mask(s).",
        flush=True,
    )

    print(
        f"   Cell table: "
        f"{finalCellPath}",
        flush=True,
    )

    print(
        f"   Mask table: "
        f"{finalMaskPath}",
        flush=True,
    )

    print(
        f"   ROI checkpoints: "
        f"{tempDir}",
        flush=True,
    )


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "PINT segmentation "
            "quantification worker"
        )
    )

    parser.add_argument(
        "job",
        help=(
            "Path to quantification "
            "job JSON"
        ),
    )

    args = parser.parse_args()

    try:
        run_job(
            args.job
        )

    except Exception as e:
        traceback.print_exc()

        # ----------------------------------------------------
        # Best-effort failure status
        # ----------------------------------------------------

        try:
            jobPath = Path(
                args.job
            )

            job = json.loads(
                jobPath.read_text(
                    encoding="utf-8"
                )
            )

            statusPath = Path(
                job["status_path"]
            )

            _write_json_atomic(
                statusPath,
                {
                    "state": "failed",
                    "message": str(e),
                },
            )

        except Exception:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()