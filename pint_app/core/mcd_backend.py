from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import numpy as np
import tifffile
import imageio

import re
import os
import warnings
from PIL import Image

try:
    from readimc import MCDFile
except ImportError:
    MCDFile = None

from pint_app.core.channel_names import (
    _make_unique_channel_names,
)

warnings.filterwarnings(
    "ignore",
    category=Image.DecompressionBombWarning,
    module=r"PIL\.Image",
)

ACQUISITION_COLUMNS = [
    "Slide index",
    "Slide ID",
    "Slide description",
    "Acquisition index",
    "Acquisition ID",
    "Acquisition description",
    "Width (px)",
    "Height (px)",
    "Width (µm)",
    "Height (µm)",
    "Channels",
    "Panoramas on slide",
]

PANORAMA_COLUMNS = [
    "Slide index",
    "Slide ID",
    "Slide description",
    "Panorama index",
    "Panorama ID",
    "Panorama description",
    "Width (µm)",
    "Height (µm)",
]


def _require_readimc() -> None:
    if MCDFile is None:
        raise RuntimeError(
            "MCD support requires the 'readimc' package. "
            "Install it by updating the PINT environment."
        )


def _clean_description(
    value: Any,
    fallback: str,
) -> str:
    """
    Convert optional MCD descriptions to readable strings.
    """
    if value is None:
        return fallback

    text = str(value).strip()

    if not text:
        return fallback

    return text


def _optional_number(
    obj: Any,
    attribute: str,
) -> int | float | None:
    """
    Safely retrieve an optional numeric MCD metadata attribute.

    Different MCD software versions may omit some metadata fields.
    """
    value = getattr(
        obj,
        attribute,
        None,
    )

    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric.is_integer():
        return int(numeric)

    return numeric


def _get_pixel_dimensions(
    acquisition: Any,
) -> tuple[int | None, int | None]:
    """
    Retrieve acquisition pixel dimensions when exposed by readimc.

    Property names have varied across MCD metadata versions, so this
    checks a small set of known/likely names without parsing pixel data.
    """
    widthCandidates = (
        "width_px",
        "width_pixels",
        "num_pixels_x",
    )

    heightCandidates = (
        "height_px",
        "height_pixels",
        "num_pixels_y",
    )

    width = None
    height = None

    for attribute in widthCandidates:
        width = _optional_number(
            acquisition,
            attribute,
        )

        if width is not None:
            break

    for attribute in heightCandidates:
        height = _optional_number(
            acquisition,
            attribute,
        )

        if height is not None:
            break

    return width, height


def inspect_mcd_file(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Inspect MCD metadata without loading acquisition or panorama pixels.

    Returns:
    - acquisition metadata table
    - panorama metadata table
    - file-level summary dictionary
    """
    _require_readimc()

    mcdPath = Path(path).expanduser().resolve()

    if not mcdPath.is_file():
        raise FileNotFoundError(
            f"MCD file does not exist: {mcdPath}"
        )

    if mcdPath.suffix.lower() != ".mcd":
        raise ValueError(
            "The selected file does not have an .mcd extension."
        )

    acquisitionRows = []
    panoramaRows = []

    with MCDFile(mcdPath) as mcd:
        slides = list(mcd.slides)

        for slideIndex, slide in enumerate(slides):
            slideId = getattr(
                slide,
                "id",
                slideIndex,
            )

            slideDescription = _clean_description(
                getattr(
                    slide,
                    "description",
                    None,
                ),
                f"Slide {slideId}",
            )

            panoramas = list(
                getattr(
                    slide,
                    "panoramas",
                    [],
                )
                or []
            )

            acquisitions = list(
                getattr(
                    slide,
                    "acquisitions",
                    [],
                )
                or []
            )

            for panoramaIndex, panorama in enumerate(
                panoramas
            ):
                panoramaId = getattr(
                    panorama,
                    "id",
                    panoramaIndex,
                )

                panoramaRows.append(
                    {
                        "Slide index": slideIndex,
                        "Slide ID": slideId,
                        "Slide description":
                            slideDescription,
                        "Panorama index":
                            panoramaIndex,
                        "Panorama ID":
                            panoramaId,
                        "Panorama description":
                            _clean_description(
                                getattr(
                                    panorama,
                                    "description",
                                    None,
                                ),
                                f"Panorama {panoramaId}",
                            ),
                        "Width (µm)":
                            _optional_number(
                                panorama,
                                "width_um",
                            ),
                        "Height (µm)":
                            _optional_number(
                                panorama,
                                "height_um",
                            ),
                    }
                )

            for acquisitionIndex, acquisition in enumerate(
                acquisitions
            ):
                acquisitionId = getattr(
                    acquisition,
                    "id",
                    acquisitionIndex,
                )

                channelNames = list(
                    getattr(
                        acquisition,
                        "channel_names",
                        [],
                    )
                    or []
                )

                channelLabels = list(
                    getattr(
                        acquisition,
                        "channel_labels",
                        [],
                    )
                    or []
                )

                channelCount = max(
                    len(channelNames),
                    len(channelLabels),
                )

                widthPx, heightPx = (
                    _get_pixel_dimensions(
                        acquisition
                    )
                )

                acquisitionRows.append(
                    {
                        "Slide index":
                            slideIndex,
                        "Slide ID":
                            slideId,
                        "Slide description":
                            slideDescription,
                        "Acquisition index":
                            acquisitionIndex,
                        "Acquisition ID":
                            acquisitionId,
                        "Acquisition description":
                            _clean_description(
                                getattr(
                                    acquisition,
                                    "description",
                                    None,
                                ),
                                f"Acquisition {acquisitionId}",
                            ),
                        "Width (px)":
                            widthPx,
                        "Height (px)":
                            heightPx,
                        "Width (µm)":
                            _optional_number(
                                acquisition,
                                "width_um",
                            ),
                        "Height (µm)":
                            _optional_number(
                                acquisition,
                                "height_um",
                            ),
                        "Channels":
                            channelCount,
                        "Panoramas on slide":
                            len(panoramas),
                    }
                )

    acquisitionDf = pd.DataFrame(
        acquisitionRows,
        columns=ACQUISITION_COLUMNS,
    )

    panoramaDf = pd.DataFrame(
        panoramaRows,
        columns=PANORAMA_COLUMNS,
    )

    summary = {
        "path": str(mcdPath),
        "file_name": mcdPath.name,
        "file_size_bytes": mcdPath.stat().st_size,
        "slides": (
            int(
                acquisitionDf["Slide index"].nunique()
            )
            if not acquisitionDf.empty
            else int(
                panoramaDf["Slide index"].nunique()
            )
            if not panoramaDf.empty
            else 0
        ),
        "acquisitions": len(acquisitionDf),
        "panoramas": len(panoramaDf),
    }

    return (
        acquisitionDf,
        panoramaDf,
        summary,
    )

def _safe_image_name(
    value: object,
    fallback: str,
) -> str:
    text = str(value or "").strip()

    if not text:
        text = fallback

    text = re.sub(
        r"[^\w\-+.]+",
        "_",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    ).strip("_")

    return text or fallback

def _make_mcd_roi_name(
    acquisition: Any,
    *,
    acquisition_index: int,
) -> str:
    """
    Construct the canonical PINT sample name for an MCD acquisition.

    The slide description is deliberately excluded because it should
    not become part of the biological ROI/sample identifier.

    Example:
        acquisition ID:          1
        acquisition description: 3_4_1(1)

    becomes:
        ROI001_3_4_1(1)
    """

    acquisitionId = getattr(
        acquisition,
        "id",
        acquisition_index + 1,
    )

    try:
        roiId = f"{int(acquisitionId):03d}"
    except (TypeError, ValueError):
        roiId = str(
            acquisitionId
        ).strip()

    description = str(
        getattr(
            acquisition,
            "description",
            None,
        )
        or ""
    ).strip()

    # Keep characters PINT commonly uses in ROI names,
    # including parentheses.
    description = re.sub(
        r"[^\w\-+.()]+",
        "_",
        description,
        flags=re.UNICODE,
    )

    description = re.sub(
        r"_+",
        "_",
        description,
    ).strip("_")

    if description:
        return (
            f"ROI{roiId}_"
            f"{description}"
        )

    return f"ROI{roiId}"

def load_mcd_acquisitions(
    path: str | Path,
    selected_acquisitions: pd.DataFrame,
    *,
    standardize_channel_names: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, list[str]],
]:
    """
    Load selected MCD acquisitions into PINT-compatible dictionaries.

    selected_acquisitions must contain:
    - Slide index
    - Acquisition index
    """
    _require_readimc()

    mcdPath = Path(path).expanduser().resolve()

    if not mcdPath.is_file():
        raise FileNotFoundError(
            f"MCD file does not exist: {mcdPath}"
        )

    if (
        selected_acquisitions is None
        or selected_acquisitions.empty
    ):
        raise ValueError(
            "No MCD acquisitions were selected."
        )

    requiredColumns = {
        "Slide index",
        "Acquisition index",
    }

    missingColumns = (
        requiredColumns
        - set(selected_acquisitions.columns)
    )

    if missingColumns:
        raise ValueError(
            "Selected acquisition metadata is missing: "
            + ", ".join(sorted(missingColumns))
        )

    imagesDict = {}
    channelNamesDict = {}

    referenceName = None
    referenceChannels = None
    mismatches = []

    with MCDFile(mcdPath) as mcd:
        slides = list(mcd.slides)

        for _, selectedRow in (
            selected_acquisitions.iterrows()
        ):
            slideIndex = int(
                selectedRow["Slide index"]
            )

            acquisitionIndex = int(
                selectedRow["Acquisition index"]
            )

            if (
                slideIndex < 0
                or slideIndex >= len(slides)
            ):
                raise ValueError(
                    f"Invalid slide index: {slideIndex}"
                )

            slide = slides[slideIndex]
            acquisitions = list(
                slide.acquisitions
            )

            if (
                acquisitionIndex < 0
                or acquisitionIndex
                >= len(acquisitions)
            ):
                raise ValueError(
                    "Invalid acquisition index "
                    f"{acquisitionIndex} for slide "
                    f"{slideIndex}."
                )

            acquisition = acquisitions[
                acquisitionIndex
            ]

            rawChannelNames = list(
                acquisition.channel_names
                or []
            )

            channelLabels = list(
                acquisition.channel_labels
                or []
            )

            if standardize_channel_names:
                channelNames = _make_unique_channel_names(
                    rawChannelNames,
                    channelLabels,
                )

            else:
                channelNames = []

                nChannels = max(
                    len(rawChannelNames),
                    len(channelLabels),
                )

                for i in range(nChannels):
                    metal = (
                        str(rawChannelNames[i]).strip()
                        if i < len(rawChannelNames)
                        and rawChannelNames[i] is not None
                        else ""
                    )

                    label = (
                        str(channelLabels[i]).strip()
                        if i < len(channelLabels)
                        and channelLabels[i] is not None
                        else ""
                    )

                    if label and metal:
                        name = f"{label}({metal})"
                    elif label:
                        name = label
                    elif metal:
                        name = metal
                    else:
                        name = f"Channel{i + 1}"

                    channelNames.append(name)

            sampleName = _make_mcd_roi_name(
                acquisition,
                acquisition_index=acquisitionIndex,
            )

            # Avoid accidental duplicate dictionary keys.
            uniqueSampleName = sampleName
            duplicateNumber = 2

            while uniqueSampleName in imagesDict:
                uniqueSampleName = (
                    f"{sampleName}_{duplicateNumber}"
                )
                duplicateNumber += 1

            print(
                "▶️ Reading MCD acquisition: "
                f"{uniqueSampleName}",
                flush=True,
            )

            imageArray = mcd.read_acquisition(
                acquisition
            )

            imageArray = np.asarray(
                imageArray,
                dtype=np.float32,
            )

            if imageArray.ndim != 3:
                raise ValueError(
                    f"Acquisition '{uniqueSampleName}' "
                    f"returned shape {imageArray.shape}; "
                    "expected channels × height × width."
                )

            if imageArray.shape[0] != len(
                channelNames
            ):
                raise ValueError(
                    f"Acquisition '{uniqueSampleName}' "
                    f"contains {imageArray.shape[0]} image "
                    f"channels but {len(channelNames)} "
                    "channel names."
                )

            normalizedChannels = [
                " ".join(
                    str(name)
                    .strip()
                    .split()
                ).upper()
                for name in channelNames
            ]

            if referenceChannels is None:
                referenceChannels = (
                    normalizedChannels
                )
                referenceName = uniqueSampleName

            elif (
                normalizedChannels
                != referenceChannels
            ):
                firstDifference = next(
                    (
                        i
                        for i, (
                            observed,
                            expected,
                        ) in enumerate(
                            zip(
                                normalizedChannels,
                                referenceChannels,
                            )
                        )
                        if observed != expected
                    ),
                    None,
                )

                if len(normalizedChannels) != len(
                    referenceChannels
                ):
                    mismatchText = (
                        f"{len(channelNames)} channels; "
                        f"expected "
                        f"{len(referenceChannels)}"
                    )
                elif firstDifference is not None:
                    mismatchText = (
                        "channel order/name differs at "
                        f"position {firstDifference + 1}: "
                        f"'{channelNames[firstDifference]}' "
                        "versus "
                        f"'{referenceChannels[firstDifference]}'"
                    )
                else:
                    mismatchText = (
                        "channel layout differs"
                    )

                mismatches.append(
                    f"- {uniqueSampleName}: "
                    f"{mismatchText}"
                )

                # Do not retain an acquisition that will
                # ultimately be rejected.
                del imageArray
                continue

            imagesDict[uniqueSampleName] = (
                imageArray
            )

            channelNamesDict[
                uniqueSampleName
            ] = channelNames

            print(
                f"✅ Loaded {uniqueSampleName}: "
                f"shape {imageArray.shape}",
                flush=True,
            )

    if mismatches:
        imagesDict.clear()
        channelNamesDict.clear()

        raise ValueError(
            "Selected MCD acquisitions do not have "
            "a consistent channel layout. All loaded "
            "ROIs must contain the same channels in the "
            "same order.\n\nReference acquisition: "
            f"{referenceName}\n\nMismatches:\n"
            + "\n".join(mismatches)
        )

    if not imagesDict:
        raise ValueError(
            "No MCD acquisitions were loaded."
        )

    return imagesDict, channelNamesDict

def _get_acquisition_export_channels(
    acquisition: Any,
) -> list[str]:
    """
    Build unique, readable OME channel names for one acquisition.

    Target labels are preferred. Metal-isotope names are retained as
    fallbacks and used to disambiguate duplicate labels.
    """
    channelNames = list(
        getattr(
            acquisition,
            "channel_names",
            [],
        )
        or []
    )

    channelLabels = list(
        getattr(
            acquisition,
            "channel_labels",
            [],
        )
        or []
    )

    return _make_unique_channel_names(
        channelNames,
        channelLabels,
    )

def _get_mcd_acquisition_by_indices(
    mcd: Any,
    *,
    slide_index: int,
    acquisition_index: int,
) -> tuple[Any, Any]:
    """
    Retrieve one slide and acquisition by their stored table indices.
    """
    slides = list(mcd.slides)

    if (
        slide_index < 0
        or slide_index >= len(slides)
    ):
        raise IndexError(
            f"Slide index {slide_index} is invalid. "
            f"The MCD contains {len(slides)} slide(s)."
        )

    slide = slides[slide_index]

    acquisitions = list(
        getattr(
            slide,
            "acquisitions",
            [],
        )
        or []
    )

    if (
        acquisition_index < 0
        or acquisition_index >= len(acquisitions)
    ):
        raise IndexError(
            f"Acquisition index {acquisition_index} is invalid "
            f"for slide {slide_index}. The slide contains "
            f"{len(acquisitions)} acquisition(s)."
        )

    return slide, acquisitions[acquisition_index]

def _make_acquisition_export_stem(
    slide: Any,
    acquisition: Any,
    *,
    slide_index: int,
    acquisition_index: int,
) -> str:
    """
    Construct the canonical PINT ROI filename stem for an MCD acquisition.

    The slide metadata is intentionally not included in the sample name.
    """

    return _make_mcd_roi_name(
        acquisition,
        acquisition_index=acquisition_index,
    )

def _write_mcd_acquisition_ome_tiff(
    output_path: str | Path,
    image: np.ndarray,
    acquisition: Any,
    *,
    image_name: str,
) -> Path:
    """
    Write one MCD acquisition as a float32 CYX OME-TIFF.

    The file contains:
    - float32 channel data;
    - CYX axis metadata;
    - unique channel display names;
    - physical X/Y pixel sizes when present.
    """
    outputPath = Path(output_path)

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 3:
        raise ValueError(
            f"Cannot export acquisition '{image_name}': "
            f"expected a CYX array but received shape {image.shape}."
        )

    channelNames = (
        _get_acquisition_export_channels(
            acquisition
        )
    )

    if len(channelNames) != image.shape[0]:
        raise ValueError(
            f"Cannot export acquisition '{image_name}': "
            f"pixel data contains {image.shape[0]} channels, "
            f"but metadata contains {len(channelNames)} names."
        )

    metadata: dict[str, Any] = {
        "axes": "CYX",
        "Name": image_name,
        "Channel": {
            "Name": channelNames,
        },
    }

    pixelSizeX = getattr(
        acquisition,
        "pixel_size_x_um",
        None,
    )

    pixelSizeY = getattr(
        acquisition,
        "pixel_size_y_um",
        None,
    )

    if pixelSizeX is not None:
        metadata["PhysicalSizeX"] = float(
            pixelSizeX
        )
        metadata["PhysicalSizeXUnit"] = "µm"

    if pixelSizeY is not None:
        metadata["PhysicalSizeY"] = float(
            pixelSizeY
        )
        metadata["PhysicalSizeYUnit"] = "µm"

    # Standard TIFF has a practical 4 GiB limit. Enable BigTIFF before
    # approaching that boundary.
    useBigTiff = image.nbytes >= int(
        3.8 * 1024**3
    )

    outputPath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporaryPath = outputPath.with_name(
        outputPath.name + ".partial"
    )

    try:
        tifffile.imwrite(
            temporaryPath,
            image,
            ome=True,
            metadata=metadata,
            photometric="minisblack",
            bigtiff=useBigTiff,
        )

        os.replace(
            temporaryPath,
            outputPath,
        )

    except Exception:
        try:
            temporaryPath.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise

    return outputPath

def export_mcd_acquisitions_as_ome_tiff(
    mcd_path: str | Path,
    selected_acquisitions: pd.DataFrame,
    output_folder: str | Path,
    *,
    output_subfolder: str | None = None,
    progress_callback: (
        Callable[[int, int, str], None]
        | None
    ) = None,
) -> list[Path]:
    """
    Export selected MCD acquisitions as separate float32 OME-TIFFs.

    Acquisitions are processed sequentially so only one ROI pixel array
    needs to be held in memory by this function at a time.
    """
    _require_readimc()

    mcdPath = Path(
        mcd_path
    ).expanduser().resolve()

    outputFolder = (
        Path(output_folder)
        .expanduser()
        .resolve()
        / "acquisitions"
    )

    if output_subfolder:
        outputFolder = (
            outputFolder
            / _safe_image_name(
                output_subfolder,
                "MCD",
            )
        )

    if not mcdPath.is_file():
        raise FileNotFoundError(
            f"MCD file does not exist: {mcdPath}"
        )

    if (
        selected_acquisitions is None
        or selected_acquisitions.empty
    ):
        raise ValueError(
            "No MCD acquisitions were selected."
        )

    requiredColumns = {
        "Slide index",
        "Acquisition index",
    }

    missingColumns = (
        requiredColumns
        - set(selected_acquisitions.columns)
    )

    if missingColumns:
        raise ValueError(
            "Selected acquisition metadata is missing: "
            + ", ".join(
                sorted(missingColumns)
            )
        )

    outputFolder.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(
        selected_acquisitions
    )

    exportedPaths: list[Path] = []
    usedStems: set[str] = set()

    with MCDFile(mcdPath) as mcd:
        for exportIndex, (_, row) in enumerate(
            selected_acquisitions.iterrows(),
            start=1,
        ):
            slideIndex = int(
                row["Slide index"]
            )

            acquisitionIndex = int(
                row["Acquisition index"]
            )

            slide, acquisition = (
                _get_mcd_acquisition_by_indices(
                    mcd,
                    slide_index=slideIndex,
                    acquisition_index=acquisitionIndex,
                )
            )

            stem = _make_acquisition_export_stem(
                slide,
                acquisition,
                slide_index=slideIndex,
                acquisition_index=acquisitionIndex,
            )

            uniqueStem = stem
            duplicateNumber = 2

            while uniqueStem in usedStems:
                uniqueStem = (
                    f"{stem}_{duplicateNumber}"
                )
                duplicateNumber += 1

            usedStems.add(uniqueStem)

            if progress_callback is not None:
                progress_callback(
                    exportIndex - 1,
                    total,
                    uniqueStem,
                )

            print(
                f"▶️ Reading MCD ROI "
                f"{exportIndex}/{total}: {uniqueStem}",
                flush=True,
            )

            image = mcd.read_acquisition(
                acquisition
            )

            image = np.asarray(
                image,
                dtype=np.float32,
            )

            outputPath = (
                outputFolder
                / f"{uniqueStem}.ome.tiff"
            )

            try:
                _write_mcd_acquisition_ome_tiff(
                    outputPath,
                    image,
                    acquisition,
                    image_name=uniqueStem,
                )

            finally:
                # Release the local reference before reading the next ROI.
                del image

            exportedPaths.append(
                outputPath
            )

            print(
                f"✅ Exported {outputPath.name}",
                flush=True,
            )

            if progress_callback is not None:
                progress_callback(
                    exportIndex,
                    total,
                    uniqueStem,
                )

    return exportedPaths

def _panorama_to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert a decoded panorama image to PNG-compatible uint8.

    Typical readimc panoramas are already uint8. Other integer or floating
    dtypes are converted defensively without changing the image dimensions.
    """
    image = np.asarray(image)

    if image.ndim not in {2, 3}:
        raise ValueError(
            "Panorama image must be two-dimensional grayscale or "
            f"three-dimensional color data, but received shape {image.shape}."
        )

    if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
        raise ValueError(
            "Panorama color data must have 1, 3, or 4 channels, "
            f"but received shape {image.shape}."
        )

    if image.dtype == np.uint8:
        return image

    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255

    finiteMask = np.isfinite(image)

    if not finiteMask.any():
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    finiteValues = image[finiteMask]

    imageMin = float(np.min(finiteValues))
    imageMax = float(np.max(finiteValues))

    if imageMax <= imageMin:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    # Preserve common 8-bit-like and 16-bit-like ranges when possible.
    if imageMin >= 0 and imageMax <= 255:
        scaled = image

    elif imageMin >= 0 and imageMax <= 65535:
        scaled = image / 257.0

    else:
        scaled = (
            image.astype(np.float32)
            - imageMin
        ) / (
            imageMax - imageMin
        )

        scaled *= 255.0

    scaled = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    return np.clip(
        np.round(scaled),
        0,
        255,
    ).astype(np.uint8)

def _make_panorama_export_stem(
    slide: Any,
    panorama: Any,
    *,
    slide_index: int,
    panorama_index: int,
) -> str:
    """
    Construct a readable, filesystem-safe panorama filename.
    """
    slideId = getattr(
        slide,
        "id",
        slide_index + 1,
    )

    panoramaId = getattr(
        panorama,
        "id",
        panorama_index + 1,
    )

    slideDescription = _safe_image_name(
        getattr(
            slide,
            "description",
            None,
        ),
        f"Slide{slideId}",
    )

    panoramaDescription = _safe_image_name(
        getattr(
            panorama,
            "description",
            None,
        ),
        f"Panorama{panoramaId}",
    )

    return (
        f"{slideDescription}_"
        f"Panorama{panoramaId}_"
        f"{panoramaDescription}"
    )

def export_mcd_panoramas(
    mcd_path: str | Path,
    output_folder: str | Path,
    *,
    slide_indices: Iterable[int] | None = None,
    output_subfolder: str | None = None,
    progress_callback: (
        Callable[[int, int, str], None]
        | None
    ) = None,
) -> list[Path]:
    """
    Export real panorama images from an MCD file as PNG.

    Parameters
    ----------
    mcd_path
        Source MCD file.

    output_folder
        User-selected root output folder. A `panoramas` subfolder is created.

    slide_indices
        Optional zero-based slide indices. When omitted, panoramas from all
        slides are exported.

    progress_callback
        Optional callback receiving:
        completed_count, total_count, panorama_name

    Returns
    -------
    list[Path]
        Paths of successfully exported PNG files.
    """
    _require_readimc()

    mcdPath = (
        Path(mcd_path)
        .expanduser()
        .resolve()
    )

    outputFolder = (
        Path(output_folder)
        .expanduser()
        .resolve()
        / "panoramas"
    )

    if output_subfolder:
        outputFolder = (
            outputFolder
            / _safe_image_name(
                output_subfolder,
                "MCD",
            )
        )

    if not mcdPath.is_file():
        raise FileNotFoundError(
            f"MCD file does not exist: {mcdPath}"
        )

    outputFolder.mkdir(
        parents=True,
        exist_ok=True,
    )

    requestedSlideIndices = (
        None
        if slide_indices is None
        else {
            int(index)
            for index in slide_indices
        }
    )

    exportedPaths: list[Path] = []
    usedStems: set[str] = set()

    with MCDFile(mcdPath) as mcd:
        slides = list(mcd.slides)

        if requestedSlideIndices is not None:
            invalidIndices = sorted(
                index
                for index in requestedSlideIndices
                if index < 0 or index >= len(slides)
            )

            if invalidIndices:
                raise ValueError(
                    "Invalid MCD slide indices requested: "
                    + ", ".join(
                        str(index)
                        for index in invalidIndices
                    )
                )

        panoramaJobs = []

        for slideIndex, slide in enumerate(slides):
            if (
                requestedSlideIndices is not None
                and slideIndex not in requestedSlideIndices
            ):
                continue

            panoramas = list(
                getattr(
                    slide,
                    "panoramas",
                    [],
                )
                or []
            )

            for panoramaIndex, panorama in enumerate(
                panoramas
            ):
                panoramaJobs.append(
                    (
                        slideIndex,
                        slide,
                        panoramaIndex,
                        panorama,
                    )
                )

        total = len(panoramaJobs)

        if total == 0:
            if requestedSlideIndices is None:
                raise ValueError(
                    "The selected MCD file does not contain any "
                    "exportable panorama images."
                )

            raise ValueError(
                "The selected slides do not contain any "
                "exportable panorama images."
            )

        for exportIndex, (
            slideIndex,
            slide,
            panoramaIndex,
            panorama,
        ) in enumerate(
            panoramaJobs,
            start=1,
        ):
            stem = _make_panorama_export_stem(
                slide,
                panorama,
                slide_index=slideIndex,
                panorama_index=panoramaIndex,
            )

            uniqueStem = stem
            duplicateNumber = 2

            while uniqueStem in usedStems:
                uniqueStem = (
                    f"{stem}_{duplicateNumber}"
                )
                duplicateNumber += 1

            usedStems.add(uniqueStem)

            if progress_callback is not None:
                progress_callback(
                    exportIndex - 1,
                    total,
                    uniqueStem,
                )

            print(
                f"▶️ Reading MCD panorama "
                f"{exportIndex}/{total}: {uniqueStem}",
                flush=True,
            )

            panoramaImage = mcd.read_panorama(
                panorama
            )

            if panoramaImage is None:
                raise ValueError(
                    f"readimc returned no image data for panorama "
                    f"'{uniqueStem}'."
                )

            panoramaUint8 = _panorama_to_uint8(
                panoramaImage
            )

            outputPath = (
                outputFolder
                / f"{uniqueStem}.png"
            )

            temporaryPath = outputPath.with_name(
                outputPath.name + ".partial"
            )

            try:
                imageio.v3.imwrite(
                    temporaryPath,
                    panoramaUint8,
                    extension=".png",
                )

                os.replace(
                    temporaryPath,
                    outputPath,
                )

            except Exception:
                try:
                    temporaryPath.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

                raise

            finally:
                del panoramaImage
                del panoramaUint8

            exportedPaths.append(
                outputPath
            )

            print(
                f"✅ Exported {outputPath.name}",
                flush=True,
            )

            if progress_callback is not None:
                progress_callback(
                    exportIndex,
                    total,
                    uniqueStem,
                )

    return exportedPaths

def inspect_mcd_files(
    paths: list[str | Path],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """
    Inspect multiple MCD files without loading image pixels.

    Returns:
    - combined acquisition metadata
    - combined panorama metadata
    - MCD file registry
    - combined summary
    """
    if not paths:
        raise ValueError(
            "No MCD files were selected."
        )

    acquisitionTables = []
    panoramaTables = []
    registryRows = []
    fileSummaries = []

    seenPaths = set()

    for mcdFileIndex, path in enumerate(paths):
        resolvedPath = (
            Path(path)
            .expanduser()
            .resolve()
        )

        normalizedPath = str(resolvedPath)

        if normalizedPath in seenPaths:
            continue

        seenPaths.add(normalizedPath)

        (
            acquisitionDf,
            panoramaDf,
            fileSummary,
        ) = inspect_mcd_file(
            resolvedPath
        )

        fileName = resolvedPath.name

        acquisitionDf = acquisitionDf.copy()
        acquisitionDf.insert(
            0,
            "MCD file name",
            fileName,
        )
        acquisitionDf.insert(
            0,
            "MCD file index",
            mcdFileIndex,
        )

        panoramaDf = panoramaDf.copy()
        panoramaDf.insert(
            0,
            "MCD file name",
            fileName,
        )
        panoramaDf.insert(
            0,
            "MCD file index",
            mcdFileIndex,
        )

        acquisitionTables.append(
            acquisitionDf
        )

        panoramaTables.append(
            panoramaDf
        )

        registryRows.append(
            {
                "MCD file index": mcdFileIndex,
                "MCD file name": fileName,
                "MCD file path": normalizedPath,
            }
        )

        fileSummary = dict(fileSummary)
        fileSummary["MCD file index"] = (
            mcdFileIndex
        )

        fileSummaries.append(
            fileSummary
        )

    if not registryRows:
        raise ValueError(
            "No valid MCD files were inspected."
        )

    combinedAcquisitions = (
        pd.concat(
            acquisitionTables,
            ignore_index=True,
        )
        if acquisitionTables
        else pd.DataFrame()
    )

    combinedPanoramas = (
        pd.concat(
            panoramaTables,
            ignore_index=True,
        )
        if panoramaTables
        else pd.DataFrame()
    )

    registryDf = pd.DataFrame(
        registryRows
    )

    summary = {
        "files": len(registryDf),
        "slides": sum(
            int(item.get("slides", 0))
            for item in fileSummaries
        ),
        "acquisitions": len(
            combinedAcquisitions
        ),
        "panoramas": len(
            combinedPanoramas
        ),
        "file_size_bytes": sum(
            int(
                item.get(
                    "file_size_bytes",
                    0,
                )
            )
            for item in fileSummaries
        ),
        "file_summaries": fileSummaries,
    }

    return (
        combinedAcquisitions,
        combinedPanoramas,
        registryDf,
        summary,
    )