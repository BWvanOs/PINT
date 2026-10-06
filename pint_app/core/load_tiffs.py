from pathlib import Path
import re

import numpy as np
import tifffile

from xml.etree import ElementTree as ET

from pint_app.core.channel_names import (
    _normalize_channel_name,
    _make_unique_channel_names,
)


# ============================================================
# Channel-name handling
# ============================================================

def _split_mcd_viewer_channel_name(
    channel_name: object,
) -> tuple[str, str]:
    """
    Split an MCD Viewer channel name into:

        (metal/isotope, target)

    MCD Viewer examples:

        "aSMA(Pr141Di)"       -> ("Pr141Di", "aSMA")
        "CD4(Nd145Di)"        -> ("Nd145Di", "CD4")
        "HLA-DR(Eu151Di)"     -> ("Eu151Di", "HLA-DR")
        "CollagenI(Y89Di)"    -> ("Y89Di", "CollagenI")

    Channels without a biological target remain unchanged:

        "Kr80Di"              -> ("", "Kr80Di")
        "Nd144Di"             -> ("", "Nd144Di")
        "Sm152Di"             -> ("", "Sm152Di")

    The expression inside parentheses must contain a number and end
    in "Di" to be considered an MCD isotope/channel identifier.
    """

    name = _normalize_channel_name(channel_name)

    if not name:
        return "", ""

    # MCD Viewer format:
    #
    # Target(Metal123Di)
    #
    # Examples:
    #   CD3(Gd160Di)
    #   aSMA(Pr141Di)
    #   CollagenI(Y89Di)
    #
    # Also deliberately allows unusual MCD channel identifiers such as:
    #   ArAr80Di
    #   BCKG190Di
    #
    match = re.match(
        r"^(.+?)\s*\(\s*([A-Za-z]+\d+Di)\s*\)$",
        name,
        flags=re.IGNORECASE,
    )

    if match:
        target = match.group(1).strip()
        metal = match.group(2).strip()

        return metal, target

    # No target(metal) structure:
    # keep the original channel name
    return "", name

def _standardize_mcd_channel_names(
    channel_names: list[str],
) -> list[str]:
    """
    Convert TIFF channel names to the same convention used when
    acquisitions are loaded directly from MCD files.

    Examples:

        ["170Er_CD3", "147Sm_CD20"]

    becomes:

        ["CD3", "CD20"]

    Duplicate biological targets retain their metals:

        ["170Er_CD3", "142Nd_CD3", "147Sm_CD20"]

    becomes:

        ["CD3 [170Er]", "CD3 [142Nd]", "CD20"]

    Duplicate detection is performed case-insensitively by
    _make_unique_channel_names().
    """

    metals = []
    labels = []

    for name in channel_names:
        metal, label = _split_mcd_viewer_channel_name(name)

        metals.append(metal)
        labels.append(label)

    return _make_unique_channel_names(
        metals,
        labels,
    )


def _normalize_ch_names(
    names: list[str],
) -> list[str]:
    """
    Normalize channel names for comparison between images.

    This does not alter the displayed channel names.
    """

    return [
        " ".join(
            str(name)
            .strip()
            .split()
        ).casefold()
        for name in names
    ]


# ============================================================
# TIFF metadata reading
# ============================================================

def _channel_names_from_page_tags(
    tif: tifffile.TiffFile,
) -> list[str] | None:
    """
    Preferred channel-name source.

    Read TIFF PageName tag 285 for each page/channel.

    Returns:
        list[str] if at least one PageName exists
        None otherwise

    Missing individual names are replaced with Channel1,
    Channel2, etc.
    """

    names = []
    has_any = False

    for i, page in enumerate(tif.pages):
        tag = page.tags.get(285)

        name = None

        if tag is not None:
            value = tag.value

            if isinstance(value, bytes):
                value = value.decode(
                    "utf-8",
                    "ignore",
                )

            name = str(value).strip()

        if name:
            has_any = True
            names.append(name)

        else:
            names.append(None)

    if not has_any:
        return None

    return [
        name if name else f"Channel{i + 1}"
        for i, name in enumerate(names)
    ]


def _channel_names_from_ome_xml(
    tif: tifffile.TiffFile,
) -> list[str] | None:
    """
    Read channel names directly from OME-XML.

    Uses Python's standard XML parser rather than ome-types so TIFF
    loading does not depend on pydantic / pydantic-core.
    """

    try:
        omeXml = tif.ome_metadata

        if not omeXml:
            return None

        root = ET.fromstring(omeXml)

        # OME XML uses namespaces. The {*} wildcard allows this
        # to work across OME schema versions without hard-coding
        # a particular namespace URI.
        pixels = root.find(".//{*}Image/{*}Pixels")

        if pixels is None: return None

        channels = pixels.findall("{*}Channel")

        if not channels: return None

        names = []

        for i, channel in enumerate(channels):
            name = (
                channel.attrib.get("Name")
                or channel.attrib.get("ID")
                or f"Channel{i + 1}"
            )

            names.append(str(name).strip())

        return names or None

    except Exception as e:
        print(
            "[OME] Could not parse channel names "
            f"from OME-XML: {e}"
        )

        return None

def inspect_tiff_stack(
    filePath: str | Path,
    *,
    standardize_channel_names: bool = True,
) -> dict:
    """
    Inspect a TIFF stack without loading pixel data.

    Returns path, sample name, dimensions, dtype,
    and channel names.
    """

    filePath = Path(
        filePath
    )

    with tifffile.TiffFile(
        filePath
    ) as tif:

        nChannels = len(
            tif.pages
        )

        if nChannels == 0:
            raise ValueError(
                f"No TIFF pages found: {filePath}"
            )

        firstPage = tif.pages[0]

        height = int(
            firstPage.imagelength
        )

        width = int(
            firstPage.imagewidth
        )

        dtype = str(
            firstPage.dtype
        )

        channelNames = (
            _channel_names_from_page_tags(
                tif
            )
        )

        if not channelNames:
            channelNames = (
                _channel_names_from_ome_xml(
                    tif
                )
            )

    if not channelNames:
        channelNames = [
            f"Channel{i + 1}"
            for i in range(nChannels)
        ]

    if len(channelNames) < nChannels:
        channelNames = (
            channelNames
            + [
                f"Channel{i + 1}"
                for i in range(
                    len(channelNames),
                    nChannels,
                )
            ]
        )

    elif len(channelNames) > nChannels:
        channelNames = channelNames[
            :nChannels
        ]

    if standardize_channel_names:
        channelNames = (
            _standardize_mcd_channel_names(
                channelNames
            )
        )

    return {
        "SampleName": filePath.stem,
        "ImagePath": str(filePath),
        "Channels": channelNames,
        "NChannels": nChannels,
        "Height": height,
        "Width": width,
        "DType": dtype,
    }


# ============================================================
# Main TIFF loader
# ============================================================

def load_tiffs_raw(
    folderPath: str,
    *,
    validate_consistent: bool = True,
    standardize_channel_names: bool = True,
    file_paths: list[str | Path] | None = None,
):
    """
    Load (OME-)TIFFs as raw multi-page TIFF files.

    Each TIFF page/frame is treated as one channel.

    Returns:
        imagesDict:
            {sampleName: imageArray [C, Y, X]}

        channelNamesDict:
            {sampleName: [channel names]}
    """

    folderPath = Path(folderPath)

    if file_paths is None:
        tiffFiles = sorted(
            path
            for path in folderPath.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in {".tif", ".tiff"}
            )
        )

    else:
        tiffFiles = sorted(
            Path(path)
            for path in file_paths
        )

    imagesDict: dict[str, np.ndarray] = {}
    channelNamesDict: dict[str, list[str]] = {}


    # First image becomes the reference for consistency checks.
    ref_sample: str | None = None
    ref_nC: int | None = None
    ref_names_norm: list[str] | None = None
    ref_names_raw: list[str] | None = None

    mismatches: list[str] = []

    for filePath in tiffFiles:

        with tifffile.TiffFile(filePath) as tif:

            pages = [
                page.asarray()
                for page in tif.pages
            ]

            imageArray = np.stack(pages, axis=0,)

            # Preferred:
            # TIFF PageName tag 285
            ch_names = _channel_names_from_page_tags(tif)

            # Fallback:
            # OME-XML metadata
            if not ch_names: ch_names = _channel_names_from_ome_xml(tif)

        sampleName = filePath.stem.replace(".ome", "",)
        nC = int(imageArray.shape[0])

        # ----------------------------------------------------
        # Ensure exactly one name per channel
        # ----------------------------------------------------

        if not ch_names:

            ch_names = [
                f"Channel{i + 1}"
                for i in range(nC)
            ]

        else:

            if len(ch_names) < nC:

                ch_names = ch_names + [
                    f"Channel{i + 1}"
                    for i in range(
                        len(ch_names),
                        nC,
                    )
                ]

            elif len(ch_names) > nC:

                ch_names = ch_names[:nC]

        # ----------------------------------------------------
        # Convert MCD Viewer names to the same naming scheme
        # used by direct MCD loading.
        # ----------------------------------------------------

        raw_ch_names = list(ch_names)

        if standardize_channel_names:
            ch_names = _standardize_mcd_channel_names(
                ch_names
            )
        # Useful temporary diagnostic:
        #
        # print("RAW:", raw_ch_names)
        # print("PINT:", ch_names)

        imagesDict[sampleName] = imageArray

        channelNamesDict[sampleName] = ch_names

        # Console preview

        preview = ", ".join(
            ch_names[:5]
        )

        if nC > 5:
            preview += "..."

        print(
            f"Loaded {filePath.name} "
            f"→ shape {imageArray.shape}, "
            f"channels=[{preview}]"
        )

        # Consistency validation

        if validate_consistent:

            names_norm = _normalize_ch_names(
                ch_names
            )

            if ref_sample is None:

                ref_sample = sampleName
                ref_nC = nC
                ref_names_norm = names_norm
                ref_names_raw = ch_names

                continue

            assert ref_nC is not None
            assert ref_names_norm is not None
            assert ref_names_raw is not None

            # Channel count differs
            if nC != ref_nC:

                mismatches.append(
                    f"- {sampleName}: "
                    f"{nC} channels "
                    f"(expected {ref_nC} "
                    f"like {ref_sample})"
                )

                continue

            # Channel names/order differ
            if names_norm != ref_names_norm:

                first_diff = next(
                    (
                        i
                        for i, (current, reference)
                        in enumerate(
                            zip(
                                names_norm,
                                ref_names_norm,
                            )
                        )
                        if current != reference
                    ),
                    None,
                )

                if first_diff is None:
                    first_diff = 0

                mismatches.append(
                    f"- {sampleName}: "
                    f"channel list/order differs "
                    f"from {ref_sample} "
                    f"(first diff at index "
                    f"{first_diff}: "
                    f"got "
                    f"'{ch_names[first_diff]}' "
                    f"vs expected "
                    f"'{ref_names_raw[first_diff]}')"
                )

    # --------------------------------------------------------
    # Report all consistency problems together
    # --------------------------------------------------------

    if validate_consistent and mismatches:

        message = (
            "Inconsistent channel layout across images. "
            "All images must have the same number of "
            "channels and the same channel names in the "
            "same order.\n\n"
            "Mismatches:\n"
            + "\n".join(mismatches)
        )

        raise ValueError(message)

    return imagesDict, channelNamesDict


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python load_tiffs.py "
            "/path/to/ome-tiffs"
        )

        raise SystemExit(1)

    folder = sys.argv[1]

    imagesDict, channelNamesDict = load_tiffs_raw(
        folder
    )

    firstSample = next(
        iter(imagesDict)
    )

    print(
        "First sample shape:",
        imagesDict[firstSample].shape,
    )

    print(
        "Channel names:",
        channelNamesDict[firstSample],
    )