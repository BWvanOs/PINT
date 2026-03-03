from pathlib import Path
import tifffile
import numpy as np

def _channel_names_from_page_tags(tif: tifffile.TiffFile):
    """
    Preferred: read TIFF 'PageName' (tag 285) per page.
    Returns list[str] or None if not present.
    """
    names = []
    has_any = False
    for i, page in enumerate(tif.pages):
        tag = page.tags.get("PageName") or page.tags.get(285)
        name = None
        if tag is not None:
            val = tag.value
            if isinstance(val, bytes):
                val = val.decode("utf-8", "ignore")
            name = str(val).strip()
        if name:
            has_any = True
            names.append(name)
        else:
            names.append(None)
    if has_any:
        # fill missing with Channel#
        return [n if n else f"Channel{i+1}" for i, n in enumerate(names)]
    return None

def _channel_names_from_ome_xml(tif: tifffile.TiffFile):
    """
    Best-effort fallback: parse channel names from OME-XML (may be malformed).
    Returns list[str] or None.
    """
    try:
        ome_xml = tif.ome_metadata
        if not ome_xml:
            return None
        from ome_types import from_xml  # optional; only used if available
        ome = from_xml(ome_xml)
        chans = getattr(ome.images[0].pixels, "channels", None)
        if not chans:
            return None
        out = []
        for i, ch in enumerate(chans):
            nm = getattr(ch, "name", None) or getattr(ch, "id", None) or f"Channel{i+1}"
            if isinstance(nm, str):
                nm = nm.strip()
            out.append(nm)
        return out or None
    except Exception as e:
        print(f"[OME] Could not parse channel names from OME-XML: {e}")
        return None

def load_tiffs_raw(folderPath: str, *, validate_consistent: bool = True):
    """
    Load IMC OME-TIFFs as raw multi-page TIFFs.
    Each page/frame = one channel.

    Returns:
        imagesDict: {sampleName: imageArray [C, Y, X]}
        channelNamesDict: {sampleName: [channel names]}
    """
    folderPath = Path(folderPath)
    tiffFiles = sorted(folderPath.glob("*.ome.tif*"))

    imagesDict: dict[str, np.ndarray] = {}
    channelNamesDict: dict[str, list[str]] = {}

    # Reference (first image) for consistency checks
    ref_sample: str | None = None
    ref_nC: int | None = None
    ref_names_norm: list[str] | None = None
    ref_names_raw: list[str] | None = None

    mismatches: list[str] = []

    for f in tiffFiles:
        with tifffile.TiffFile(f) as tif:
            pages = [page.asarray() for page in tif.pages]
            imageArray = np.stack(pages, axis=0)

            ch_names = _channel_names_from_page_tags(tif)
            if not ch_names:
                ch_names = _channel_names_from_ome_xml(tif)

        sampleName = f.stem.replace(".ome", "")
        nC = int(imageArray.shape[0])

        # Ensure we have exactly nC names
        if not ch_names:
            ch_names = [f"Channel{i+1}" for i in range(nC)]
        else:
            if len(ch_names) < nC:
                ch_names = ch_names + [f"Channel{i+1}" for i in range(len(ch_names), nC)]
            elif len(ch_names) > nC:
                ch_names = ch_names[:nC]

        imagesDict[sampleName] = imageArray
        channelNamesDict[sampleName] = ch_names

        # Preview log
        preview = ", ".join(ch_names[:5]) + ("..." if nC > 5 else "")
        print(f"Loaded {f.name} → shape {imageArray.shape}, channels=[{preview}]")

        # ------------------ consistency validation ------------------
        if validate_consistent:
            names_norm = _normalize_ch_names(ch_names)

            if ref_sample is None:
                ref_sample = sampleName
                ref_nC = nC
                ref_names_norm = names_norm
                ref_names_raw = ch_names
            else:
                assert ref_nC is not None and ref_names_norm is not None and ref_names_raw is not None

                if nC != ref_nC:
                    mismatches.append(
                        f"- {sampleName}: {nC} channels (expected {ref_nC} like {ref_sample})"
                    )
                    continue  # no point checking names/order if count differs

                if names_norm != ref_names_norm:
                    # Provide a compact diff: first index where it differs
                    first_diff = next((i for i, (a, b) in enumerate(zip(names_norm, ref_names_norm)) if a != b), None)
                    if first_diff is None:
                        first_diff = 0
                    mismatches.append(
                        f"- {sampleName}: channel list/order differs from {ref_sample} "
                        f"(first diff at index {first_diff}: got '{ch_names[first_diff]}' vs expected '{ref_names_raw[first_diff]}')"
                    )

    if validate_consistent and mismatches:
        msg = (
            "Inconsistent channel layout across images. "
            "All images must have the same number of channels and the same channel names in the same order.\n\n"
            "Mismatches:\n" + "\n".join(mismatches)
        )
        raise ValueError(msg)

    return imagesDict, channelNamesDict

def _normalize_ch_names(names: list[str]) -> list[str]:
    # conservative normalization: keep case? I'd upper() to avoid "CD45" vs "cd45"
    return [" ".join(str(x).strip().split()).upper() for x in names]

# Optional CLI test
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_tiffs.py /path/to/ome-tiffs")
        raise SystemExit(1)
    folder = sys.argv[1]
    imagesDict, channelNamesDict = load_tiffs_raw(folder)
    firstSample = next(iter(imagesDict))
    print("First sample shape:", imagesDict[firstSample].shape)
    print("Channel names:", channelNamesDict[firstSample])
