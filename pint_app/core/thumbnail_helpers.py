import numpy as np
import base64
import io
from PIL import Image

THUMBNAIL_MAX_WIDTH = 250
THUMBNAIL_MAX_HEIGHT = 350

THUMBNAIL_PARAM_COLUMNS = (
    "DoWinsor",
    "Low",
    "High",
    "DoThr",
    "ThrVal",
    "DoAbsThr",
    "AbsThrVal",
    "Noise",
    "NStr",
    "WinSz",
    "DoNorm",
    "NormScope",
    "DoAsinh",
    "Cofac",
)

def _thumbnail_target_shape(
    source_height: int,
    source_width: int,
    *,
    max_width: int = THUMBNAIL_MAX_WIDTH,
    max_height: int = THUMBNAIL_MAX_HEIGHT,
) -> tuple[int, int]:
    """
    Calculate thumbnail dimensions while preserving aspect ratio.

    Images are never enlarged beyond their original dimensions.
    """
    source_height = int(source_height)
    source_width = int(source_width)

    if source_height < 1 or source_width < 1:
        raise ValueError(
            "Thumbnail source image has invalid dimensions."
        )

    scale = min(
        float(max_width) / float(source_width),
        float(max_height) / float(source_height),
        1.0,
    )

    target_width = max(
        1,
        int(round(source_width * scale)),
    )

    target_height = max(
        1,
        int(round(source_height * scale)),
    )

    return target_height, target_width

def _processed_image_to_uint8(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a processed two-dimensional image to display-ready uint8.

        PINT-normalized arrays are already in [0, 1]. If normalization was
        disabled, this reproduces Matplotlib's automatic min/max display scaling.
        """
        image = np.asarray(
            image,
            dtype=np.float32,
        )

        ##Catch if input is incorrect
        if image.ndim != 2:
            raise ValueError(
                "Thumbnail input must be a two-dimensional channel image."
            )

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

        # Normalized PINT images can be converted directly. Non-normalized
        # images are scaled as Matplotlib would scale them for display.
        if imageMin >= 0.0 and imageMax <= 1.0:
            scaled = image
        else:
            scaled = (
                image - imageMin
            ) / (
                imageMax - imageMin
            )

        scaled = np.nan_to_num(
            scaled,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        scaled = np.clip(
            scaled,
            0.0,
            1.0,
        )

        return np.round(
            scaled * 255.0
        ).astype(np.uint8)

def _max_pool_uint8_to_shape(
    image: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Downscale a uint8 image using variable-size maximum-pooling regions.

    Every source pixel belongs to one output region. Small bright structures
    are therefore retained rather than averaged away.
    """
    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    source_height, source_width = image.shape

    target_height = min(
        int(target_height),
        source_height,
    )

    target_width = min(
        int(target_width),
        source_width,
    )

    if (
        target_height == source_height
        and target_width == source_width
    ):
        return image.copy()

    rowStarts = np.floor(
        np.arange(target_height)
        * source_height
        / target_height
    ).astype(np.int64)

    colStarts = np.floor(
        np.arange(target_width)
        * source_width
        / target_width
    ).astype(np.int64)

    # np.maximum.reduceat reduces each interval from one start index
    # up to the next. The final interval continues to the image boundary.
    # This handles images without integer scaling (eg 1100x700 into 250x~)
    pooledRows = np.maximum.reduceat(
        image,
        rowStarts,
        axis=0,
    )

    pooled = np.maximum.reduceat(
        pooledRows,
        colStarts,
        axis=1,
    )

    return pooled[
        :target_height,
        :target_width,
    ].astype(
        np.uint8,
        copy=False,
    )

##Downscale to unint8 to prevent the cache from exploding in size
def _area_downscale_uint8(
    image: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Downscale uint8 image using area averaging.

    This produces a smooth, anti-aliased overview but can dilute isolated
    bright structures.
    """
    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    source_height, source_width = image.shape

    target_height = min(
        int(target_height),
        source_height,
    )

    target_width = min(
        int(target_width),
        source_width,
    )

    if (
        target_height == source_height
        and target_width == source_width
    ):
        return image.copy()

    pilImage = Image.fromarray(
        image,
        mode="L",
    )

    resized = pilImage.resize(
        (
            target_width,
            target_height,
        ),
        resample=Image.Resampling.BOX,
    )

    return np.asarray(
        resized,
        dtype=np.uint8,
    )

def _uint8_thumbnail_to_png_bytes(
    image: np.ndarray,
) -> bytes:
    """
    Encode a grayscale uint8 thumbnail as compressed PNG bytes.
    """
    buffer = io.BytesIO()

    Image.fromarray(
        np.asarray(image, dtype=np.uint8),
        mode="L",
    ).save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return buffer.getvalue()

def png_bytes_to_data_uri(
    pngBytes: bytes,
) -> str:
    encoded = base64.b64encode(
        pngBytes
    ).decode("ascii")

    return (
        "data:image/png;base64,"
        + encoded
    )

##If you change a channel it will only invalidate part of the cache. 
def build_thumbnail_cache_entry_from_image(
    processed,
    render_mode,
):

    sourceHeight, sourceWidth = processed.shape

    targetHeight, targetWidth = _thumbnail_target_shape(
        sourceHeight,
        sourceWidth,
    )

    displayUint8 = _processed_image_to_uint8(
        processed
    )

    if render_mode == "signal":
        thumbnailUint8 = _max_pool_uint8_to_shape(
            displayUint8,
            targetHeight,
            targetWidth,
        )

    elif render_mode == "smooth":
        thumbnailUint8 = _area_downscale_uint8(
            displayUint8,
            targetHeight,
            targetWidth,
        )

    else:
        raise ValueError(
            f"Unknown thumbnail rendering mode: {render_mode}"
        )

    del displayUint8

    pngBytes = (
        _uint8_thumbnail_to_png_bytes(
            thumbnailUint8
        )
    )

    return {
        "png_bytes": pngBytes,
        "width": int(targetWidth),
        "height": int(targetHeight),
        "source_width": int(sourceWidth),
        "source_height": int(sourceHeight),
    }

def thumbnail_parameter_signature(
    parameter_row,
) -> tuple:
    """
    Build a stable cache signature from one channel's processing settings.

    The parameter row is supplied by viewer.py so this helper does not need
    access to Shiny reactive state.
    """
    if parameter_row is None:
        return ("missing-parameters",)

    return tuple(
        str(parameter_row.get(columnName, ""))
        for columnName in THUMBNAIL_PARAM_COLUMNS
    )

def make_thumbnail_cache_key(
    sample_name: str,
    channel_name: str,
    render_mode: str,
    parameter_row,
) -> tuple:
    """
    Construct the cache key for one thumbnail.

    Including the processing parameter signature means changing a channel's
    processing settings automatically produces a different cache key.
    """
    return (
        str(sample_name),
        str(channel_name),
        str(render_mode),
        THUMBNAIL_MAX_WIDTH,
        THUMBNAIL_MAX_HEIGHT,
        thumbnail_parameter_signature(
            parameter_row
        ),
    )

def generate_sample_thumbnails(
    sample_name: str,
    channel_names: list[str],
    render_mode: str,
    cache: dict,
    *,
    get_parameter_row,
    process_channel,
) -> tuple[dict, int, int, list[str], int]:
    """
    Generate thumbnails for all requested channels in one sample.

    `get_parameter_row` and `process_channel` are supplied by viewer.py.
    This keeps the thumbnail module independent of Shiny reactive state.
    """
    nGenerated = 0
    nReused = 0
    failures = []

    for channelName in channel_names:

        parameterRow = get_parameter_row(
            channelName
        )

        cacheKey = make_thumbnail_cache_key(
            sample_name,
            channelName,
            render_mode,
            parameterRow,
        )

        if cacheKey in cache:
            nReused += 1
            continue

        try:
            processed = process_channel(
                sample_name,
                channelName,
            )

            if processed is None:
                raise ValueError(
                    f"Could not process channel '{channelName}'."
                )

            cache[cacheKey] = (
                build_thumbnail_cache_entry_from_image(
                    processed,
                    render_mode,
                )
            )

            nGenerated += 1

        except Exception as e:
            failures.append(
                f"{sample_name} / {channelName}: {e}"
            )

    return (
        cache,
        nGenerated,
        nReused,
        failures,
        len(channel_names),
    )

def generate_channel_thumbnails(
    channel_name: str,
    sample_names: list[str],
    render_mode: str,
    cache: dict,
    *,
    get_parameter_row,
    process_channel,
) -> tuple[dict, int, int, list[str], int]:
    """
    Generate one channel thumbnail across all supplied samples.
    """
    nGenerated = 0
    nReused = 0
    failures = []

    parameterRow = get_parameter_row(
        channel_name
    )

    for sampleName in sample_names:

        cacheKey = make_thumbnail_cache_key(
            sampleName,
            channel_name,
            render_mode,
            parameterRow,
        )

        if cacheKey in cache:
            nReused += 1
            continue

        try:
            processed = process_channel(
                sampleName,
                channel_name,
            )

            if processed is None:
                raise ValueError(
                    f"Could not process channel '{channel_name}'."
                )

            cache[cacheKey] = (
                build_thumbnail_cache_entry_from_image(
                    processed,
                    render_mode,
                )
            )

            nGenerated += 1

        except Exception as e:
            failures.append(
                f"{sampleName} / {channel_name}: {e}"
            )

    return (
        cache,
        nGenerated,
        nReused,
        failures,
        len(sample_names),
    )