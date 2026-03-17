"""Pure image processing helpers used by the Shiny viewer and batch analysis.

Design intent
  - Side-effect free (no Shiny session/input access).
  - Deterministic given inputs.
  - Shared across interactive viewer and batch pipelines.
"""

from __future__ import annotations

import numpy as np

from scipy.ndimage import convolve, percentile_filter


def clamp01(x: float) -> float:
    """Clamp a float to [0, 1]."""
    return float(np.clip(float(x), 0.0, 1.0))


def sanitize_cofactor(cofactor: int | float) -> int:
    """Validate/clamp the arcsinh cofactor.

    The viewer currently expects an integer cofactor and keeps it within a
    reasonable range.
    """
    try:
        cofac = int(float(cofactor))
    except Exception:
        cofac = 5
    return max(2, min(10, cofac))


def winsorize_with_bounds(
    img: np.ndarray,
    lo_q: float,
    hi_q: float,
    min_upper_bound: float = 5.0,
) -> tuple[np.ndarray, float, float]:
    """Winsorize an image by quantiles and also return the (low, high) bounds used.

    Important:
      - The upper winsor bound is floored at `min_upper_bound`.
      - This prevents rare positive staining from being clipped into background.

    Example:
      q_high = 3.7, min_upper_bound = 5.0 -> use 5.0 as upper clip bound
    """
    lo_q = clamp01(lo_q)
    hi_q = clamp01(hi_q)

    try:
        min_upper_bound = float(min_upper_bound)
    except Exception:
        min_upper_bound = 5.0

    if hi_q <= lo_q:
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        return img, mn, mx

    q_low, q_high = np.nanquantile(img, [lo_q, hi_q])

    if np.isnan(q_low) or np.isnan(q_high):
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        return img, mn, mx

    if np.isfinite(min_upper_bound):
        q_high = max(float(q_high), float(min_upper_bound))

    # Safety: make sure upper is not below lower
    if q_high < q_low:
        q_high = q_low

    out = np.clip(img, q_low, q_high)
    return out, float(q_low), float(q_high)


def apply_winsor(
    img: np.ndarray,
    lo_q: float,
    hi_q: float,
    min_upper_bound: float = 5.0,
) -> np.ndarray:
    """Winsorize an image by clipping to quantiles.

    The upper clip bound is floored at `min_upper_bound` so sparse true signal
    is not clipped down into the Hyperion background range.
    """
    return winsorize_with_bounds(
        img,
        lo_q,
        hi_q,
        min_upper_bound=min_upper_bound,
    )[0]


def apply_threshold_absolute(img: np.ndarray, thr_abs: float) -> np.ndarray:
    """Zero pixels below an absolute cutoff value (IMC dual-count background)."""
    try:
        thr_abs = float(thr_abs)
    except Exception:
        thr_abs = 0.0
    if not np.isfinite(thr_abs) or thr_abs <= 0.0:
        return img
    return np.where(img >= thr_abs, img, 0.0)


def apply_threshold_fraction_of_max(img: np.ndarray, thr_fraction: float) -> np.ndarray:
    """Zero pixels below `thr_fraction * max(img)`.

    `thr_fraction` is expected in [0, 1]. Values <= 0 return img unchanged.
    """
    thr_fraction = clamp01(thr_fraction)
    if thr_fraction <= 0.0:
        return img
    m = float(np.nanmax(img))
    cutoff = thr_fraction * (m if m > 0 else 1.0)
    return np.where(img >= cutoff, img, 0.0)


def strength_to_percentile(s: float, eps: float = 0.005) -> float:
    """Map a UI "strength" slider (0..1) to a local percentile cutoff.

    Higher strength -> lower percentile (more aggressive).
    """
    s = clamp01(s)
    return 1.0 - eps - s * (1.0 - eps)


def apply_speckle_suppress(
    img: np.ndarray,
    size: int,
    perc: float,
    neighbor_limit: int = 2,
) -> np.ndarray:
    """Suppress isolated bright pixels.

    Algorithm:
      1) Mark pixels "bright" if center > local percentile in a size×size window.
      2) Count how many of 8 neighbors are also bright.
      3) Remove (set to 0) bright pixels that have <= neighbor_limit bright neighbors.

    Notes:
      - size is forced odd
      - perc expected in [0,1] (e.g. 0.995)
    """
    size = int(size)
    if size < 1:
        return img
    if size % 2 == 0:
        size += 1

    perc = clamp01(perc)

    thr = percentile_filter(img, percentile=perc * 100.0, size=size)
    bright = (img > thr) & (img > 0)

    k = np.ones((3, 3), dtype=np.float32)
    k[1, 1] = 0.0
    neighbor_count = convolve(bright.astype(np.float32), k, mode="reflect")

    remove = bright & (neighbor_count <= float(neighbor_limit))
    out = img.copy()
    out[remove] = 0.0
    return out


def arcsinh_transform(img: np.ndarray, cofactor: int | float = 5) -> np.ndarray:
    """Apply arcsinh transform with a given cofactor."""
    cofac = sanitize_cofactor(cofactor)
    return np.arcsinh(img / float(cofac))


def normalize_minmax(img: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Normalize image to [0,1] using min/max.

    If vmin/vmax are not provided, uses nanmin/nanmax of `img`.
    """
    x = img.astype(np.float32, copy=False)
    if vmin is None:
        vmin = float(np.nanmin(x))
    if vmax is None:
        vmax = float(np.nanmax(x))

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        return np.zeros_like(x, dtype=np.float32)

    y = (x - vmin) / (vmax - vmin)
    return np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def global_minmax_for_channel(images_dict: dict, channels_dict: dict, channel_name: str):
    """Compute global (min,max) for a channel across all samples.

    Returns None if the channel is missing everywhere or has a degenerate range.
    """
    gmin = np.inf
    gmax = -np.inf
    found = False

    for sample, arr in images_dict.items():
        chlist = channels_dict.get(sample, [])
        if channel_name not in chlist:
            continue
        idx = chlist.index(channel_name)
        ch = arr[idx]
        mn = float(np.nanmin(ch))
        mx = float(np.nanmax(ch))
        if np.isfinite(mn) and np.isfinite(mx):
            gmin = min(gmin, mn)
            gmax = max(gmax, mx)
            found = True

    if not found or not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        return None

    return float(gmin), float(gmax)


def global_winsor_range_for_channel(
    images_dict: dict,
    channels_dict: dict,
    channel_name: str,
    lo: float,
    hi: float,
    min_upper_bound: float = 5.0,
):
    """Compute global winsor range for a channel.

    The global range is defined as:
      - min over samples of q_lo
      - max over samples of adjusted q_hi

    Where adjusted q_hi = max(raw q_hi, min_upper_bound)
    """
    lo = clamp01(lo)
    hi = clamp01(hi)
    if hi <= lo:
        return None

    try:
        min_upper_bound = float(min_upper_bound)
    except Exception:
        min_upper_bound = 5.0

    gmin = np.inf
    gmax = -np.inf
    found = False

    for sample, arr in images_dict.items():
        chlist = channels_dict.get(sample, [])
        if channel_name not in chlist:
            continue

        idx = chlist.index(channel_name)
        qlo, qhi = np.nanquantile(arr[idx], [lo, hi])

        if np.isnan(qlo) or np.isnan(qhi):
            continue

        qhi = max(float(qhi), float(min_upper_bound))

        gmin = min(gmin, float(qlo))
        gmax = max(gmax, float(qhi))
        found = True

    if not found or not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        return None

    return float(gmin), float(gmax)


def image_winsor_range(
    images_dict: dict,
    channels_dict: dict,
    channel_name: str,
    sample: str,
    lo: float,
    hi: float,
    min_upper_bound: float = 5.0,
):
    """Compute winsor range for a single sample/channel."""
    if not sample or sample not in images_dict:
        return None

    chlist = channels_dict.get(sample, [])
    if channel_name not in chlist:
        return None

    lo = clamp01(lo)
    hi = clamp01(hi)
    if hi <= lo:
        return None

    try:
        min_upper_bound = float(min_upper_bound)
    except Exception:
        min_upper_bound = 5.0

    idx = chlist.index(channel_name)
    qlo, qhi = np.nanquantile(images_dict[sample][idx], [lo, hi])

    if np.isnan(qlo) or np.isnan(qhi):
        return None

    qhi = max(float(qhi), float(min_upper_bound))
    return float(qlo), float(qhi)

def process_image_pipeline(
    img: np.ndarray,
    *,
    do_winsor: bool = False,
    winsor_low: float = 0.0,
    winsor_high: float = 1.0,
    do_abs_threshold: bool = False,
    abs_threshold: float = 0.0,
    do_fraction_threshold: bool = False,
    thr_fraction: float = 0.0,
    do_noise: bool = False,
    noise_strength: float = 0.0,
    window_size: int = 3,
    do_asinh: bool = False,
    asinh_cofactor: int | float = 5,
    do_norm: bool = True,
    norm_vmin: float | None = None,
    norm_vmax: float | None = None,
    winsor_min_upper_bound: float = 5.0,
) -> np.ndarray:
    """
    Apply the same processing order used by the viewer/batch pipeline.

    Order:
      1) winsorize
      2) absolute threshold
      3) fraction-of-max threshold
      4) speckle suppression
      5) arcsinh
      6) min-max normalization (optional)

    If do_norm=True:
      - norm_vmin/norm_vmax can be provided for global normalization
      - if they are None, per-image min/max is used
    """
    out = img.astype(np.float32, copy=True)

    if do_winsor:
        lo = clamp01(winsor_low)
        hi = clamp01(winsor_high)
        if hi > lo:
            out = apply_winsor(
                out,
                lo,
                hi,
                min_upper_bound=winsor_min_upper_bound,
            )

    if do_abs_threshold:
        out = apply_threshold_absolute(out, abs_threshold)

    if do_fraction_threshold:
        thr_fraction = clamp01(thr_fraction)
        if thr_fraction > 0.0:
            out = apply_threshold_fraction_of_max(out, thr_fraction)

    if do_noise:
        wsize = max(1, int(window_size))
        s = clamp01(noise_strength)
        perc = strength_to_percentile(s)
        out = apply_speckle_suppress(out, size=wsize, perc=perc, neighbor_limit=2)

    if do_asinh:
        cofac = sanitize_cofactor(asinh_cofactor)
        out = arcsinh_transform(out, cofac)

    if do_norm:
        out = normalize_minmax(out, vmin=norm_vmin, vmax=norm_vmax)

    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)