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


def winsorize_with_bounds(img: np.ndarray, lo_q: float, hi_q: float) -> tuple[np.ndarray, float, float]:
    """Winsorize an image by quantiles and also return the (low, high) bounds used.

    If the quantiles are invalid (hi<=lo) the image is returned unchanged and the
    bounds are the raw min/max.
    """
    lo_q = clamp01(lo_q)
    hi_q = clamp01(hi_q)

    if hi_q <= lo_q:
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        return img, mn, mx

    q_low, q_high = np.nanquantile(img, [lo_q, hi_q])
    out = np.clip(img, q_low, q_high)
    return out, float(q_low), float(q_high)


def apply_winsor(img: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    """Winsorize an image by clipping to quantiles."""
    return winsorize_with_bounds(img, lo_q, hi_q)[0]


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


def apply_threshold_absolute(img: np.ndarray, thr_abs: float) -> np.ndarray:
    """Zero pixels below an absolute cutoff value.

    Useful for IMC "dual count" background suppression, where you may want to
    hard-zero everything below e.g. 2.5.

    Parameters
    ----------
    img:
        Input image.
    thr_abs:
        Absolute cutoff in the same units as the image (raw counts / duals).
        Values <= 0 return img unchanged.
    """
    try:
        thr_abs = float(thr_abs)
    except Exception:
        thr_abs = 0.0

    if not np.isfinite(thr_abs) or thr_abs <= 0.0:
        return img

    return np.where(img >= thr_abs, img, 0.0)


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


def normalize_minmax(img: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Normalize image to [0,1] using min/max.

    If vmin/vmax are not provided, uses nanmin/nanmax of `img`.
    """
    if vmin is None:
        vmin = float(np.nanmin(img))
    if vmax is None:
        vmax = float(np.nanmax(img))
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return img
    return (img - float(vmin)) / (float(vmax) - float(vmin))


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
):
    """Compute global winsor range for a channel.

    The global range is defined as:
      - min over samples of q_lo
      - max over samples of q_hi

    Returns None if no sample provides a valid quantile pair.
    """
    lo = clamp01(lo)
    hi = clamp01(hi)
    if hi <= lo:
        return None

    gmin, gmax = np.inf, -np.inf
    found = False

    for sample, arr in images_dict.items():
        chlist = channels_dict.get(sample, [])
        if channel_name not in chlist:
            continue
        idx = chlist.index(channel_name)
        qlo, qhi = np.nanquantile(arr[idx], [lo, hi])
        if np.isnan(qlo) or np.isnan(qhi):
            continue
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

    idx = chlist.index(channel_name)
    qlo, qhi = np.nanquantile(images_dict[sample][idx], [lo, hi])
    if np.isnan(qlo) or np.isnan(qhi):
        return None
    return float(qlo), float(qhi)
