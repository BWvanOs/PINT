#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.ndimage import percentile_filter, convolve
from tifffile import imwrite

# Reuse your loader
from load_tiffs import load_tiffs_raw


# ---------------------- helpers ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Batch-process IMC stacks using viewer parameters.")
    p.add_argument("--input-dir", required=True, help="Folder containing OME-TIFF images")
    p.add_argument("--params-csv", required=True, help="CSV saved by the viewer (per-channel params)")
    p.add_argument("--output-dir", required=True, help="Destination folder (e.g., '<input>/normalized images')")
    return p.parse_args()


def winsorize(arr: np.ndarray, lo_q: float, hi_q: float) -> tuple[np.ndarray, float, float]:
    """Clip by quantiles; return clipped image and (floor, ceil) used."""
    if hi_q <= lo_q:
        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
        return arr, mn, mx
    q_low, q_high = np.quantile(arr, [lo_q, hi_q])
    return np.clip(arr, q_low, q_high), float(q_low), float(q_high)


def to_uint16_stack(stack_f32: np.ndarray, is_unit_flags: list[bool] | None = None) -> np.ndarray:
    """
    Convert to uint16.
    - For channels already in [0,1] (is_unit_flags[i] == True), preserve that scale.
    - For others, scale per-channel to [0,1] before mapping to 0..65535.
    """
    C, H, W = stack_f32.shape
    out = np.zeros((C, H, W), dtype=np.uint16)
    if is_unit_flags is None:
        is_unit_flags = [False] * C

    for i in range(C):
        img = stack_f32[i]
        if is_unit_flags[i]:
            scaled = np.clip(img, 0.0, 1.0)
        else:
            mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
            if mx > mn:
                scaled = (img - mn) / (mx - mn)
            else:
                scaled = np.zeros_like(img, dtype=np.float32)
        out[i] = np.clip(np.round(scaled * 65535.0), 0, 65535).astype(np.uint16)
    return out


def strength_to_percentile(s: float, eps: float = 0.005) -> float:
    """Map Strength s∈[0,1] to percentile p∈(0,1): p = 1 - eps - s*(1 - eps)."""
    s = float(np.clip(s, 0.0, 1.0))
    return 1.0 - eps - s * (1.0 - eps)


def speckle_suppress_percentile_with_neighbors(
    img: np.ndarray,
    p: float,            # percentile in 0..1
    size: int,           # odd window for local percentile
    neighbor_limit: int = 2,
) -> np.ndarray:
    """
    Old method + cluster check:
      - Mark 'bright' if center > local P (computed over size×size)
      - Remove only if ≤ neighbor_limit of the 8 neighbors are also bright
      - Treat zeros as background (must be > 0 to be considered bright)
    """
    if size % 2 == 0:
        size += 1

    thr = percentile_filter(img, percentile=p * 100.0, size=size)
    bright = (img > thr) & (img > 0)

    k = np.ones((3, 3), dtype=np.float32)
    k[1, 1] = 0.0
    neighbor_count = convolve(bright.astype(np.float32), k, mode="reflect")

    remove = bright & (neighbor_count <= float(neighbor_limit))

    out = img.copy()
    out[remove] = 0.0
    return out


def channels_needing_global(params: pd.DataFrame) -> set[str]:
    """Return channel names that request global normalization."""
    need = set()
    if "Channel" not in params.columns:
        return need
    norm_scope = params.get("NormScope", pd.Series(["page"] * len(params)))
    do_norm    = params.get("DoNorm", pd.Series([True] * len(params)))
    for ch, scope, dn in zip(params["Channel"].astype(str), norm_scope, do_norm):
        if str(scope).strip().lower() == "global" and bool(dn):
            need.add(ch)
    return need


def precompute_global_minmax(
    images: dict[str, np.ndarray],
    channels: dict[str, list[str]],
    needed: set[str],
    params: pd.DataFrame,
) -> dict[str, tuple[float, float] | None]:
    """
    Compute global (min, max) per channel across ALL samples for channels in `needed`.

    If that channel has DoWinsor=True and valid Low/High in params, we:
      - compute per-image winsor quantiles [Low, High]
      - take global min(q_lo) and max(q_hi) as (gmin, gmax)

    Otherwise we fall back to raw global min/max.

    Returns {channel_name: (gmin, gmax) or None}.
    """
    out: dict[str, tuple[float, float] | None] = {ch: None for ch in needed}
    if not needed:
        return out

    # Quick lookup of winsor settings per channel
    winsor_cfg: dict[str, tuple[bool, float, float]] = {}
    if "Channel" in params.columns:
        for _, row in params.iterrows():
            ch = str(row["Channel"])
            if ch not in needed:
                continue
            do_win = bool(row.get("DoWinsor", False))
            lo = float(row.get("Low", 0.0))
            hi = float(row.get("High", 1.0))
            winsor_cfg[ch] = (do_win, lo, hi)

    for ch in needed:
        do_win, lo, hi = winsor_cfg.get(ch, (False, 0.0, 1.0))

        gmin = np.inf
        gmax = -np.inf
        found = False

        for sample, arr in images.items():
            chlist = channels.get(sample, [])
            if ch not in chlist:
                continue

            idx = chlist.index(ch)
            frame = arr[idx].astype(np.float32)

            if do_win and hi > lo:
                # Use the winsor quantiles as that image's effective range
                q_low, q_high = np.quantile(frame, [lo, hi])
                mn, mx = float(q_low), float(q_high)
            else:
                # Fallback: raw range
                mn = float(np.nanmin(frame))
                mx = float(np.nanmax(frame))

            if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                gmin = min(gmin, mn)
                gmax = max(gmax, mx)
                found = True

        if found and np.isfinite(gmin) and np.isfinite(gmax) and gmax > gmin:
            out[ch] = (float(gmin), float(gmax))
        else:
            out[ch] = None

    return out


# ---------------------- main ----------------------
def main():
    args = parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the parameter table saved by the viewer and also save a copy in the output folder
    params = pd.read_csv(args.params_csv)
    (out_dir / "parameter_table_used.csv").write_text(params.to_csv(index=False))

    # Friendly defaults if some columns are missing
    defaults = {
        "DoWinsor": False, "Low": 0.0, "High": 1.0,
        "DoThr": False, "ThrVal": 0.0,
        "Noise": False, "NStr": 0.0, "NPrctl": 0.995, "WinSz": 3,
        "DoNorm": True, "NormScope": "page",
        "DoAsinh": False, "Cofac": 5,
    }
    for col, val in defaults.items():
        if col not in params.columns:
            params[col] = val

    # Make sure columns have reasonable types/values
    params["Channel"]   = params["Channel"].astype(str)
    params["NormScope"] = params["NormScope"].astype(str).str.lower().fillna("page")

    # Load stacks
    imgs, chs = load_tiffs_raw(str(in_dir))

    # Precompute global min/max where requested (RAW frames, across all samples)
    need_global = channels_needing_global(params)
    global_minmax = precompute_global_minmax(imgs, chs, need_global)  # {ch: (gmin,gmax) | None}

    # Collect per-frame results (like your R 'results')
    results_rows: list[dict] = []

    for img_name, stack in imgs.items():
        chlist = chs[img_name]  # channel names
        C, H, W = stack.shape
        processed = np.zeros((C, H, W), dtype=np.float32)
        # track which channels ended up in [0,1] already (for uint16 writing)
        ch_is_unit = [False] * C

        for j, ch in enumerate(chlist):
            frame = stack[j].astype(np.float32).copy()

            preMin = float(np.nanmin(frame))
            preMax = float(np.nanmax(frame))

            # Per-channel params (fallback to defaults)
            row = params.loc[params["Channel"] == ch]
            r = row.iloc[0] if not row.empty else pd.Series(defaults)

            doWin   = bool(r.get("DoWinsor", defaults["DoWinsor"]))
            lo_q    = float(r.get("Low",      defaults["Low"]))
            hi_q    = float(r.get("High",     defaults["High"]))

            doThr   = bool(r.get("DoThr",     defaults["DoThr"]))
            thrN    = float(r.get("ThrVal",   defaults["ThrVal"]))     # 0..1

            doNoise = bool(r.get("Noise",     defaults["Noise"]))
            sVal    = float(r.get("NStr",     defaults["NStr"]))       # UI strength 0..1
            pCsv    = float(r.get("NPrctl",   defaults["NPrctl"]))     # if present in CSV
            winSz   = int(r.get("WinSz",      defaults["WinSz"]))

            doNorm  = bool(r.get("DoNorm",    defaults["DoNorm"]))
            scope   = str(r.get("NormScope",  defaults["NormScope"])).lower()
            doAsinh = bool(r.get("DoAsinh",   defaults["DoAsinh"]))
            cofac   = float(r.get("Cofac",    defaults["Cofac"]))

            # Resolve percentile used: prefer NPrctl if provided; else derive from strength
            if np.isfinite(pCsv) and 0.0 < pCsv < 1.0:
                pUsed = pCsv
            else:
                pUsed = strength_to_percentile(sVal)

            # 1) Winsorization
            if doWin:
                frame, rawFloor, rawCeil = winsorize(frame, lo_q, hi_q)
            else:
                rawFloor, rawCeil = preMin, preMax

            # 2) Global threshold (cutoff = thr * max(after winsor))
            if doThr and thrN > 0.0:
                mx = float(np.nanmax(frame))
                if mx > 0.0:
                    cutoff = thrN * mx
                    frame[frame < cutoff] = 0.0
                    thr_used = float(cutoff)
                else:
                    thr_used = np.nan
            else:
                thr_used = np.nan

            # 3) Local percentile speckle removal + neighbor rule (viewer parity)
            pct_used = np.nan
            if doNoise and winSz >= 1 and (0.0 < pUsed < 1.0):
                frame = speckle_suppress_percentile_with_neighbors(
                    frame, p=pUsed, size=winSz, neighbor_limit=2
                )
                pct_used = float(pUsed)

            # 4) Optional arcsinh (after denoise, before normalization)
            asinh_applied = False
            asinh_cofac = np.nan
            if doAsinh:
                if cofac <= 0:
                    cofac = 5.0
                frame = np.arcsinh(frame / float(cofac))
                asinh_applied = True
                asinh_cofac = float(cofac)

            # 5) Optional normalization
            normalized = False
            gmin_used = np.nan
            gmax_used = np.nan
            if doNorm:
                if scope == "global":
                    gpair = global_minmax.get(ch, None)
                    if gpair and gpair[1] > gpair[0]:
                        gmin, gmax = gpair
                        frame = (frame - gmin) / (gmax - gmin)
                        normalized = True
                        ch_is_unit[j] = True
                        gmin_used, gmax_used = float(gmin), float(gmax)
                    else:
                        # Fallback to per-page if global unavailable/degenerate
                        mn, mx = float(np.nanmin(frame)), float(np.nanmax(frame))
                        if mx > mn:
                            frame = (frame - mn) / (mx - mn)
                            normalized = True
                            ch_is_unit[j] = True
                else:
                    # Per-page normalization
                    mn, mx = float(np.nanmin(frame)), float(np.nanmax(frame))
                    if mx > mn:
                        frame = (frame - mn) / (mx - mn)
                        normalized = True
                        ch_is_unit[j] = True

            processed[j] = frame

            # Record parameters/results
            results_rows.append({
                "Image":          img_name,
                "Channel":        ch,
                "Page":           j + 1,  # 1-based
                "MinValueRaw":    preMin,
                "MaxValueRaw":    preMax,
                "WinsorLower":    float(rawFloor),
                "WinsorUpper":    float(rawCeil),
                "Threshold":      thr_used,
                "Strength":       float(sVal),
                "Percentile":     pct_used,
                "WindowSize":     int(winSz),
                "NeighborsRule":  "remove if ≤2 bright neighbors",
                "Normalized":     bool(normalized),
                "NormScope":      scope,
                "GlobalMinUsed":  gmin_used,
                "GlobalMaxUsed":  gmax_used,
                "AsinhApplied":   bool(asinh_applied),
                "AsinhCofactor":  asinh_cofac,
            })

        # Write outputs
        out16 = out_dir / f"{img_name} Normalized.tiff"
        out32 = out_dir / f"{img_name} Normalized 32bit.tiff"

        # 32-bit float (exact processed values)
        imwrite(
            out32,
            processed.astype(np.float32),
            dtype=np.float32,
            ome=True,
            metadata={"axes": "CYX"},
        )

        # 16-bit:
        # - channels already normalized to [0,1] keep that scale
        # - others get per-channel stretch (so they're usable)
        u16 = to_uint16_stack(processed, is_unit_flags=ch_is_unit)
        imwrite(
            out16,
            u16,
            dtype=np.uint16,
            ome=True,
            metadata={"axes": "CYX"},
        )

        print(f"✓ Wrote: {out16.name}, {out32.name}")

    # Save results sheet(s)
    results = pd.DataFrame(results_rows)
    results_csv  = out_dir / "Parameters Of all images.csv"
    results_xlsx = out_dir / "Parameters Of all images.xlsx"
    results.to_csv(results_csv, index=False)
    try:
        results.to_excel(results_xlsx, index=False)  # requires openpyxl
        print(f"✓ Saved: {results_xlsx.name}")
    except Exception as e:
        print(f"⚠️ Could not write XLSX ({e}); CSV saved instead: {results_csv.name}")

    print("All done.")


if __name__ == "__main__":
    main()
