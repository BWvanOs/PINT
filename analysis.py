#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd

from scipy.ndimage import percentile_filter
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


def local_percentile(arr: np.ndarray, p01_100: float, size: int) -> np.ndarray:
    """Local percentile (p in 0..100) with odd window size."""
    if size < 1:
        size = 1
    if size % 2 == 0:
        size += 1
    return percentile_filter(arr, percentile=p01_100, size=size)


def to_uint16_stack(stack_f32: np.ndarray) -> np.ndarray:
    """Scale each channel independently to [0, 65535] and cast to uint16."""
    c, h, w = stack_f32.shape
    out = np.zeros((c, h, w), dtype=np.uint16)
    for i in range(c):
        img = stack_f32[i]
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        if mx > mn:
            scaled = (img - mn) / (mx - mn)
        else:
            scaled = np.zeros_like(img, dtype=np.float32)
        out[i] = np.clip(np.round(scaled * 65535.0), 0, 65535).astype(np.uint16)
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
        "Noise": False, "NStr": 0.0, "WinSz": 3,
        "DoNorm": True,
        "DoAsinh": False, "Cofac": 5,
    }
    for col, val in defaults.items():
        if col not in params.columns:
            params[col] = val

    # Load stacks
    imgs, chs = load_tiffs_raw(str(in_dir))

    # Collect per-frame results (like your R 'results')
    results_rows: list[dict] = []

    for img_name, stack in imgs.items():
        chlist = chs[img_name]  # channel names
        C, H, W = stack.shape
        processed = np.zeros((C, H, W), dtype=np.float32)

        for j, ch in enumerate(chlist):
            frame = stack[j].astype(np.float32).copy()

            preMin = float(np.nanmin(frame))
            preMax = float(np.nanmax(frame))

            # Per-channel params (fallback to defaults)
            row = params.loc[params["Channel"] == ch]
            r = row.iloc[0] if not row.empty else pd.Series(defaults)

            doWin  = bool(r.get("DoWinsor", defaults["DoWinsor"]))
            lo_q   = float(r.get("Low",      defaults["Low"]))
            hi_q   = float(r.get("High",     defaults["High"]))

            doThr  = bool(r.get("DoThr",     defaults["DoThr"]))
            thrN   = float(r.get("ThrVal",   defaults["ThrVal"]))     # 0..1

            doNoise = bool(r.get("Noise",    defaults["Noise"]))
            pVal    = float(r.get("NStr",    defaults["NStr"]))       # 0..1 (percentile)
            winSz   = int(r.get("WinSz",     defaults["WinSz"]))

            doNorm  = bool(r.get("DoNorm",   defaults["DoNorm"]))
            doAsinh = bool(r.get("DoAsinh",  defaults["DoAsinh"]))
            cofac   = float(r.get("Cofac",   defaults["Cofac"]))

            # 1) Winsorization
            if doWin:
                frame, rawFloor, rawCeil = winsorize(frame, lo_q, hi_q)
            else:
                rawFloor, rawCeil = preMin, preMax

            # 2) Global threshold (mirror viewer: cutoff = thr * max(after winsor))
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

            # 3) Local percentile speckle removal (bright-speckle removal)
            #    Zero values ABOVE the local percentile threshold
            if doNoise and pVal > 0.0 and winSz > 1:
                lp = local_percentile(frame, pVal * 100.0, winSz)
                frame[frame > lp] = 0.0
                pct_used = float(pVal)
            else:
                pct_used = np.nan

            # 4) Arcsinh (after denoise, before normalization)
            asinh_applied = False
            asinh_cofac = np.nan
            if doAsinh:
                if cofac <= 0:
                    cofac = 5.0
                frame = np.arcsinh(frame / float(cofac))
                asinh_applied = True
                asinh_cofac = float(cofac)

            # 5) Optional normalization to [0,1] (viewer always normalizes for display;
            #    here it's controlled by DoNorm, per your design)
            normalized = False
            if doNorm:
                mn, mx = float(np.nanmin(frame)), float(np.nanmax(frame))
                if mx > mn:
                    frame = (frame - mn) / (mx - mn)
                else:
                    frame[:] = 0.0
                normalized = True

            processed[j] = frame

            # Record parameters like the R 'results'
            results_rows.append({
                "Image":          img_name,
                "Page":           j + 1,  # 1-based index
                "MinValue":       preMin,
                "MaxValue":       preMax,
                "WinsorLower":    float(rawFloor),
                "WinsorUpper":    float(rawCeil),
                "Threshold":      thr_used,
                "Percentile":     pct_used,
                "Normalized":     bool(normalized),
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

        # 16-bit (per-channel min–max scaling)
        u16 = to_uint16_stack(processed)
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
