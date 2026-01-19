#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imwrite

# Reuse your loader
from load_tiffs import load_tiffs_raw

from pint_app.core.processing import (
    winsorize_with_bounds,
    apply_threshold_fraction_of_max,
    strength_to_percentile,
    apply_speckle_suppress,
    arcsinh_transform,
    sanitize_cofactor,
    normalize_minmax,
    global_minmax_for_channel,
    global_winsor_range_for_channel,
)
from pint_app.core.params import coerce_params_df, channels_needing_global, DEFAULTS


# ---------------------- helpers ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Batch-process IMC stacks using viewer parameters.")
    p.add_argument("--input-dir", required=True, help="Folder containing OME-TIFF images")
    p.add_argument("--params-csv", required=True, help="CSV saved by the viewer (per-channel params)")
    p.add_argument("--output-dir", required=True, help="Destination folder (e.g., '<input>/normalized images')")
    return p.parse_args()

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

    # Lookup per-channel winsor settings (so global norms can match viewer behaviour)
    winsor_cfg: dict[str, tuple[bool, float, float]] = {}
    if params is not None and not params.empty and "Channel" in params.columns:
        for _, row in params.iterrows():
            ch = str(row.get("Channel"))
            if ch not in needed:
                continue
            do_win = bool(row.get("DoWinsor", False))
            lo = float(row.get("Low", 0.0))
            hi = float(row.get("High", 1.0))
            winsor_cfg[ch] = (do_win, lo, hi)

    for ch in needed:
        do_win, lo, hi = winsor_cfg.get(ch, (False, 0.0, 1.0))

        if do_win and hi > lo:
            out[ch] = global_winsor_range_for_channel(images, channels, ch, lo, hi)
        else:
            out[ch] = global_minmax_for_channel(images, channels, ch)

    return out


# ---------------------- main ----------------------
def main():
    args = parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the parameter table saved by the viewer and coerce it to the current schema.
    params_raw = pd.read_csv(args.params_csv)
    params = coerce_params_df(params_raw)
    params["NormScope"] = params["NormScope"].astype(str).str.lower().fillna("page")

    # Save the exact table actually used (after schema coercion) for provenance.
    (out_dir / "parameter_table_used.csv").write_text(params.to_csv(index=False))

    # Load stacks
    imgs, chs = load_tiffs_raw(str(in_dir))

    #Precompute global min/max where requested (RAW frames, across all samples)
    need_global = channels_needing_global(params)
    global_minmax = precompute_global_minmax(imgs, chs, need_global, params)  # {ch: (gmin,gmax) | None}


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
            r = row.iloc[0] if not row.empty else pd.Series(DEFAULTS)

            doWin   = bool(r.get("DoWinsor", DEFAULTS["DoWinsor"]))
            lo_q    = float(r.get("Low",      DEFAULTS["Low"]))
            hi_q    = float(r.get("High",     DEFAULTS["High"]))

            doThr   = bool(r.get("DoThr",     DEFAULTS["DoThr"]))
            thrN    = float(r.get("ThrVal",   DEFAULTS["ThrVal"]))     # 0..1

            doNoise = bool(r.get("Noise",     DEFAULTS["Noise"]))
            sVal    = float(r.get("NStr",     DEFAULTS["NStr"]))       # UI strength 0..1
            pCsv    = float(r.get("NPrctl",   DEFAULTS["NPrctl"]))     # if present in CSV
            winSz   = int(r.get("WinSz",      DEFAULTS["WinSz"]))

            doNorm  = bool(r.get("DoNorm",    DEFAULTS["DoNorm"]))
            scope   = str(r.get("NormScope",  DEFAULTS["NormScope"])).lower()
            doAsinh = bool(r.get("DoAsinh",   DEFAULTS["DoAsinh"]))
            cofac   = float(r.get("Cofac",    DEFAULTS["Cofac"]))

            # Resolve percentile used: prefer NPrctl if provided; else derive from strength
            if np.isfinite(pCsv) and 0.0 < pCsv < 1.0:
                pUsed = pCsv
            else:
                pUsed = strength_to_percentile(sVal)

            # 1) Winsorization
            if doWin:
                frame, rawFloor, rawCeil = winsorize_with_bounds(frame, lo_q, hi_q)
            else:
                rawFloor, rawCeil = preMin, preMax

            # 2) Global threshold (cutoff = thr * max(after winsor))
            if doThr and thrN > 0.0:
                mx = float(np.nanmax(frame))
                if mx > 0.0:
                    cutoff = thrN * mx
                    frame = apply_threshold_fraction_of_max(frame, thrN)
                    thr_used = float(cutoff)
                else:
                    thr_used = np.nan
            else:
                thr_used = np.nan

            # 3) Local percentile speckle removal + neighbor rule (viewer parity)
            pct_used = np.nan
            if doNoise and winSz >= 1 and (0.0 < pUsed < 1.0):
                frame = apply_speckle_suppress(frame, size=winSz, perc=pUsed, neighbor_limit=2)
                pct_used = float(pUsed)

            # 4) Optional arcsinh (after denoise, before normalization)
            asinh_applied = False
            asinh_cofac = np.nan
            if doAsinh:
                used_cofac = sanitize_cofactor(cofac)
                frame = arcsinh_transform(frame, used_cofac)
                asinh_applied = True
                asinh_cofac = float(used_cofac)

            # 5) Optional normalization
            normalized = False
            gmin_used = np.nan
            gmax_used = np.nan
            if doNorm:
                if scope == "global":
                    gpair = global_minmax.get(ch, None)
                    if gpair and gpair[1] > gpair[0]:
                        gmin, gmax = gpair
                        frame = normalize_minmax(frame, gmin, gmax)
                        normalized = True
                        ch_is_unit[j] = True
                        gmin_used, gmax_used = float(gmin), float(gmax)
                    else:
                        # Fallback to per-page if global unavailable/degenerate
                        mn, mx = float(np.nanmin(frame)), float(np.nanmax(frame))
                        if mx > mn:
                            frame = normalize_minmax(frame, mn, mx)
                            normalized = True
                            ch_is_unit[j] = True
                else:
                    # Per-page normalization
                    mn, mx = float(np.nanmin(frame)), float(np.nanmax(frame))
                    if mx > mn:
                        frame = normalize_minmax(frame, mn, mx)
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
