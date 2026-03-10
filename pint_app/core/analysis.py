#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imwrite

from pint_app.core.load_tiffs import load_tiffs_raw
from pint_app.core.processing import (
    winsorize_with_bounds,
    apply_threshold_absolute,
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


# ----------------------> helpers <----------------------
## Use these arguments to run the script from the command line
WINSOR_MIN_UPPER_BOUND = 5.0

def parse_args():
    p = argparse.ArgumentParser(description="Batch-process IMC stacks using viewer parameters.")
    p.add_argument("--input-dir", required=True, help="Folder containing OME-TIFF images")
    p.add_argument("--params-csv", required=True, help="CSV saved by the viewer (per-channel params)")
    p.add_argument("--output-dir", required=True, help="Destination folder (e.g., '<input>/normalized images')")
    return p.parse_args()

## This is used to export the normalized images to unsigned 16bit TIFF's to use for downstream analysis
## The 16bit TIFFs are an addition to the 32bit output. Most downstream tools will take the 32bit images.
def to_uint16_stack(stack_f32: np.ndarray, is_unit_flags: list[bool] | None = None) -> np.ndarray:
    """
    Convert normalized images to usingned int16 TIFF stakcs.
    - For channels already min/max scaled [0,1], flagged by (is_unit_flags[i] == True), preserve that scale.
    - For non-scaled images, scale per-channel to [0,1] before mapping to 0-65535.
    - Note that these 16 bit images are thus normalized, regardless of pipeline setting. This is not true for the 32bit version!
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

## Precompute global min/max ranges per channel across all images for channels that use global normalization.
## This is computed once at the start of the batch run and reused for every image, to avoid recomputing the whole range again
## If winsorization is enabled for a channel (DoWinsor=True and High > Low), the global range is based on
## per-image winsor quantiles; so the range is "corrected" for the winsorization settings. If not, it will fall back to raw global min/max.
## Important note: this function is a reflection of the "params" table; if "params" change.
def precompute_global_minmax(
    images: dict[str, np.ndarray],
    channels: dict[str, list[str]],
    needed: set[str],
    params: pd.DataFrame,
) -> dict[str, tuple[float, float] | None]:
    """
    Compute global (min, max) per channel across all samples for channels that are input in 'needed'.
    'needed' contains all the channel names that are input by the user, where te input is global normalization'

    If that channel has DoWinsor=True and valid Low/High (so High > Low) in params:
      - compute per-image winsor quantiles [Low, High]
      - take global min(q_lo) and max(q_hi) as (gmin, gmax)
    In determines the bounds of the channel and returns them

    If that channel has invalid Low/High winsor parameters, fall back to raw global min/max.

    Returns dictionary: {channel_name: (gmin, gmax) or None}.
    """
    out: dict[str, tuple[float, float] | None] = {ch: None for ch in needed}
    if not needed:
        return out

    ##Lookup per-channel winsor settings (so global norms can match viewer behaviour)
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
            out[ch] = global_winsor_range_for_channel(
                images,
                channels,
                ch,
                lo,
                hi,
                min_upper_bound=WINSOR_MIN_UPPER_BOUND,
            )
        else:
            out[ch] = global_minmax_for_channel(images, channels, ch)

    return out


# ---------------------- main ----------------------
def main():
    ##------------------------------------> Parse CLi argument <---------------------------##
    ##Pars the arguments from CLI.
    args = parse_args()
    ##Resolve the paths
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    ##Make sure the output directory actually exist
    out_dir.mkdir(parents=True, exist_ok=True)

    ##------------------------------------> Load images and coerce all parameters <---------------------------##
    ##Read the parameter table saved by the viewer and coerce it to the current schema.
    ##It runs it through coerce_params_df to format it correctly and add column that were missing if the input CSV was from an older version
    params_raw = pd.read_csv(args.params_csv)
    params = coerce_params_df(params_raw)
    ##Normalize formatting of the normalization scope to lower case. Also replace any empty values, NA's by "page" should there be empty values
    params["NormScope"] = params["NormScope"].astype(str).str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA}).fillna("page")
    ##Save the exact table actually used (after schema coercion).
    (out_dir / "parameter_table_used.csv").write_text(params.to_csv(index=False))


    ##------------------------------------> Load the image stacks <---------------------------##
    ##Load image stacks through load_tiffs_raw, return np.ndarray dict with shape C, H, W
    ##Channels are loaded as list of channel names with length C, matching the image array
    imgs, chs = load_tiffs_raw(str(in_dir))

    ##------------------------------------> Precompute the global image ranges if required <---------------------------##
    #Precompute global min/max where requested (RAW frames, across all samples)
    need_global = channels_needing_global(params)
    global_minmax = precompute_global_minmax(imgs, chs, need_global, params)  # {ch: (gmin,gmax) | None}
    #Collect per-frame results and derived values
    results_rows: list[dict] = []

    ##------------------------------------> image processing pipeline <---------------------------##
    ##------------------------------------> process each image <---------------------------##
    ##for each image, take name and value 
    for img_name, stack in imgs.items():
        chlist = chs[img_name]  ##channel names
        C, H, W = stack.shape ##stack shape names (C=channel(0), H = heigth(1), W = width(2))
        processed = np.zeros((C, H, W), dtype=np.float32) ##zero initialized array 
        ## This one tracks which which channels are already in [0,1] for the for uint16 writing
        ch_is_unit = [False] * C

        ##------------------------------------> process each frame <---------------------------##
        ##j 
        for j, ch in enumerate(chlist):
            ##Take channel j from the stack, change it to float32 and make a copy
            ##Copy ensures a copy to be made of the underlying data and not to use the same underlying data 'stack' 
            frame = stack[j].astype(np.float32).copy()

            ##------------------------------------> Featch all parameters for the channel<---------------------------##
            ##Pre save the raw min/max
            preMin = float(np.nanmin(frame))
            preMax = float(np.nanmax(frame))

            #Get the per channel parameters, if not available, use the fallbacks from params.py
            row = params.loc[params["Channel"] == ch]
            r = row.iloc[0] if not row.empty else pd.Series(DEFAULTS)
            ##Winsorization
            doWin   = bool(r.get("DoWinsor", DEFAULTS["DoWinsor"]))
            lo_q    = float(r.get("Low", DEFAULTS["Low"]))
            hi_q    = float(r.get("High", DEFAULTS["High"]))
            ##Percentile thresholding
            doThr   = bool(r.get("DoThr", DEFAULTS["DoThr"]))
            thrN    = float(r.get("ThrVal", DEFAULTS["ThrVal"]))     
            ##Absolute thresholding value
            doAbsThr = bool(r.get("DoAbsThr", DEFAULTS["DoAbsThr"]))
            absThr   = float(r.get("AbsThrVal", DEFAULTS["AbsThrVal"]))
            ##Noise removal parameters
            doNoise = bool(r.get("Noise", DEFAULTS["Noise"]))
            sVal    = float(r.get("NStr", DEFAULTS["NStr"])) 
            pCsv    = float(r.get("NPrctl", DEFAULTS["NPrctl"]))  
            winSz   = int(r.get("WinSz", DEFAULTS["WinSz"]))
            #Normalization and transformation parameters
            doNorm  = bool(r.get("DoNorm", DEFAULTS["DoNorm"]))
            scope   = str(r.get("NormScope", DEFAULTS["NormScope"])).lower()
            doAsinh = bool(r.get("DoAsinh", DEFAULTS["DoAsinh"]))
            cofac   = float(r.get("Cofac", DEFAULTS["Cofac"]))
            #Try to resolve the percentile that is used, prefer NPrctl if provided; else derive from strength
            if np.isfinite(pCsv) and 0.0 < pCsv < 1.0:
                pUsed = pCsv
            else:
                pUsed = strength_to_percentile(sVal)

            ##------------------------------------> Calling the thresholding/norm etc function <----------------------------##
            #1)Winsorization, clipping values
            if doWin:
                frame, rawFloor, rawCeil = winsorize_with_bounds(
                    frame,
                    lo_q,
                    hi_q,
                    min_upper_bound=WINSOR_MIN_UPPER_BOUND,
                )
            else:
                rawFloor, rawCeil = preMin, preMax

            #2)Global absolute threshold to zero out pixels <- absThr
            abs_thr_used = np.nan
            if doAbsThr and np.isfinite(absThr) and absThr > 0.0:
                frame = apply_threshold_absolute(frame, absThr)
                abs_thr_used = float(absThr)

            #3)Fraction-of-max threshold, zeros out a fraction of pixels. So 0.1 zero's out lowers 10% pixels. 
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

            #4)Local percentile speckle removal. Removes bright specles (if <2 birght neigbours)
            pct_used = np.nan
            if doNoise and winSz >= 1 and (0.0 < pUsed < 1.0):
                frame = apply_speckle_suppress(frame, size=winSz, perc=pUsed, neighbor_limit=2)
                pct_used = float(pUsed)

            #5)Optional arcsinh transformation of data
            asinh_applied = False
            asinh_cofac = np.nan
            if doAsinh:
                used_cofac = sanitize_cofactor(cofac)
                frame = arcsinh_transform(frame, used_cofac)
                asinh_applied = True
                asinh_cofac = float(used_cofac)

            #6)Optional normalization
            ##Checks if global, if not go to per page. If global it uses the global minimax previously calculated
            ##in _precompute_global_minmax. Both cases it's just 0-1 minmaxing
            normalized = False
            ##incase global fails, let the user know
            requested_scope = scope
            effective_scope = scope
            fallback_reason = ""

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
                        effective_scope = "global" ##Redundant but explicit
                    else:
                        ##Fallback to per-page if global unavailable/degenerate or something went wrong with the precomputing
                        ##Let's userknow
                        mn, mx = float(np.nanmin(frame)), float(np.nanmax(frame))
                        if mx > mn:
                            frame = normalize_minmax(frame, mn, mx)
                            normalized = True
                            ch_is_unit[j] = True
                            effective_scope = "page"
                            fallback_reason = "global bounds missing/degenerate"
                else:
                    #Per-page normalization
                    mn, mx = float(np.nanmin(frame)), float(np.nanmax(frame))
                    if mx > mn:
                        frame = normalize_minmax(frame, mn, mx)
                        normalized = True
                        ch_is_unit[j] = True
                        effective_scope = "page"

            processed[j] = frame

            #Record parameters/results for the user if they ever need to get back to what they did
            results_rows.append({
                "Image": img_name,
                "Channel": ch,
                "Page": j + 1,  #arrays start with 1 ;)
                "MinValueRaw": preMin,
                "MaxValueRaw": preMax,
                "WinsorLower": float(rawFloor),
                "WinsorUpper": float(rawCeil),
                "Threshold": thr_used,
                "AbsThreshold": abs_thr_used,
                "Strength": float(sVal),
                "Percentile": pct_used,
                "WindowSize": int(winSz),
                "NeighborsRule": "remove if ≤2 bright neighbors",
                "Normalized": bool(normalized),
                "NormScopeRequested": requested_scope,
                "NormScopeUsed": effective_scope,
                "NormFallbackReason": fallback_reason,
                "GlobalMinUsed": gmin_used,
                "GlobalMaxUsed": gmax_used,
                "AsinhApplied": bool(asinh_applied),
                "AsinhCofactor": asinh_cofac,
            })

        #Write outputs
        out16 = out_dir / f"{img_name} Normalized uint16.tiff"
        out32 = out_dir / f"{img_name} Normalized 32bit.tiff"

        #32-bit float (exact processed values)
        imwrite(
            out32,
            processed.astype(np.float32),
            dtype=np.float32,
            ome=True,
            metadata={"axes": "CYX"},
        )

        ##16-bit:
        ##if channels already normalized to [0,1] it will keep that scale
        ##all the others get per-channel stretch (so they're usable)
        u16 = to_uint16_stack(processed, is_unit_flags=ch_is_unit)
        imwrite(
            out16,
            u16,
            dtype=np.uint16,
            ome=True,
            metadata={"axes": "CYX"},
        )

        print(f"✓ Wrote: {out16.name}, {out32.name}")

    ##Save results sheet, no excel. It failed too often
    results = pd.DataFrame(results_rows)
    results_csv = out_dir / "Parameters Of all images.csv"
    results.to_csv(results_csv, index=False)
    print(f"✓ Saved: {results_csv.name}")

    print("All done.")


if __name__ == "__main__":
    main()
