"""Parameter table schema and helpers.

The viewer maintains per-channel settings in a pandas DataFrame. The analysis
script consumes the same CSV. This module is the shared, single source of truth
for:
  - expected columns
  - defaults
  - coercion/validation helpers

Design constraints
  - No Shiny imports.
  - Functions should be deterministic given their inputs.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


PARAM_COLUMNS: list[str] = [
    "Channel", "DoWinsor", "Low", "High",
    "DoThr", "ThrVal",
    "Noise", "NStr", "NPrctl", "WinSz",
    "DoNorm", "NormScope",
    "DoAsinh", "Cofac",
]


DEFAULTS: dict[str, Any] = {
    "Channel": "",
    "DoWinsor": False,
    "Low": 0.0,
    "High": 0.99,
    "DoThr": False,
    "ThrVal": 0.0,
    "Noise": False,
    "NStr": 0.1,
    "NPrctl": 0.995,
    "WinSz": 3,
    "DoNorm": True,
    "NormScope": "page",
    "DoAsinh": False,
    "Cofac": 5,
}


DISPLAY_RENAME_MAP: dict[str, str] = {
    "Channel": "Ch",
    "DoWinsor": "Win",
    "Low": "Low",
    "High": "High",
    "DoThr": "Thr?",
    "ThrVal": "ThrVal",
    "Noise": "Noise",
    "NStr": "NStr",
    "NPrctl": "NPerc",
    "WinSz": "WinSz",
    "DoNorm": "Norm?",
    "DoAsinh": "Asinh?",
    "Cofac": "Cofac",
    "NormScope": "Scope",
}


def make_params_df(
    channel_names: Sequence[str],
    *,
    do_winsor: bool,
    low: float,
    high: float,
    do_threshold: bool,
    thr_val: float,
    do_noise: bool,
    noise_strength: float,
    noise_percentile: float,
    window_size: int,
    do_norm: bool,
    norm_scope: str,
    do_asinh: bool,
    cofactor: int,
) -> pd.DataFrame:
    """Create a new parameter DataFrame for the given channel list."""
    rows: list[dict[str, Any]] = []
    for ch in channel_names:
        rows.append(
            {
                "Channel": str(ch),
                "DoWinsor": bool(do_winsor),
                "Low": float(low),
                "High": float(high),
                "DoThr": bool(do_threshold),
                "ThrVal": float(thr_val),
                "Noise": bool(do_noise),
                "NStr": float(noise_strength),
                "NPrctl": float(noise_percentile),
                "WinSz": int(window_size),
                "DoNorm": bool(do_norm),
                "NormScope": str(norm_scope or "page"),
                "DoAsinh": bool(do_asinh),
                "Cofac": int(cofactor),
            }
        )
    df = pd.DataFrame(rows)
    return coerce_params_df(df, canonical_channels=channel_names)


def update_channel_row(df: pd.DataFrame, channel: str, updates: Mapping[str, Any]) -> pd.DataFrame:
    """Return a copy of `df` with the row for `channel` updated."""
    if df is None or df.empty or not channel:
        return df
    if "Channel" not in df.columns:
        return df

    idx_list = df.index[df["Channel"].astype(str) == str(channel)].tolist()
    if not idx_list:
        return df

    i = idx_list[0]
    new_df = df.copy()
    for k, v in updates.items():
        if k in new_df.columns:
            new_df.at[i, k] = v
    return new_df.reset_index(drop=True)


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the parameter table with user-friendly headers."""
    if df is None or df.empty:
        return pd.DataFrame({"Info": ["No parameters yet"]})
    return df.rename(columns=DISPLAY_RENAME_MAP).reset_index(drop=True)


def _to_bool(v: Any) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    try:
        return bool(int(float(v)))
    except Exception:
        return False


def coerce_params_df(
    df_in: pd.DataFrame,
    *,
    canonical_channels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Coerce a params DataFrame to the current schema.

    - Adds missing columns with defaults
    - Drops unknown columns
    - Coerces types
    - Optionally reorders rows to `canonical_channels`
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=PARAM_COLUMNS)

    df = df_in.copy()

    if "Channel" not in df.columns:
        raise ValueError("Parameter table missing required 'Channel' column.")

    df["Channel"] = df["Channel"].astype(str)

    if canonical_channels is not None:
        canon = [str(x) for x in list(canonical_channels)]
        # Reorder rows to canonical channel order
        df = df.set_index("Channel").reindex(canon).reset_index()

    # Add missing columns
    for col in PARAM_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col)

    # Drop unknown/deprecated columns and enforce column order
    df = df[list(PARAM_COLUMNS)].copy()

    bool_cols = ["DoWinsor", "DoThr", "Noise", "DoNorm", "DoAsinh"]
    float_cols = ["Low", "High", "ThrVal", "NStr", "NPrctl"]
    int_cols = ["WinSz", "Cofac"]

    for c in bool_cols:
        df[c] = df[c].map(_to_bool)

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(DEFAULTS[c]).astype(float)

    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(DEFAULTS[c]).astype(int)

    if "NormScope" in df.columns:
        df["NormScope"] = df["NormScope"].astype(str).fillna(DEFAULTS["NormScope"])

    return df.reset_index(drop=True)


def validate_and_normalize_import(
    df_in: pd.DataFrame,
    canonical_channels: Sequence[str],
) -> pd.DataFrame:
    """Validate an imported parameter CSV and coerce it to the current schema.

    Raises ValueError with a clear message if the CSV is incompatible.
    """
    if df_in is None or df_in.empty:
        raise ValueError("CSV is empty.")

    if "Channel" not in df_in.columns:
        raise ValueError("CSV missing required 'Channel' column.")

    canon = [str(x) for x in list(canonical_channels)]
    csv_channels = [str(x) for x in df_in["Channel"].astype(str).tolist()]

    if set(csv_channels) != set(canon) or len(csv_channels) != len(canon):
        raise ValueError(
            "CSV channels do not match current image channels.\n"
            f"CSV: {csv_channels}\nExpected: {canon}"
        )

    return coerce_params_df(df_in, canonical_channels=canon)


def channels_needing_global(params: pd.DataFrame) -> set[str]:
    """Return channel names that request global normalization."""
    need: set[str] = set()
    if params is None or params.empty or "Channel" not in params.columns:
        return need

    norm_scope = params.get("NormScope", pd.Series(["page"] * len(params)))
    do_norm = params.get("DoNorm", pd.Series([True] * len(params)))

    for ch, scope, dn in zip(params["Channel"].astype(str), norm_scope, do_norm):
        if str(scope).strip().lower() == "global" and bool(dn):
            need.add(ch)

    return need
