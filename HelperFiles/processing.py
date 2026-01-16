"""Parameter table helpers.

The Shiny viewer keeps channel-level settings in a pandas DataFrame.
These utilities centralize:
- default table creation
- updating a single channel row
- table display formatting
- importing/validating a parameter CSV

The functions here are framework-agnostic: no Shiny `input`/`session`.
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
                "Channel": ch,
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
    return df.reindex(columns=PARAM_COLUMNS).reset_index(drop=True)


def update_channel_row(df: pd.DataFrame, channel: str, updates: Mapping[str, Any]) -> pd.DataFrame:
    """Return a copy of `df` with the row for `channel` updated."""
    if df is None or df.empty or not channel:
        return df
    if "Channel" not in df.columns:
        return df

    idx_list = df.index[df["Channel"] == channel].tolist()
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


def validate_and_normalize_import(
    df_in: pd.DataFrame,
    canonical_channels: Sequence[str],
    target_cols: Sequence[str] = PARAM_COLUMNS,
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

    defaults: dict[str, Any] = {
        "Channel": "",
        "DoWinsor": False,
        "Low": 0.0,
        "High": 1.0,
        "DoThr": False,
        "ThrVal": 0.0,
        "Noise": False,
        "NStr": 1.0,
        "NPrctl": 0.995,
        "WinSz": 3,
        "DoNorm": True,
        "NormScope": "page",
        "DoAsinh": False,
        "Cofac": 5,
    }

    # Reorder rows to canonical channel order
    df_norm = df_in.set_index("Channel").reindex(canon).reset_index()

    # Add missing columns
    for col in target_cols:
        if col not in df_norm.columns:
            df_norm[col] = defaults.get(col)

    # Drop unknown/deprecated columns and enforce column order
    df_norm = df_norm[list(target_cols)].copy()

    bool_cols = [c for c in target_cols if c in ["DoWinsor", "DoThr", "Noise", "DoNorm", "DoAsinh"]]
    float_cols = [c for c in target_cols if c in ["Low", "High", "ThrVal", "NStr", "NPrctl"]]
    int_cols = [c for c in target_cols if c in ["WinSz", "Cofac"]]

    for c in bool_cols:
        df_norm[c] = df_norm[c].map(_to_bool)

    for c in float_cols:
        df_norm[c] = pd.to_numeric(df_norm[c], errors="coerce").fillna(defaults[c]).astype(float)

    for c in int_cols:
        df_norm[c] = pd.to_numeric(df_norm[c], errors="coerce").fillna(defaults[c]).astype(int)

    # NormScope as string
    if "NormScope" in df_norm.columns:
        df_norm["NormScope"] = df_norm["NormScope"].astype(str).fillna(defaults["NormScope"])

    return df_norm.reset_index(drop=True)