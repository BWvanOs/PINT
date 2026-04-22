from pathlib import Path
import pandas as pd
from tifffile import imread
import numpy as np


def validate_mask_input_table(
    df: pd.DataFrame,
    cell_id_col: str = "CellName",
    x_col: str = "Location_Center_X",
    y_col: str = "Location_Center_Y",
    mask_name_col: str = "CellMaskName",
) -> pd.DataFrame:
    """
    Validate that the input cell table contains the required columns for mask linking.
    Returns a copy with standardized string/numeric types.
    """
    required_cols = [cell_id_col, x_col, y_col, mask_name_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_cols)}"
        )

    out = df.copy()

    out[cell_id_col] = out[cell_id_col].astype(str)
    out[mask_name_col] = out[mask_name_col].astype(str)
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")

    return out

def list_mask_files(mask_dir: str | Path, suffix: str = ".tiff") -> pd.DataFrame:
    """
    List mask files and derive a stem name used for matching.
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        raise ValueError(f"Mask directory does not exist: {mask_dir}")

    files = sorted(mask_dir.glob(f"*{suffix}"))

    if not files:
        return pd.DataFrame(columns=["MaskFile", "MaskPath", "CellMaskName"])

    return pd.DataFrame(
        {
            "MaskFile": [f.name for f in files],
            "MaskPath": [str(f) for f in files],
            "CellMaskName": [strip_known_mask_suffix(f.stem) for f in files],
        }
    )

def match_cellmask_names_to_files(
    df: pd.DataFrame,
    mask_files_df: pd.DataFrame,
    mask_name_col: str = "CellMaskName",
) -> pd.DataFrame:
    """
    Match unique CellMaskName values in the cell table to files in the mask folder.
    Assumes exact match between CellMaskName and mask filename stem.
    """
    unique_masks = pd.DataFrame(
        {mask_name_col: sorted(df[mask_name_col].dropna().astype(str).unique())}
    )

    match_df = unique_masks.merge(
        mask_files_df,
        left_on=mask_name_col,
        right_on="CellMaskName",
        how="left",
    )

    match_df["MaskExists"] = match_df["MaskPath"].notna()
    return match_df

def split_mask_matches(match_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched = match_df.loc[match_df["MaskExists"]].copy()
    missing = match_df.loc[~match_df["MaskExists"]].copy()
    return matched, missing

def get_cells_for_mask_name(
    df: pd.DataFrame,
    mask_name: str,
    mask_name_col: str = "CellMaskName",
) -> pd.DataFrame:
    """
    Return the subset of cell rows belonging to one mask.
    """
    return df.loc[df[mask_name_col].astype(str) == str(mask_name)].copy()

def strip_known_mask_suffix(name: str) -> str:
    return (
        str(name)
        .removesuffix(" Normalized 32bit_UNIT16")
        .removesuffix(".tiff")
        .removesuffix(".tif")
    )

def load_cellmask(mask_path: str | Path) -> np.ndarray:
    """
    Load a labeled cell mask TIFF and return a 2D matrix.
    """
    mask = imread(mask_path)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return np.asarray(mask)