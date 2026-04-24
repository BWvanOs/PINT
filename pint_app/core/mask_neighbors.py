from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

from pint_app.core.mask_viz import read_mask_tiff, match_mask_centroids_to_cells


ProgressFn = Callable[[str], None]


def _emit(progress: Optional[ProgressFn], msg: str) -> None:
    if progress is not None:
        progress(msg)


def build_touching_adjacency(cellMask: np.ndarray) -> pd.DataFrame:
    """
    Build an undirected adjacency table from a labeled mask image.

    Two cells are adjacent if different nonzero labels touch along a shared
    edge in the raster. Uses 4-connectivity (right and down comparisons).
    """
    if cellMask is None or cellMask.size == 0:
        return pd.DataFrame(columns=["CellValueMask_1", "CellValueMask_2"])

    edges: set[tuple[int, int]] = set()

    # Right neighbors
    a = cellMask[:, :-1]
    b = cellMask[:, 1:]
    diff = (a != b) & (a > 0) & (b > 0)
    if np.any(diff):
        for l1, l2 in zip(a[diff], b[diff]):
            edge = tuple(sorted((int(l1), int(l2))))
            edges.add(edge)

    # Down neighbors
    a = cellMask[:-1, :]
    b = cellMask[1:, :]
    diff = (a != b) & (a > 0) & (b > 0)
    if np.any(diff):
        for l1, l2 in zip(a[diff], b[diff]):
            edge = tuple(sorted((int(l1), int(l2))))
            edges.add(edge)

    if not edges:
        return pd.DataFrame(columns=["CellValueMask_1", "CellValueMask_2"])

    return pd.DataFrame(sorted(edges), columns=["CellValueMask_1", "CellValueMask_2"])


def annotate_touching_edges(
    edgeDf: pd.DataFrame,
    matchedData: pd.DataFrame,
    *,
    mask_label_col: str = "CellValueMask",
    cluster_col: str,
    mask_name_col: Optional[str] = None,
    sample_col: Optional[str] = None,
    condition_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Annotate touching edges with metadata from the matched cell table.
    """
    if edgeDf.empty:
        cols = ["CellValueMask_1", "CellValueMask_2", "cell_cluster", "neighbor_cluster"]
        if mask_name_col:
            cols.append(mask_name_col)
        if sample_col:
            cols.append(sample_col)
        if condition_col:
            cols.append(condition_col)
        return pd.DataFrame(columns=cols)

    keepCols = [mask_label_col, cluster_col]
    for extra in [mask_name_col, sample_col, condition_col]:
        if extra and extra in matchedData.columns and extra not in keepCols:
            keepCols.append(extra)

    meta = matchedData[keepCols].drop_duplicates(subset=[mask_label_col]).copy()

    left = meta.rename(columns={
        mask_label_col: "CellValueMask_1",
        cluster_col: "cell_cluster",
    })
    right = meta.rename(columns={
        mask_label_col: "CellValueMask_2",
        cluster_col: "neighbor_cluster",
    })

    if mask_name_col and mask_name_col in left.columns:
        left = left.rename(columns={mask_name_col: "mask_name_left"})
    if sample_col and sample_col in left.columns:
        left = left.rename(columns={sample_col: "sample_left"})
    if condition_col and condition_col in left.columns:
        left = left.rename(columns={condition_col: "condition_left"})

    if mask_name_col and mask_name_col in right.columns:
        right = right.rename(columns={mask_name_col: "mask_name_right"})
    if sample_col and sample_col in right.columns:
        right = right.rename(columns={sample_col: "sample_right"})
    if condition_col and condition_col in right.columns:
        right = right.rename(columns={condition_col: "condition_right"})

    out = edgeDf.merge(left, on="CellValueMask_1", how="left")
    out = out.merge(right, on="CellValueMask_2", how="left")

    if "mask_name_left" in out.columns:
        out[mask_name_col] = out["mask_name_left"].combine_first(out.get("mask_name_right"))
    if "sample_left" in out.columns:
        out[sample_col] = out["sample_left"].combine_first(out.get("sample_right"))
    if "condition_left" in out.columns:
        out[condition_col] = out["condition_left"].combine_first(out.get("condition_right"))

    dropCols = [
        c for c in [
            "mask_name_left", "mask_name_right",
            "sample_left", "sample_right",
            "condition_left", "condition_right",
        ]
        if c in out.columns
    ]
    if dropCols:
        out = out.drop(columns=dropCols)

    return out


def build_touching_edges_for_one_mask(
    *,
    maskPath: str,
    matchingData: pd.DataFrame,
    xCol: str,
    yCol: str,
    clusterCol: str,
    maskNameCol: Optional[str] = None,
    sampleCol: Optional[str] = None,
    conditionCol: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      matchedData: one row per mask label matched to cell metadata
      annotatedEdges: touching edge table with metadata
    """
    cellMask = read_mask_tiff(maskPath)

    matchedData = match_mask_centroids_to_cells(
        cellMask=cellMask,
        matchingData=matchingData,
        xCol=xCol,
        yCol=yCol,
    )

    edgeDf = build_touching_adjacency(cellMask)

    annotatedEdges = annotate_touching_edges(
        edgeDf,
        matchedData,
        cluster_col=clusterCol,
        mask_name_col=maskNameCol,
        sample_col=sampleCol,
        condition_col=conditionCol,
    )

    return matchedData, annotatedEdges


def build_touching_edges_for_pushed_dataset(
    neighborhoodInput: dict,
    *,
    progress: Optional[ProgressFn] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build touching edges across all masks in a pushed neighborhood dataset.

    Returns:
      allMatchedData
      allAnnotatedEdges
    """
    if neighborhoodInput is None:
        raise ValueError("No pushed neighborhood dataset available.")

    df = neighborhoodInput["cell_table"].copy()
    colMap = neighborhoodInput["column_map"]
    maskNameCol = colMap["mask_name_col"]
    xCol = colMap["x_col"]
    yCol = colMap["y_col"]
    clusterCol = colMap["cluster_col"]
    sampleCol = colMap.get("sample_col", None)
    conditionCol = colMap.get("condition_col", None)

    if "MaskPath" not in df.columns:
        raise ValueError("Pushed cell table is missing 'MaskPath'.")

    matchedFrames: list[pd.DataFrame] = []
    edgeFrames: list[pd.DataFrame] = []

    maskTable = (
        df[[maskNameCol, "MaskPath"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if maskTable.empty:
        raise ValueError("No valid masks available in pushed dataset.")

    for i, row in maskTable.iterrows():
        maskName = str(row[maskNameCol])
        maskPath = str(row["MaskPath"])

        _emit(progress, f"[{i+1}/{len(maskTable)}] Building touching graph for {maskName}")

        matchingData = df.loc[df[maskNameCol].astype(str) == maskName].copy()
        if matchingData.empty:
            _emit(progress, f"[{i+1}/{len(maskTable)}] Skipped {maskName}: no rows in cell table")
            continue

        try:
            matchedData, annotatedEdges = build_touching_edges_for_one_mask(
                maskPath=maskPath,
                matchingData=matchingData,
                xCol=xCol,
                yCol=yCol,
                clusterCol=clusterCol,
                maskNameCol=maskNameCol,
                sampleCol=sampleCol,
                conditionCol=conditionCol,
            )
        except Exception as e:
            _emit(progress, f"[{i+1}/{len(maskTable)}] Failed {maskName}: {e}")
            continue

        matchedFrames.append(matchedData)
        edgeFrames.append(annotatedEdges)

        _emit(progress, f"[{i+1}/{len(maskTable)}] Done {maskName} ({len(annotatedEdges):,} touching edges)")

    allMatchedData = pd.concat(matchedFrames, ignore_index=True) if matchedFrames else pd.DataFrame()
    allAnnotatedEdges = pd.concat(edgeFrames, ignore_index=True) if edgeFrames else pd.DataFrame()

    return allMatchedData, allAnnotatedEdges