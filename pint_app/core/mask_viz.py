from pathlib import Path
import colorsys
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from tifffile import imread


def read_mask_tiff(maskPath: str | Path) -> np.ndarray:
    """
    Read a mask TIFF and return a 2D numpy array.
    """
    cellMask = np.asarray(imread(str(maskPath)))
    cellMask = np.squeeze(cellMask)

    if cellMask.ndim == 3:
        # Fallback: just take the first plane/channel if still 3D
        if cellMask.shape[0] == 1:
            cellMask = cellMask[0, :, :]
        else:
            cellMask = cellMask[:, :, 0]

    if cellMask.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeezing; got shape {cellMask.shape}")

    return np.asarray(cellMask)


def compute_mask_centroids(cellMask: np.ndarray) -> pd.DataFrame:
    """
    Compute centroid coordinates and pixel counts for each non-zero mask label.
    """
    maskLabels = np.unique(cellMask)
    maskLabels = maskLabels[maskLabels > 0]

    records = []
    for label in maskLabels:
        idx = np.argwhere(cellMask == label)
        records.append(
            {
                "CellValueMask": int(label),
                "MaskCentroid_X": float(idx[:, 1].mean()),
                "MaskCentroid_Y": float(idx[:, 0].mean()),
                "PixelCount": int(idx.shape[0]),
            }
        )

    return pd.DataFrame(records)


def match_mask_centroids_to_cells(
    cellMask: np.ndarray,
    matchingData: pd.DataFrame,
    xCol: str,
    yCol: str,
) -> pd.DataFrame:
    """
    For each mask object, match the nearest cell row based on centroid distance.
    """
    if matchingData is None or matchingData.empty:
        raise ValueError("matchingData is empty")

    tempDf = matchingData.copy()
    tempDf[xCol] = pd.to_numeric(tempDf[xCol], errors="coerce")
    tempDf[yCol] = pd.to_numeric(tempDf[yCol], errors="coerce")
    tempDf = tempDf.loc[tempDf[xCol].notna() & tempDf[yCol].notna()].copy()

    if tempDf.empty:
        raise ValueError("No valid X/Y coordinates available after coercion")

    maskCentroids = compute_mask_centroids(cellMask)

    cellX = tempDf[xCol].to_numpy(dtype=float)
    cellY = tempDf[yCol].to_numpy(dtype=float)

    nearestMatchIdx = []
    matchDistance = []

    for i in range(maskCentroids.shape[0]):
        dx = cellX - maskCentroids.iloc[i]["MaskCentroid_X"]
        dy = cellY - maskCentroids.iloc[i]["MaskCentroid_Y"]
        d2 = dx ** 2 + dy ** 2
        bestIdx = int(np.argmin(d2))
        nearestMatchIdx.append(bestIdx)
        matchDistance.append(float(np.sqrt(d2[bestIdx])))

    matchedData = pd.concat(
        [
            maskCentroids.reset_index(drop=True),
            tempDf.iloc[nearestMatchIdx].reset_index(drop=True),
        ],
        axis=1,
    )

    matchedData["MatchDistance"] = matchDistance
    return matchedData


def get_cell_borders(cellMask: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of border pixels between touching mask labels.
    """
    nr, nc = cellMask.shape
    borderMask = np.zeros((nr, nc), dtype=bool)

    borderMask[1:, :] |= cellMask[1:, :] != cellMask[:-1, :]
    borderMask[:-1, :] |= cellMask[:-1, :] != cellMask[1:, :]
    borderMask[:, 1:] |= cellMask[:, 1:] != cellMask[:, :-1]
    borderMask[:, :-1] |= cellMask[:, :-1] != cellMask[:, 1:]

    borderMask[cellMask == 0] = False
    return borderMask


def upscale_matrix(mat: np.ndarray, scaleFactor: int = 3) -> np.ndarray:
    """
    Nearest-neighbor upscale for 2D or 3D matrices.
    """
    out = np.repeat(mat, scaleFactor, axis=0)
    out = np.repeat(out, scaleFactor, axis=1)
    return out


def make_distinct_colors(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    """
    Generate distinct colors for category labels.
    """
    labels = sorted([str(x) for x in labels])
    n = len(labels)

    if n == 0:
        return {}

    colorMap = {}
    for i, label in enumerate(labels):
        h = i / max(n, 1)
        l = 0.50
        s = 0.65
        colorMap[label] = colorsys.hls_to_rgb(h, l, s)

    return colorMap


def make_mask_plot_data(
    cellMask: np.ndarray,
    matchedData: pd.DataFrame,
    clusterCol: str,
    scaleFactor: int = 2,
    background: str = "white",
    borderColor: str = "white",
    missingColor: str = "#808080",
) -> dict:
    """
    Convert a cell mask + matched cell annotations into an RGB image matrix and legend colors.
    """
    if clusterCol not in matchedData.columns:
        raise ValueError(f"Cluster column '{clusterCol}' not found in matchedData")

    labelInfo = matchedData.loc[:, ["CellValueMask", clusterCol]].copy()
    labelInfo = labelInfo.drop_duplicates(subset=["CellValueMask"])
    labelInfo[clusterCol] = labelInfo[clusterCol].astype("string")
    labelInfo[clusterCol] = labelInfo[clusterCol].fillna("Unknown")
    labelInfo[clusterCol] = labelInfo[clusterCol].replace("<NA>", "Unknown")

    allClusters = sorted(labelInfo[clusterCol].astype(str).unique())
    clusterColors = make_distinct_colors(allClusters)

    bgRgb = np.array(to_rgb(background), dtype=float)
    borderRgb = np.array(to_rgb(borderColor), dtype=float)
    missingRgb = np.array(to_rgb(missingColor), dtype=float)

    nr, nc = cellMask.shape
    colorMat = np.zeros((nr, nc, 3), dtype=float)
    colorMat[:, :, :] = bgRgb

    assignedMaskValues = []

    for _, row in labelInfo.iterrows():
        maskValue = int(row["CellValueMask"])
        clusterName = str(row[clusterCol])
        rgb = clusterColors.get(clusterName, missingRgb)
        colorMat[cellMask == maskValue] = rgb
        assignedMaskValues.append(maskValue)

    assignedMaskValues = np.asarray(assignedMaskValues, dtype=cellMask.dtype)

    missingIdx = (cellMask != 0) & (~np.isin(cellMask, assignedMaskValues))
    colorMat[missingIdx] = missingRgb

    cellMaskBig = upscale_matrix(cellMask, scaleFactor=scaleFactor)
    colorMatBig = upscale_matrix(colorMat, scaleFactor=scaleFactor)

    borderMaskBig = get_cell_borders(cellMaskBig)
    colorMatBig[borderMaskBig] = borderRgb

    return {
        "colorMat": colorMatBig,
        "clusterColors": clusterColors,
        "labelInfo": labelInfo,
    }