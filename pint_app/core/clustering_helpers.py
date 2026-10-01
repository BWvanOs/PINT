import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
import pacmap

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def run_pca(
    X: np.ndarray,
    obs: pd.DataFrame,
    source_columns: list[str],
    display_columns: list[str],
    *,
    n_pcs: int,
    random_seed: int = 1,
) -> dict:
    """
    Run PCA on a prepared clustering feature matrix.

    Parameters
    ----------
    X
        Numeric cells × features matrix.
    obs
        Cell-level metadata containing the stable cell identifier.
    source_columns
        Original feature column names.
    display_columns
        User-facing feature names.
    n_pcs
        Number of principal components to calculate.
    random_seed
        Random seed supplied to sklearn PCA.

    Returns
    -------
    dict
        PCA feature matrix, scores, loadings, variance table,
        and feature-name metadata.
    """
    if X is None:
        raise ValueError(
            "No feature matrix was supplied for PCA."
        )

    if X.ndim != 2:
        raise ValueError(
            "PCA feature matrix must be two-dimensional."
        )

    nCells, nFeatures = X.shape

    if nCells < 2:
        raise ValueError(
            "Need at least 2 cells for PCA."
        )

    if nFeatures < 1:
        raise ValueError(
            "Need at least 1 feature for PCA."
        )

    nPcs = max(
        1,
        min(
            int(n_pcs),
            nFeatures,
            nCells - 1,
        ),
    )

    if nPcs < 2:
        raise ValueError(
            "Need at least 2 PCs. "
            "Check number of cells/features."
        )

    pca = PCA(
        n_components=nPcs,
        random_state=int(random_seed),
    )

    scores = pca.fit_transform(X)

    scoreCols = [
        f"PC_{i}"
        for i in range(1, nPcs + 1)
    ]

    scoresDf = obs.copy()

    for i, columnName in enumerate(scoreCols):
        scoresDf[columnName] = (
            scores[:, i]
            .astype(np.float32)
        )

    loadingsDf = pd.DataFrame(
        pca.components_.T,
        columns=scoreCols,
    )

    loadingsDf.insert(
        0,
        "Feature",
        display_columns,
    )

    loadingsDf.insert(
        0,
        "SourceColumn",
        source_columns,
    )

    varianceDf = pd.DataFrame(
        {
            "PC": scoreCols,
            "ExplainedVarianceRatio":
                pca.explained_variance_ratio_,
            "ExplainedVariancePercent":
                pca.explained_variance_ratio_ * 100,
            "CumulativeVariancePercent":
                np.cumsum(
                    pca.explained_variance_ratio_
                ) * 100,
        }
    )

    return {
        "feature_matrix": X,
        "source_columns": source_columns,
        "display_columns": display_columns,
        "scores": scoresDf,
        "loadings": loadingsDf,
        "variance": varianceDf,
        "n_cells": nCells,
        "n_features": nFeatures,
        "n_pcs": nPcs,
    }

def run_leiden(
    pca_df: pd.DataFrame,
    *,
    cell_id_col: str,
    n_dims: int,
    n_neighbors: int,
    resolution: float,
    seed: int,
) -> dict:
    """
    Run Leiden clustering on PCA coordinates.

    Returns cluster labels together with metadata describing
    the actual parameters used.
    """
    if pca_df is None or pca_df.empty:
        raise ValueError(
            "No PCA scores were supplied for Leiden clustering."
        )

    if cell_id_col not in pca_df.columns:
        raise ValueError(
            f"Missing required cell ID column: {cell_id_col}"
        )

    pcCols = [
        columnName
        for columnName in pca_df.columns
        if columnName.startswith("PC_")
    ]

    if len(pcCols) < 2:
        raise ValueError(
            "PCA scores do not contain enough PC columns."
        )

    usePcCols = pcCols[
        :min(int(n_dims), len(pcCols))
    ]

    Xpca = pca_df.loc[
        :,
        usePcCols,
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    nCells = Xpca.shape[0]

    if nCells < 3:
        raise ValueError(
            "Need at least 3 cells for graph clustering."
        )

    nNeighbors = max(
        2,
        min(
            int(n_neighbors),
            nCells - 1,
        ),
    )

    nn = NearestNeighbors(
        n_neighbors=nNeighbors + 1,
        metric="euclidean",
        algorithm="auto",
    )

    nn.fit(Xpca)

    distances, indices = nn.kneighbors(
        Xpca
    )

    edges = []
    weights = []

    for i in range(nCells):
        for j, dist in zip(
            indices[i, 1:],
            distances[i, 1:],
        ):
            if i == j:
                continue

            a = int(i)
            b = int(j)

            if a < b:
                edges.append((a, b))
            else:
                edges.append((b, a))

            weights.append(
                float(
                    1.0 / (1.0 + dist)
                )
            )

    if not edges:
        raise ValueError(
            "No graph edges were created."
        )

    edgeDf = pd.DataFrame(
        edges,
        columns=[
            "source",
            "target",
        ],
    )

    edgeDf["weight"] = weights

    edgeDf = (
        edgeDf
        .groupby(
            ["source", "target"],
            as_index=False,
        )["weight"]
        .max()
    )

    graph = ig.Graph(
        n=nCells,
        edges=list(
            map(
                tuple,
                edgeDf[
                    ["source", "target"]
                ].to_numpy(),
            )
        ),
    )

    graph.es["weight"] = (
        edgeDf["weight"]
        .tolist()
    )

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=float(resolution),
        seed=int(seed),
    )

    labels = np.array(
        partition.membership,
        dtype=int,
    )

    labelsDf = pca_df[
        [cell_id_col]
    ].copy()

    labelsDf["PINT_Leiden_cluster"] = [
        f"Cluster_{value}"
        for value in labels
    ]

    return {
        "labels": labelsDf,
        "n_cells": nCells,
        "n_clusters": int(
            labelsDf[
                "PINT_Leiden_cluster"
            ].nunique()
        ),
        "n_dims": len(usePcCols),
        "n_neighbors": nNeighbors,
        "resolution": float(resolution),
    }

import pacmap


def run_pacmap(
    pca_df: pd.DataFrame,
    *,
    cell_id_col: str,
    n_dims: int,
    n_neighbors: int,
    mn_ratio: float,
    fp_ratio: float,
    seed: int,
) -> dict:
    """
    Run PaCMAP on selected PCA dimensions.

    Returns the embedding dataframe together with metadata
    describing the actual parameters used.
    """
    if pca_df is None or pca_df.empty:
        raise ValueError(
            "No PCA scores were supplied for PaCMAP."
        )

    if cell_id_col not in pca_df.columns:
        raise ValueError(
            f"Missing required cell ID column: {cell_id_col}"
        )

    pcCols = [
        columnName
        for columnName in pca_df.columns
        if columnName.startswith("PC_")
    ]

    if len(pcCols) < 2:
        raise ValueError(
            "PCA scores do not contain enough PC columns."
        )

    usePcCols = pcCols[
        :min(int(n_dims), len(pcCols))
    ]

    Xpca = pca_df.loc[
        :,
        usePcCols,
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    nCells = Xpca.shape[0]

    if nCells < 3:
        raise ValueError(
            "Need at least 3 cells for PaCMAP."
        )

    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        MN_ratio=float(mn_ratio),
        FP_ratio=float(fp_ratio),
        random_state=int(seed),
    )

    embedding = reducer.fit_transform(
        Xpca,
        init="pca",
    )

    embeddingDf = pca_df[
        [cell_id_col]
    ].copy()

    embeddingDf["PaCMAP_1"] = (
        embedding[:, 0]
        .astype(np.float32)
    )

    embeddingDf["PaCMAP_2"] = (
        embedding[:, 1]
        .astype(np.float32)
    )

    return {
        "embedding": embeddingDf,
        "n_cells": nCells,
        "n_dims": len(usePcCols),
        "n_neighbors": int(n_neighbors),
        "mn_ratio": float(mn_ratio),
        "fp_ratio": float(fp_ratio),
    }

def prepare_clustering_feature_matrix(
    df: pd.DataFrame,
    feature_map: pd.DataFrame,
    *,
    cell_id_col: str,
    active_cell_ids=None,
    transform: str = "asinh",
    cofactor: float = 5,
    scale_data: bool = True,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    list[str],
    list[str],
]:
    """
    Build the numeric feature matrix used for PCA and clustering.

    The function is independent of Shiny state. All data, selected features,
    preprocessing settings, and optional cell subsets are supplied explicitly.
    """
    if df is None or df.empty:
        raise ValueError(
            "No clustering dataset supplied."
        )

    if cell_id_col not in df.columns:
        raise ValueError(
            f"Required cell ID column is missing: {cell_id_col}"
        )

    if feature_map is None or feature_map.empty:
        raise ValueError(
            "No valid feature columns selected for PCA/clustering."
        )

    requiredMapColumns = {
        "ChannelNamesForClustering",
        "ChannelNameToDisplay",
    }

    missingMapColumns = (
        requiredMapColumns
        - set(feature_map.columns)
    )

    if missingMapColumns:
        raise ValueError(
            "Feature map is missing required column(s): "
            + ", ".join(
                sorted(missingMapColumns)
            )
        )

    featureMap = feature_map.copy()

    sourceCols = (
        featureMap[
            "ChannelNamesForClustering"
        ]
        .tolist()
    )

    displayCols = (
        featureMap[
            "ChannelNameToDisplay"
        ]
        .tolist()
    )

    missingCols = [
        columnName
        for columnName in sourceCols
        if columnName not in df.columns
    ]

    if missingCols:
        raise ValueError(
            "Selected feature columns are missing "
            "from clustering data: "
            + ", ".join(missingCols)
        )

    # Optional subset for subclustering.
    if active_cell_ids is not None:
        activeIds = set(
            map(
                str,
                active_cell_ids,
            )
        )

        useDf = df.loc[
            df[cell_id_col]
            .astype(str)
            .isin(activeIds)
        ].copy()

    else:
        useDf = df.copy()

    if useDf.empty:
        raise ValueError(
            "No cells available for the current clustering run."
        )

    obs = useDf[
        [cell_id_col]
    ].copy()

    Xdf = useDf.loc[
        :,
        sourceCols,
    ].copy()

    Xdf = Xdf.apply(
        pd.to_numeric,
        errors="coerce",
    )

    Xdf = Xdf.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove features containing no usable numeric values.
    allMissing = (
        Xdf.columns[
            Xdf.isna().all()
        ]
        .tolist()
    )

    if allMissing:
        keepMask = (
            ~Xdf.columns.isin(
                allMissing
            )
        )

        Xdf = Xdf.loc[
            :,
            keepMask,
        ].copy()

        featureMap = featureMap.loc[
            keepMask
        ].copy()

        sourceCols = (
            featureMap[
                "ChannelNamesForClustering"
            ]
            .tolist()
        )

        displayCols = (
            featureMap[
                "ChannelNameToDisplay"
            ]
            .tolist()
        )

    if Xdf.shape[1] == 0:
        raise ValueError(
            "No numeric feature columns remain after filtering."
        )

    # Missing values are replaced by the marker median.
    medians = Xdf.median(
        axis=0,
        numeric_only=True,
    )

    Xdf = (
        Xdf
        .fillna(medians)
        .fillna(0)
    )

    X = Xdf.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    transform = str(
        transform or "asinh"
    ).strip().lower()

    cofactor = float(cofactor)

    if transform == "asinh":
        if cofactor <= 0:
            raise ValueError(
                "asinh cofactor must be > 0."
            )

        X = np.arcsinh(
            X / cofactor
        ).astype(
            np.float32,
            copy=False,
        )

    elif transform == "log1p":
        X = np.clip(
            X,
            a_min=0,
            a_max=None,
        )

        X = np.log1p(
            X
        ).astype(
            np.float32,
            copy=False,
        )

    elif transform == "none":
        pass

    else:
        raise ValueError(
            f"Unknown transform: {transform}"
        )

    if bool(scale_data):
        scaler = StandardScaler(
            copy=True
        )

        X = (
            scaler
            .fit_transform(X)
            .astype(
                np.float32,
                copy=False,
            )
        )

    return (
        X,
        obs,
        sourceCols,
        displayCols,
    )

def set_clustering_columns_by_suffix(
    column_map: pd.DataFrame,
    suffix: str,
    include: bool,
) -> tuple[pd.DataFrame, int]:

    updated = column_map.copy()

    source_names = (
        updated["ChannelNamesForClustering"]
        .fillna("")
        .astype(str)
    )

    mask = source_names.str.endswith(
        suffix,
        na=False,
    )

    n_matched = int(mask.sum())

    updated.loc[
        mask,
        "IncludeForClustering",
    ] = include

    return updated, n_matched

def format_deg_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    ordinaryCols = [
        "Mean_in",
        "Mean_out",
        "Mean_difference",
        "log2FC",
        "pct_in",
        "pct_out",
    ]

    for col in ordinaryCols:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            ).round(3)

    for col in [
        "p_value",
        "p_adj_BH",
    ]:
        if col in out.columns:
            values = pd.to_numeric(
                out[col],
                errors="coerce",
            )

            out[col] = values.map(
                lambda x: (
                    f"{x:.3e}"
                    if pd.notna(x)
                    else ""
                )
            )

    return out