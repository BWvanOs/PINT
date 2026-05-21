from __future__ import annotations

from typing import Optional, Callable
import numpy as np
import pandas as pd


ProgressFn = Callable[[str], None]


def _emit(progress: Optional[ProgressFn], msg: str) -> None:
    if progress is not None:
        progress(msg)


def _bh_adjust(pvals: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg FDR correction.
    """
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)

    valid = np.isfinite(p)
    if valid.sum() == 0:
        return pd.Series(out, index=pvals.index)

    pv = p[valid]
    n = len(pv)

    order = np.argsort(pv)
    ranked = pv[order]

    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[valid] = tmp

    return pd.Series(out, index=pvals.index)


def summarize_touching_observed(
    edgeDf: pd.DataFrame,
    *,
    sample_col: str,
    cluster_col_left: str = "cell_cluster",
    cluster_col_right: str = "neighbor_cluster",
) -> pd.DataFrame:
    if edgeDf is None or edgeDf.empty:
        return pd.DataFrame(columns=[sample_col, "cell_cluster", "neighbor_cluster", "n_interactions"])

    out = (
        edgeDf
        .groupby([sample_col, cluster_col_left, cluster_col_right], sort=False)
        .size()
        .reset_index(name="n_interactions")
        .rename(columns={
            cluster_col_left: "cell_cluster",
            cluster_col_right: "neighbor_cluster",
        })
    )
    return out


def summarize_cluster_frequency(
    matchedDf: pd.DataFrame,
    *,
    sample_col: str,
    cluster_col: str,
) -> pd.DataFrame:
    if matchedDf is None or matchedDf.empty:
        return pd.DataFrame(columns=[sample_col, "cell_cluster", "n_cells"])

    out = (
        matchedDf
        .groupby([sample_col, cluster_col], sort=False)
        .size()
        .reset_index(name="n_cells")
        .rename(columns={cluster_col: "cell_cluster"})
    )
    return out


def normalize_observed_interactions(
    observedDf: pd.DataFrame,
    clusterFreqDf: pd.DataFrame,
    *,
    sample_col: str,
) -> pd.DataFrame:
    if observedDf is None or observedDf.empty:
        return pd.DataFrame(columns=[
            sample_col, "cell_cluster", "neighbor_cluster",
            "n_interactions", "n_cells", "normalized"
        ])

    out = observedDf.merge(
        clusterFreqDf,
        on=[sample_col, "cell_cluster"],
        how="left",
    )
    out["normalized"] = out["n_interactions"] / out["n_cells"]
    return out


def expected_touching_stats_permutation(
    edgeDf: pd.DataFrame,
    matchedDf: pd.DataFrame,
    *,
    sample_col: str,
    cluster_col: str,
    mask_label_col: str = "CellValueMask",
    n_perm: int = 1000,
    random_seed: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
) -> pd.DataFrame:
    """
    Fixed touching-graph permutation that returns:
      - expected count
      - permutation SD
      - empirical p-values against enrichment/depletion/two-sided tails

    Returns:
      sample, cell_cluster, neighbor_cluster,
      observed, expected, perm_sd,
      p_gt, p_lt, p_two_sided
    """
    if n_perm <= 0:
        return pd.DataFrame(columns=[
            sample_col, "cell_cluster", "neighbor_cluster",
            "observed", "expected", "perm_sd",
            "p_gt", "p_lt", "p_two_sided"
        ])

    if edgeDf is None or edgeDf.empty or matchedDf is None or matchedDf.empty:
        return pd.DataFrame(columns=[
            sample_col, "cell_cluster", "neighbor_cluster",
            "observed", "expected", "perm_sd",
            "p_gt", "p_lt", "p_two_sided"
        ])

    rng = np.random.default_rng(random_seed)
    outFrames: list[pd.DataFrame] = []

    for sampleValue, sampleMatchedDf in matchedDf.groupby(sample_col, sort=False):
        sampleEdgeDf = edgeDf.loc[edgeDf[sample_col].astype(str) == str(sampleValue)].copy()

        if sampleEdgeDf.empty or sampleMatchedDf.empty:
            continue

        _emit(progress, f"[{sampleValue}] Preparing fixed touching graph for permutations")

        meta = sampleMatchedDf[[mask_label_col, cluster_col]].drop_duplicates(
            subset=[mask_label_col]
        ).copy()

        # Clean mask labels
        meta[mask_label_col] = pd.to_numeric(meta[mask_label_col], errors="coerce")

        # Clean cluster labels
        clusterClean = meta[cluster_col].astype("string").str.strip()

        badCluster = (
            clusterClean.isna()
            | (clusterClean == "")
            | clusterClean.str.lower().isin(["nan", "none", "na"])
        )

        badMaskLabel = meta[mask_label_col].isna()

        badRows = badCluster | badMaskLabel
        nBadRows = int(badRows.sum())

        if nBadRows > 0:
            _emit(
                progress,
                f"[{sampleValue}] Dropping {nBadRows:,} cells with empty/missing "
                "cluster labels or invalid mask labels"
            )

        meta = meta.loc[~badRows].copy()
        meta[cluster_col] = clusterClean.loc[~badRows].astype(str)
        meta[mask_label_col] = meta[mask_label_col].astype(int)

        if meta.empty:
            _emit(progress, f"[{sampleValue}] No valid clustered cells; skipping")
            continue

        validLabels = set(meta[mask_label_col].to_numpy())

        # Make sure edge mask labels are numeric too, otherwise isin() can silently fail
        sampleEdgeDf["CellValueMask_1"] = pd.to_numeric(
            sampleEdgeDf["CellValueMask_1"],
            errors="coerce",
        )
        sampleEdgeDf["CellValueMask_2"] = pd.to_numeric(
            sampleEdgeDf["CellValueMask_2"],
            errors="coerce",
        )

        sampleEdgeDf = sampleEdgeDf.dropna(
            subset=["CellValueMask_1", "CellValueMask_2"]
        ).copy()

        sampleEdgeDf["CellValueMask_1"] = sampleEdgeDf["CellValueMask_1"].astype(int)
        sampleEdgeDf["CellValueMask_2"] = sampleEdgeDf["CellValueMask_2"].astype(int)

        sampleEdgeDf = sampleEdgeDf.loc[
            sampleEdgeDf["CellValueMask_1"].isin(validLabels)
            & sampleEdgeDf["CellValueMask_2"].isin(validLabels)
        ].copy()

        if sampleEdgeDf.empty:
            _emit(progress, f"[{sampleValue}] No valid touching edges; skipping")
            continue

        labels = meta[mask_label_col].astype(int).to_numpy()
        clusterStrings = meta[cluster_col].astype(str).to_numpy()

        labelToIdx = {lab: i for i, lab in enumerate(labels)}

        i_idx = sampleEdgeDf["CellValueMask_1"].map(labelToIdx).to_numpy(dtype=int)
        j_idx = sampleEdgeDf["CellValueMask_2"].map(labelToIdx).to_numpy(dtype=int)

        codes, uniques = pd.factorize(clusterStrings, sort=False)
        k = len(uniques)

        if k == 0:
            continue

        # observed directed counts
        observedPairCodes = codes[i_idx] * k + codes[j_idx]
        observedCounts = np.bincount(observedPairCodes, minlength=k * k)

        reverseObservedPairCodes = codes[j_idx] * k + codes[i_idx]
        observedCounts = observedCounts + np.bincount(reverseObservedPairCodes, minlength=k * k)

        # accumulate permutation summaries
        sumCounts = np.zeros(k * k, dtype=np.float64)
        sumSqCounts = np.zeros(k * k, dtype=np.float64)
        geCounts = np.zeros(k * k, dtype=np.int64)  # perm >= obs
        leCounts = np.zeros(k * k, dtype=np.int64)  # perm <= obs

        _emit(
            progress,
            f"[{sampleValue}] Starting touching-label permutations: "
            f"{len(meta):,} cells, {len(sampleEdgeDf):,} touching edges, "
            f"{k:,} clusters, n={int(n_perm):,}"
        )

        for _ in range(int(n_perm)):
            shuffled = rng.permutation(codes)

            pairCodes = shuffled[i_idx] * k + shuffled[j_idx]
            permCounts = np.bincount(pairCodes, minlength=k * k)

            reversePairCodes = shuffled[j_idx] * k + shuffled[i_idx]
            permCounts = permCounts + np.bincount(reversePairCodes, minlength=k * k)

            sumCounts += permCounts
            sumSqCounts += permCounts.astype(np.float64) ** 2
            geCounts += (permCounts >= observedCounts)
            leCounts += (permCounts <= observedCounts)

        _emit(progress, f"[{sampleValue}] Permutations done")

        expected = sumCounts / float(n_perm)
        variance = (sumSqCounts / float(n_perm)) - (expected ** 2)
        variance = np.maximum(variance, 0.0)
        permSd = np.sqrt(variance)

        # empirical tails with +1 correction
        p_gt = (geCounts + 1) / (int(n_perm) + 1)
        p_lt = (leCounts + 1) / (int(n_perm) + 1)
        p_two = 2.0 * np.minimum(p_gt, p_lt)
        p_two = np.clip(p_two, 0.0, 1.0)

        linear = np.arange(k * k, dtype=int)
        cellCode = linear // k
        neighCode = linear % k

        statDf = pd.DataFrame({
            "cell_cluster": uniques.take(cellCode).astype(str),
            "neighbor_cluster": uniques.take(neighCode).astype(str),
            "observed": observedCounts.astype(float),
            "expected": expected.astype(float),
            "perm_sd": permSd.astype(float),
            "p_gt": p_gt.astype(float),
            "p_lt": p_lt.astype(float),
            "p_two_sided": p_two.astype(float),
        })

        # keep rows that are relevant in either observed or expected space
        statDf = statDf.loc[(statDf["observed"] > 0) | (statDf["expected"] > 0)].copy()
        statDf[sample_col] = sampleValue
        outFrames.append(statDf)

    if not outFrames:
        return pd.DataFrame(columns=[
            sample_col, "cell_cluster", "neighbor_cluster",
            "observed", "expected", "perm_sd",
            "p_gt", "p_lt", "p_two_sided"
        ])

    return pd.concat(outFrames, ignore_index=True)


def chance_correct_touching_interactions(
    edgeDf: pd.DataFrame,
    matchedDf: pd.DataFrame,
    *,
    sample_col: str,
    cluster_col: str,
    mask_label_col: str = "CellValueMask",
    n_perm: int = 1000,
    random_seed: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
    ) -> pd.DataFrame:
        """
        Full touching-based neighborhood summary with empirical p-values and FDR.
        """

        _emit(progress, "Summarizing observed touching interactions")
        observedDf = summarize_touching_observed(edgeDf, sample_col=sample_col)

        _emit(
            progress,
            f"Observed touching summary contains {len(observedDf):,} interaction rows"
        )

        _emit(progress, "Summarizing cluster frequencies")
        clusterFreqDf = summarize_cluster_frequency(
            matchedDf,
            sample_col=sample_col,
            cluster_col=cluster_col,
        )

        _emit(
            progress,
            f"Cluster frequency table contains {len(clusterFreqDf):,} sample-cluster rows"
        )

        _emit(progress, "Normalizing observed touching interactions by cluster abundance")
        normalizedObsDf = normalize_observed_interactions(
            observedDf,
            clusterFreqDf,
            sample_col=sample_col,
        )

        if normalizedObsDf.empty:
            _emit(progress, "No normalized touching interactions found; stopping chance correction")

            return pd.DataFrame(columns=[
                sample_col, "cell_cluster", "neighbor_cluster",
                "n_interactions", "n_cells", "normalized",
                "observed", "expected", "perm_sd",
                "normalized_expected", "ChanceCorrectedInteraction",
                "p_gt", "p_lt", "p_two_sided",
                "p_adj_gt", "p_adj_lt", "p_adj_two_sided",
                "Direction"
            ])

        _emit(
            progress,
            f"Computing permutation expectation on fixed touching graph "
            f"({int(n_perm):,} permutations per sample)"
        )

        expectedStatsDf = expected_touching_stats_permutation(
            edgeDf=edgeDf,
            matchedDf=matchedDf,
            sample_col=sample_col,
            cluster_col=cluster_col,
            mask_label_col=mask_label_col,
            n_perm=n_perm,
            random_seed=random_seed,
            progress=progress,
        )

        if expectedStatsDf.empty:
            _emit(
                progress,
                "Permutation expectation returned no rows; expected values will be set to zero"
            )

            expectedNormDf = pd.DataFrame(columns=[
                sample_col, "cell_cluster", "neighbor_cluster",
                "observed", "expected", "perm_sd",
                "p_gt", "p_lt", "p_two_sided",
                "n_cells", "normalized_expected"
            ])
        else:
            _emit(
                progress,
                f"Normalizing permutation expectations for {len(expectedStatsDf):,} interaction rows"
            )

            expectedNormDf = expectedStatsDf.merge(
                clusterFreqDf,
                on=[sample_col, "cell_cluster"],
                how="left",
            )

            expectedNormDf["normalized_expected"] = (
                expectedNormDf["expected"] / expectedNormDf["n_cells"]
            )

        _emit(progress, "Merging observed and expected touching interactions")

        out = normalizedObsDf.merge(
            expectedNormDf[
                [
                    sample_col, "cell_cluster", "neighbor_cluster",
                    "observed", "expected", "perm_sd",
                    "normalized_expected",
                    "p_gt", "p_lt", "p_two_sided",
                ]
            ],
            on=[sample_col, "cell_cluster", "neighbor_cluster"],
            how="left",
        )

        _emit(progress, "Calculating chance-corrected interaction scores")

        out["observed"] = out["observed"].fillna(out["n_interactions"].astype(float))
        out["expected"] = out["expected"].fillna(0.0)
        out["perm_sd"] = out["perm_sd"].fillna(0.0)
        out["normalized_expected"] = out["normalized_expected"].fillna(0.0)

        out["p_gt"] = out["p_gt"].fillna(1.0)
        out["p_lt"] = out["p_lt"].fillna(1.0)
        out["p_two_sided"] = out["p_two_sided"].fillna(1.0)

        out["ChanceCorrectedInteraction"] = (
            out["normalized"] - out["normalized_expected"]
        )

        out["Direction"] = np.where(
            out["ChanceCorrectedInteraction"] > 0,
            "Enriched",
            np.where(
                out["ChanceCorrectedInteraction"] < 0,
                "Depleted",
                "Neutral",
            ),
        )

        _emit(progress, "Adjusting empirical p-values using Benjamini-Hochberg FDR")

        # adjust within each analysis run across all tested interaction rows
        out["p_adj_gt"] = _bh_adjust(out["p_gt"])
        out["p_adj_lt"] = _bh_adjust(out["p_lt"])
        out["p_adj_two_sided"] = _bh_adjust(out["p_two_sided"])

        _emit(
            progress,
            f"Chance correction complete: {len(out):,} interaction rows"
        )

        return out
    

def make_sample_interaction_matrix(
    resultsDf: pd.DataFrame,
    *,
    sample_col: str,
    value_col: str = "ChanceCorrectedInteraction",
) -> pd.DataFrame:
    """
    Build a wide sample-by-interaction matrix from long interaction results.
    If multiple rows map to the same sample and interaction pair, average them.
    """
    if resultsDf is None or resultsDf.empty:
        return pd.DataFrame(columns=[sample_col])

    temp = resultsDf.copy()
    temp["InteractionPair"] = (
        temp["cell_cluster"].astype(str) + "__" + temp["neighbor_cluster"].astype(str)
    )

    mat = (
        temp.pivot_table(
            index=sample_col,
            columns="InteractionPair",
            values=value_col,
            aggfunc="mean",
            fill_value=0.0,
        )
        .reset_index()
    )

    mat.columns.name = None
    return mat


def aggregate_interaction_matrix(
    matrixDf: pd.DataFrame,
    metadataDf: pd.DataFrame,
    *,
    sample_col: str,
    aggregate_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate a sample-by-feature interaction matrix to a higher level
    (for example patient level) by taking the mean across rows sharing
    the same aggregate_col.
    """
    if matrixDf is None or matrixDf.empty:
        return pd.DataFrame(), pd.DataFrame()

    if metadataDf is None or metadataDf.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta = metadataDf[[sample_col, aggregate_col, group_col]].drop_duplicates().copy()

    df = matrixDf.merge(meta, on=sample_col, how="inner")
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    featureCols = [c for c in df.columns if c not in [sample_col, aggregate_col, group_col]]

    aggregatedMatrixDf = (
        df.groupby(aggregate_col, sort=False)[featureCols]
        .mean()
        .reset_index()
    )

    aggregatedMetaDf = (
        df[[aggregate_col, group_col]]
        .drop_duplicates()
        .groupby(aggregate_col, sort=False)
        .first()
        .reset_index()
    )

    return aggregatedMatrixDf, aggregatedMetaDf


def permanova_one_factor(
    matrixDf: pd.DataFrame,
    metadataDf: pd.DataFrame,
    *,
    sample_col: str,
    group_col: str,
    n_perm: int = 999,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Simple one-factor permutation MANOVA on a sample-by-feature matrix.
    Uses Euclidean-space sums of squares logic.
    """
    if matrixDf is None or matrixDf.empty:
        return pd.DataFrame()

    if metadataDf is None or metadataDf.empty:
        return pd.DataFrame()

    df = matrixDf.merge(metadataDf[[sample_col, group_col]].drop_duplicates(), on=sample_col, how="inner")
    if df.empty:
        return pd.DataFrame()

    featureCols = [c for c in df.columns if c not in [sample_col, group_col]]
    if len(featureCols) == 0:
        return pd.DataFrame()

    X = df[featureCols].to_numpy(dtype=float)
    groups = df[group_col].astype(str).to_numpy()

    n = X.shape[0]
    uniqueGroups = np.unique(groups)
    g = len(uniqueGroups)

    if n < 2 or g < 2:
        return pd.DataFrame([{
            "N": n,
            "Groups": g,
            "PseudoF": np.nan,
            "PValue": np.nan,
            "Permutations": n_perm,
        }])

    def compute_pseudo_f(Xmat: np.ndarray, grp: np.ndarray) -> float:
        grandCentroid = Xmat.mean(axis=0, keepdims=True)

        ssBetween = 0.0
        ssWithin = 0.0

        for lev in np.unique(grp):
            Xi = Xmat[grp == lev]
            if Xi.shape[0] == 0:
                continue
            centroid = Xi.mean(axis=0, keepdims=True)
            ssBetween += Xi.shape[0] * float(((centroid - grandCentroid) ** 2).sum())
            ssWithin += float(((Xi - centroid) ** 2).sum())

        dfBetween = len(np.unique(grp)) - 1
        dfWithin = Xmat.shape[0] - len(np.unique(grp))

        if dfBetween <= 0 or dfWithin <= 0 or ssWithin <= 0:
            return np.nan

        msBetween = ssBetween / dfBetween
        msWithin = ssWithin / dfWithin
        if msWithin <= 0:
            return np.nan

        return msBetween / msWithin

    observedF = compute_pseudo_f(X, groups)

    rng = np.random.default_rng(random_seed)
    permFs = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        permGroups = rng.permutation(groups)
        permFs[i] = compute_pseudo_f(X, permGroups)

    if np.isnan(observedF):
        pval = np.nan
    else:
        pval = (np.sum(permFs >= observedF) + 1) / (n_perm + 1)

    return pd.DataFrame([{
        "N": n,
        "Groups": g,
        "PseudoF": observedF,
        "PValue": pval,
        "Permutations": n_perm,
    }])