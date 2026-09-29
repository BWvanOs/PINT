from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ProgressCallback = Callable[[str, int, int], None]


def _benjamini_hochberg(
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Benjamini-Hochberg multiple-testing correction.

    NaN p-values remain NaN.
    """
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        p_values.shape,
        np.nan,
        dtype=float,
    )

    valid_mask = np.isfinite(
        p_values
    )

    if not valid_mask.any():
        return adjusted

    valid_p = p_values[
        valid_mask
    ]

    order = np.argsort(
        valid_p
    )

    ranked_p = valid_p[
        order
    ]

    n_tests = len(
        ranked_p
    )

    ranks = np.arange(
        1,
        n_tests + 1,
        dtype=float,
    )

    ranked_adjusted = (
        ranked_p
        * n_tests
        / ranks
    )

    # BH adjusted values must be monotonic when
    # traversing from the largest p-value downward.
    ranked_adjusted = np.minimum.accumulate(
        ranked_adjusted[::-1]
    )[::-1]

    ranked_adjusted = np.clip(
        ranked_adjusted,
        0.0,
        1.0,
    )

    original_order_adjusted = np.empty(
        n_tests,
        dtype=float,
    )

    original_order_adjusted[
        order
    ] = ranked_adjusted

    adjusted[
        valid_mask
    ] = original_order_adjusted

    return adjusted


def _prepare_numeric_feature(
    series: pd.Series,
) -> np.ndarray:
    """
    Convert one feature column to a floating-point NumPy array.

    Invalid and infinite values are represented as NaN.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float,
        copy=False,
    )

    values = values.copy()

    values[
        ~np.isfinite(values)
    ] = np.nan

    return values


def _report_progress(
    message: str,
    current: int,
    total: int,
    callback: ProgressCallback | None,
) -> None:
    """
    Send progress to the UI when available and always print it
    to the terminal for long-running analyses.
    """
    print(
        message,
        flush=True,
    )

    if callback is not None:
        callback(
            message,
            current,
            total,
        )


def find_all_cluster_markers(
    data_df: pd.DataFrame,
    *,
    cluster_col: str,
    feature_cols: list[str],
    only_positive: bool = True,
    min_pct: float = 0.10,
    min_log2fc: float = 0.25,
    detection_threshold: float = 0.0,
    pseudocount: float = 1e-9,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """
    Find marker features for every cluster versus all remaining cells.

    This provides a Seurat FindAllMarkers-like first implementation
    for PINT.

    For each cluster and feature, PINT calculates:
    - arithmetic mean inside the cluster
    - arithmetic mean outside the cluster
    - log2 fold-change of those means
    - fraction of cells above detection_threshold
    - Mann-Whitney U / Wilcoxon rank-sum p-value
    - Benjamini-Hochberg adjusted p-value

    Multiple-testing correction is performed separately for each
    cluster-versus-rest comparison.

    Parameters
    ----------
    data_df:
        Cell-level master dataframe.

    cluster_col:
        Column containing the cluster annotation to compare.

    feature_cols:
        Numeric marker/gene/channel columns to test.

    only_positive:
        If True, retain only features enriched in the cluster.

    min_pct:
        A feature is considered only when its detected fraction is at
        least this high in either the cluster or the remaining cells.

    min_log2fc:
        Minimum absolute log2 fold-change. When only_positive=True,
        this is interpreted as minimum positive log2FC.

    detection_threshold:
        Values greater than this threshold count as detected.

        For count-like Xenium data, zero is usually sensible.
        For IMC, another threshold may eventually be more meaningful.

    pseudocount:
        Small value added to means before calculating their ratio.

    progress_callback:
        Optional callback with signature:
            callback(message, current_cluster, total_clusters)

    Returns
    -------
    pd.DataFrame
        Long marker table containing one row per tested cluster-feature
        combination that passed the prefilters.
    """
    if data_df is None or data_df.empty:
        raise ValueError(
            "The clustering dataset is empty."
        )

    if cluster_col not in data_df.columns:
        raise ValueError(
            f"Cluster column {cluster_col!r} is not present "
            "in the clustering dataset."
        )

    if not feature_cols:
        raise ValueError(
            "No marker features were selected."
        )

    missing_features = [
        feature
        for feature in feature_cols
        if feature not in data_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Marker feature column(s) are missing from the dataset: "
            + ", ".join(
                missing_features[:20]
            )
        )

    if not 0 <= min_pct <= 1:
        raise ValueError(
            "min_pct must be between 0 and 1."
        )

    if min_log2fc < 0:
        raise ValueError(
            "min_log2fc cannot be negative."
        )

    if pseudocount <= 0:
        raise ValueError(
            "pseudocount must be greater than zero."
        )

    cluster_series = (
        data_df[cluster_col]
        .astype("string")
    )

    valid_cluster_mask = (
        cluster_series.notna()
        & cluster_series.str.strip().ne("")
    )

    if not valid_cluster_mask.any():
        raise ValueError(
            f"Cluster column {cluster_col!r} does not contain "
            "any valid cluster names."
        )

    cluster_series = cluster_series.loc[
        valid_cluster_mask
    ]

    working_df = data_df.loc[
        valid_cluster_mask,
        feature_cols,
    ]

    cluster_names = (
        cluster_series
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    total_clusters = len(
        cluster_names
    )

    if total_clusters < 2:
        raise ValueError(
            "At least two clusters are required for "
            "cluster-versus-rest marker analysis."
        )

    print(
        "▶️ Starting cluster marker analysis: "
        f"{len(working_df):,} cells, "
        f"{len(feature_cols):,} features, "
        f"{total_clusters:,} clusters.",
        flush=True,
    )

    # Convert every feature only once rather than once per cluster.
    numeric_features = {
        feature: _prepare_numeric_feature(
            working_df[feature]
        )
        for feature in feature_cols
    }

    cluster_values = (
        cluster_series
        .astype(str)
        .to_numpy()
    )

    all_results = []

    for cluster_index, cluster_name in enumerate(
        cluster_names,
        start=1,
    ):
        in_cluster = (
            cluster_values
            == cluster_name
        )

        out_cluster = ~in_cluster

        n_in = int(
            in_cluster.sum()
        )

        n_out = int(
            out_cluster.sum()
        )

        _report_progress(
            (
                f"▶️ DEG {cluster_index}/{total_clusters}: "
                f"{cluster_name} vs all other cells "
                f"({n_in:,} vs {n_out:,} cells)..."
            ),
            cluster_index - 1,
            total_clusters,
            progress_callback,
        )

        cluster_results = []

        for feature in feature_cols:
            values = numeric_features[
                feature
            ]

            x = values[
                in_cluster
            ]

            y = values[
                out_cluster
            ]

            x = x[
                np.isfinite(x)
            ]

            y = y[
                np.isfinite(y)
            ]

            if (
                len(x) == 0
                or len(y) == 0
            ):
                continue

            mean_in = float(
                np.mean(x)
            )

            mean_out = float(
                np.mean(y)
            )

            pct_in = float(
                np.mean(
                    x > detection_threshold
                )
            )

            pct_out = float(
                np.mean(
                    y > detection_threshold
                )
            )

            # Similar to Seurat's min.pct concept:
            # do not test features essentially absent everywhere.
            if max(
                pct_in,
                pct_out,
            ) < min_pct:
                continue

            # A log ratio is meaningful for the non-negative
            # expression/intensity values PINT currently expects.
            if (
                mean_in < 0
                or mean_out < 0
            ):
                log2fc = np.nan
            else:
                log2fc = float(
                    np.log2(
                        (
                            mean_in
                            + pseudocount
                        )
                        /
                        (
                            mean_out
                            + pseudocount
                        )
                    )
                )

            if np.isfinite(
                log2fc
            ):
                if only_positive:
                    if log2fc < min_log2fc:
                        continue
                else:
                    if abs(
                        log2fc
                    ) < min_log2fc:
                        continue

            try:
                _, p_value = mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided",
                    method="auto",
                )

                p_value = float(
                    p_value
                )

            except ValueError:
                # Usually occurs only for pathological/empty
                # inputs that escaped earlier validation.
                p_value = np.nan

            cluster_results.append(
                {
                    "Cluster": cluster_name,
                    "Feature": feature,
                    "n_in": n_in,
                    "n_out": n_out,
                    "Mean_in": mean_in,
                    "Mean_out": mean_out,
                    "Mean_difference": (
                        mean_in
                        - mean_out
                    ),
                    "log2FC": log2fc,
                    "pct_in": pct_in,
                    "pct_out": pct_out,
                    "p_value": p_value,
                }
            )

        if cluster_results:
            cluster_df = pd.DataFrame(
                cluster_results
            )

            cluster_df[
                "p_adj_BH"
            ] = _benjamini_hochberg(
                cluster_df[
                    "p_value"
                ].to_numpy()
            )

            cluster_df = (
                cluster_df
                .sort_values(
                    [
                        "p_adj_BH",
                        "log2FC",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                    na_position="last",
                )
                .reset_index(
                    drop=True
                )
            )

            all_results.append(
                cluster_df
            )

            n_significant = int(
                (
                    cluster_df[
                        "p_adj_BH"
                    ]
                    < 0.05
                ).sum()
            )

            _report_progress(
                (
                    f"✅ Finished {cluster_name}: "
                    f"{len(cluster_df):,} marker(s) passed filters; "
                    f"{n_significant:,} with BH-adjusted p < 0.05."
                ),
                cluster_index,
                total_clusters,
                progress_callback,
            )

        else:
            _report_progress(
                (
                    f"✅ Finished {cluster_name}: "
                    "no features passed the current marker filters."
                ),
                cluster_index,
                total_clusters,
                progress_callback,
            )

    if not all_results:
        print(
            "⚠️ Cluster marker analysis finished, but no features "
            "passed the selected filters.",
            flush=True,
        )

        return pd.DataFrame(
            columns=[
                "Cluster",
                "Feature",
                "n_in",
                "n_out",
                "Mean_in",
                "Mean_out",
                "Mean_difference",
                "log2FC",
                "pct_in",
                "pct_out",
                "p_value",
                "p_adj_BH",
            ]
        )

    result = pd.concat(
        all_results,
        ignore_index=True,
    )

    print(
        "✅ Cluster marker analysis complete: "
        f"{len(result):,} cluster-marker result(s) retained "
        f"across {total_clusters:,} clusters.",
        flush=True,
    )

    return result