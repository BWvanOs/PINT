"""PINT six-ROI regression tests with independently known outcomes.

Run from repository root:
    python -m pytest tests/test_neighborhood_integration.py -v

Requires locally implemented aggregate_interaction_matrix_weighted and
zero-preserving make_sample_interaction_matrix.
"""
import numpy as np
import pandas as pd
import pytest

from pint_app.core.mask_neighbors import build_touching_edges_for_pushed_dataset
from pint_app.core.mask_neighbors_stats import (
    chance_correct_touching_interactions,
    make_sample_interaction_matrix,
    aggregate_interaction_matrix_weighted,
    permanova_one_factor,
)
from tests.fixtures.synthetic_neighborhood import (
    ROI_TYPES, generate_dataset, expected_edges, observed_directed,
    exact_expected_directed, exact_sample_score,
)


@pytest.fixture
def synthetic(tmp_path):
    data = generate_dataset(tmp_path)
    matched, edges = build_touching_edges_for_pushed_dataset(data)
    return data, matched, edges


def _run_permutations(matched, edges, n_perm=4000):
    return chance_correct_touching_interactions(
        edgeDf=edges,
        matchedDf=matched,
        sample_col="ROIName",  # Must be ROI, NOT SampleName.
        cluster_col="Cluster",
        n_perm=n_perm,
        random_seed=42,
    )


def _row(df, roi, source, target):
    rows = df.loc[
        df["ROIName"].eq(roi)
        & df["cell_cluster"].eq(source)
        & df["neighbor_cluster"].eq(target)
    ]
    assert len(rows) == 1, f"Expected exactly one {roi} {source}->{target} row"
    return rows.iloc[0]


def test_mask_matching_and_expected_edges(synthetic):
    data, matched, edges = synthetic
    assert len(matched) == sum(len(t[2]) for t in ROI_TYPES.values())
    assert set(matched.ROIName) == set(ROI_TYPES)
    for roi, (sample, _, types) in ROI_TYPES.items():
        cells = matched.loc[matched.ROIName == roi]
        assert sorted(cells.CellValueMask.astype(int)) == list(range(1, len(types)+1))
        assert set(cells.SampleName) == {sample}
        assert cells.sort_values("CellValueMask").Cluster.tolist() == list(types)
        graph = edges.loc[edges.ROIName == roi]
        got = set(map(tuple, graph[["CellValueMask_1", "CellValueMask_2"]].astype(int).to_numpy()))
        assert got == expected_edges(roi)
    # Same cell-mask label is reused in all six ROIs: intended and safe.
    assert (matched.CellValueMask == 1).sum() == 6


def test_roi_permutations_preserve_cell_identity_and_exact_observed(synthetic):
    _, matched, edges = synthetic
    stats = _run_permutations(matched, edges)
    assert set(stats.ROIName) == set(ROI_TYPES)
    for roi, (_, _, types) in ROI_TYPES.items():
        for source in set(types):
            for target in set(types):
                row = _row(stats, roi, source, target)
                assert row["observed"] == observed_directed(roi, source, target)
                assert row["n_cells"] == types.count(source)
                # Monte Carlo estimate compared against independent exact formula.
                assert row["expected"] == pytest.approx(
                    exact_expected_directed(roi, source, target), abs=0.15
                )
                assert row["ChanceCorrectedInteraction"] == pytest.approx(
                    (row["observed"] - row["expected"]) / row["n_cells"]
                )


def test_true_zero_and_absent_population_are_not_confused(synthetic):
    _, matched, edges = synthetic
    stats = _run_permutations(matched, edges, n_perm=100)
    # One C cell in ROI003: no C-C edge can exist in any label permutation.
    zero = _row(stats, "ROI003", "C", "C")
    assert zero["observed"] == 0
    assert zero["expected"] == 0
    assert zero["ChanceCorrectedInteraction"] == 0
    assert zero["p_two_sided"] == 1

    matrix = make_sample_interaction_matrix(stats, sample_col="ROIName")
    assert matrix.loc[matrix.ROIName == "ROI003", "C__C"].iloc[0] == 0
    # ROI006 has no B cells, so A->B must be undefined, not neutral.
    assert pd.isna(matrix.loc[matrix.ROIName == "ROI006", "A__B"].iloc[0])


def test_source_cell_weighted_aggregation_matches_known_answer(synthetic):
    _, matched, edges = synthetic
    stats = _run_permutations(matched, edges)
    meta = matched[["ROIName", "SampleName", "Condition"]].drop_duplicates()
    matrix, out_meta = aggregate_interaction_matrix_weighted(
        resultsDf=stats,
        metadataDf=meta,
        sample_col="ROIName",
        aggregate_col="SampleName",
        group_col="Condition",
    )
    assert set(matrix.SampleName) == {"1_1", "1_2"}
    assert len(out_meta) == 2
    for sample in ("1_1", "1_2"):
        measured = matrix.loc[matrix.SampleName == sample, "A__B"].iloc[0]
        assert measured == pytest.approx(
            exact_sample_score(sample, "A", "B"), abs=0.12
        )
    # C only exists in ROI003 of 1_1: C->C is zero in 1_1,
    # undefined for all of 1_2.
    assert matrix.loc[matrix.SampleName == "1_1", "C__C"].iloc[0] == 0
    assert pd.isna(matrix.loc[matrix.SampleName == "1_2", "C__C"].iloc[0])


def test_entire_sample_without_results_is_kept_as_nan(synthetic):
    _, matched, edges = synthetic
    stats = _run_permutations(matched, edges, n_perm=100)
    # Simulate only sample 1_1 having any valid result rows.
    stats = stats.loc[stats.ROIName.isin(["ROI001", "ROI002", "ROI003"])].copy()
    meta = matched[["ROIName", "SampleName", "Condition"]].drop_duplicates()
    matrix, _ = aggregate_interaction_matrix_weighted(
        resultsDf=stats,
        metadataDf=meta,
        sample_col="ROIName",
        aggregate_col="SampleName",
        group_col="Condition",
    )
    assert set(matrix.SampleName) == {"1_1", "1_2"}
    assert pd.isna(matrix.loc[matrix.SampleName == "1_2", "A__B"].iloc[0])


def test_two_samples_do_not_falsely_create_permanova_replication(synthetic):
    _, matched, edges = synthetic
    stats = _run_permutations(matched, edges, n_perm=100)
    meta = matched[["ROIName", "SampleName", "Condition"]].drop_duplicates()
    matrix, out_meta = aggregate_interaction_matrix_weighted(
        resultsDf=stats,
        metadataDf=meta,
        sample_col="ROIName",
        aggregate_col="SampleName",
        group_col="Condition",
    )
    # Only choose shared, finite features, as per the PINT viewer policy.
    complete = [col for col in matrix if col != "SampleName"
                and np.isfinite(matrix[col].to_numpy(dtype=float)).all()]
    assert complete
    result = permanova_one_factor(
        matrixDf=matrix[["SampleName"] + complete],
        metadataDf=out_meta,
        sample_col="SampleName",
        group_col="Condition",
        n_perm=19, random_seed=42,
    )
    # One sample per condition -> no residual DF, so p cannot be inferred.
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["PseudoF"])
    assert pd.isna(result.iloc[0]["PValue"])


def test_permanova_four_independent_samples():
    # Separate fixture: 2 biological samples per condition.
    # This tests the statistical routine, not the viewer's feature filter.
    matrix = pd.DataFrame({
        "SampleName": ["S1", "S2", "S3", "S4"],
        "A__B": [0.0, 0.2, 1.0, 1.2],
        "B__A": [0.0, 0.1, 0.9, 1.1],
    })
    meta = pd.DataFrame({
        "SampleName": ["S1", "S2", "S3", "S4"],
        "Condition": ["Control", "Control", "Inflamed", "Inflamed"],
    })
    out = permanova_one_factor(
        matrixDf=matrix, metadataDf=meta,
        sample_col="SampleName", group_col="Condition",
        n_perm=99, random_seed=42,
    )
    assert len(out) == 1
    assert np.isfinite(out.iloc[0]["PseudoF"])
    assert out.iloc[0]["PseudoF"] > 0
    assert 0 < out.iloc[0]["PValue"] <= 1
    assert out.iloc[0]["N"] == 4
