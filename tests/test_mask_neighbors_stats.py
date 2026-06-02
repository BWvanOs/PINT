import numpy as np
import pandas as pd

from pint_app.core.mask_neighbors import (
    build_touching_adjacency,
    annotate_touching_edges,
)

from pint_app.core.mask_neighbors_stats import (
    expected_touching_stats_permutation,
    chance_correct_touching_interactions,
)


def _make_annotated_edges(cellMask, labels, clusters):
    matchedDf = pd.DataFrame({
        "CellValueMask": labels,
        "Cluster": clusters,
        "Sample": ["S1"] * len(labels),
    })

    edgeDf = build_touching_adjacency(cellMask)

    annotatedEdgeDf = annotate_touching_edges(
        edgeDf,
        matchedDf,
        mask_label_col="CellValueMask",
        cluster_col="Cluster",
        sample_col="Sample",
    )

    return matchedDf, edgeDf, annotatedEdgeDf


def test_two_touching_cells_are_counted_bidirectionally():
    cellMask = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
    ], dtype=np.uint32)

    matchedDf, edgeDf, annotatedEdgeDf = _make_annotated_edges(
        cellMask,
        labels=[1, 2],
        clusters=["A", "B"],
    )

    statsDf = expected_touching_stats_permutation(
        edgeDf=annotatedEdgeDf,
        matchedDf=matchedDf,
        sample_col="Sample",
        cluster_col="Cluster",
        mask_label_col="CellValueMask",
        n_perm=10,
        random_seed=1,
    )

    obs = {
        (row.cell_cluster, row.neighbor_cluster): row.observed
        for row in statsDf.itertuples()
    }

    assert obs[("A", "B")] == 1
    assert obs[("B", "A")] == 1


def test_three_cell_chain_counts_a_b_contacts_correctly():
    cellMask = np.array([
        [1, 1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3, 3],
    ], dtype=np.uint32)

    matchedDf, edgeDf, annotatedEdgeDf = _make_annotated_edges(
        cellMask,
        labels=[1, 2, 3],
        clusters=["A", "B", "A"],
    )

    resultDf = chance_correct_touching_interactions(
        edgeDf=annotatedEdgeDf,
        matchedDf=matchedDf,
        sample_col="Sample",
        cluster_col="Cluster",
        mask_label_col="CellValueMask",
        n_perm=10,
        random_seed=1,
    )

    rowAB = resultDf[
        (resultDf["cell_cluster"] == "A")
        & (resultDf["neighbor_cluster"] == "B")
    ].iloc[0]

    rowBA = resultDf[
        (resultDf["cell_cluster"] == "B")
        & (resultDf["neighbor_cluster"] == "A")
    ].iloc[0]

    assert rowAB["observed"] == 2
    assert rowBA["observed"] == 2

    assert rowAB["n_cells"] == 2
    assert rowBA["n_cells"] == 1

    assert rowAB["observed_per_cell"] == 1.0
    assert rowBA["observed_per_cell"] == 2.0

    assert rowAB["ChanceCorrectedInteraction"] == (
        rowAB["observed_per_cell"] - rowAB["expected_per_cell"]
    )

    assert rowBA["ChanceCorrectedInteraction"] == (
        rowBA["observed_per_cell"] - rowBA["expected_per_cell"]
    )


def test_same_cluster_touch_is_counted_as_two_directed_self_interactions():
    cellMask = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
    ], dtype=np.uint32)

    matchedDf, edgeDf, annotatedEdgeDf = _make_annotated_edges(
        cellMask,
        labels=[1, 2],
        clusters=["A", "A"],
    )

    statsDf = expected_touching_stats_permutation(
        edgeDf=annotatedEdgeDf,
        matchedDf=matchedDf,
        sample_col="Sample",
        cluster_col="Cluster",
        mask_label_col="CellValueMask",
        n_perm=10,
        random_seed=1,
    )

    rowAA = statsDf[
        (statsDf["cell_cluster"] == "A")
        & (statsDf["neighbor_cluster"] == "A")
    ].iloc[0]

    assert rowAA["observed"] == 2