from __future__ import annotations
from typing import Callable, Optional, Tuple, List
from typing import Optional
import numpy as np
import pandas as pd


ProgressFn = Callable[[str], None]

def _emit(progress: Optional[ProgressFn], msg: str) -> None:
    if progress is not None:
        progress(msg)


def get_neighbors_one_sample(
    df: pd.DataFrame,
    radius: float,
    *,
    x_col: str = "X_Location",
    y_col: str = "Y_Location",
    cluster_col: str = "ClusterName",
    cellid_col: str = "CellID",
) -> pd.DataFrame:
    """
    For a single sample dataframe, return neighbor pairs within `radius`.
    Equivalent to your R get_neighbors(df, radius), but uses CellID if present.
    """
    if df.empty:
        return pd.DataFrame(columns=["cell", "neighbor", "cell_cluster", "neighbor_cluster"])

    coords = df[[x_col, y_col]].to_numpy(dtype=float, copy=False)
    clusters = df[cluster_col].astype("string").to_numpy()

    # Prefer CellID values if present, otherwise use 1..n like your R loop index
    if cellid_col in df.columns:
        cell_ids = df[cellid_col].astype("string").to_numpy()
    else:
        cell_ids = np.arange(1, len(df) + 1).astype(str)

    # ---- Fast path: SciPy KD-tree if available ----
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(coords)
        # neighbors_list[i] = list of indices within radius of point i (including itself)
        neighbors_list = tree.query_ball_point(coords, r=radius)

        cell_out: list[str] = []
        neigh_out: list[str] = []
        cell_cl_out: list[str] = []
        neigh_cl_out: list[str] = []

        for i, nbrs in enumerate(neighbors_list):
            # remove self
            nbrs = [j for j in nbrs if j != i]
            if not nbrs:
                continue

            cell_out.extend([cell_ids[i]] * len(nbrs))
            neigh_out.extend(cell_ids[nbrs].tolist())
            cell_cl_out.extend([clusters[i]] * len(nbrs))
            neigh_cl_out.extend(clusters[nbrs].tolist())

        return pd.DataFrame(
            {
                "cell": cell_out,
                "neighbor": neigh_out,
                "cell_cluster": cell_cl_out,
                "neighbor_cluster": neigh_cl_out,
            }
        )

    except Exception:
        # ---- Fallback: O(n^2) distance computation (OK for small samples) ----
        # Compute squared distances via broadcasting; avoid sqrt by comparing to radius^2
        r2 = float(radius) ** 2
        diffs = coords[:, None, :] - coords[None, :, :]
        dist2 = (diffs ** 2).sum(axis=2)

        cell_out: list[str] = []
        neigh_out: list[str] = []
        cell_cl_out: list[str] = []
        neigh_cl_out: list[str] = []

        n = dist2.shape[0]
        for i in range(n):
            # within radius, excluding self (dist2 > 0)
            nbrs = np.where((dist2[i] <= r2) & (dist2[i] > 0))[0]
            if nbrs.size == 0:
                continue

            cell_out.extend([cell_ids[i]] * int(nbrs.size))
            neigh_out.extend(cell_ids[nbrs].tolist())
            cell_cl_out.extend([clusters[i]] * int(nbrs.size))
            neigh_cl_out.extend(clusters[nbrs].tolist())

        return pd.DataFrame(
            {
                "cell": cell_out,
                "neighbor": neigh_out,
                "cell_cluster": cell_cl_out,
                "neighbor_cluster": neigh_cl_out,
            }
        )

def observed_neighbors(
    df: pd.DataFrame,
    radius: float,
    *,
    sample_col: str = "SampleNumber",
    x_col: str = "X_Location",
    y_col: str = "Y_Location",
    cluster_col: str = "ClusterName",
    cellid_col: str = "CellID",
    progress: Optional[ProgressFn] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["cell", "neighbor", "cell_cluster", "neighbor_cluster", sample_col])

    out_frames: list[pd.DataFrame] = []
    for sample_value, sdf in df.groupby(sample_col, sort=False):
        _emit(progress, f"[{sample_value}] Starting neighbor association ({len(sdf):,} cells)")

        res = get_neighbors_one_sample(
            sdf,
            radius,
            x_col=x_col,
            y_col=y_col,
            cluster_col=cluster_col,
            cellid_col=cellid_col,
        )

        _emit(progress, f"[{sample_value}] Neighbor association done ({len(res):,} pairs)")

        res[sample_col] = sample_value
        out_frames.append(res)

    if not out_frames:
        return pd.DataFrame(columns=["cell", "neighbor", "cell_cluster", "neighbor_cluster", sample_col])

    return pd.concat(out_frames, ignore_index=True)


def summarize_observed_interactions(
    neighbors_df: pd.DataFrame,
    original_df: pd.DataFrame,
    *,
    sample_col: str = "SampleNumber",
    cluster_col: str = "ClusterName",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Replicates the R steps:
      raw_interactions
      cluster_frequency
      normalized_interactions

    Returns:
      raw_interactions: [SampleNumber, cell_cluster, neighbor_cluster, n_interactions]
      cluster_frequency: [SampleNumber, cell_cluster, n_cells]
      normalized_interactions: raw_interactions joined with cluster_frequency and normalized
    """
    if neighbors_df.empty:
        raw = pd.DataFrame(columns=[sample_col, "cell_cluster", "neighbor_cluster", "n_interactions"])
    else:
        raw = (
            neighbors_df
            .groupby([sample_col, "cell_cluster", "neighbor_cluster"], sort=False)
            .size()
            .reset_index(name="n_interactions")
        )

    cluster_frequency = (
        original_df
        .groupby([sample_col, cluster_col], sort=False)
        .size()
        .reset_index(name="n_cells")
        .rename(columns={cluster_col: "cell_cluster"})
    )

    normalized = raw.merge(cluster_frequency, on=[sample_col, "cell_cluster"], how="left")
    if not normalized.empty:
        normalized["normalized"] = normalized["n_interactions"] / normalized["n_cells"]
    else:
        normalized["normalized"] = pd.Series(dtype=float)

    return raw, cluster_frequency, normalized


# ----------------------------
# New: fast, fixed neighbor index pairs for permutations
# ----------------------------
def _neighbor_index_pairs_one_sample(
    coords: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build directed neighbor index pairs (i -> j) within radius, excluding i==j.

    Returns:
      i_idx, j_idx (same length), both int arrays indexing rows in the sample dataframe.
    """
    if coords.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Fast path: SciPy KD-tree
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(coords)
        neighbors_list = tree.query_ball_point(coords, r=float(radius))

        i_out: List[int] = []
        j_out: List[int] = []

        for i, nbrs in enumerate(neighbors_list):
            # remove self
            if not nbrs:
                continue
            for j in nbrs:
                if j != i:
                    i_out.append(i)
                    j_out.append(j)

        return np.asarray(i_out, dtype=np.int32), np.asarray(j_out, dtype=np.int32)

    except Exception:
        # Fallback: O(n^2) for small samples
        r2 = float(radius) ** 2
        diffs = coords[:, None, :] - coords[None, :, :]
        dist2 = (diffs ** 2).sum(axis=2)

        mask = (dist2 <= r2) & (dist2 > 0)
        i_idx, j_idx = np.where(mask)
        return i_idx.astype(np.int32), j_idx.astype(np.int32)


def expected_counts_permutation(
    df: pd.DataFrame,
    radius: float,
    n_perm: int,
    *,
    sample_col: str = "SampleNumber",
    x_col: str = "X_Location",
    y_col: str = "Y_Location",
    cluster_col: str = "ClusterName",
    random_seed: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
    progress_every: int = 50,  # emit every N permutations (set to 0/None to disable)
) -> pd.DataFrame:
    if n_perm <= 0 or df.empty:
        return pd.DataFrame(columns=[sample_col, "cell_cluster", "neighbor_cluster", "expected"])

    rng = np.random.default_rng(random_seed)
    out_frames: List[pd.DataFrame] = []

    for sample_value, sdf in df.groupby(sample_col, sort=False):
        _emit(progress, f"[{sample_value}] Building fixed neighbor index list for permutations")

        coords = sdf[[x_col, y_col]].to_numpy(dtype=float, copy=False)
        clusters = sdf[cluster_col].astype("string").to_numpy()

        i_idx, j_idx = _neighbor_index_pairs_one_sample(coords, radius)

        _emit(progress, f"[{sample_value}] Fixed neighbor list done ({i_idx.size:,} directed pairs)")

        if i_idx.size == 0:
            _emit(progress, f"[{sample_value}] No neighbor pairs within radius; skipping permutations")
            continue

        codes, uniques = pd.factorize(clusters, sort=False)
        k = int(len(uniques))
        if k == 0:
            continue

        sum_counts = np.zeros(k * k, dtype=np.int64)

        _emit(progress, f"[{sample_value}] Starting permutations (n={n_perm:,})")

        for p in range(int(n_perm)):
            shuffled = rng.permutation(codes)
            pair_codes = shuffled[i_idx] * k + shuffled[j_idx]
            counts = np.bincount(pair_codes, minlength=k * k)
            sum_counts += counts

            if progress_every and ((p + 1) % progress_every == 0):
                _emit(progress, f"[{sample_value}] Permutations: {p + 1:,}/{n_perm:,}")

        _emit(progress, f"[{sample_value}] Permutations done")

        expected = sum_counts.astype(np.float64) / float(n_perm)

        linear = np.arange(k * k, dtype=np.int32)
        cell_code = (linear // k).astype(np.int32)
        neigh_code = (linear % k).astype(np.int32)

        exp_df = pd.DataFrame(
            {
                "cell_cluster": uniques.take(cell_code).astype(str),
                "neighbor_cluster": uniques.take(neigh_code).astype(str),
                "expected": expected,
            }
        )
        exp_df = exp_df.loc[exp_df["expected"] > 0].copy()
        exp_df[sample_col] = sample_value
        out_frames.append(exp_df)

    if not out_frames:
        return pd.DataFrame(columns=[sample_col, "cell_cluster", "neighbor_cluster", "expected"])

    return pd.concat(out_frames, ignore_index=True)



def chance_corrected_interactions(
    df: pd.DataFrame,
    radius: float,
    n_perm: int,
    *,
    sample_col: str = "SampleNumber",
    x_col: str = "X_Location",
    y_col: str = "Y_Location",
    cluster_col: str = "ClusterName",
    cellid_col: str = "CellID",
    random_seed: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
) -> pd.DataFrame:
    _emit(progress, "Starting observed neighbor calculation")
    neighbors_df = observed_neighbors(
        df,
        radius,
        sample_col=sample_col,
        x_col=x_col,
        y_col=y_col,
        cluster_col=cluster_col,
        cellid_col=cellid_col,
        progress=progress,
    )
    _emit(progress, "Observed neighbor calculation done; summarizing")

    _, cluster_frequency, normalized_obs = summarize_observed_interactions(
        neighbors_df,
        df,
        sample_col=sample_col,
        cluster_col=cluster_col,
    )

    if normalized_obs.empty:
        _emit(progress, "No observed interactions; returning empty result")
        cols = [
            sample_col, "cell_cluster", "neighbor_cluster",
            "n_interactions", "n_cells", "normalized",
            "expected", "normalized_expected", "ChanceCorrectedInteraction",
        ]
        return pd.DataFrame(columns=cols)

    _emit(progress, "Starting permutation-based expectation")
    expected_df = expected_counts_permutation(
        df,
        radius,
        n_perm,
        sample_col=sample_col,
        x_col=x_col,
        y_col=y_col,
        cluster_col=cluster_col,
        random_seed=random_seed,
        progress=progress,
    )
    _emit(progress, "Permutation-based expectation done; normalizing and chance-correcting")

    if expected_df.empty:
        expected_norm = pd.DataFrame(
            columns=[sample_col, "cell_cluster", "neighbor_cluster", "expected", "n_cells", "normalized_expected"]
        )
    else:
        expected_norm = expected_df.merge(cluster_frequency, on=[sample_col, "cell_cluster"], how="left")
        expected_norm["normalized_expected"] = expected_norm["expected"] / expected_norm["n_cells"]

    out = normalized_obs.merge(
        expected_norm[[sample_col, "cell_cluster", "neighbor_cluster", "expected", "normalized_expected"]],
        on=[sample_col, "cell_cluster", "neighbor_cluster"],
        how="left",
    )
    out["expected"] = out["expected"].fillna(0.0)
    out["normalized_expected"] = out["normalized_expected"].fillna(0.0)
    out["ChanceCorrectedInteraction"] = out["normalized"] - out["normalized_expected"]

    _emit(progress, "Done")
    return out
