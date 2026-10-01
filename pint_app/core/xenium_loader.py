import json
import zarr
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import sparse

def open_xenium_zarr(
    path: str | Path,
):
    """
    Open a Xenium Zarr store read-only.

    Supports zipped Zarr stores and unpacked Zarr directories.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Xenium Zarr resource does not exist: {path}"
        )

    if path.suffix.lower() == ".zip":
        store = zarr.storage.ZipStore(
            str(path),
            mode="r",
        )
    else:
        store = zarr.storage.LocalStore(
            str(path)
        )

    root = zarr.open_group(
        store=store,
        mode="r",
    )

    return root, store

def _format_zarr_attribute(
    value,
    max_items=5,
    max_chars=150,
):
    """
    Produce a compact preview of a Zarr attribute.

    Avoid displaying thousands of gene names or IDs
    inside a single Shiny DataGrid cell.
    """
    if isinstance(value, dict):
        return (
            f"Dictionary with {len(value):,} entries: "
            f"{list(value.keys())[:max_items]}"
        )

    if isinstance(value, (list, tuple, np.ndarray)):
        preview = [
            str(item)
            for item in value[:max_items]
        ]

        return (
            f"{len(value):,} items: "
            + ", ".join(preview)
            + (" ..." if len(value) > max_items else "")
        )

    text = str(value)

    if len(text) > max_chars:
        return text[:max_chars] + " ..."

    return text

def inspect_zarr_group(
    group,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Recursively describe a Zarr hierarchy without loading array contents.
    """
    rows = []

    # Group attributes
    try:
        for key, value in group.attrs.items():
            rows.append(
                {
                    "Path": prefix or "/",
                    "Kind": "attribute",
                    "Shape": "",
                    "Dtype": "",
                    "Name": str(key),
                    "Value": _format_zarr_attribute(value),
                }
            )
    except Exception:
        pass

    # Arrays directly inside this group
    for name, array in group.arrays():
        arrayPath = (
            f"{prefix}/{name}"
            if prefix
            else f"/{name}"
        )

        rows.append(
            {
                "Path": arrayPath,
                "Kind": "array",
                "Shape": str(array.shape),
                "Dtype": str(array.dtype),
                "Name": str(name),
                "Value": "",
            }
        )

        try:
            for key, value in array.attrs.items():
                rows.append(
                    {
                        "Path": arrayPath,
                        "Kind": "array attribute",
                        "Shape": "",
                        "Dtype": "",
                        "Name": str(key),
                        "Value": _format_zarr_attribute(value),
                    }
                )
        except Exception:
            pass

    # Child groups
    for name, child in group.groups():
        childPrefix = (
            f"{prefix}/{name}"
            if prefix
            else f"/{name}"
        )

        rows.extend(
            inspect_zarr_group(
                child,
                prefix=childPrefix,
            ).to_dict(
                orient="records"
            )
        )

    return pd.DataFrame(rows)

def decode_xenium_cell_ids(encoded_ids):
    """
    Convert Xenium's two-column integer cell IDs
    to their original Xenium Explorer string IDs.
    """
    encoded_ids = np.asarray(encoded_ids)

    if (
        encoded_ids.ndim != 2
        or encoded_ids.shape[1] != 2
    ):
        raise ValueError(
            "Expected Xenium cell IDs with shape (n_cells, 2)."
        )

    alphabet = "abcdefghijklmnop"

    def decode_prefix(value):
        hex_string = f"{int(value):08x}"

        return "".join(
            alphabet[int(digit, 16)]
            for digit in hex_string
        )

    return [
        f"{decode_prefix(prefix)}-{int(suffix)}"
        for prefix, suffix in encoded_ids
    ]

def load_xenium_count_matrix(matrix_path):
    """
    Load Xenium's sparse count matrix.

    Returns
    -------
    X
        CSR matrix, cells x features.
    cell_ids
        Original Xenium string cell IDs.
    var
        Feature metadata.
    """
    root, store = open_xenium_zarr(
        matrix_path
    )

    try:
        group = root["cell_features"]

        n_cells = int(
            group.attrs["number_cells"]
        )

        n_features = int(
            group.attrs["number_features"]
        )

        print(
            f"Loading Xenium matrix: "
            f"{n_cells:,} cells x "
            f"{n_features:,} features.",
            flush=True,
        )

        # Original IDs, in matrix column order.
        cell_ids = decode_xenium_cell_ids(
            group["cell_id"][:]
        )

        # Feature annotations from the matrix metadata.
        var = pd.DataFrame(
            {
                "feature_name": list(
                    group.attrs["feature_keys"]
                ),
                "feature_id": list(
                    group.attrs["feature_ids"]
                ),
                "feature_type": list(
                    group.attrs["feature_types"]
                ),
            }
        )

        if len(var) != n_features:
            raise ValueError(
                "Feature metadata length does not match "
                "the matrix dimensions."
            )

        if len(cell_ids) != n_cells:
            raise ValueError(
                "Cell ID count does not match "
                "the matrix dimensions."
            )

        # Xenium CSC representation:
        # rows = features, columns = cells.
        csc = group["csc"]

        data = csc["data"][:]
        indices = csc["indices"][:]
        indptr = csc["indptr"][:]

        if len(indptr) != n_cells + 1:
            raise ValueError(
                "Unexpected CSC matrix orientation."
            )

        if len(data) != len(indices):
            raise ValueError(
                "Sparse matrix data and indices "
                "have different lengths."
            )

        feature_by_cell = sparse.csc_matrix(
            (
                data,
                indices,
                indptr,
            ),
            shape=(
                n_features,
                n_cells,
            ),
        )

        # PINT convention: cells x features.
        X = feature_by_cell.transpose().tocsr()

        if X.shape != (
            n_cells,
            n_features,
        ):
            raise ValueError(
                "Unexpected final matrix dimensions."
            )

        print(
            f"Loaded sparse matrix: "
            f"{X.shape[0]:,} cells, "
            f"{X.shape[1]:,} features, "
            f"{X.nnz:,} stored entries.",
            flush=True,
        )

        return X, cell_ids, var

    finally:
        store.close()

def load_xenium_count_matrix(matrix_path):
    """
    Load Xenium's sparse count matrix.

    Returns
    -------
    X
        CSR matrix, cells x features.
    cell_ids
        Original Xenium string cell IDs.
    var
        Feature metadata.
    """
    root, store = open_xenium_zarr(
        matrix_path
    )

    try:
        group = root["cell_features"]

        n_cells = int(
            group.attrs["number_cells"]
        )

        n_features = int(
            group.attrs["number_features"]
        )

        print(
            f"Loading Xenium matrix: "
            f"{n_cells:,} cells x "
            f"{n_features:,} features.",
            flush=True,
        )

        # Original IDs, in matrix column order.
        cell_ids = decode_xenium_cell_ids(
            group["cell_id"][:]
        )

        # Feature annotations from the matrix metadata.
        var = pd.DataFrame(
            {
                "feature_name": list(
                    group.attrs["feature_keys"]
                ),
                "feature_id": list(
                    group.attrs["feature_ids"]
                ),
                "feature_type": list(
                    group.attrs["feature_types"]
                ),
            }
        )

        if len(var) != n_features:
            raise ValueError(
                "Feature metadata length does not match "
                "the matrix dimensions."
            )

        if len(cell_ids) != n_cells:
            raise ValueError(
                "Cell ID count does not match "
                "the matrix dimensions."
            )

        # Xenium CSC representation:
        # rows = features, columns = cells.
        csc = group["csc"]

        data = csc["data"][:]
        indices = csc["indices"][:]
        indptr = csc["indptr"][:]

        if len(indptr) != n_cells + 1:
            raise ValueError(
                "Unexpected CSC matrix orientation."
            )

        if len(data) != len(indices):
            raise ValueError(
                "Sparse matrix data and indices "
                "have different lengths."
            )

        feature_by_cell = sparse.csc_matrix(
            (
                data,
                indices,
                indptr,
            ),
            shape=(
                n_features,
                n_cells,
            ),
        )

        # PINT convention: cells x features.
        X = feature_by_cell.transpose().tocsr()

        if X.shape != (
            n_cells,
            n_features,
        ):
            raise ValueError(
                "Unexpected final matrix dimensions."
            )

        print(
            f"Loaded sparse matrix: "
            f"{X.shape[0]:,} cells, "
            f"{X.shape[1]:,} features, "
            f"{X.nnz:,} stored entries.",
            flush=True,
        )

        return X, cell_ids, var

    finally:
        store.close()


def load_xenium_cell_metadata(cells_path):
    """
    Load Xenium cell IDs and cell-level spatial metadata.

    Does not load masks or polygon boundaries.
    """
    root, store = open_xenium_zarr(
        cells_path
    )

    try:
        cell_ids = decode_xenium_cell_ids(
            root["cell_id"][:]
        )

        summary = root["cell_summary"][:]

        # Known documented columns.
        column_names = [
            "x_centroid",
            "y_centroid",
            "cell_area",
            "nucleus_x_centroid",
            "nucleus_y_centroid",
            "nucleus_area",
            "z_level",
            "nucleus_count",
        ]

        # Account for older/newer schemas rather
        # than silently assigning incorrect columns.
        if summary.shape[1] not in (7, 8):
            raise ValueError(
                "Unexpected Xenium cell_summary schema: "
                f"{summary.shape}. Inspect its attributes "
                "before assigning column names."
            )

        columns = column_names[
            :summary.shape[1]
        ]

        obs = pd.DataFrame(
            summary,
            columns=columns,
        )

        obs.insert(
            0,
            "cell_id",
            cell_ids,
        )

        if obs["cell_id"].duplicated().any():
            raise ValueError(
                "Duplicate Xenium cell IDs found "
                "in cells.zarr.zip."
            )

        print(
            f"Loaded cell metadata: "
            f"{len(obs):,} cells.",
            flush=True,
        )

        return obs

    finally:
        store.close()


def load_xenium_dataset(manifest_path):
    """
    Load Xenium counts and cell metadata.

    Returns a PINT-compatible sparse dataset while
    preserving the original Xenium cell identifiers.
    """
    manifest_path = Path(
        manifest_path
    ).resolve()

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    resources = manifest[
        "xenium_explorer_files"
    ]

    root = manifest_path.parent

    matrix_path = root / resources[
        "cell_features_zarr_filepath"
    ]

    cells_path = root / resources[
        "cells_zarr_filepath"
    ]

    # Load both resources independently.
    X, matrix_ids, var = (
        load_xenium_count_matrix(
            matrix_path
        )
    )

    obs = load_xenium_cell_metadata(
        cells_path
    )

    # Validate matrix identifiers.
    if len(set(matrix_ids)) != len(matrix_ids):
        raise ValueError(
            "Duplicate Xenium cell IDs found "
            "in the count matrix."
        )

    # Both resources must describe exactly
    # the same cells.
    if set(matrix_ids) != set(
        obs["cell_id"]
    ):
        raise ValueError(
            "The cell IDs in the count matrix "
            "do not match cells.zarr.zip."
        )

    # Reorder metadata to match matrix rows.
    obs = (
        obs
        .set_index(
            "cell_id",
            drop=False,
        )
        .loc[matrix_ids]
        .reset_index(drop=True)
    )

    # Existing PINT analyses use PINT_Cell_ID.
    # Preserve the original Xenium ID.
    obs.insert(
        0,
        "PINT_Cell_ID",
        obs["cell_id"],
    )

    # Keep matrix-feature position explicit.
    var.insert(
        0,
        "feature_index",
        np.arange(
            len(var),
            dtype=np.int32,
        ),
    )

    summary = {
        "source": "xenium",
        "n_cells": X.shape[0],
        "n_features": X.shape[1],
        "n_nonzero": X.nnz,
        "matrix_dtype": str(X.dtype),
        "manifest_path": str(manifest_path),
    }

    return {
        "X": X,
        "obs": obs,
        "var": var,
        "summary": summary,
        "manifest": manifest,
    }