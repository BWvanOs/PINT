"""Deterministic six-ROI synthetic dataset for PINT neighborhood regression tests.

Masks are vertical stripes: 4-connected touching edges are exactly
(1,2), (2,3), ..., (n-1,n). Labels restart at 1 in every ROI.

This file is test data and an independent analytical oracle, NOT a
reimplementation of PINT's permutation code.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from tifffile import imwrite

# Each array index is the mask label minus 1.
# Two samples, three ROIs each. Deliberately reused mask labels.
ROI_TYPES = {
    "ROI001": ("1_1", "Inflamed", ("A", "B", "A", "B")),
    "ROI002": ("1_1", "Inflamed", ("A", "A", "A", "B", "B", "B")),
    "ROI003": ("1_1", "Inflamed", ("A", "A", "B", "B", "C")),
    "ROI004": ("1_2", "Control", ("A", "A", "B", "B")),
    "ROI005": ("1_2", "Control", ("A", "B", "B", "A")),
    "ROI006": ("1_2", "Control", ("A", "A", "A")),
}


def generate_dataset(folder):
    """Write six small TIFF masks and return PINT's pushed-dataset dict.

    Cell coordinates coincide exactly with mask centroids, so matching
    behavior is deterministic and independent of spatial tolerance.
    """
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    mask_records = []
    for roi, (sample, condition, types) in ROI_TYPES.items():
        # Every stripe is 3 pixels wide and 4 pixels tall.
        mask = np.repeat(
            np.arange(1, len(types) + 1, dtype=np.uint16)[None, :],
            repeats=4,
            axis=0,
        )
        mask = np.repeat(mask, 3, axis=1)
        mask_path = folder / f"{roi}.tif"
        imwrite(mask_path, mask)
        mask_records.append({
            "ROIName": roi, "SampleName": sample, "Condition": condition,
            "MaskFile": mask_path.name, "MaskPath": str(mask_path),
            "MaskExists": True,
        })
        for i, cell_type in enumerate(types):
            records.append({
                "ROIName": roi,
                "SampleName": sample,
                "Condition": condition,
                "Cluster": cell_type,
                "CellID": f"{roi}_cell_{i+1}",
                "X": float(3 * i + 1),
                "Y": 1.5,
                "MaskPath": str(mask_path),
            })
    return {
        "cell_table": pd.DataFrame(records),
        "mask_match_table": pd.DataFrame(mask_records),
        "column_map": {
            "mask_name_col": "ROIName",
            "sample_col": "SampleName",  # Intentional: biological sample != ROI
            "cluster_col": "Cluster",
            "condition_col": "Condition",
            "x_col": "X",
            "y_col": "Y",
        },
    }


def expected_edges(roi):
    """Oracle: stripe number i only touches stripe i+1."""
    n = len(ROI_TYPES[roi][2])
    return {(i, i + 1) for i in range(1, n)}


def observed_directed(roi, source, target):
    """Count directed contacts from the *specified* stripe sequence."""
    types = ROI_TYPES[roi][2]
    return sum(
        int(types[i] == source and types[i + 1] == target)
        + int(types[i + 1] == source and types[i] == target)
        for i in range(len(types) - 1)
    )


def exact_expected_directed(roi, source, target):
    """Exact null expectation for a chain under label permutation.

    Counts are preserved within the ROI. For n cells and m=n-1 edges:
      A != B: each edge contributes 2*nA*nB / [n*(n-1)] to A->B.
      A == B: each edge contributes 2*nA*(nA-1) / [n*(n-1)] to A->A.
    These formulas follow from sampling endpoint types without replacement.
    """
    types = ROI_TYPES[roi][2]
    n = len(types)
    n_a = types.count(source)
    n_b = types.count(target)
    if n < 2:
        return 0.0
    if source == target:
        return 2 * n_a * (n_a - 1) / n
    return 2 * n_a * n_b / n


def exact_sample_score(sample, source, target):
    """Source-cell weighted ground truth (independent of PINT outputs).

    An ROI contributes only if BOTH types occur in it. This is the
    explicit missingness policy adopted for the tests.
    """
    total_excess = 0.0
    source_cells = 0
    for roi, (sample_name, _, types) in ROI_TYPES.items():
        if sample_name != sample or source not in types or target not in types:
            continue
        total_excess += (
            observed_directed(roi, source, target)
            - exact_expected_directed(roi, source, target)
        )
        source_cells += types.count(source)
    return total_excess / source_cells if source_cells else float("nan")
