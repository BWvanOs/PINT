from __future__ import annotations

from typing import Iterable

import matplotlib.cm as cm
import matplotlib.colors as mcolors


VIRIDIS_PALETTES = {
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
}


def normalize_cluster_name(value: object) -> str:
    if value is None:
        return "Unassigned"

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "na"}:
        return "Unassigned"

    return text


def get_sorted_cluster_names(values: Iterable[object]) -> list[str]:
    names = {
        normalize_cluster_name(v)
        for v in values
    }

    return sorted(names, key=lambda x: x.lower())


def make_palette_color_map(
    cluster_names: list[str],
    palette_name: str,
) -> dict[str, str]:
    """
    Create a stable cluster_name -> hex color mapping.

    Important:
    - Uses sorted cluster names.
    - Samples evenly across the selected matplotlib colormap.
    - Does not depend on which mask/image is currently shown.
    """
    cluster_names = sorted(
        [normalize_cluster_name(x) for x in cluster_names],
        key=lambda x: x.lower(),
    )

    if len(cluster_names) == 0:
        return {}

    palette_name = palette_name or "viridis"

    if palette_name not in VIRIDIS_PALETTES:
        palette_name = "viridis"

    cmap = cm.get_cmap(palette_name)

    if len(cluster_names) == 1:
        positions = [0.5]
    else:
        positions = [
            i / (len(cluster_names) - 1)
            for i in range(len(cluster_names))
        ]

    return {
        cluster_name: mcolors.to_hex(cmap(pos), keep_alpha=False)
        for cluster_name, pos in zip(cluster_names, positions)
    }


def make_custom_color_map(
    cluster_names: list[str],
    input,
    *,
    default_color: str = "#808080",
) -> dict[str, str]:
    cluster_names = sorted(
        [normalize_cluster_name(x) for x in cluster_names],
        key=lambda x: x.lower(),
    )

    color_map: dict[str, str] = {}

    for i, cluster_name in enumerate(cluster_names):
        input_id = f"mask_cluster_color_{i}"

        try:
            value = getattr(input, input_id)()
        except Exception:
            value = default_color

        value = str(value or "").strip()

        if not value.startswith("#") or len(value) != 7:
            value = default_color

        color_map[cluster_name] = value

    return color_map