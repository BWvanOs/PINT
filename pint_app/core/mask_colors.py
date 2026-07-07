from __future__ import annotations

from typing import Iterable

import matplotlib as mpl
import matplotlib.colors as mcolors


import matplotlib as mpl
import matplotlib.colors as mcolors

DEFAULT_PALETTE = "viridis"
CUSTOM_PALETTE_NAMES = {"custom", "manual"}


def is_custom_palette(palette_name: str | None) -> bool:
    return str(palette_name or "").strip().lower() in CUSTOM_PALETTE_NAMES


def resolve_matplotlib_palette_name(palette_name: str | None) -> str:
    palette_name = str(palette_name or "").strip()

    if palette_name in mpl.colormaps:
        return palette_name

    return DEFAULT_PALETTE

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

    cluster_names = sorted(
        [normalize_cluster_name(x) for x in cluster_names],
        key=lambda x: x.lower(),
    )

    if len(cluster_names) == 0:
        return {}

    palette_name = resolve_matplotlib_palette_name(palette_name)
    cmap = mpl.colormaps[palette_name]

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