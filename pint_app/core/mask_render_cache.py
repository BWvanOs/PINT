from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Hashable


def file_signature(path: str | Path) -> tuple[str, int, int]:
    """
    Return a stable file signature for cache keys.

    Includes:
    - resolved absolute path
    - modification time in nanoseconds
    - file size in bytes

    This prevents stale cache hits if a file is overwritten at the same path.
    """
    p = Path(path)
    stat = p.stat()

    return (
        str(p.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def make_hashable(value: Any) -> Hashable:
    """
    Convert common values into hashable cache-key-safe objects.

    Useful when some settings are lists, dicts, Paths, or NumPy/Pandas scalar-like
    values.
    """
    if isinstance(value, Path):
        return str(value.resolve())

    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(k), make_hashable(v))
                for k, v in value.items()
            )
        )

    if isinstance(value, (list, tuple, set)):
        return tuple(make_hashable(v) for v in value)

    try:
        hash(value)
        return value
    except TypeError:
        return str(value)


def make_mask_render_cache_key(
    *,
    mask_path: str | Path,
    data_version: int = 0,
    render_settings: dict[str, Any] | None = None,
) -> tuple[Hashable, ...]:
    """
    Build a cache key for a rendered mask image.

    Do not include browser size, browser zoom, or Shiny output dimensions here.
    Only include values that actually change the rendered image content.
    """
    settings = render_settings or {}

    return (
        file_signature(mask_path),
        int(data_version),
        make_hashable(settings),
    )


class MaskRenderCache:
    """
    Small LRU cache for rendered mask images.

    Intended for session-level use in the Shiny server. This avoids recomputing
    rendered masks when users switch samples back and forth or when the browser
    redraws the output.
    """

    def __init__(self, max_items: int = 25):
        self.max_items = int(max_items)
        self._cache: OrderedDict[Hashable, Any] = OrderedDict()

    def get(self, key: Hashable) -> Any | None:
        if key not in self._cache:
            return None

        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def set(self, key: Hashable, value: Any) -> None:
        if key in self._cache:
            self._cache.pop(key)

        self._cache[key] = value

        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def info(self) -> dict[str, int]:
        return {
            "items": len(self._cache),
            "max_items": self.max_items,
        }