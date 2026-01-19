"""Small list/selection helpers."""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def cycle_list(lst: Sequence[T], current: T | None, step: int) -> T | None:
    """Cycle through a sequence with wrap-around.

    If `current` is not present, it is treated as "before the first element".
    """
    if not lst:
        return None
    try:
        i = lst.index(current)  # type: ignore[arg-type]
    except ValueError:
        i = -1
    return lst[(i + step) % len(lst)]


def order_by_canonical(canonical: Sequence[str], current: Sequence[str]) -> list[str]:
    """Order `current` using `canonical` as a template; keep fallbacks.

    Returns:
      - intersection in canonical order if possible, else
      - original `current` list.
    """
    ordered = [ch for ch in canonical if ch in current]
    return ordered or list(current)