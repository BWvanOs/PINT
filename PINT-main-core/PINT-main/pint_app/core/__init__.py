"""Core (non-UI) utilities shared by the viewer and batch analysis.

This package holds *reusable* logic:
- parameter schema and coercion
- image processing and normalization primitives
- small utilities (selection helpers, formatting, dialogs)

The Shiny viewer and the batch analysis entry points should import from here
so that behaviour stays consistent across interactive and batch workflows.
"""

from __future__ import annotations

__all__ = [
    "processing",
    "params",
    "dialogs",
    "formatting",
    "selection",
    "winsor_quantiles",
]
