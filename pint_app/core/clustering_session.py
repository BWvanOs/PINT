from __future__ import annotations

import pickle
from pathlib import Path


PINT_CLUSTERING_SESSION_FORMAT = "PINT clustering session"
PINT_CLUSTERING_SESSION_VERSION = 1


def save_clustering_session(
    path: str | Path,
    state: dict,
) -> None:
    """
    Save the current PINT clustering state.

    This is an internal PINT session format. Only load session files
    created by PINT or obtained from a trusted source.
    """
    path = Path(path)

    payload = {
        "format": PINT_CLUSTERING_SESSION_FORMAT,
        "version": PINT_CLUSTERING_SESSION_VERSION,
        "state": state,
    }

    with path.open("wb") as handle:
        pickle.dump(
            payload,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_clustering_session(
    path: str | Path,
) -> dict:
    """
    Load and validate a PINT clustering session.
    """
    path = Path(path)

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            "The selected file is not a valid PINT clustering session."
        )

    if payload.get("format") != PINT_CLUSTERING_SESSION_FORMAT:
        raise ValueError(
            "The selected file is not a PINT clustering session."
        )

    version = payload.get("version")

    if version != PINT_CLUSTERING_SESSION_VERSION:
        raise ValueError(
            f"Unsupported PINT clustering session version: {version!r}."
        )

    state = payload.get("state")

    if not isinstance(state, dict):
        raise ValueError(
            "The PINT clustering session does not contain valid state data."
        )

    return state