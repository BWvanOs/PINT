# pint_app/cli.py
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

def _root() -> Path:
    # repo root is the parent dir of the package directory
    return Path(__file__).resolve().parents[1]

def main():
    """Launch the main app script PINT.py"""
    script = _root() / "PINT.py"
    if not script.exists():
        print(f"[pint] Could not find {script}", file=sys.stderr)
        sys.exit(2)
    # forward any extra CLI args to PINT.py
    sys.exit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))

def viewer():
    """Run the Shiny viewer directly."""
    viewer_py = _root() / "viewer.py"
    if not viewer_py.exists():
        print(f"[pint-viewer] Could not find {viewer_py}", file=sys.stderr)
        sys.exit(2)
    sys.exit(subprocess.call([
        sys.executable, "-m", "shiny", "run", "--port", "8000", str(viewer_py), *sys.argv[1:]
    ]))

def analysis():
    """Run the batch analysis script directly."""
    analysis_py = _root() / "analysis.py"
    if not analysis_py.exists():
        print(f"[pint-analysis] Could not find {analysis_py}", file=sys.stderr)
        sys.exit(2)
    sys.exit(subprocess.call([sys.executable, str(analysis_py), *sys.argv[1:]]))
