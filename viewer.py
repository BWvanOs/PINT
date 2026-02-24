"""Backward-compatible shim.

Historically the Shiny viewer was at repo root called `viewer.py`.
It has moved to `pint_app.apps.viewer` in a restructuring of the app. .

This shim keeps old imports working, will be removed in the future!:
  - `import viewer; viewer.app`
  - `python viewer.py`
"""

from pint_app.apps.viewer import app  # re-export

if __name__ == "__main__":
    # Running this file directly isn't the recommended path anymore,
    # but keep it functional for convenience.
    import subprocess, sys
    subprocess.run([
        sys.executable, "-m", "uvicorn", "pint_app.asgi:app",
        "--host", "127.0.0.1", "--port", "8000",
    ], check=True)
