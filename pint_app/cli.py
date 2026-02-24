import sys, subprocess
from pathlib import Path

def _root() -> Path:
    # repo root = parent of this package directory
    return Path(__file__).resolve().parents[1]

def main():
    """Launch the main app script PINT.py."""
    script = _root() / "PINT.py"
    if not script.exists():
        print(f"[pint] Could not find {script}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))

def viewer():
    """Run the (mounted) Shiny server (viewer at /, neighborhood at /neighborhood)."""
    # Ensure the ASGI entry point exists
    asgi_app = "pint_app.asgi:app"

    cmd = [sys.executable, "-m", "uvicorn", asgi_app, "--host", "127.0.0.1", "--port", "8000", *sys.argv[1:],]
    raise SystemExit(subprocess.call(cmd))

def analysis():
    """Run batch analysis via the installed module entry point."""
    cmd = [sys.executable, "-m", "pint_app.core.analysis", *sys.argv[1:]]
    try:
        raise SystemExit(subprocess.call(cmd))
    except FileNotFoundError as e:
        print(f"[pint-analysis] Python executable not found: {e}", file=sys.stderr)
        raise SystemExit(2)
