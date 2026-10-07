from pathlib import Path
import subprocess, sys, threading, webbrowser, time

def run_viewer():
    url = "http://127.0.0.1:8000/"

    def _open():
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    cmd = [
    sys.executable, "-m", "uvicorn", "pint_app.asgi:app",
    "--host", "127.0.0.1", "--port", "8000",
    "--ws-ping-interval", "120",
    "--ws-ping-timeout", "120",
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_viewer()
