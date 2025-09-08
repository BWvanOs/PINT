# main.py
from pathlib import Path
import subprocess, sys, threading, webbrowser, time

def run_viewer():
    viewer_path = Path(__file__).with_name("viewer.py")
    url = "http://127.0.0.1:8000"

    # Open the browser shortly after the server starts
    def _open():
        time.sleep(1.0)  # small delay so server is listening
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    # Launch the Shiny app on a fixed port so the URL matches
    cmd = [sys.executable, "-m", "shiny", "run", "--port", "8000", str(viewer_path)]
    # During development you can add "--reload" just after "run" if you like:
    # cmd = [sys.executable, "-m", "shiny", "run", "--reload", "--port", "8000", str(viewer_path)]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_viewer()
