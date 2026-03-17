"""Cross-platform file/folder picker dialogs.

These helpers are intentionally UI-framework agnostic so they can be reused
from both the Shiny viewer and any future CLI tooling.

Preference order:
1) tkinter (Windows/macOS and many Linux installs)
2) zenity (common on Linux desktops), if unavailable use tkinter

All functions return a "faillback failed" if failed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

##Check if user is running Linux
def _is_linux() -> bool:
    return sys.platform.startswith("linux")

##Check if zenity is available
def _zenity_available() -> bool:
    return shutil.which("zenity") is not None


def pick_open_csv_dialog(title: str = "Select parameter CSV", initialdir: str | None = None) -> str:
    """Open a file selection dialog for a CSV file."""
    #Linux prefer zenity (GNOME-style), tkinter is unusable on hihg DPI screens
    if _is_linux() and _zenity_available():
        try:
            cmd = ["zenity", "--file-selection", "--title", title]
            if initialdir:
                cmd += ["--filename", os.path.join(initialdir, "")]
            #zenity filters are a bit particular; this works well in practice
            cmd += ["--file-filter=CSV files | *.csv", "--file-filter=All files | *"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.stdout.strip() if res.returncode == 0 else ""
        except Exception:
            pass

    #Fallback is tkinter, need to find an alternative because it doesn't work well
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        try:
            dpi = root.winfo_fpixels("1i")          
            scale = max(1.0, float(dpi) / 96.0)     
            root.tk.call("tk", "scaling", scale)    
        except Exception:
            pass

        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass

        path = filedialog.askopenfilename(
            title=title,
            initialdir=initialdir or os.getcwd(),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return path or ""
    except Exception:
        print("[file dialog] tkinter fallback failed", file=sys.stderr)
        return ""


def pick_save_csv_dialog(
    title: str = "Save parameters CSV",
    initialdir: str | None = None,
    initialfile: str = "parameter_table.csv",
) -> str:
    """Open a save-as dialog for a CSV file."""
    #Linux: prefer zenity (GNOME-style) selector
    if _is_linux() and _zenity_available():
        try:
            #zenity save mode uses --save and can prefill filename in the selection
            start = initialdir or os.getcwd()
            suggested = os.path.join(start, initialfile)
            cmd = [
                "zenity",
                "--file-selection",
                "--save",
                "--confirm-overwrite",
                "--title", title,
                "--filename", suggested,
                "--file-filter=CSV files | *.csv",
                "--file-filter=All files | *",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            out = res.stdout.strip() if res.returncode == 0 else ""
            if out and not out.lower().endswith(".csv"):
                out += ".csv"
            return out
        except Exception:
            pass

    #Fallback, again tkinter if zenity is unavailable
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        ##Try to scale tkinter a but to make is usable.
        try:
            dpi = root.winfo_fpixels("1i")          
            scale = max(1.0, float(dpi) / 96.0)     
            root.tk.call("tk", "scaling", scale)    
        except Exception:
            pass

        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass

        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=initialdir or os.getcwd(),
            initialfile=initialfile,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return path or ""
    except Exception:
        print("[file dialog] tkinter fallback failed", file=sys.stderr)
        return ""


def pick_folder_dialog(title: str = "Select folder", initialdir: str | None = None) -> str:
    """Open a folder selection dialog."""
    # Linux: prefer zenity (GNOME-style)
    if _is_linux() and _zenity_available():
        try:
            cmd = ["zenity", "--file-selection", "--directory", "--title", title]
            if initialdir:
                cmd += ["--filename", os.path.join(initialdir, "")]
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.stdout.strip() if res.returncode == 0 else ""
        except Exception:
            pass

    # Fallback: tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        try:
            dpi = root.winfo_fpixels("1i")          
            scale = max(1.0, float(dpi) / 96.0)     
            root.tk.call("tk", "scaling", scale)    
        except Exception:
            pass

        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass

        path = filedialog.askdirectory(title=title, initialdir=initialdir or os.getcwd())
        root.destroy()
        return path or ""
    except Exception:
        print("[file dialog] tkinter fallback failed", file=sys.stderr)
        return ""

def pick_save_png_dialog(
    title: str = "Save composite PNG",
    initialdir: str | None = None,
    initialfile: str = "composite.png",
) -> str:
    """Open a save-as dialog for a PNG file."""
    # Linux: prefer zenity (GNOME-style) selector
    if _is_linux() and _zenity_available():
        try:
            start = initialdir or os.getcwd()
            suggested = os.path.join(start, initialfile)
            cmd = [
                "zenity",
                "--file-selection",
                "--save",
                "--confirm-overwrite",
                "--title", title,
                "--filename", suggested,
                "--file-filter=PNG files | *.png",
                "--file-filter=All files | *",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            out = res.stdout.strip() if res.returncode == 0 else ""
            if out and not out.lower().endswith(".png"):
                out += ".png"
            return out
        except Exception:
            pass

    # Fallback: tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        try:
            dpi = root.winfo_fpixels("1i")
            scale = max(1.0, float(dpi) / 96.0)
            root.tk.call("tk", "scaling", scale)
        except Exception:
            pass

        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass

        path = filedialog.asksaveasfilename(
            title=title,
            initialdir=initialdir or os.getcwd(),
            initialfile=initialfile,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        root.destroy()
        return path or ""
    except Exception:
        print("[file dialog] tkinter fallback failed", file=sys.stderr)
        return ""

    root.destroy()
    return path or ""