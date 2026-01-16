"""Cross-platform file/folder picker dialogs.

These helpers are intentionally UI-framework agnostic so they can be reused
from both the Shiny viewer and any future CLI tooling.

Preference order:
1) tkinter (Windows/macOS and many Linux installs)
2) zenity (common on Linux desktops)

All functions return an empty string on cancel/failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def pick_open_csv_dialog(title: str = "Select parameter CSV", initialdir: str | None = None) -> str:
    """Open a file selection dialog for a CSV file."""
    # Prefer tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
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
        pass

    # Linux fallback: zenity
    try:
        if shutil.which("zenity"):
            res = subprocess.run(
                ["zenity", "--file-selection", "--title", title, "--file-filter=*.csv"],
                capture_output=True,
                text=True,
            )
            return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        pass

    return ""


def pick_save_csv_dialog(
    title: str = "Save parameters CSV",
    initialdir: str | None = None,
    initialfile: str = "parameter_table.csv",
) -> str:
    """Open a save-as dialog for a CSV file."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
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
        return ""


def pick_folder_dialog(title: str = "Select folder") -> str:
    """Open a folder selection dialog."""
    # Prefer tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path or ""
    except Exception:
        pass

    # Linux fallback: zenity
    try:
        if shutil.which("zenity"):
            res = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title", title],
                capture_output=True,
                text=True,
            )
            return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        pass

    return ""