from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, ui, render, reactive
from load_tiffs import load_tiffs_raw
from scipy.ndimage import percentile_filter, convolve, grey_opening, uniform_filter
import os, sys, subprocess
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*Tight layout not applied.*")


app_ui = ui.page_sidebar(
    # ---- Right collapsible sidebar (positional arg #1) ----
    ui.sidebar(
        ui.h4("Current Parameters"),
        ui.tags.div(ui.output_table("param_table"), class_="param-table-wrap"),
        open="closed",          # collapsed by default
        id="sidebar",
        class_="sidebar-col",
        width="850px",          # tweak as needed
    ),

    # ---- HEAD/CSS (positional arg #2) ----
    ui.head_content(
        ui.tags.style("""
            :root{
                /* You can tweak these two and nothing else */
                --controls-h: 450px;     /* total height of the top area (toolbar + panels) */
                --controls-top-h: 170px; /* height of the toolbar row */
            }

            /* Page skeleton: fixed top area + growing viewer */
            .flex-col { display:flex; flex-direction:column; height:100vh; }

            .controls-fixed {
                flex: 0 0 var(--controls-h);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .controls-top  { flex: 0 0 var(--controls-top-h); overflow: visible; }

            /* No CSS layout for the second row (panels) —
                sizes/spacing come only from your Python row/column props. */

            /* Viewer grows to fill remaining space */
            .viewer-fill { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }

            /* Sidebar & parameter table (unchanged, lightweight) */
            .sidebar-col { display:flex; flex-direction:column; height:100%; }
            .param-table-wrap table {
                font-size: 12px;
                width: 100% !important;
                table-layout: auto;
                border-collapse: collapse;
            }
            .param-table-wrap td, .param-table-wrap th {
                padding: 2px 4px;
                white-space: nowrap;
                text-overflow: ellipsis;
                overflow: hidden;
                text-align: left;
            }
            .param-table-wrap th { font-weight: 600; text-align: left; }

            /* Make sure the sidebar overlays other content when open */
            .bslib-sidebar-layout > .bslib-sidebar { z-index: 1050; }
            .bslib-sidebar-layout .bslib-sidebar-toggle { z-index: 1060; }

            /* Tighten the toolbar only (kills the “ghost row”) */
            .controls-top .shiny-input-container { margin-bottom: 0 !important; }
            .controls-top .row { --bs-gutter-y: 0; margin-bottom: 0; }
            .controls-top .col { padding-left: 0; padding-right: 0; }
            .controls-fixed hr { margin: 6px 0; }"""
        )
    ), 

    # ---- MAIN CONTENT (positional arg #3) ----
    ui.row(
        ui.column(
            12,
            ui.tags.div(
                # ===== Fixed-height controls (toolbar + panels) =====
                ui.tags.div(
                    # --- Top control bar: path + load + sample + channel ---
                    # --- Top control bar: path + load + sample + channel ---
                    ui.row(
                        ui.column(5, ui.input_text("path", "Folder path", value="", width="100%")),
                        ui.column(1, ui.input_action_button("load", "Load images", class_="w-100")),
                        ui.column(1),  # spacer

                        # Sample + Channel stacked
                        ui.column(
                            2,
                            ui.row(
                                ui.column(8, ui.input_select("sample", "Sample", choices=[], selected=None, width="100%")),
                                ui.column(2, ui.input_action_button("prev_sample", "←", class_="btn-sm")),
                                ui.column(2, ui.input_action_button("next_sample", "→", class_="btn-sm")),
                                class_="align-items-center gy-0",
                            ),
                            ui.row(
                                ui.column(8, ui.input_select("channel", "Channel", choices=[], selected=None, width="100%")),
                                ui.column(2, ui.input_action_button("prev_channel", "←", class_="btn-sm")),
                                ui.column(2, ui.input_action_button("next_channel", "→", class_="btn-sm")),
                                class_="align-items-center gy-0",
                            ),
                        ),

                        ui.column(1),  # spacer

                        # Export / Import / Process
                        ui.column(
                            1,
                            ui.row(
                                ui.div(
                                    ui.input_action_button("export_params", "Export CSV", class_="btn btn-secondary w-100"),
                                    class_="mb-1",
                                ),
                            ),
                            ui.row(
                                ui.div(
                                    ui.input_action_button("import_params", "Import CSV", class_="btn btn-secondary w-100"),
                                    class_="mb-1",
                                ),
                            ),
                            ui.row(
                                ui.div(
                                    ui.input_action_button(
                                        "perform_analysis",
                                        "Process Images",
                                        class_="btn btn-primary text-white w-100 h-100",
                                    ),
                                    class_="d-flex h-100 align-items-stretch",
                                ),
                            ),
                        ),

                        ui.column(1),  # spacer

                        # IMPORTANT: this is part of the SAME ui.row(...) call; note the comma above.
                        class_="controls-top align-items-center gy-0",
                    ),

                    ui.hr(),

                    ui.row(
                        # --- PANEL 1: Winsorization ---
                        ui.column(
                            3,
                            ui.card(
                                ui.card_header("Winsorization"),
                                ui.row(
                                    ui.column(6, ui.input_slider("winsor_low", "Lower quantile (0–1)", min=0.0, max=1.0, value=0.00, step=0.01)),
                                    ui.column(6, ui.input_slider("winsor_high", "Upper quantile (0–1)", min=0.0, max=1.0, value=0.99, step=0.01)),
                                ),
                                ui.row(
                                    ui.column(6, ui.input_checkbox("doWinsor", "doWinsorize", value=True)),
                                    ui.column(6, ui.input_action_button("apply_one", "Update channel", class_="btn btn-primary w-100")),
                                ),
                            ),
                        ),

                        # --- PANEL 2: Global Threshold ---
                        ui.column(
                            3,
                            ui.card(
                                ui.card_header("Global Threshold"),
                                ui.row(ui.column(12, ui.input_slider("threshold_val", "Threshold (0-1)", min=0.0, max=1.0, value=0.1, step=0.01))),
                                ui.row(
                                    ui.column(6, ui.input_checkbox("doThreshold", "Apply threshold", value=True)),
                                    ui.column(6, ui.input_action_button("apply_threshold", "Update channel", class_="btn btn-primary w-100")),
                                ),
                            ),
                        ),

                        # --- PANEL 3: Noise Removal ---
                        ui.column(
                            3,
                            ui.card(
                                ui.card_header("Sliding Window Noise Removal"),
                                ui.row(
                                    ui.column(6, ui.input_slider("noise_strength", "Denoise strength (0–1)", min=0.0, max=1.0, value=0.1, step=0.01)),
                                    ui.column(6, ui.input_numeric("window_size", "Window size (odd)", value=3, min=1, step=2)),
                                ),
                                ui.row(
                                    ui.column(6, ui.input_checkbox("doNoise", "Apply noise removal", value=True)),
                                    ui.column(6, ui.input_action_button("apply_noise", "Update channel", class_="btn btn-primary w-100")),
                                    ui.row(
                                        ui.column(
                                            12,
                                            ui.output_ui("noise_tooltip"),   # dynamic helper + tooltip
                                        ),
                                    ),
                                ),
                            ),
                        ),

                        # --- PANEL 4: Normalization & Transform ---
                        ui.column(
                            3,
                            ui.card(
                                ui.card_header("Normalization and Transformation"),
                                ui.row(
                                    # LEFT: normalization controls
                                    ui.column(
                                        6,
                                        ui.tags.div(
                                            ui.input_checkbox("doNorm", "Normalize the channel?", value=True),
                                            class_="d-flex align-items-center mb-2",
                                        ),
                                        ui.input_radio_buttons(
                                            "norm_scope",
                                            "Normalize using",
                                            choices={
                                                "page":   "Per page (each sample seperate)",
                                                "global": "Global min/max across channel",
                                            },
                                            selected="page",
                                            inline=True,
                                        ),
                                        ui.output_ui("norm_scope_hint"),
                                    ),

                                    # RIGHT: arcsinh controls stacked, then Apply button
                                    ui.column(
                                        6,
                                        ui.tags.div(
                                            ui.input_checkbox("doAsinh", "Arcsinh transform data", value=False),
                                            class_="d-flex align-items-center mb-2",
                                        ),
                                        ui.input_select(
                                            "asinh_cofactor",
                                            "Cofactor",
                                            choices=[str(i) for i in range(2, 11)],
                                            selected="5",
                                            width="100%",
                                        ),
                                        ui.input_action_button(
                                            "apply_norm",
                                            "Apply norm/transform",
                                            class_="btn btn-primary w-100 mt-2",
                                        ),
                                    ),
                                    class_="align-items-start",
                                ),
                            ),
                        ),
                    ),
                    class_="controls-fixed",
                ),

                # ===== Plot area =====
                ui.tags.div(
                    ui.output_plot("img_viewer", fill=True, height="100%"),
                    class_="viewer-fill",
                ),
                class_="flex-col",
            ),
        ),
    ),

    # ---- KEYWORD ARGS (must be last) ----
    position="right",
)



# --------------- Server ---------------
from scipy.ndimage import median_filter

def server(input, output, session):
    # ---------- state ----------
    images = reactive.Value({})                  # {sample: np.ndarray[C,Y,X]}
    channels = reactive.Value({})                # {sample: [channel names]}
    canonical_channels = reactive.Value([])      # list[str], from first image only
    params_df = reactive.Value(
        pd.DataFrame(columns=[
            "Channel", "DoWinsor", "Low", "High",
            "DoThr", "ThrVal",
            "Noise", "NStr", "NPrctl", "WinSz",
            "DoNorm",
            "DoAsinh", "Cofac",
            "NormScope",                # <-- NEW
        ])
    )

    loading = reactive.Value(False)
    setting_selects = reactive.Value(False)
    syncing_controls = reactive.Value(False)
    data_loaded = reactive.Value(False)
    last_loaded_folder = reactive.Value("") 
    
    # =============> helper funtions go here! <============
    def _pick_open_csv_dialog(title="Select parameter CSV", initialdir=None) -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
            import os as _os
            root = tk.Tk(); root.withdraw()
            try: root.wm_attributes("-topmost", 1)
            except Exception: pass
            path = filedialog.askopenfilename(
                title=title,
                initialdir=initialdir or _os.getcwd(),
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            root.destroy()
            return path or ""
        except Exception:
            try:
                import shutil, subprocess
                if shutil.which("zenity"):
                    res = subprocess.run(
                        ["zenity", "--file-selection", "--title", title, "--file-filter=*.csv"],
                        capture_output=True, text=True
                    )
                    return res.stdout.strip() if res.returncode == 0 else ""
            except Exception:
                pass
            return ""

    def _pick_save_csv_dialog(title="Save parameters CSV", initialdir=None, initialfile="parameter_table.csv") -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
            import os as _os
            root = tk.Tk(); root.withdraw()
            try: root.wm_attributes("-topmost", 1)
            except Exception: pass
            path = filedialog.asksaveasfilename(
                title=title,
                initialdir=initialdir or _os.getcwd(),
                initialfile=initialfile,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            root.destroy()
            return path or ""
        except Exception:
            return ""

    def pick_folder_dialog(title="Select folder with OME-TIFFs") -> str:
        # Prefer Tk (works on Win/macOS/Linux if tkinter is installed)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            try:
                # bring dialog to front on some WMs
                root.wm_attributes("-topmost", 1)
            except Exception:
                pass
            path = filedialog.askdirectory(title=title)
            root.destroy()
            return path or ""
        except Exception:
            # Linux fallback if Tk isn't available
            if shutil.which("zenity"):
                res = subprocess.run(
                    ["zenity", "--file-selection", "--directory", "--title", title],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    return res.stdout.strip()
            return ""

    def _ensure_param_columns():
        required = ["Channel","DoWinsor","Low","High",
                    "DoThr","ThrVal",
                    "Noise","NStr", "NPrctl", "WinSz",
                    "DoNorm"]  # if you're adding Normalize
        defaults = {
            "Channel": "",
            "DoWinsor": False, "Low": 0.0, "High": 1.0,
            "DoThr": False, "ThrVal": 0.0,
            "Noise": False, "NStr": 1.0, "NPrctl": 0.995, "WinSz": 3,
            "DoNorm": True,
        }
        df = params_df.get()
        if df.empty:
            params_df.set(pd.DataFrame(columns=required))
            return
        for col in required:
            if col not in df.columns:
                df[col] = defaults[col]
        # enforce column order
        params_df.set(df[required].reset_index(drop=True))

    def _prefill_params(first_chlist: list[str]) -> None:
        # read the current UI slider once and map to a percentile
        s = float(input.noise_strength())
        p = _strength_to_percentile(s)

        rows = [{
            "Channel": ch,
            "DoWinsor": bool(input.doWinsor()),
            "Low": float(input.winsor_low()),
            "High": float(input.winsor_high()),
            "DoThr": bool(input.doThreshold()),
            "ThrVal": float(input.threshold_val()),
            "Noise": bool(input.doNoise()),
            "NStr": s,
            "NPrctl": p,
            "WinSz": int(input.window_size()),
            "DoNorm": bool(input.doNorm()),
            "DoAsinh": bool(input.doAsinh()),
            "Cofac": int(float(input.asinh_cofactor() or 5)),
            "NormScope": (input.norm_scope() or "page"),   # <-- NEW
        } for ch in first_chlist]

        df = pd.DataFrame(rows)
        params_df.set(df.reindex(columns=params_df.get().columns).reset_index(drop=True))


    def _sync_controls_from_table(channel: str) -> None:
        if not channel:
            return
        df = params_df.get()
        if df.empty:
            return
        m = (df["Channel"] == channel)
        if not m.any():
            return
        row = df.loc[m].iloc[0]
        syncing_controls.set(True)
        try:
            session.send_input_message("doWinsor", {"value": bool(row.get("DoWinsor", False))})
            session.send_input_message("winsor_low", {"value": float(row.get("Low", 0.0))})
            session.send_input_message("winsor_high", {"value": float(row.get("High", 1.0))})

            session.send_input_message("doThreshold", {"value": bool(row.get("DoThr", False))})
            session.send_input_message("threshold_val", {"value": float(row.get("ThrVal", 0.0))})

            session.send_input_message("doNoise", {"value": bool(row.get("Noise", False))})
            winsz = int(row.get("WinSz", int(input.window_size())))
            session.send_input_message("noise_strength", {"value": float(row.get("NStr", 1.0))})
            session.send_input_message("window_size", {"value": winsz})

            # Normalize checkbox (if present)
            session.send_input_message("doNorm", {"value": bool(row.get("DoNorm", True))})
            session.send_input_message("norm_scope", {"value": str(row.get("NormScope", "page"))})

            ##arcsinh controls
            session.send_input_message("doAsinh", {"value": bool(row.get("DoAsinh", False))})
            session.send_input_message("asinh_cofactor", {"value": str(int(row.get("Cofac", 5)))})
        finally:
            syncing_controls.set(False)
    
    def _cycle(lst, current, step):
        if not lst:
            return None
        try:
            i = lst.index(current)
        except ValueError:
            i = -1
        return lst[(i + step) % len(lst)]


    def _apply_winsor(cur: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
        q_low, q_high = np.quantile(cur, [lo_q, hi_q])
        return np.clip(cur, q_low, q_high)
    
    def _strength_to_percentile(s: float, eps: float = 0.005) -> float:
        # s∈[0,1]  ->  p = 1 - eps - s*(1-eps)  (right = stronger = lower percentile)
        s = float(np.clip(s, 0.0, 1.0))
        return 1.0 - eps - s * (1.0 - eps)


    def _apply_speckle_suppress(
        img: np.ndarray,
        size: int,
        perc: float,
        neighbor_limit: int = 2,    # ≤ 2 bright adjacent neighbors ⇒ remove
    ) -> np.ndarray:
        """
        Old method: center is 'bright' if > local percentile in a (size×size) window.
        Extra check: remove only if AT MOST `neighbor_limit` of the 8 adjacent pixels
        (3×3 neighborhood, center excluded) are also bright.
        """
        if size % 2 == 0:
            size += 1

        # 1) local percentile threshold on the raw image
        thr = percentile_filter(img, percentile=perc * 100.0, size=size)
        bright = (img > thr) & (img > 0)

        # 2) count bright pixels in the immediate 8-neighborhood (3×3, center=0)
        k = np.ones((3, 3), dtype=np.float32)
        k[1, 1] = 0.0
        neighbor_count = convolve(bright.astype(np.float32), k, mode="reflect")

        # 3) remove if bright and has ≤ neighbor_limit bright neighbors
        remove = bright & (neighbor_count <= float(neighbor_limit))

        out = img.copy()
        out[remove] = 0.0
        return out

    # Cache for global min/max per channel (invalidated on load)
    _global_minmax_cache = reactive.Value({})   # {channel_name: (gmin, gmax)}

    def _invalidate_global_cache():
        _global_minmax_cache.set({})

    def _global_minmax_for_channel(images_dict: dict, channels_dict: dict, channel_name: str):
        """Return (gmin, gmax) across all samples for the given channel name.
        Uses & updates a reactive cache. Skips samples that lack this channel.
        """
        cache = _global_minmax_cache.get()
        if channel_name in cache:
            return cache[channel_name]

        gmin = np.inf
        gmax = -np.inf
        found_any = False

        for sample, arr in images_dict.items():
            chlist = channels_dict.get(sample, [])
            if channel_name not in chlist:
                continue
            idx = chlist.index(channel_name)
            ch = arr[idx]  # raw 2D page
            # robust against NaNs
            mn = float(np.nanmin(ch))
            mx = float(np.nanmax(ch))
            if np.isfinite(mn) and np.isfinite(mx):
                gmin = min(gmin, mn)
                gmax = max(gmax, mx)
                found_any = True

        if not found_any or not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
            # store sentinel to avoid recomputation loops
            cache[channel_name] = None
            _global_minmax_cache.set(cache)
            return None

        cache[channel_name] = (gmin, gmax)
        _global_minmax_cache.set(cache)
        return (gmin, gmax)

    # ---------- load images ----------
    @reactive.Effect
    @reactive.event(input.load)
    def _do_load():
        if loading.get():
            return
        loading.set(True)
        try:
            # Try the text box first; if empty, open the dialog.
            folder = (input.path() or "").strip()
            if not folder:
                # If you added pick_folder_dialog(), use it here:
                try:
                    folder = pick_folder_dialog()
                except Exception:
                    folder = ""
            if not folder:
                print("🛑 Load canceled (no folder selected).")
                return

            # Reflect the path in the UI text box
            session.send_input_message("path", {"value": folder})
            print(">>> Load triggered with folder:", folder)

            imgs, chs = load_tiffs_raw(folder)
            if not imgs:
                print("⚠️ No images found in selected folder.")
                return

            images.set(imgs)
            channels.set(chs)

            _invalidate_global_cache()

            samples = list(imgs.keys())
            first_sample = samples[0]
            first_chlist = chs[first_sample]
            first_channel = first_chlist[0] if first_chlist else None

            # set canonical + prefill table
            canonical_channels.set(list(first_chlist))
            _prefill_params(first_chlist)

            # update selects under guard
            setting_selects.set(True)
            try:
                ui.update_select("sample",  choices=samples,      selected=first_sample,  session=session)
                if first_channel:
                    ui.update_select("channel", choices=first_chlist, selected=first_channel, session=session)
            finally:
                setting_selects.set(False)

            if first_channel:
                _sync_controls_from_table(first_channel)

            # ✅ mark as loaded and remember folder
            data_loaded.set(True)
            last_loaded_folder.set(folder)

            print(">>> Post-load selected:", first_sample, first_channel)

        finally:
            loading.set(False)

    # ---------- react to sample change ----------
    @reactive.Effect
    @reactive.event(input.next_sample)
    def _next_sample():
        if loading.get() or not images.get():
            return
        samples = list(images.get().keys())
        cur = input.sample() or (samples[0] if samples else None)
        nxt = _cycle(samples, cur, +1)
        if nxt:
            # Don’t set setting_selects here; we want _on_sample_change to run
            ui.update_select("sample", choices=samples, selected=nxt, session=session)

    @reactive.Effect
    @reactive.event(input.prev_sample)
    def _prev_sample():
        if loading.get() or not images.get():
            return
        samples = list(images.get().keys())
        cur = input.sample() or (samples[0] if samples else None)
        prv = _cycle(samples, cur, -1)
        if prv:
            ui.update_select("sample", choices=samples, selected=prv, session=session)


    @reactive.Effect
    @reactive.event(input.sample)
    def _on_sample_change():
        if loading.get() or setting_selects.get():
            return
        s = input.sample()
        if not s:
            return
        chlist_current = channels.get().get(s, [])
        if not chlist_current:
            return
        canon = canonical_channels.get()
        ordered = [ch for ch in canon if ch in chlist_current] or chlist_current
        sel = input.channel()
        if sel not in ordered:
            sel = ordered[0]
        setting_selects.set(True)
        try:
            ui.update_select("channel", choices=ordered, selected=sel, session=session)
        finally:
            setting_selects.set(False)
        _sync_controls_from_table(sel)

    @reactive.Effect
    @reactive.event(input.next_channel)
    def _next_channel():
        if loading.get():
            return
        s = input.sample()
        if not s:
            return
        chlist_current = channels.get().get(s, [])
        if not chlist_current:
            return
        canon = canonical_channels.get() or []
        ordered = [ch for ch in canon if ch in chlist_current] or chlist_current
        cur = input.channel() or ordered[0]
        nxt = _cycle(ordered, cur, +1)
        if nxt:
            # Let _on_channel_change run; no setting_selects here
            ui.update_select("channel", choices=ordered, selected=nxt, session=session)

    @reactive.Effect
    @reactive.event(input.prev_channel)
    def _prev_channel():
        if loading.get():
            return
        s = input.sample()
        if not s:
            return
        chlist_current = channels.get().get(s, [])
        if not chlist_current:
            return
        canon = canonical_channels.get() or []
        ordered = [ch for ch in canon if ch in chlist_current] or chlist_current
        cur = input.channel() or ordered[0]
        prv = _cycle(ordered, cur, -1)
        if prv:
            ui.update_select("channel", choices=ordered, selected=prv, session=session)


    # ---------- react to channel change ----------
    @reactive.Effect
    @reactive.event(input.channel)
    def _on_channel_change():
        if loading.get() or setting_selects.get():
            return
        c = input.channel()
        if not c:
            return
        _sync_controls_from_table(c)

    # ---------- update table ----------
    @reactive.Effect
    @reactive.event(input.apply_one)
    def _apply_one_channel():
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        if df.empty:
            return
        idx = df.index[df["Channel"] == c].tolist()
        if not idx:
            return
        i = idx[0]
        new_df = df.copy()
        new_df.at[i, "DoWinsor"] = bool(input.doWinsor())
        new_df.at[i, "Low"]      = float(input.winsor_low())
        new_df.at[i, "High"]     = float(input.winsor_high())
        params_df.set(new_df.reset_index(drop=True))

    @reactive.Effect
    @reactive.event(input.apply_threshold)
    def _apply_threshold_channel():
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        if df.empty:
            return
        idx_list = df.index[df["Channel"] == c].tolist()
        if not idx_list:
            return

        i = idx_list[0]
        new_df = df.copy()

        # clamp and persist to table
        try:
            thr_val = float(input.threshold_val())
        except Exception:
            thr_val = 0.0
        thr_val = max(0.0, min(1.0, thr_val))

        new_df.at[i, "DoThr"]  = bool(input.doThreshold())
        new_df.at[i, "ThrVal"] = thr_val

        params_df.set(new_df.reset_index(drop=True))

    @reactive.Effect
    @reactive.event(input.apply_noise)
    def _apply_noise_channel():
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        if df.empty:
            return
        idx = df.index[df["Channel"] == c].tolist()
        if not idx:
            return
        i = idx[0]
        new_df = df.copy()

        s = float(input.noise_strength())
        p = _strength_to_percentile(s)

        new_df.at[i, "Noise"]  = bool(input.doNoise())
        new_df.at[i, "NStr"]   = s         # UI strength (0..1)
        new_df.at[i, "NPrctl"] = p         # derived percentile actually used
        new_df.at[i, "WinSz"]  = int(input.window_size())

        params_df.set(new_df.reset_index(drop=True))



    # ---------- plot ----------
    @output
    @render.plot
    def img_viewer():
        # one figure only; decent size/dpi so the browser has pixels to work with
        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        try:
            imgs = images.get()
            s = input.sample()
            c = input.channel()

            if not imgs or not s or not c or s not in imgs:
                ax.text(0.5, 0.5, "No image", ha="center", va="center")
                ax.set_axis_off()
                return fig

            arr = imgs[s]
            chlist = channels.get().get(s, [])
            if c not in chlist:
                ax.text(0.5, 0.5, f"Channel {c!r} not found", ha="center", va="center")
                ax.set_axis_off()
                return fig

            idx = chlist.index(c)
            img = arr[idx, :, :].astype(np.float32)

            # Step 1: Winsorize (actually run this under doWinsor)
            if input.doWinsor():
                lo = max(0.0, min(1.0, float(input.winsor_low())))
                hi = max(0.0, min(1.0, float(input.winsor_high())))
                if hi > lo:
                    img = _apply_winsor(img, lo, hi)

            # Step 2: Global threshold (optional)
            if input.doThreshold():
                thr = float(input.threshold_val())
                if thr > 0.0:
                    cutoff = thr * (np.nanmax(img) if np.nanmax(img) > 0 else 1.0)
                    img = np.where(img >= cutoff, img, 0.0)

            # Step 3: Speckle suppression (old percentile rule + 2-neighbor check)
            if input.doNoise():
                wsize = max(1, int(input.window_size()))
                if wsize % 2 == 0:
                    wsize += 1
                s = float(input.noise_strength())
                p = _strength_to_percentile(s)
                img = _apply_speckle_suppress(img, size=wsize, perc=p, neighbor_limit=2)

            # Step 4: OPTIONAL arcsinh (note: now after denoise/threshold in this pipeline)
            if input.doAsinh():
                try:
                    cofac = int(float(input.asinh_cofactor()))
                except Exception:
                    cofac = 5
                cofac = max(2, min(10, cofac))
                img = np.arcsinh(img / float(cofac))

            #            # Step 5: Final normalization for display
            if bool(input.doNorm()):
                scope = (input.norm_scope() or "page")
                if scope == "global":
                    gpair = _global_minmax_for_channel(images.get(), channels.get(), c)
                    if gpair is not None:
                        gmin, gmax = gpair
                        if gmax > gmin:
                            img = (img - gmin) / (gmax - gmin)
                        else:
                            mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
                            if mx > mn:
                                img = (img - mn) / (mx - mn)
                    else:
                        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
                        if mx > mn:
                            img = (img - mn) / (mx - mn)
            else:
                # no normalization: just use processed img as-is
                pass

            # --- Step 6: Render ---
            ax.imshow(img, cmap="gray")
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

        except Exception as e:
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            return fig

            
    @output
    @render.ui
    def noise_tooltip():
        s = float(input.noise_strength())
        p = _strength_to_percentile(s)            # p in [0..1]
        p_txt = f"P{p*100:.1f}"

        long = (
            f"Pixels above their local {p_txt} are marked ‘bright.’ "
            "A pixel is removed only if ≤ 2 of its 8 neighbors are also bright. "
            "Increase sensitivity to flag more pixels as bright (more aggressive denoising). "
            "Decrease to preserve detail in bright patches."
        )

        # Small, muted line with a hover tooltip and a compact readout
        return ui.tags.small(
            ui.tags.span("ℹ️ ", class_="me-1"),
            f"Local cutoff ≈ {p_txt}",
            title=long,
            class_="text-muted"
        )


    @reactive.Effect
    @reactive.event(input.apply_norm)
    def _apply_norm_channel():
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        if df.empty:
            return
        idx = df.index[df["Channel"] == c].tolist()
        if not idx:
            return
        i = idx[0]
        new_df = df.copy()
        new_df.at[i, "DoNorm"] = bool(input.doNorm())
        params_df.set(new_df.reset_index(drop=True))


    @output
    @render.table
    def param_table():
        df = params_df.get()
        if df.empty:
            return pd.DataFrame({"Info": ["No parameters yet"]})
        rename_map = {
            "Channel": "Ch",
            "DoWinsor": "Win",
            "Low": "Low",
            "High": "High",
            "DoThr": "Thr?",
            "ThrVal": "ThrVal",
            "Noise": "Noise",
            "NStr": "NStr",
            "NPrctl": "NPerc",
            "WinSz": "WinSz",
            "DoNorm": "Norm?",
            "DoAsinh": "Asinh?",
            "Cofac": "Cofac",
            "NormScope": "Scope",     # <-- NEW
        }
        return df.rename(columns=rename_map).reset_index(drop=True)
    
    @reactive.Effect
    @reactive.event(input.apply_norm)
    def _apply_options_channel():
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        if df.empty:
            return
        idx = df.index[df["Channel"] == c].tolist()
        if not idx:
            return
        i = idx[0]
        new_df = df.copy()
        new_df.at[i, "DoNorm"]  = bool(input.doNorm())
        new_df.at[i, "DoAsinh"] = bool(input.doAsinh())
        new_df.at[i, "NormScope"] = (input.norm_scope() or "page")
        # input_select returns strings, guard and clamp 2..10
        try:
            cofac = int(float(input.asinh_cofactor()))
        except Exception:
            cofac = 5
        cofac = max(2, min(10, cofac))
        new_df.at[i, "Cofac"] = cofac
        params_df.set(new_df.reset_index(drop=True))

    @reactive.Effect
    @reactive.event(input.confirm_start)
    def _run_batch_analysis():
        # Close the modal
        ui.modal_remove(session=session)

        folder = (input.path() or "").strip()
        if not folder or not os.path.isdir(folder):
            print(f"⚠️ Invalid folder: {folder!r}")
            return

        out_dir = os.path.join(folder, "normalized images")
        os.makedirs(out_dir, exist_ok=True)

        params_path = os.path.join(out_dir, "parameter_table.csv")
        df = params_df.get().copy()
        if df.empty:
            print("⚠️ Parameter table is empty — nothing to analyze.")
            return
        df.to_csv(params_path, index=False)
        print(f"✅ Saved parameter table → {params_path}")

        script = Path(__file__).with_name("analysis.py")
        cmd = [sys.executable, str(script),
            "--input-dir", folder,
            "--params-csv", params_path,
            "--output-dir", out_dir]
        print("▶️ Running analysis:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print("✅ Analysis finished.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis script failed: {e}")


    @reactive.Effect
    @reactive.event(input.perform_analysis)
    def _confirm_start_modal():
        folder = (input.path() or "").strip()
        msg = ui.div(
            ui.p("Start image processing?"),
            ui.tags.small(f"Folder: {folder or '— (no folder chosen)'}")
        )
        m = ui.modal(
            msg,
            title="Confirm",
            easy_close=True,  # clicking outside = Cancel
            footer=ui.div(
                ui.modal_button("Cancel", class_="btn btn-secondary"),
                ui.input_action_button("confirm_start", "Start", class_="btn btn-primary ms-2"),
            ),
            size="m",
        )
        ui.modal_show(m, session=session)

    @reactive.Effect
    @reactive.event(input.export_params)
    def _export_params():
        df = params_df.get().copy()
        if df.empty:
            print("⚠️ No parameters to export.")
            return

        folder = (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()
        save_path = _pick_save_csv_dialog(initialdir=initdir, initialfile="parameter_table.csv")
        if not save_path:
            print("🛑 Export canceled.")
            return

        try:
            df.to_csv(save_path, index=False)
            print(f"✅ Exported parameters → {save_path}")
        except Exception as e:
            print(f"❌ Failed to write CSV: {e}")

    @output
    @render.ui
    def norm_scope_hint():
        if not images.get() or not input.channel():
            return ui.tags.small(" ")
        if (input.norm_scope() or "page") != "global":
            return ui.tags.small(" ")

        gpair = _global_minmax_for_channel(images.get(), channels.get(), input.channel())
        if not gpair:
            return ui.tags.small("Global range: —")
        gmin, gmax = gpair
        return ui.tags.small(f"Global range for “{input.channel()}”: [{gmin:.3g}, {gmax:.3g}]")

    @reactive.Effect
    @reactive.event(input.import_params)
    def _import_params():
        # Require a completed load (robust against stray/empty load clicks)
        if not data_loaded.get():
            print("⚠️ Load images before importing parameters.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        csv_path = _pick_open_csv_dialog(initialdir=initdir)
        if not csv_path:
            print("🛑 Import canceled.")
            return

        # 3) Read & validate
        try:
            df_in = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Failed to read CSV: {e}")
            return

        if "Channel" not in df_in.columns:
            print("❌ CSV missing required 'Channel' column.")
            return

        canon = list(canonical_channels.get() or [])
        csv_channels = [str(x) for x in df_in["Channel"].astype(str).tolist()]
        if set(csv_channels) != set(canon) or len(csv_channels) != len(canon):
            print("❌ CSV channels do not match current image channels.")
            print(f"   CSV:     {csv_channels}")
            print(f"   Expected:{canon}")
            return

        # 4) Normalize columns, order, and types
        target_cols = params_df.get().columns.tolist()
        defaults = {
            "Channel": "",
            "DoWinsor": False, "Low": 0.0, "High": 1.0,
            "DoThr": False, "ThrVal": 0.0,
            "Noise": False, "NStr": 1.0, "NPrctl": 0.995, "WinSz": 3,
            "DoNorm": True,
            "DoAsinh": False, "Cofac": 5,
        }

        df_in = df_in.set_index("Channel").reindex(canon).reset_index()

        # Add any missing columns with defaults
        for col in target_cols:
            if col not in df_in.columns:
                df_in[col] = defaults.get(col)

        # Keep only target columns, in order
        df_in = df_in[target_cols].copy()

        # Coerce types
        bool_cols  = [c for c in target_cols if c in ["DoWinsor","DoThr","Noise","DoNorm","DoAsinh"]]
        float_cols = [c for c in target_cols if c in ["Low","High","ThrVal","NStr"]]
        int_cols   = [c for c in target_cols if c in ["WinSz","Cofac"]]

        def _to_bool(v):
            s = str(v).strip().lower()
            if s in ("true","1","yes","y","t"): return True
            if s in ("false","0","no","n","f"): return False
            try: return bool(int(float(v)))
            except Exception: return False

        for c in bool_cols:
            df_in[c] = df_in[c].map(_to_bool)

        for c in float_cols:
            df_in[c] = pd.to_numeric(df_in[c], errors="coerce").fillna(defaults[c]).astype(float)

        for c in int_cols:
            df_in[c] = pd.to_numeric(df_in[c], errors="coerce").fillna(defaults[c]).astype(int)

        # 5) Commit and sync UI
        params_df.set(df_in.reset_index(drop=True))
        sel = input.channel()
        if sel:
            _sync_controls_from_table(sel)

        print(f"✅ Imported parameters from {csv_path}")

from shiny import App  # (you already have this at the top)

app = App(app_ui, server)