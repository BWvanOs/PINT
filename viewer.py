from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, ui, render, reactive
from load_tiffs import load_tiffs_raw
from scipy.ndimage import percentile_filter
import os, sys, subprocess


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
        width="650px",          # tweak as needed
    ),

    # ---- HEAD/CSS (positional arg #2) ----
    ui.head_content(
        ui.tags.style("""
            :root{
                /* Tweak these to taste */
                --controls-h: 450px;    /* total height of the top controls area */
                --controls-top-h: 75px; /* height of the toolbar row */
                --controls-gap: 3px;
            }

            .flex-col { display:flex; flex-direction:column; height:100vh; }

            .controls-fixed {
                flex: 0 0 var(--controls-h);
                display:flex; flex-direction:column;
                gap: var(--controls-gap);
                overflow:hidden;
            }
                      
                        /* Make the panels row and its columns fill the fixed area height */
            .controls-panels > .row { height: 100%; }
            .controls-panels > .row > [class^="col"],
            .controls-panels > .row > [class*=" col"] { display: flex; }
            .controls-panels .card { flex: 1 1 auto; margin-bottom: 0; }

            .controls-top    { flex: 0 0 var(--controls-top-h); overflow:hidden; }
            .controls-panels { flex: 1 1 auto; min-height:0; overflow:hidden; }

            .controls-panels .card { height:100%; }
            .controls-panels .card-body {
                padding: 8px;
                display:flex;
                flex-direction:column;
                gap: 6px;
                overflow-x: hidden;
            }
            .controls-panels .row { margin-left: 0; margin-right: 0; }
            .controls-panels .col { padding-left: 0; padding-right: 0; }

            .viewer-fill { flex: 1 1 auto; min-height:0; display:flex; flex-direction:column; }.controls-fixed

            .sidebar-col      { display:flex; flex-direction:column; height:100%; }
            .param-table-wrap table {
                font-size: 12px;
                width: 100% !important;
                table-layout: auto;
                border-collapse: collapse;
            }
            .param-table-wrap td, .param-table-wrap th {
                padding: 2px 4px;
                white-space: nowrap;      /* keep short values on one line */
                text-overflow: ellipsis;  /* add … if too long */
                overflow: hidden;
                text-align: left;
            }
            
            .param-table-wrap th {
                font-weight: 600;     /* keep headers readable */
                text-align: left;
            }        

            /* Make sure the sidebar overlays other content when open */
            .bslib-sidebar-layout > .bslib-sidebar { z-index: 1050; }
            .bslib-sidebar-layout .bslib-sidebar-toggle { z-index: 1060; }
        """),
    ),

    # ---- MAIN CONTENT (positional arg #3) ----
    ui.row(
        ui.column(
            12,
            ui.tags.div(
                # ===== Fixed-height controls (toolbar + panels) =====
                ui.tags.div(
                    # --- Top control bar: path + load + sample + channel ---
                    ui.row(
                    # Folder path
                        ui.column(5,
                            ui.input_text("path", "Folder path", value="", width="100%"),
                        ),

                        # Load button
                        ui.column(1,
                            ui.input_action_button("load", "Load images", class_="w-100"),
                        ),

                        ui.column(1), ##spacer

                        # Sample + Channel stacked
                        ui.column(2,
                            ui.row(
                                ui.column(8, ui.input_select("sample", "Sample", choices=[], selected=None, width="100%"),),
                                ui.column(2, ui.input_action_button("prev_sample", "←", class_="btn-sm"),),
                                ui.column(2, ui.input_action_button("next_sample", "→", class_="btn-sm"),),
                                class_="align-items-center"
                            ),
                            ui.row(
                                ui.column(8, ui.input_select("channel", "Channel", choices=[], selected=None, width="100%"),),
                                ui.column(2, ui.input_action_button("prev_channel", "←", class_="btn-sm"),),
                                ui.column(2, ui.input_action_button("next_channel", "→", class_="btn-sm"),),
                                class_="align-items-center"
                            ),
                        ),

                        ui.column(1), ##spacer

                        # Perform analysis button
                        ui.column(1,
                            ui.row(
                                ui.div(
                                    ui.input_action_button(
                                        "perform_analysis",
                                        "Perform analysis",
                                        class_="btn btn-primary text-white w-100 h-100"
                                    ),
                                    class_="d-flex h-100 align-items-stretch"  # make the container stretch full height, only it doesn't work. What the...
                                ),
                            ),
                        ),

                        ui.column(1), ##spacer

                        class_="controls-top align-items-center",  # center contents vertically
                    ),

                    ui.hr(),

                    ui.row(
                        # --- PANEL 1: Winsorization ---
                        ui.column(3,
                            ui.card(
                                ui.card_header("Winsorization"),
                                # Sliders stacked
                                ui.row(
                                    ui.column(6, ui.input_slider("winsor_low", "Lower quantile (0–1)", min=0.0, max=1.0, value=0.00, step=0.01),),
                                    ui.column(6, ui.input_slider("winsor_high", "Upper quantile (0–1)", min=0.0, max=1.0, value=0.99, step=0.01),),
                                ),
                                # Checkbox + button side-by-side
                                ui.row(
                                    ui.column(4, ui.input_checkbox("doWinsor", "doWinsorize", value=True)),
                                    ui.column(6, ui.input_action_button("apply_one", "Update channel", class_="btn btn-primary w-100")),
                                    ui.column(2),  # spacer
                                ),
                            ),
                        ),

                        # --- PANEL 2: Global Threshold ---
                        ui.column(3,
                            ui.card(
                                ui.card_header("Global Threshold"),
                                ui.row(
                                    ui.column(12, ui.input_slider("threshold_val", "Threshold (0-1)", min=0.0, max=1.0, value=0.1, step=0.01),),
                                ),
                                ui.row(
                                    ui.column(4, ui.input_checkbox("doThreshold", "Apply threshold", value=True)),
                                    ui.column(6, ui.input_action_button("apply_threshold", "Update channel", class_="btn btn-primary w-100")),
                                    ui.column(2
                                    ),
                                ),
                            ),
                        ),

                        # --- PANEL 3: Noise Removal ---
                        ui.column(3,
                            ui.card(
                                ui.card_header("Sliding Window Noise Removal"),
                                ui.row(
                                    ui.column(6, ui.input_slider("noise_strength", "Strength (0-1)", min=0.0, max=1.0, value=1, step=0.01)),
                                    ui.column(6, ui.input_numeric("window_size", "Window size", value=3, min=1, step=2)),
                                ),
                                ui.row(
                                    ui.column(4, ui.input_checkbox("doNoise", "Apply noise removal", value=True)),
                                    ui.column(6, ui.input_action_button("apply_noise", "Update channel", class_="btn btn-primary w-100")),
                                    ui.column(2),
                                ),
                            ),
                        ),
                        ##PANEL4
                        ui.column(3,   # narrow column as you requested
                            ui.card(
                                ui.card_header("Normalization and Transformation"),
                                ui.row(
                                    ui.column(
                                        8,
                                        ui.tags.div(
                                            ui.input_checkbox("doAsinh", "Arcsinh transform data", value=False),
                                            class_="d-flex align-items-center"
                                        ),
                                    ),
                                    ui.column(
                                        4,
                                        ui.input_select(
                                            "asinh_cofactor",
                                            "Cofactor",
                                            choices=[str(i) for i in range(2, 11)],
                                            selected="5",
                                            width="100%"
                                        ),
                                    ),
                                    class_="align-items-center",
                                ),
                                ui.row(
                                    ui.column(
                                        6,
                                        ui.tags.div(
                                            ui.input_checkbox("doNorm", "Normalize the channel?", value=True),
                                            class_="d-flex align-items-center"
                                        ),
                                    ),
                                    ui.column(
                                        6,
                                        ui.input_action_button("apply_norm", "Apply norm/transform", class_="btn btn-primary w-100"),
                                    ),
                                    class_="align-items-center",
                                ),
                            ),
                        ),

                        class_="controls-panels",
                        ),
                    ),
                    
                    # ===== Plot area =====
                    ui.tags.div(
                        ui.output_plot("img_viewer", fill=True, height="100%"),
                        #ui.output_text_verbatim("dbg"),
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
            "Noise", "NStr", "WinSz",
            "DoNorm",
            "DoAsinh", "Cofac",
        ])
    )

    loading = reactive.Value(False)
    setting_selects = reactive.Value(False)
    syncing_controls = reactive.Value(False)

    # ---------- helpers ----------
    def _ensure_param_columns():
        required = ["Channel","DoWinsor","Low","High",
                    "DoThr","ThrVal",
                    "Noise","NStr","WinSz",
                    "DoNorm"]  # if you're adding Normalize
        defaults = {
            "Channel": "",
            "DoWinsor": False, "Low": 0.0, "High": 1.0,
            "DoThr": False, "ThrVal": 0.0,
            "Noise": False, "NStr": 1.0, "WinSz": 3,
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
        rows = [{
            "Channel": ch,
            "DoWinsor": bool(input.doWinsor()),
            "Low": float(input.winsor_low()),
            "High": float(input.winsor_high()),
            "DoThr": bool(input.doThreshold()),
            "ThrVal": float(input.threshold_val()),
            "Noise": bool(input.doNoise()),
            "NStr": float(input.noise_strength()),
            "WinSz": int(input.window_size()),
            "DoNorm": bool(input.doNorm()),
            "DoAsinh": bool(input.doAsinh()),
            "Cofac": int(float(input.asinh_cofactor() or 5)),
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

            ##arcsinh controls
            session.send_input_message("doAsinh", {"value": bool(row.get("DoAsinh", False))})
            session.send_input_message("asinh_cofactor", {"value": str(int(row.get("Cofac", 5)))})
        finally:
            syncing_controls.set(False)

    def _apply_winsor(cur: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
        q_low, q_high = np.quantile(cur, [lo_q, hi_q])
        return np.clip(cur, q_low, q_high)

    def _apply_sliding_window(img: np.ndarray, size: int, perc: float, remove: str = "bright") -> np.ndarray:
        """
        Remove speckles using a local percentile threshold.

        Parameters
        ----------
        img : 2D array
        size : odd int (e.g., 3,5,7)
        perc : 0..1 (e.g., 0.9 = 90th percentile)
        remove : "bright" (zero pixels > local pct) or "dark" (zero pixels < local pct)
        """
        if size <= 1 or perc <= 0:
            return img
        if size % 2 == 0:
            size += 1

        local_thresh = percentile_filter(img, percentile=perc * 100, size=size)

        if remove == "bright":
            # keep pixels <= local pct; zero brighter-than-local outliers
            return np.where(img <= local_thresh, img, 0.0)
        else:
            # keep pixels >= local pct; zero darker-than-local outliers
            return np.where(img >= local_thresh, img, 0.0)

    
    # ---------- load images ----------
    @reactive.Effect
    @reactive.event(input.load)
    def _do_load():
        if loading.get():
            return
        loading.set(True)
        try:
            folder = input.path().strip()
            print(">>> Load triggered with folder:", folder)

            imgs, chs = load_tiffs_raw(folder)
            images.set(imgs)
            channels.set(chs)

            samples = list(imgs.keys())
            if not samples:
                return
            first_sample = samples[0]
            first_chlist = chs[first_sample]
            first_channel = first_chlist[0]

            # set canonical + prefill table
            canonical_channels.set(list(first_chlist))
            _prefill_params(first_chlist)

            # update selects under guard
            setting_selects.set(True)
            try:
                ui.update_select("sample",  choices=samples,      selected=first_sample,  session=session)
                ui.update_select("channel", choices=first_chlist, selected=first_channel, session=session)
            finally:
                setting_selects.set(False)

            # sync knobs from the table row for the first channel
            _sync_controls_from_table(first_channel)

            print(">>> Post-load selected:", first_sample, first_channel)

        finally:
            loading.set(False)

    # ---------- react to sample change ----------
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
        new_df.at[i, "Noise"]  = bool(input.doNoise())
        new_df.at[i, "NStr"]   = float(input.noise_strength())
        new_df.at[i, "WinSz"]  = int(input.window_size())
        params_df.set(new_df.reset_index(drop=True))

    # ---------- plot ----------
    @output
    @render.plot
    def img_viewer():
        try:
            imgs = images.get()
            s = input.sample()
            c = input.channel()
            fig, ax = plt.subplots()

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

            # Winsorize
            # Step 1: Winsorize
            if input.doWinsor():
                lo = max(0.0, min(1.0, float(input.winsor_low())))
                hi = max(0.0, min(1.0, float(input.winsor_high())))
                if hi > lo:
                    img = _apply_winsor(img, lo, hi)

            # Step 2: Threshold
            if input.doThreshold():
                thr = float(input.threshold_val())
                if thr > 0.0:
                    cutoff = thr * (np.nanmax(img) if np.nanmax(img) > 0 else 1.0)
                    img = np.where(img >= cutoff, img, 0.0)

            # Step 3: Local percentile noise removal
            if input.doNoise():
                wsize = max(1, int(input.window_size()))
                if wsize % 2 == 0:
                    wsize += 1
                perc = float(input.noise_strength())  # 0..1
                img = _apply_sliding_window(img, wsize, perc, remove="bright")

            # Step 4: Arcsinh transform (after denoising, before normalization)
            if input.doAsinh():
                try:
                    cofac = int(float(input.asinh_cofactor()))
                except Exception:
                    cofac = 5
                cofac = max(2, min(10, cofac))
                # standard asinh transform for CyTOF/IMC: asinh(x / cofactor)
                img = np.arcsinh(img / float(cofac))
            
            # Final normalization for display
            mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
            if mx > mn:
                img = (img - mn) / (mx - mn)

            ax.imshow(img, cmap="gray")
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            return fig

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
            "WinSz": "WinSz",
            "DoNorm": "Norm?",
            "DoAsinh": "Asinh?",
            "Cofac": "Cofac",
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
        # input_select returns strings, guard and clamp 2..10
        try:
            cofac = int(float(input.asinh_cofactor()))
        except Exception:
            cofac = 5
        cofac = max(2, min(10, cofac))
        new_df.at[i, "Cofac"] = cofac
        params_df.set(new_df.reset_index(drop=True))

    @reactive.Effect
    @reactive.event(input.perform_analysis)
    def _run_batch_analysis():
        folder = (input.path() or "").strip()
        if not folder or not os.path.isdir(folder):
            print(f"⚠️ Invalid folder: {folder!r}")
            return

        out_dir = os.path.join(folder, "normalized images")
        os.makedirs(out_dir, exist_ok=True)

        # Save current parameter table to the output folder
        params_path = os.path.join(out_dir, "parameter_table.csv")
        df = params_df.get().copy()
        if df.empty:
            print("⚠️ Parameter table is empty — nothing to analyze.")
            return
        df.to_csv(params_path, index=False)
        print(f"✅ Saved parameter table → {params_path}")

        # Call the external analysis script with input dir, params csv and output dir
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


from shiny import App  # (you already have this at the top)

app = App(app_ui, server)