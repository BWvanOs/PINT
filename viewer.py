from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, ui, render, reactive
from load_tiffs import load_tiffs_raw
from scipy.ndimage import percentile_filter

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
        width="600px",          # tweak as needed
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

            .viewer-fill { flex: 1 1 auto; min-height:0; display:flex; flex-direction:column; }

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
                text-alighn: left;
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
                        ui.column(5, ui.input_text("path", "Folder path", value="", width="100%")),
                        ui.column(1, ui.input_action_button("load", "Load images", class_="w-100")),
                        ui.column(1),  # spacer
                        ui.column(
                            2,
                            ui.tags.div(
                                ui.tags.label("Sample", class_="form-label mb-0 me-2"),
                                ui.input_select("sample", None, choices=[], selected=None, width="100%"),
                                ui.input_action_button("prev_sample", "←", class_="btn-sm"),
                                ui.input_action_button("next_sample", "→", class_="btn-sm"),
                                class_="d-flex align-items-center gap-2",
                            ),
                        ),
                        ui.column(
                            2,
                            ui.tags.div(
                                ui.tags.label("Channel", class_="form-label mb-0 me-2"),
                                ui.input_select("channel", None, choices=[], selected=None, width="100%"),
                                ui.input_action_button("prev_channel", "←", class_="btn-sm"),
                                ui.input_action_button("next_channel", "→", class_="btn-sm"),
                                class_="d-flex align-items-center gap-2",
                            ),
                        ),
                        class_="controls-top",
                    ),

                    ui.hr(),

                    # --- Second controls row: three vertical panels ---
                    ui.row(
                        ui.column(
                            4,
                            ui.card(
                                ui.card_header("Winsorization"),
                                ui.row(
                                    ui.column(
                                        12,
                                        ui.input_slider("winsor_low", "Lower quantile (0–1)", min=0.0, max=1.0, value=0.00, step=0.01),
                                    ),
                                ),
                                ui.row(
                                    ui.column(
                                        12,
                                        ui.input_slider("winsor_high", "Upper quantile (0–1)", min=0.0, max=1.0, value=0.99, step=0.01),
                                    ),
                                ),
                                ui.row(
                                    ui.column(
                                        4,
                                        ui.tags.div(
                                            ui.input_checkbox("doWinsor", "doWinsorize", value=True),
                                            class_="mt-2"
                                        ),
                                    ),
                                    ui.column(
                                        6,
                                        ui.input_action_button("apply_one", "Update channel", class_="btn btn-primary w-100 mt-2"),
                                    ),
                                    ui.column(2),
                                ),
                            ),
                        ),


                        ui.column(4,
                            ui.card(
                                ui.card_header("Global Threshold"),
                                ui.row(
                                    ui.column(
                                        12,
                                        ui.input_slider(
                                            "threshold_val", 
                                            "Threshold (0–1)", 
                                            min=0.0, max=1.0, value=0.1, step=0.01
                                        ),
                                    ),
                                ),
                                ui.row(
                                    ui.column(
                                        4,
                                        ui.tags.div(
                                            ui.input_checkbox("doThreshold", "Apply threshold", value=True),
                                            class_="mt-2"
                                        ),
                                    ),
                                    ui.column(
                                        6,
                                        ui.input_action_button(
                                            "apply_threshold", "Update channel", 
                                            class_="btn btn-primary w-100 mt-2"
                                        ),
                                    ),
                                    ui.column(2),
                                ),
                            ),
                        ),
                        ##PANEL 3 LAYOUT
                        ui.column(
                            4,
                            ui.card(
                                ui.card_header("Sliding Window Noise Removal"),
                                ui.row(
                                    ui.column(
                                        12,
                                        ui.input_slider(
                                            "noise_strength",
                                            "Strength (0–1)",
                                            min=0.0, max=1.0, value=0.5, step=0.01,
                                        ),
                                    ),
                                ),
                                ui.row(
                                    ui.column(
                                        12,
                                        ui.input_numeric(
                                            "window_size", "Window size", value=3, min=1, step=1
                                        ),
                                    ),
                                ),
                                ui.row(
                                    ui.column(
                                        4,
                                        ui.tags.div(
                                            ui.input_checkbox("doNoise", "Apply noise removal", value=False),
                                            class_="mt-2",
                                        ),
                                    ),
                                    ui.column(
                                        6,
                                        ui.input_action_button(
                                            "apply_noise",
                                            "Update channel",
                                            class_="btn btn-primary w-100 mt-2",
                                        ),
                                    ),
                                    ui.column(2),
                                ),
                            ),
                        ),
                        class_="controls-panels",
                    ),
                    class_="controls-fixed",
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
            "Noise", "NStr", "WinSz"
        ])
    )

    loading = reactive.Value(False)
    setting_selects = reactive.Value(False)
    syncing_controls = reactive.Value(False)

    # ---------- helpers ----------
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
        } for ch in first_chlist]
        df = pd.DataFrame(rows, columns=params_df.get().columns)
        params_df.set(df.reset_index(drop=True))

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
            session.send_input_message("doWinsor", {"value": bool(row["DoWinsor"])})
            session.send_input_message("winsor_low", {"value": float(row["Low"])})
            session.send_input_message("winsor_high", {"value": float(row["High"])})
            session.send_input_message("doThreshold", {"value": bool(row["DoThr"])})
            session.send_input_message("threshold_val", {"value": float(row["ThrVal"])})
            session.send_input_message("doNoise", {"value": bool(row["Noise"])})
            session.send_input_message("noise_strength", {"value": float(row["NStr"])})
            session.send_input_message("window_size", {"value": int(row["WinSz"])})
        finally:
            syncing_controls.set(False)

    def _apply_winsor(cur: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
        q_low, q_high = np.quantile(cur, [lo_q, hi_q])
        return np.clip(cur, q_low, q_high)

    def _apply_sliding_window(img: np.ndarray, size: int, perc: float) -> np.ndarray:
        """
        Zero-out pixels that are below the local percentile threshold.
        
        Parameters
        ----------
        img : np.ndarray
            2D array of image intensities (after winsorization + thresholding).
        size : int
            Window size (odd integer, e.g. 3, 5, 7).
        perc : float
            Percentile threshold (0–1). Example: 0.5 means 50th percentile.
        """
        if size <= 1 or perc <= 0:
            return img

        if size % 2 == 0:
            size += 1  # enforce odd window size

        # Compute local percentile per neighborhood (scipy expects 0–100)
        local_thresh = percentile_filter(img, percentile=perc * 100, size=size)

        # Keep pixels above local percentile, zero out the rest
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
                perc = float(input.noise_strength())  # e.g. 0.5 = 50th percentile
                img = _apply_sliding_window(img, wsize, perc)

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
        }
        return df.rename(columns=rename_map).reset_index(drop=True)

from shiny import App  # (you already have this at the top)

app = App(app_ui, server)
