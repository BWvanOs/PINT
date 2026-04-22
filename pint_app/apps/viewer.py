from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, ui, render, reactive
from scipy.ndimage import grey_opening, uniform_filter
import os, sys, subprocess
import shutil
import warnings

from tifffile import imwrite, imread

from pint_app.core.load_tiffs import load_tiffs_raw
from pint_app.core.formatting import fmt1
from pint_app.core.dialogs import (
    pick_folder_dialog,
    pick_open_csv_dialog,
    pick_save_csv_dialog,
    pick_save_png_dialog,
    pick_save_tiff_dialog,
)
from pint_app.core.processing import (
    clamp01,
    apply_winsor,
    apply_threshold_absolute,
    apply_threshold_fraction_of_max,
    arcsinh_transform,
    apply_speckle_suppress,
    strength_to_percentile,
    normalize_minmax,
    global_minmax_for_channel as compute_global_minmax_for_channel,
    global_winsor_range_for_channel as compute_global_winsor_range_for_channel,
    image_winsor_range as compute_image_winsor_range,
    process_image_pipeline,
)
from pint_app.core.selection import cycle_list, order_by_canonical
from pint_app.core.params import (
    PARAM_COLUMNS,
    make_params_df,
    update_channel_row,
    format_for_display,
    validate_and_normalize_import,
)

from matplotlib.patches import Patch

from pint_app.core.load_masks import (
    validate_mask_input_table,
    list_mask_files,
    match_cellmask_names_to_files,
    split_mask_matches,
    get_cells_for_mask_name,
    strip_known_mask_suffix,
)

from pint_app.core.mask_viz import (
    read_mask_tiff,
    match_mask_centroids_to_cells,
    make_mask_plot_data,
)

warnings.filterwarnings("ignore", category=UserWarning, message=".*Tight layout not applied.*")

COMPOSITE_PALETTE = {
    "Cyan":    (0.00, 1.00, 1.00),
    "Magenta": (1.00, 0.00, 1.00),
    "Yellow":  (1.00, 1.00, 0.00),
    "Green":   (0.00, 1.00, 0.00),
    "Red":     (1.00, 0.00, 0.00),
    "Blue":    (0.00, 0.40, 1.00),
    "Orange":  (1.00, 0.55, 0.00),
    "White":   (1.00, 1.00, 1.00),
}

COMPOSITE_COLOR_CHOICES = list(COMPOSITE_PALETTE.keys())
MAX_COMPOSITE_CHANNELS = 8


def make_composite_slot(slotIdx: int):
    defaultColor = COMPOSITE_COLOR_CHOICES[(slotIdx - 1) % len(COMPOSITE_COLOR_CHOICES)]
    return ui.row(
        ui.column(
            1,
            ui.input_checkbox(f"comp_enable_{slotIdx}", "", value=(slotIdx <= 3)),
        ),
        ui.column(
            6,
            ui.input_select(
                f"comp_channel_{slotIdx}",
                "",
                choices=[],
                selected=None,
                width="100%",
            ),
        ),
        ui.column(
            3,
            ui.input_select(
                f"comp_color_{slotIdx}",
                "",
                choices=COMPOSITE_COLOR_CHOICES,
                selected=defaultColor,
                width="100%",
            ),
        ),
        ui.column(
            2,
            ui.input_slider(
                f"comp_gain_{slotIdx}",
                "",
                min=0.1,
                max=3.0,
                value=1.0,
                step=0.1,
            ),
        ),
        class_="align-items-end gy-0",
    )

app_ui = ui.page_sidebar(
    # This is the right side of the window that collapses when opening the sidebar.
    ui.sidebar(
        ui.h4("Current Parameters"),
        ui.tags.div(ui.output_table("param_table"), class_="param-table-wrap"),
        open="closed",          # collapsed by default
        id="sidebar",
        class_="sidebar-col",
        width="850px",          # tweak as needed
    ),

    # This is CSS to fix scaling issues with the viewer. Not that this is made by chatgpt, so edit at your own risk.
    ui.head_content(
        ui.tags.style("""
            :root{
                /* You can tweak these two and nothing else! */
                --controls-h: 350px;     /* total height of the top area (toolbar + panels) */
                --controls-top-h: 100px; /* height of the toolbar row */
            }

            /* Page skeleton: fixed top area + growing viewer */
            .flex-col { display:flex; flex-direction:column; height:100vh; }

            .controls-fixed {
                flex: 0 0 auto;
                min-height: var(--controls-h);
                display: flex;
                flex-direction: column;
                overflow: visible;
            }
            .controls-top  {
                flex: 0 0 auto;
                min-height: var(--controls-top-h);
                overflow: visible;
            }
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
            .param-table-wrap th { font-weight: 750; text-align: left; }

            /* Make sure the sidebar overlays other content when open */
            .bslib-sidebar-layout > .bslib-sidebar { z-index: 1050; }
            .bslib-sidebar-layout .bslib-sidebar-toggle { z-index: 1060; }

            .nav-tabs {
                margin-bottom: 10px;
            }

            .nav-tabs .nav-link {
                font-weight: 600;
            }

            .nav-tabs .nav-link.active {
                background-color: #f8f9fa;
                border-color: #dee2e6 #dee2e6 #fff;
            }          

            /* Tighten the toolbar only (kills the “ghost row”) */
            .controls-top .shiny-input-container { margin-bottom: 0 !important; }
            .controls-top .row { --bs-gutter-y: 0; margin-bottom: 0; }
            .controls-top .col { padding-left: 0; padding-right: 0; }
            .controls-fixed hr { margin: 6px 0; }"""
        )
    ), 

    # This is the main content of the image handler and normalization settings tools
    ui.row(
        ui.column(
            12,
            ui.tags.div(
                # Toolbar + panels (so essentially everything but the table)
                ui.tags.div(
                    
                    ##line to seperate the UI elements
                    ui.hr(),

                    ui.navset_tab(
                        ui.nav_panel(
                            "PINT",
                            #Top bar with path, load images, sample channel selector and export functions
                            ui.row(
                                ## Folder path
                                ui.column(
                                    4,
                                    ui.input_text("path", "Folder path", value="", width="100%"),
                                ),
                                ## Load button
                                ui.column(
                                    1,
                                    ui.input_action_button("load", "Load images", class_="w-100"),
                                ),
                                ui.column(1),
                                ## Sample selector
                                ui.column(
                                    2,
                                    ui.row(
                                        ui.column(
                                            8,
                                            ui.input_select("sample", "Sample", choices=[], selected=None, width="100%"),
                                        ),
                                        ui.column(2, ui.input_action_button("prev_sample", "←", class_="btn-sm w-100")),
                                        ui.column(2, ui.input_action_button("next_sample", "→", class_="btn-sm w-100")),
                                        class_="align-items-center gy-0",
                                    ),
                                ),

                                ## Channel selector
                                ui.column(
                                    2,
                                    ui.row(
                                        ui.column(
                                            8,
                                            ui.input_select("channel", "Channel", choices=[], selected=None, width="100%"),
                                        ),
                                        ui.column(2, ui.input_action_button("prev_channel", "←", class_="btn-sm w-100")),
                                        ui.column(2, ui.input_action_button("next_channel", "→", class_="btn-sm w-100")),
                                        class_="align-items-center gy-0",
                                    ),
                                ),
                                ## Export / import / process
                                ui.column(
                                    2,
                                    ui.row(
                                        ui.column(
                                            4,
                                            ui.input_action_button("export_params", "Export CSV", class_="btn btn-secondary w-100"),
                                        ),
                                        ui.column(
                                            4,
                                            ui.input_action_button("import_params", "Import CSV", class_="btn btn-secondary w-100"),
                                        ),
                                        ui.column(
                                            4,
                                            ui.input_action_button(
                                                "perform_analysis",
                                                "Process Images",
                                                class_="btn btn-primary text-white w-100",
                                            ),
                                        ),
                                        class_="align-items-end gy-0",
                                    ),
                                ),
                                #IMPORTANT: this is part of the SAME ui.row(...) call; note the comma! If you move this it will break everything
                                class_="controls-top align-items-end gy-0 mb-2",
                            ),
                            ui.row(
                                # Panel 1 winsorization
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

                                # Panel 2 Thresholding
                                ui.column(
                                    3,
                                    ui.card(
                                        ui.card_header("Thresholding"),
                                        ui.row(
                                            ui.column(6, ui.input_slider("abs_threshold_val", "Absolute threshold (counts)", min=0.0, max=100.0, value=1, step=0.1)),
                                            ui.column(6, ui.input_slider("thr_fraction_val", "Fraction of max (0–1)", min=0.0, max=1.0, value=0.1, step=0.01)),
                                        ),
                                        ui.row(
                                            ui.column(
                                                3,
                                                ui.tags.div(
                                                    ui.input_checkbox("doAbsThreshold", "Do Abs thresholding", value=True),
                                                ),
                                            ),
                                            ui.column(
                                                3,
                                                ui.tags.div(
                                                    ui.input_checkbox("doThreshold", "Do Perc thresholding", value=False),
                                                ),
                                            ),
                                            ui.column(6, ui.input_action_button("apply_threshold", "Update channel", class_="btn btn-primary w-100")),
                                        ),
                                    ),
                                ),

                                # Panel3 Sliding windows noise removal
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
                                                    ui.output_ui("noise_tooltip"),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),

                                # Panel4: Normalization and transformation
                                ui.column(
                                    3,
                                    ui.card(
                                        ui.card_header("Normalization and Transformation"),
                                        ui.row(
                                            ui.column(
                                                7,
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
                                            ui.column(
                                                5,
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
                        ),

                        ui.nav_panel(
                            "Image creator",
                            ui.row(
                                ui.column(
                                    9,
                                    ui.card(
                                        ui.card_header("Composite channels (processed images)"),
                                        ui.row(
                                            ui.column(
                                                6,
                                                ui.row(
                                                    ui.column(1, ui.tags.strong("On")),
                                                    ui.column(6, ui.tags.strong("Channel")),
                                                    ui.column(3, ui.tags.strong("Color")),
                                                    ui.column(2, ui.tags.strong("Gain")),
                                                    class_="mb-1",
                                                ),
                                                *[make_composite_slot(i) for i in range(1, 5)],
                                            ),
                                            ui.column(
                                                6,
                                                ui.row(
                                                    ui.column(1, ui.tags.strong("On")),
                                                    ui.column(6, ui.tags.strong("Channel")),
                                                    ui.column(3, ui.tags.strong("Color")),
                                                    ui.column(2, ui.tags.strong("Gain")),
                                                    class_="mb-1",
                                                ),
                                                *[make_composite_slot(i) for i in range(5, 9)],
                                            ),
                                        ),
                                    ),
                                ),
                                ui.column(
                                    3,
                                    ui.card(
                                        ui.card_header("Composite export"),
                                        ui.input_action_button("fill_composite_from_current", "Fill from first 8 channels", class_="btn btn-secondary w-100 mb-2"),
                                        ui.input_action_button("save_composite_tiff", "Save composite TIFF", class_="btn btn-primary w-100 mb-2"),
                                        ui.input_action_button("export_creator_composites_all", "Export composite TIFF for all images", class_="btn btn-secondary w-100"),
                                        ui.br(),
                                        ui.br(),
                                        ui.output_ui("composite_summary"),
                                    ),
                                ),
                            ),
                        ),

                        ui.nav_panel(
                            "Mask visualization",
                            ui.row(
                                ui.column(
                                    4,
                                    ui.card(
                                        ui.card_header("Mask input"),
                                        ui.input_text("mask_csv_path", "Cell table CSV", value="", width="100%"),
                                        ui.row(
                                            ui.column(6, ui.input_action_button("browse_mask_csv", "Browse CSV", class_="btn btn-secondary w-100")),
                                            ui.column(6, ui.input_action_button("load_mask_csv", "Load cell table", class_="btn btn-primary w-100")),
                                        ),
                                        ui.br(),
                                        ui.input_text("mask_path", "Mask folder", value="", width="100%"),
                                        ui.row(
                                            ui.column(6, ui.input_action_button("browse_mask_path", "Browse folder", class_="btn btn-secondary w-100")),
                                            ui.column(6, ui.input_action_button("match_masks", "Match masks", class_="btn btn-primary w-100")),
                                        ),
                                        ui.br(),
                                        ui.input_select("mask_cell_id_col", "Cell ID column", choices=[], selected=None, width="100%"),
                                        ui.input_select("mask_x_col", "X coordinate column", choices=[], selected=None, width="100%"),
                                        ui.input_select("mask_y_col", "Y coordinate column", choices=[], selected=None, width="100%"),
                                        ui.input_select("mask_name_col", "Mask name column", choices=[], selected=None, width="100%"),
                                        ui.input_select("mask_cluster_col", "Cluster column", choices=[], selected=None, width="100%"),
                                        ui.input_select("mask_condition_col", "Condition column", choices=[], selected=None, width="100%"),
                                        ui.input_select("mask_sample_col", "SampleNumber column", choices=[], selected=None, width="100%"),
                                        ui.input_numeric("mask_scale_factor", "Scale factor", value=5, min=1, max=20, step=1),
                                    ),
                                    ui.br(),
                                    ui.card(
                                        ui.card_header("Matched masks"),
                                        ui.output_ui("mask_table_summary"),
                                        ui.output_ui("mask_match_summary"),
                                        ui.input_select("selected_mask_name", "Select mask", choices=[], selected=None, width="100%"),
                                        ui.output_ui("selected_mask_summary"),
                                    ),
                                ),
                                ui.column(
                                    8,
                                    ui.card(
                                        ui.card_header("Visualization"),
                                        ui.div(
                                            ui.output_plot("mask_viewer", fill=True, height="100%"),
                                            style="height: 75vh;"
                                        ),
                                    ),
                                ),
                            ),
                        ),

                        ui.nav_panel(
                            "Neigborhood",
                            ui.row(
                                ui.column(
                                    2,
                                    ui.card(
                                        ui.card_header("Neighborhood analysis"),
                                        ui.tags.a(
                                            "Open neighborhood analysis",
                                            href="/neighborhood/",
                                            target="_blank",
                                            role="button",
                                            class_="btn btn-secondary w-100",
                                            style="pointer-events: auto;",
                                        ),
                                    ),
                                ),
                                ui.column(10),
                            ),
                        ),

                        id="viewer_mode",
                    ),
                    class_="controls-fixed",
                ),

                ##Main plot area --> here images will be rendered
                ui.output_ui("main_viewer_ui"),
                class_="flex-col",
            ),
        ),
    ),

    #IMPORTANT: This must be the last argument!
    position="right",
)



## <----------------> SERVER! <-------------------> ##
from scipy.ndimage import median_filter


def server(input, output, session):
    #Make all the reactive states of the app, will be filled later
    images = reactive.Value({})                  #Dictionary of images, will be filled by tiffloader
    channels = reactive.Value({})                #Dic of channel names
    canonical_channels = reactive.Value([])      #Reference list fr the tabel
    ##Below is the table for all the values the user can adjust . This table is loaded into the collapsable side bar
    params_df = reactive.Value(pd.DataFrame(columns=PARAM_COLUMNS))

    loading = reactive.Value(False)
    setting_selects = reactive.Value(False)
    syncing_controls = reactive.Value(False)
    data_loaded = reactive.Value(False)
    last_loaded_folder = reactive.Value("")  #This stores the last path used to load images so saving throws them into the same folder update here If you want this to cahnge
    mask_input_df = reactive.Value(pd.DataFrame())
    mask_files_df = reactive.Value(pd.DataFrame())
    mask_match_df = reactive.Value(pd.DataFrame())
    matched_masks_df = reactive.Value(pd.DataFrame())
    missing_masks_df = reactive.Value(pd.DataFrame())
    selected_mask_match = reactive.Value(pd.DataFrame())
   
    
    ## <----------------> Helper functions <-------------------> ##
    def _get_winsor_settings():
        """
        Reads the UI winsor settings, which is clamped to [0,1]. 
        Output is input for winsor_quantiles
        """
        do_w = bool(input.doWinsor())
        lo = clamp01(input.winsor_low())
        hi = clamp01(input.winsor_high())
        return do_w, lo, hi

    #Stores the values from the winzorization so you don't have to recompute every time you return to this channel.
    _global_range_cache = reactive.Value({})  ##(channel, kind, lo, hi) -> (gmin,gmax). eg ("CD45", "winsor", 0.01, 0.99): (12.3, 845.7)
    def _global_range_for_channel(images_dict: dict, channels_dict: dict,
                                channel_name: str, do_winsor: bool, lo: float, hi: float):
        """
        If winsor is enabled and hi>lo*: global range is min of per-image q_lo, max of per-image q_hi. This return the global high and low for a channel
        Otherwise: fallback (so if doWindor is FALSE) is to retturn the raw global min/max.
        *There is no check on this in the interface
        """
        ##key construct a cache key. If there is an exact match (so same channel, values and wins) in the global cache it'll pull it from there. If not it will compute it from there
        key = (channel_name, "winsor" if (do_winsor and hi > lo) else "raw",
            round(lo, 6), round(hi, 6))
        cache = _global_range_cache.get()
        if key in cache:
            return cache[key]
        ##If not in the cache, run the winsorization on the array:
        ##This is the CPU heavy part
        if do_winsor and hi > lo:
            # Compute global winsor range (min of per-sample q_lo, max of per-sample q_hi)
            result = compute_global_winsor_range_for_channel(images_dict, channels_dict, channel_name, lo, hi)
        ##Fallback if winsorization is off
        else:
            result = _global_minmax_for_channel(images_dict, channels_dict, channel_name)
        ##Stores it in the global cache and also return the results
        cache[key] = result
        _global_range_cache.set(cache)
        return result


    def _image_range_for_channel(images_dict: dict, channels_dict: dict,
                                channel_name: str, sample: str,
                                do_winsor: bool, lo: float, hi: float):
        """Per-image range for current sample; winsorized if enabled (and hi>lo), else return raw values."""
        ##Same of the previous one, but then per channel. No cache is stored as the per image winsor is not that heavy.
        if not sample or sample not in images_dict:
            return None
        chlist = channels_dict.get(sample, [])
        if channel_name not in chlist:
            return None
        idx = chlist.index(channel_name)
        arr = images_dict[sample][idx]
        if do_winsor and hi > lo:
            return compute_image_winsor_range(images_dict, channels_dict, channel_name, sample, lo, hi)
        # raw fallback
        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not (np.isfinite(mn) and np.isfinite(mx)):
            return None
        return mn, mx


    # File/folder dialogs were moved to HelperFiles.dialogs

    def _prefill_params(first_chlist: list[str]) -> None:
        """
        This is loaded after the _do_load() to prepare the dataframe with parameters
        """
        # Read the current UI values once and apply them to all channels
        s = float(input.noise_strength())
        p = strength_to_percentile(s)

        df = make_params_df(
            first_chlist,
            do_winsor=bool(input.doWinsor()),
            low=float(input.winsor_low()),
            high=float(input.winsor_high()),
            do_threshold=bool(input.doThreshold()),
            thr_val=float(input.thr_fraction_val()),
            do_abs_threshold=bool(input.doAbsThreshold()),
            abs_thr_val=float(input.abs_threshold_val()),
            do_noise=bool(input.doNoise()),
            noise_strength=s,
            noise_percentile=p,
            window_size=int(input.window_size()),
            do_norm=bool(input.doNorm()),
            norm_scope=(input.norm_scope() or "page"),
            do_asinh=bool(input.doAsinh()),
            cofactor=int(float(input.asinh_cofactor() or 5)),
        )
        params_df.set(df)

    def _sync_controls_from_table(channel: str) -> None:
        """
        This handles the defaults for the empty dataframe when loading the parameters
        """
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
        ##This is how each value is determines. Eg, doWinsor -> read out current value and return it. Fallback is FALSE
        try:
            session.send_input_message("doWinsor", {"value": bool(row.get("DoWinsor", False))})
            session.send_input_message("winsor_low", {"value": float(row.get("Low", 0.0))})
            session.send_input_message("winsor_high", {"value": float(row.get("High", 1.0))})

            session.send_input_message("doThreshold", {"value": bool(row.get("DoThr", False))})
            session.send_input_message("thr_fraction_val", {"value": float(row.get("ThrVal", 0.0))})

            session.send_input_message("doAbsThreshold", {"value": bool(row.get("DoAbsThr", False))})
            session.send_input_message("abs_threshold_val", {"value": float(row.get("AbsThrVal", 0.0))})

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
    
    # The following helpers have been moved to HelperFiles/*.
    # We keep these thin wrappers to minimize churn in the rest of viewer.py.
    def _cycle(lst, current, step):
        return cycle_list(lst, current, step)

    WINSOR_MIN_UPPER_BOUND = 5.0

    def _apply_winsor(cur: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
        return apply_winsor(cur, lo_q, hi_q, min_upper_bound=WINSOR_MIN_UPPER_BOUND)

    def _strength_to_percentile(s: float, eps: float = 0.005) -> float:
        return strength_to_percentile(s, eps=eps)

    def _apply_speckle_suppress(
        img: np.ndarray,
        size: int,
        perc: float,
        neighbor_limit: int = 2,
    ) -> np.ndarray:
        return apply_speckle_suppress(img, size=size, perc=perc, neighbor_limit=neighbor_limit)

    ##Cache for global min/max per channel (invalidated on load)
    ##Also stored previous values so moving back and forth is cached
    _global_minmax_cache = reactive.Value({})   ##looks like {channel_name: (gmin, gmax)}

    def _invalidate_global_cache():
        _global_minmax_cache.set({})
        _global_range_cache.set({})


    def _global_minmax_for_channel(images_dict: dict, channels_dict: dict, channel_name: str):
        """Return (gmin, gmax) across all samples for the given channel name.
        Uses & updates a reactive cache. Skips samples that lack this channel, 
        so it is compatible with samples with a different channel layout to harmonize samples
        """
        ##Checks cache first. This is invalidated on loading. This solved to problem that the cache has old values when loading images
        ##Invalid channel will return 0. Only needs 1 image in the channel to return something to return something relevant
        ##entry examples:
        ##  "CD45": (12.3, 845.7), 
        ##  "PanCK": (0.0, 501.2),
        ##  "FOXP3": None,
        cache = _global_minmax_cache.get()
        if channel_name in cache:
            return cache[channel_name]

        result = compute_global_minmax_for_channel(images_dict, channels_dict, channel_name)
        cache[channel_name] = result
        _global_minmax_cache.set(cache)
        return result

    def _get_channel_param_row(channelName: str):
        df = params_df.get()
        if df.empty or not channelName:
            return None
        m = (df["Channel"] == channelName)
        if not m.any():
            return None
        return df.loc[m].iloc[0]

    def _process_channel_from_table(sampleName: str, channelName: str) -> np.ndarray | None:
        imgs = images.get()
        chs = channels.get()

        if not sampleName or not channelName or sampleName not in imgs:
            return None

        chlist = chs.get(sampleName, [])
        if channelName not in chlist:
            return None

        idx = chlist.index(channelName)
        img = imgs[sampleName][idx, :, :].astype(np.float32)

        row = _get_channel_param_row(channelName)
        if row is None:
            return None

        doWin = bool(row.get("DoWinsor", False))
        low = float(row.get("Low", 0.0))
        high = float(row.get("High", 1.0))

        doAbsThr = bool(row.get("DoAbsThr", False))
        absThrVal = float(row.get("AbsThrVal", 0.0))

        doThr = bool(row.get("DoThr", False))
        thrVal = float(row.get("ThrVal", 0.0))

        doNoise = bool(row.get("Noise", False))
        noiseStrength = float(row.get("NStr", 0.0))
        winSz = int(row.get("WinSz", 3))

        doNorm = bool(row.get("DoNorm", True))
        normScope = str(row.get("NormScope", "page")).strip().lower()

        doAsinh = bool(row.get("DoAsinh", False))
        cofac = int(row.get("Cofac", 5))

        normVmin = None
        normVmax = None
        if doNorm and normScope == "global":
            gpair = _global_range_for_channel(
                images.get(),
                channels.get(),
                channelName,
                doWin,
                low,
                high,
            )
            if gpair is not None:
                normVmin, normVmax = gpair

        return process_image_pipeline(
            img,
            do_winsor=doWin,
            winsor_low=low,
            winsor_high=high,
            do_abs_threshold=doAbsThr,
            abs_threshold=absThrVal,
            do_fraction_threshold=doThr,
            thr_fraction=thrVal,
            do_noise=doNoise,
            noise_strength=noiseStrength,
            window_size=winSz,
            do_asinh=doAsinh,
            asinh_cofactor=cofac,
            do_norm=doNorm,
            norm_vmin=normVmin,
            norm_vmax=normVmax,
            winsor_min_upper_bound=WINSOR_MIN_UPPER_BOUND,
        )

    def _build_composite_rgb(sampleName: str | None = None):
        if sampleName is None:
            sampleName = input.sample()
        if not sampleName:
            return None, []

        rgb = None
        used = []

        for slotIdx in range(1, MAX_COMPOSITE_CHANNELS + 1):
            enabled = bool(getattr(input, f"comp_enable_{slotIdx}")())
            if not enabled:
                continue

            channelName = getattr(input, f"comp_channel_{slotIdx}")()
            colorName = getattr(input, f"comp_color_{slotIdx}")()
            gain = float(getattr(input, f"comp_gain_{slotIdx}")())

            if not channelName:
                continue

            proc = _process_channel_from_table(sampleName, channelName)
            if proc is None:
                continue

            if rgb is None:
                h, w = proc.shape
                rgb = np.zeros((h, w, 3), dtype=np.float32)

            colorVec = np.asarray(COMPOSITE_PALETTE.get(colorName, (1.0, 1.0, 1.0)), dtype=np.float32)
            layer = np.clip(proc[..., None] * gain * colorVec[None, None, :], 0.0, 1.0)

            # screen blend
            rgb = 1.0 - (1.0 - rgb) * (1.0 - layer)

            used.append((channelName, colorName, gain))

        if rgb is None or len(used) == 0:
            return None, []

        rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        return rgb, used

    def _sync_composite_channel_choices(sampleName: str, overwriteDefaults: bool = False) -> None:
        if not sampleName:
            return

        chlistCurrent = channels.get().get(sampleName, [])
        canon = canonical_channels.get() or []
        ordered = order_by_canonical(canon, chlistCurrent)

        choices = [""] + ordered

        for slotIdx in range(1, MAX_COMPOSITE_CHANNELS + 1):
            inputId = f"comp_channel_{slotIdx}"

            currentVal = getattr(input, inputId)()
            defaultVal = ordered[slotIdx - 1] if slotIdx <= len(ordered) else ""

            if overwriteDefaults:
                selectedVal = defaultVal
            else:
                selectedVal = currentVal if currentVal in ordered else defaultVal

            ui.update_select(
                inputId,
                choices=choices,
                selected=selectedVal,
                session=session,
            )

    def _get_selected_creator_channels():
        selected = []
        for slotIdx in range(1, MAX_COMPOSITE_CHANNELS + 1):
            enabled = bool(getattr(input, f"comp_enable_{slotIdx}")())
            if not enabled:
                continue

            channelName = getattr(input, f"comp_channel_{slotIdx}")()
            if not channelName:
                continue

            colorName = getattr(input, f"comp_color_{slotIdx}")()
            gain = float(getattr(input, f"comp_gain_{slotIdx}")())
            selected.append((channelName, colorName, gain))

        return selected

    @reactive.Effect
    @reactive.event(input.save_composite_tiff)
    def _save_composite_tiff():
        rgb, used = _build_composite_rgb()
        if rgb is None or len(used) == 0:
            print("⚠️ No composite available to save.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        sampleName = input.sample() or "sample"
        defaultName = f"{sampleName} Composite.tiff"

        savePath = pick_save_tiff_dialog(
            initialdir=initdir,
            initialfile=defaultName,
        )

        if not savePath:
            print("🛑 Save canceled.")
            return

        try:
            rgb_u16 = np.clip(np.round(rgb * 65535.0), 0, 65535).astype(np.uint16)
            imwrite(
                savePath,
                rgb_u16,
                dtype=np.uint16,
                photometric="rgb",
            )
            print(f"✅ Saved composite TIFF → {savePath}")
        except Exception as e:
            print(f"❌ Failed to save composite TIFF: {e}")

    @reactive.Effect
    @reactive.event(input.export_creator_composites_all)
    def _export_creator_composites_all():
        imgs = images.get()
        if not imgs:
            print("⚠️ No images loaded.")
            return

        selected = _get_selected_creator_channels()
        if len(selected) == 0:
            print("⚠️ No creator channels selected.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        if not folder or not os.path.isdir(folder):
            print("⚠️ Invalid input folder.")
            return

        out_dir = os.path.join(folder, "creator composite exports")
        os.makedirs(out_dir, exist_ok=True)

        # Save creator selection metadata for reproducibility
        try:
            creator_meta = pd.DataFrame(selected, columns=["Channel", "Color", "Gain"])
            creator_meta.to_csv(os.path.join(out_dir, "creator_selection.csv"), index=False)
        except Exception as e:
            print(f"⚠️ Could not save creator_selection.csv: {e}")

        for sampleName in imgs.keys():
            rgb, used = _build_composite_rgb(sampleName=sampleName)

            if rgb is None or len(used) == 0:
                print(f"⚠️ Skipped {sampleName}: no selected creator channels were present.")
                continue

            out_path = os.path.join(out_dir, f"{sampleName} Composite.tiff")

            try:
                rgb_u16 = np.clip(np.round(rgb * 65535.0), 0, 65535).astype(np.uint16)
                imwrite(
                    out_path,
                    rgb_u16,
                    dtype=np.uint16,
                    photometric="rgb",
                )
                print(f"✅ Exported composite TIFF → {out_path}")
            except Exception as e:
                print(f"❌ Failed for {sampleName}: {e}")

        print("✅ Finished exporting composite TIFFs for all images.")

    @reactive.Effect
    @reactive.event(input.fill_composite_from_current)
    def _fill_composite_from_current():
        s = input.sample()
        if not s:
            return
        _sync_composite_channel_choices(s, overwriteDefaults=True)

    @output
    @render.ui
    def composite_summary():
        rgb, used = _build_composite_rgb()
        if not used:
            return ui.tags.small("No channels selected for the composite.", class_="text-muted")

        lines = [
            ui.tags.li(f"{ch} — {col} × {gain:.1f}")
            for ch, col, gain in used
        ]
        return ui.div(
            ui.tags.small("Composite uses the stored per-channel processing settings.", class_="text-muted"),
            ui.tags.ul(*lines, class_="mt-2 mb-0"),
        )

    @reactive.Effect
    @reactive.event(input.save_composite_png)
    def _save_composite_png():
        rgb, used = _build_composite_rgb()
        if rgb is None or len(used) == 0:
            print("⚠️ No composite available to save.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        sampleName = input.sample() or "sample"
        defaultName = f"{sampleName} Composite.png"

        savePath = pick_save_png_dialog(
            initialdir=initdir,
            initialfile=defaultName,
        )

        if not savePath:
            print("🛑 Save canceled.")
            return

        try:
            plt.imsave(savePath, rgb)
            print(f"✅ Saved composite PNG → {savePath}")
        except Exception as e:
            print(f"❌ Failed to save composite PNG: {e}")

    ##<--------load images module ---------->
    @reactive.Effect
    @reactive.event(input.load)
    def _do_load():
        ##If loading is already in progress don't try to start again!
        if loading.get():
            return
        loading.set(True) 
        try:
            ##Try the text box first; if empty, open the dialog. 
            ##So if you paste the path into the box it will not open the window ad just start loading
            folder = (input.path() or "").strip()
            if not folder:
                ##So if the box is empty or not a folder
                try:
                    folder = pick_folder_dialog()
                except Exception:
                    folder = ""
            ##If the loader also isn't valid, return this:
            if not folder:
                print("🛑 Load canceled (no folder selected).")
                return

            ##Purely for visibilty, the actual folder path used is put into the text box
            session.send_input_message("path", {"value": folder})
            print(">>> Loading triggered with folder:", folder)
            ##Loads the OME.tiff, also checks if channels are consistent and throws an error if one of the images if different from the first one
            ##Error from load_tiffs_raw is passed on to the interface
            try:
                imgs, chs = load_tiffs_raw(folder)  # ideally load_tiffs_raw(folder, validate_consistent=True)
            except ValueError as e:
                ui.modal_show(
                    ui.modal(
                        ui.h4("Channel mismatch..."),
                        ui.pre(str(e)),
                        easy_close=True,
                        footer=ui.modal_button("OK"),
                    ),
                    session=session,
                )
                return
            ##If the folder didn't contain any images:
            if not imgs:
                print("⚠️ No images found in selected folder.")
                return

            ##Store images and channels
            images.set(imgs)
            channels.set(chs)
            ##Clear the cache. this prevends old settings from similar channal names to remain in memory and prevent from actually processing the images for the viewer
            _invalidate_global_cache()
            ##Picks the first images as the default display.
            samples = list(imgs.keys())
            first_sample = samples[0]
            first_chlist = chs[first_sample]
            first_channel = first_chlist[0] if first_chlist else None

            #set the channel list for the left sided table and fill it will the channels. Each chanel gets a row.
            canonical_channels.set(list(first_chlist))
            _prefill_params(first_chlist)

            ##update selects under guard, selects first image
            setting_selects.set(True)
            try:
                ui.update_select("sample", choices=samples, selected=first_sample, session=session)
                if first_channel:
                    ui.update_select("channel", choices=first_chlist, selected=first_channel, session=session)
            finally:
                setting_selects.set(False)

            _sync_composite_channel_choices(first_sample, overwriteDefaults=True)

            if first_channel:
                _sync_controls_from_table(first_channel)

            ##Tell user that everything is loaded and save the foldername for further use
            data_loaded.set(True)
            last_loaded_folder.set(folder)

        finally:
            loading.set(False)

    # ---------- react to sample change ----------
    @reactive.Effect
    @reactive.event(input.next_sample)
    def _next_sample():
        ##if loading in progress or images are not set, abort.
        if loading.get() or not images.get():
            return
        samples = list(images.get().keys())
        cur = input.sample() or (samples[0] if samples else None)
        ##Do samplenr++
        nxt = _cycle(samples, cur, +1)
        if nxt:
            ##Don’t set setting_selects here; we want _on_sample_change to run
            ui.update_select("sample", choices=samples, selected=nxt, session=session)

    @reactive.Effect
    @reactive.event(input.prev_sample)
    def _prev_sample():
        ##if loading in progress or images are not set, abort.
        if loading.get() or not images.get():
            return
        ##get list of samples
        samples = list(images.get().keys())
        ##current sample select (or first of none selected)
        cur = input.sample() or (samples[0] if samples else None)
        ##previous sample, uses cycler to wrap around
        prv = _cycle(samples, cur, -1)
        ##If valid selected, update the sample dropdown menu with current sample
        if prv:
            ui.update_select("sample", choices=samples, selected=prv, session=session)


    @reactive.Effect
    @reactive.event(input.sample)
    def _on_sample_change():
        ##again, guard against changing while loading
        if loading.get() or setting_selects.get():
            return
        ##Which sample selected? if none, bail
        s = input.sample()
        if not s:
            return
        chlist_current = channels.get().get(s, [])
        if not chlist_current:
            return
        ##reorder using cannonical channels, this fixed a problem were channels were ordered differently between samples (total channels was the same)
        canon = canonical_channels.get()
        ordered = order_by_canonical(canon, chlist_current)
        sel = input.channel()
        if sel not in ordered:
            sel = ordered[0]
        setting_selects.set(True)
        try:
            ui.update_select("channel", choices=ordered, selected=sel, session=session)
        finally:
            setting_selects.set(False)
        _sync_controls_from_table(sel)
        _sync_composite_channel_choices(s, overwriteDefaults=False)

    # ---------- react to channel change ----------
    ##Below reactive effect work the sample as their sample counterpart, check comments there on how they work
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
        ordered = order_by_canonical(canon, chlist_current)
        cur = input.channel() or ordered[0]
        nxt = _cycle(ordered, cur, +1)
        if nxt:
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
        ordered = order_by_canonical(canon, chlist_current)
        cur = input.channel() or ordered[0]
        prv = _cycle(ordered, cur, -1)
        if prv:
            ui.update_select("channel", choices=ordered, selected=prv, session=session)

    @reactive.Effect
    @reactive.event(input.channel)
    def _on_channel_change():
        if loading.get() or setting_selects.get():
            return
        c = input.channel()
        if not c:
            return
        _sync_controls_from_table(c)

    # <---------- update parameter table ---------->
    ##Below are the effects that tie the buttons to the parameter dataframe
    ##Important not is that i switched from immediately updating the parametern dataframe to making a copy and then applying
    ##This solved a bug where the image viewer seemed to respond to a direct change in the parameter table?
    ##Incidently, this will also prevent future functions that are tied to the parameter table from introducing weird behaviour 
    ##associated with reading the table and mutating the original later. 
    ##All buttons work generally the same way. 
    ##Button trigger reading of the input channel, followed by a pull of the parameter table which is the copied
    ##The user input is read, written to the copied dataframe, which in turn is written back to the params_df

    ##Winsorization
    @reactive.Effect
    @reactive.event(input.apply_one)
    def _apply_one_channel():
        ##Prevents feedback between application of change and the change of the UI
        if syncing_controls.get():
            return
        ##read input channel, if non exit
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        params_df.set(
            update_channel_row(
                df,
                c,
                {
                    "DoWinsor": bool(input.doWinsor()),
                    "Low": float(input.winsor_low()),
                    "High": float(input.winsor_high()),
                },
            )
        )

    ##Thresholding
    @reactive.Effect
    @reactive.event(input.apply_threshold)
    ##This connects the update channel threshold value button to the parameter table
    def _apply_threshold_channel():
        ##This prevents feedbackloop between the update table and "UI changed" action
        if syncing_controls.get():
            return
        #Select input channel, if none, exit
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        # Fraction-of-max threshold (0..1)
        try:
            thr_val = float(input.thr_fraction_val())
        except Exception:
            thr_val = 0.0
        thr_val = clamp01(thr_val)

        # Absolute threshold (counts)
        try:
            abs_thr_val = float(input.abs_threshold_val())
        except Exception:
            abs_thr_val = 0.0

        params_df.set(
            update_channel_row(
                df,
                c,
                {
                    "DoAbsThr": bool(input.doAbsThreshold()),
                    "AbsThrVal": float(abs_thr_val),
                    "DoThr": bool(input.doThreshold()),
                    "ThrVal": float(thr_val),
                },
            )
        )

    ##Noise removal
    @reactive.Effect
    @reactive.event(input.apply_noise)
    def _apply_noise_channel():
        ##Same thing as the previous ones
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        s = float(input.noise_strength())
        p = strength_to_percentile(s)
        params_df.set(
            update_channel_row(
                df,
                c,
                {
                    "Noise": bool(input.doNoise()),
                    "NStr": s,
                    "NPrctl": p,
                    "WinSz": int(input.window_size()),
                },
            )
        )

    ##Normalization and transformation
    @reactive.Effect
    @reactive.event(input.apply_norm)
    def _apply_options_channel():
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        try:
            cofac = int(float(input.asinh_cofactor()))
        except Exception:
            cofac = 5
        cofac = max(2, min(10, cofac))
        params_df.set(
            update_channel_row(
                df,
                c,
                {
                    "DoNorm": bool(input.doNorm()),
                    "NormScope": (input.norm_scope() or "page"),
                    "DoAsinh": bool(input.doAsinh()),
                    "Cofac": cofac,
                },
            )
        )

    # <---------- plot ---------->
    @output
    @render.ui
    def main_viewer_ui():
        mode = input.viewer_mode() or "PINT"

        if mode in ["PINT", "Image creator"]:
            return ui.tags.div(
                ui.output_plot("img_viewer", fill=True, height="100%"),
                class_="viewer-fill",
            )

        return ui.tags.div()
    
    ##Below is the actual img viewer where the image is rendered and all the different thresholding stept are visualized.
    @output
    @render.plot
    def img_viewer():
        try:
            mode = input.viewer_mode() or "PINT"
            ## -------------------- IMAGE CREATOR --------------------
            if mode == "Image creator":
                fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
                rgb, used = _build_composite_rgb()
                if rgb is None:
                    ax.text(0.5, 0.5, "No composite channels selected", ha="center", va="center")
                    ax.set_axis_off()
                    return fig

                ax.imshow(rgb)
                ax.set_axis_off()

                if used:
                    labelLines = [f"{ch} — {col} × {gain:.1f}" for ch, col, gain in used]
                    fig.text(
                        0.01,
                        0.99,
                        "\n".join(labelLines),
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4),
                    )

                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                return fig

            ## -------------------- DEFAULT PINT SINGLE-CHANNEL VIEWER --------------------
            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

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

            normVmin = None
            normVmax = None
            if bool(input.doNorm()) and (input.norm_scope() or "page") == "global":
                doW, lo, hi = _get_winsor_settings()
                gpair = _global_range_for_channel(
                    images.get(),
                    channels.get(),
                    c,
                    doW,
                    lo,
                    hi,
                )
                if gpair is not None:
                    normVmin, normVmax = gpair

            img = process_image_pipeline(
                img,
                do_winsor=bool(input.doWinsor()),
                winsor_low=float(input.winsor_low()),
                winsor_high=float(input.winsor_high()),
                do_abs_threshold=bool(input.doAbsThreshold()),
                abs_threshold=float(input.abs_threshold_val()),
                do_fraction_threshold=bool(input.doThreshold()),
                thr_fraction=float(input.thr_fraction_val()),
                do_noise=bool(input.doNoise()),
                noise_strength=float(input.noise_strength()),
                window_size=int(input.window_size()),
                do_asinh=bool(input.doAsinh()),
                asinh_cofactor=int(float(input.asinh_cofactor() or 5)),
                do_norm=bool(input.doNorm()),
                norm_vmin=normVmin,
                norm_vmax=normVmax,
                winsor_min_upper_bound=WINSOR_MIN_UPPER_BOUND,
            )

            ax.imshow(img, cmap="gray")
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

        except Exception as e:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            return fig
        

    def _sync_mask_column_choices(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        colNames = list(df.columns)

        def pick_first(existingNames: list[str], fallback=None):
            for name in existingNames:
                if name in colNames:
                    return name
            return fallback

        default = colNames[0] if colNames else None

        ui.update_select(
            "mask_cell_id_col",
            choices=colNames,
            selected=pick_first(["CellName", "ObjectNumber", "Identifier"], default),
            session=session,
        )
        ui.update_select(
            "mask_x_col",
            choices=colNames,
            selected=pick_first(["Location_Center_X", "Center_X", "X"], default),
            session=session,
        )
        ui.update_select(
            "mask_y_col",
            choices=colNames,
            selected=pick_first(["Location_Center_Y", "Center_Y", "Y"], default),
            session=session,
        )
        ui.update_select(
            "mask_name_col",
            choices=colNames,
            selected=pick_first(["CellMaskName", "ROIName", "MaskName"], default),
            session=session,
        )
        ui.update_select(
            "mask_cluster_col",
            choices=colNames,
            selected=pick_first(
                ["MergedSublineage", "MergedSubsetNames", "CellClusterNames", "Cluster", "ClusterName"],
                default,
            ),
            session=session,
        )
        ui.update_select(
            "mask_condition_col",
            choices=colNames,
            selected=pick_first(["Condition"], default),
            session=session,
        )
        ui.update_select(
            "mask_sample_col",
            choices=colNames,
            selected=pick_first(["SampleNumber", "SampleName"], default),
            session=session,
        )


    @output
    @render.plot
    def mask_viewer():
        fig = plt.figure(figsize=(12, 8), dpi=120)

        try:
            sel = selected_mask_match.get()
            inputDf = mask_input_df.get()

            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.02)
            axImg = fig.add_subplot(gs[0, 0])
            axLeg = fig.add_subplot(gs[0, 1])

            if sel is None or sel.empty:
                axImg.text(0.5, 0.5, "No matched mask selected", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            if inputDf is None or inputDf.empty:
                axImg.text(0.5, 0.5, "No cell table loaded", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            maskNameCol = input.mask_name_col() or "CellMaskName"
            xCol = input.mask_x_col()
            yCol = input.mask_y_col()
            clusterCol = input.mask_cluster_col()
            scaleFactor = int(input.mask_scale_factor() or 5)

            if not xCol or not yCol or not clusterCol or not maskNameCol:
                axImg.text(0.5, 0.5, "Please select mask, X, Y and cluster columns", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            if clusterCol not in inputDf.columns:
                axImg.text(0.5, 0.5, f"Cluster column '{clusterCol}' not found", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            row = sel.iloc[0]
            maskName = str(row[maskNameCol])
            maskPath = row.get("MaskPath", None)

            if not maskPath or pd.isna(maskPath):
                axImg.text(0.5, 0.5, "Selected mask has no valid file path", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            try:
                cellMask = read_mask_tiff(maskPath)
            except Exception as e:
                axImg.text(0.5, 0.5, f"Failed to read mask:\n{e}", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            matchingData = get_cells_for_mask_name(
                inputDf,
                mask_name=maskName,
                mask_name_col=maskNameCol,
            )

            if matchingData.empty:
                axImg.text(0.5, 0.5, "No cell rows found for selected mask", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            try:
                matchedData = match_mask_centroids_to_cells(
                    cellMask=cellMask,
                    matchingData=matchingData,
                    xCol=xCol,
                    yCol=yCol,
                )
            except Exception as e:
                axImg.text(0.5, 0.5, f"Matching failed:\n{e}", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            try:
                plotData = make_mask_plot_data(
                    cellMask=cellMask,
                    matchedData=matchedData,
                    clusterCol=clusterCol,
                    scaleFactor=scaleFactor,
                    background="white",
                    borderColor="white",
                    missingColor="#808080",
                )
            except Exception as e:
                axImg.text(0.5, 0.5, f"Plot construction failed:\n{e}", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            axImg.imshow(plotData["colorMat"], interpolation="nearest")
            axImg.set_axis_off()

            clusterColors = plotData["clusterColors"]
            handles = [
                Patch(facecolor=clusterColors[name], edgecolor="none", label=name)
                for name in sorted(clusterColors.keys())
            ]

            axLeg.set_axis_off()

            if len(handles) > 0:
                legendFontSize = 8
                if len(handles) <= 12:
                    legendFontSize = 10
                elif len(handles) >= 30:
                    legendFontSize = 6

                axLeg.legend(
                    handles=handles,
                    loc="upper left",
                    frameon=False,
                    fontsize=legendFontSize,
                    borderaxespad=0.0,
                )

            fig.suptitle(
                f"{maskName} | {matchedData.shape[0]} masks matched",
                fontsize=10,
                y=0.995,
            )
            plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
            return fig

        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            return fig


    @output
    @render.ui
    ##This is the small tooltip that gived information about how the normalization actually workd
    def noise_tooltip():
        s = float(input.noise_strength())
        p = strength_to_percentile(s)
        p_txt = f"P{p*100:.1f}"
        long = (
            f"Pixels above their local {p_txt} are marked ‘bright.’ "
            "A pixel is removed only if ≤ 2 of its 8 neighbors (not scaled with window size) are also bright. "
            "Increase sensitivity to flag more pixels as bright (more aggressive denoising). "
            "Decrease to preserve detail in bright patches."
        )
        ##Small, muted line with a hover tooltip and a compact readout
        return ui.tags.small(
            ui.tags.span("ℹ️ ", class_="me-1"),
            f"Local cutoff ≈ {p_txt}",
            title=long,
            class_="text-muted"
        )

    @output
    @render.table
    ##This is the table that is actually used to render the big parameter table on the left side
    def param_table():
        return format_for_display(params_df.get())
    
    @reactive.Effect
    @reactive.event(input.perform_analysis)
    ##The moment you hit "analyze" this will popup to give the user a chance to cancel if it was by accident
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
    @reactive.event(input.confirm_start)
    ##This will run the batch of images through he analysis.py pipeline
    def _run_batch_analysis():
        #Close the modal (the yes i'm sure button), see above block
        ui.modal_remove(session=session)
        ##reads the folder path, uses the input folder, checks if its non-empty and it exist. Strips all the white space.
        ##Throws an exception if not
        folder = (input.path() or "").strip()
        if not folder or not os.path.isdir(folder):
            print(f"⚠️ Invalid folder: {folder!r}")
            return
        ##Create new folder inside of the input one.
        out_dir = os.path.join(folder, "normalized images")
        os.makedirs(out_dir, exist_ok=True)
        ##Makes an output path for the parameter file to save it
        ##If the table is empty it will return a messege to tell the user
        params_path = os.path.join(out_dir, "parameter_table.csv")
        df = params_df.get().copy()
        if df.empty:
            print("⚠️ Parameter table is empty — nothing to analyze.")
            return
        df.to_csv(params_path, index=False)
        print(f"✅ Saved parameter table → {params_path}")
        ##Tries to open de analysis.py file from the same folder as the viewer.
        cmd = [
            sys.executable, "-m", "pint_app.core.analysis",
            "--input-dir", folder,
            "--params-csv", params_path,
            "--output-dir", out_dir,
        ]   
        print("▶️ Running analysis:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print("✅ Analysis finished.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis script failed: {e}")
    
    @output
    @render.ui
    def norm_scope_hint():
        ##If nothing loaded, exit
        if not images.get() or not input.channel():
            return ui.tags.small(" ")
        ##If not empty, but not set to global, also return nothing
        if (input.norm_scope() or "page") != "global":
            return ui.tags.small(" ")
        ##Reads the current setting winsor quantile etc
        ch = input.channel()
        do_w = bool(input.doWinsor())
        lo   = max(0.0, min(1.0, float(input.winsor_low())))
        hi   = max(0.0, min(1.0, float(input.winsor_high())))
        ##looks across all samples to get thehighest and the lowest numbers. This is the global max/min for per image winsorization
        gpair = _global_range_for_channel(images.get(), channels.get(), ch, do_w, lo, hi)
        if not gpair:
            return ui.tags.small("Global range: NO VALID RANGE")
        gmin, gmax = gpair

        try:
            s = input.sample()
        except Exception:
            s = None
        ##This is the image winsorization settings 
        ipair = _image_range_for_channel(images.get(), channels.get(), ch, s, do_w, lo, hi)

        parts = [
            ui.tags.small(
                f'Global range: “{ch}” (winsor {"on" if (do_w and hi>lo) else "off"}): '
                f'[{fmt1(gmin)}, {fmt1(gmax)}]'
            )
        ]
        ##Note that the range is the min/max range per channel but the winsorization is per image. If winsorization if turned off there it's just glabal min max
        if ipair:
            imin, imax = ipair
            label = f'Image range: “{s}” range' if s else "Image range"
            parts += [ui.br(), ui.tags.small(f"{label}: [{fmt1(imin)}, {fmt1(imax)}]")]
        return ui.div(*parts)

    @reactive.Effect
    @reactive.event(input.export_params)
    ##Export function for the parameter file
    def _export_params():
        ##copy of the parameter table, check if empty. If so, warn the user
        df = params_df.get().copy()
        if df.empty:
            print("⚠️ No parameters to export.")
            return
        ##The input folder is used as default folder in the save as thats called below it
        folder = (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()
        save_path = pick_save_csv_dialog(initialdir=initdir, initialfile="parameter_table.csv")
        ##Save or not saved warnings
        if not save_path:
            print("🛑 Export canceled.")
            return

        try:
            df.to_csv(save_path, index=False)
            print(f"✅ Exported parameters → {save_path}")
                 
        except Exception as e:
            print(f"❌ Failed to write CSV: {e}")

 
    @reactive.Effect
    @reactive.event(input.import_params)
    def _import_params():
        if not data_loaded.get():
            print("⚠️ Load images before importing parameters.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        csv_path = pick_open_csv_dialog(initialdir=initdir)
        if not csv_path:
            print("🛑 Import canceled.")
            return

        try:
            df_in = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Failed to read CSV: {e}")
            return

        if "Channel" not in df_in.columns:
            print("❌ CSV missing required 'Channel' column.")
            return

        canon = list(canonical_channels.get() or [])
        try:
            df_in = validate_and_normalize_import(df_in, canon)
        except ValueError as e:
            print(f"❌ {e}")
            return

        params_df.set(df_in.reset_index(drop=True))

        sel = input.channel()
        if sel:
            _sync_controls_from_table(sel)

        s = input.sample()
        if s:
            _sync_composite_channel_choices(s, overwriteDefaults=False)

        print(f"✅ Imported parameters from {csv_path}")

    @reactive.Effect
    @reactive.event(input.browse_mask_path)
    def _browse_mask_path():
        try:
            folder = pick_folder_dialog()
        except Exception:
            folder = ""

        if not folder:
            return

        session.send_input_message("mask_path", {"value": folder})

    @reactive.Effect
    @reactive.event(input.match_masks)
    def _match_masks():
        df = mask_input_df.get()

        if df is None or df.empty:
            print("⚠️ No cell table available for mask matching. Load a mask CSV first.")
            mask_files_df.set(pd.DataFrame())
            mask_match_df.set(pd.DataFrame())
            matched_masks_df.set(pd.DataFrame())
            missing_masks_df.set(pd.DataFrame())
            return

        mask_dir = (input.mask_path() or "").strip()
        if not mask_dir or not os.path.isdir(mask_dir):
            print("⚠️ Invalid mask folder.")
            mask_files_df.set(pd.DataFrame())
            mask_match_df.set(pd.DataFrame())
            matched_masks_df.set(pd.DataFrame())
            missing_masks_df.set(pd.DataFrame())
            return

        cell_id_col = input.mask_cell_id_col()
        x_col = input.mask_x_col()
        y_col = input.mask_y_col()
        mask_name_col = input.mask_name_col()

        try:
            df_valid = validate_mask_input_table(
                df,
                cell_id_col=cell_id_col,
                x_col=x_col,
                y_col=y_col,
                mask_name_col=mask_name_col,
            )

            files_df = list_mask_files(mask_dir)
            match_df = match_cellmask_names_to_files(
                df_valid,
                files_df,
                mask_name_col=mask_name_col,
            )
            matched_df, missing_df = split_mask_matches(match_df)

        except Exception as e:
            print(f"❌ Mask matching failed: {e}")
            mask_files_df.set(pd.DataFrame())
            mask_match_df.set(pd.DataFrame())
            matched_masks_df.set(pd.DataFrame())
            missing_masks_df.set(pd.DataFrame())
            return

        mask_input_df.set(df_valid)
        mask_files_df.set(files_df)
        mask_match_df.set(match_df)
        matched_masks_df.set(matched_df)
        missing_masks_df.set(missing_df)

        selected_choices = list(matched_df[mask_name_col].astype(str)) if not matched_df.empty else []
        selected_default = selected_choices[0] if selected_choices else None

        ui.update_select(
            "selected_mask_name",
            choices=selected_choices,
            selected=selected_default,
            session=session,
        )

        print(f"✅ Mask matching complete: {len(matched_df)} matched, {len(missing_df)} missing.")

    @reactive.Effect
    @reactive.event(input.browse_mask_csv)
    def _browse_mask_csv():
        csv_path = pick_open_csv_dialog(initialdir=os.getcwd())
        if not csv_path:
            return
        session.send_input_message("mask_csv_path", {"value": csv_path})

    @reactive.Effect
    @reactive.event(input.load_mask_csv)
    def _load_mask_csv():
        csv_path = (input.mask_csv_path() or "").strip()

        if not csv_path or not os.path.isfile(csv_path):
            print("⚠️ Invalid mask CSV path.")
            mask_input_df.set(pd.DataFrame())
            return

        try:
            df_mask = pd.read_csv(csv_path, index_col=0)
        except Exception as e:
            print(f"❌ Failed to read mask CSV: {e}")
            mask_input_df.set(pd.DataFrame())
            return

        if df_mask.empty:
            print("⚠️ Mask CSV is empty.")
            mask_input_df.set(pd.DataFrame())
            return

        mask_input_df.set(df_mask)
        _sync_mask_column_choices(df_mask)
        print(f"✅ Loaded mask cell table → {csv_path}")

    @output
    @render.ui
    def selected_mask_summary():
        sel = selected_mask_match.get()
        df = mask_input_df.get()

        if sel is None or sel.empty:
            return ui.tags.small("No matched mask selected.", class_="text-muted")

        row = sel.iloc[0]
        maskNameCol = input.mask_name_col() or "CellMaskName"
        clusterCol = input.mask_cluster_col()

        maskName = str(row[maskNameCol])
        nCells = 0
        nClusters = 0

        if df is not None and not df.empty and maskNameCol in df.columns:
            tempDf = df.loc[df[maskNameCol].astype(str) == maskName].copy()
            nCells = tempDf.shape[0]
            if clusterCol and clusterCol in tempDf.columns:
                nClusters = tempDf[clusterCol].astype(str).nunique()

        return ui.div(
            ui.tags.p(f"Selected mask: {maskName}"),
            ui.tags.p(f"Mask file: {row.get('MaskFile', '—')}"),
            ui.tags.p(f"Rows in cell table: {nCells}"),
            ui.tags.p(f"Clusters present: {nClusters}"),
        )

    @output
    @render.ui
    def mask_table_summary():
        df = mask_input_df.get()

        if df is None or df.empty:
            return ui.tags.small("No cell table loaded yet.", class_="text-muted")

        return ui.div(
            ui.tags.p(f"Rows in cell table: {len(df)}"),
            ui.tags.p(f"Columns in cell table: {len(df.columns)}"),
        )

    @output
    @render.ui
    def mask_match_summary():
        match_df = mask_match_df.get()
        matched_df = matched_masks_df.get()
        missing_df = missing_masks_df.get()

        if match_df is None or match_df.empty:
            return ui.tags.small("No mask matching performed yet.", class_="text-muted")

        return ui.div(
            ui.tags.p(f"Unique mask names in table: {len(match_df)}"),
            ui.tags.p(f"Matched mask files: {len(matched_df)}"),
            ui.tags.p(f"Missing mask files: {len(missing_df)}"),
        )

    @output
    @render.table
    def mask_match_table():
        df = mask_match_df.get()

        if df is None or df.empty:
            return pd.DataFrame(columns=["CellMaskName", "MaskFile", "MaskExists"])

        return df
    
    @reactive.Effect
    @reactive.event(input.selected_mask_name)
    def _on_selected_mask_name():
        selected_name = input.selected_mask_name()
        match_df = matched_masks_df.get()

        if match_df is None or match_df.empty or not selected_name:
            selected_mask_match.set(pd.DataFrame())
            return

        mask_name_col = input.mask_name_col() or "CellMaskName"
        sel = match_df.loc[match_df[mask_name_col].astype(str) == str(selected_name)].copy()
        selected_mask_match.set(sel)

    @output
    @render.ui
    def mask_visualization_placeholder():
        sel = selected_mask_match.get()

        if sel is None or sel.empty:
            return ui.tags.div(
                ui.tags.p("No matched mask selected."),
                class_="text-muted"
            )

        row = sel.iloc[0]
        return ui.tags.div(
            ui.tags.p("Visualization area reserved for mask rendering."),
            ui.tags.p(f"Selected file: {row.get('MaskFile', '—')}"),
            class_="text-muted"
        )


app = App(app_ui, server)