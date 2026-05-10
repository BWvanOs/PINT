from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from shiny import App, ui, render, reactive
from shiny.types import SilentException

from scipy.ndimage import grey_opening, uniform_filter, median_filter
import os, sys, subprocess
import shutil
import warnings
from datetime import datetime

from tifffile import imwrite, imread

from pint_app.core.load_tiffs import load_tiffs_raw
from pint_app.core.formatting import fmt1
from pint_app.core.mask_neighbors import build_touching_edges_for_pushed_dataset
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

from pint_app.core.mask_neighbors_stats import (
    chance_correct_touching_interactions,
    make_sample_interaction_matrix,
    permanova_one_factor,
    aggregate_interaction_matrix,
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
COMPOSITE_EMPTY_CHOICE = ">> Leave blank <<"

def make_composite_slot(slotIdx: int):
    defaultColor = COMPOSITE_COLOR_CHOICES[(slotIdx - 1) % len(COMPOSITE_COLOR_CHOICES)]

    return ui.row(
        ui.column(
            7,
            ui.input_select(
                f"comp_channel_{slotIdx}",
                "",
                choices=[COMPOSITE_EMPTY_CHOICE],
                selected=COMPOSITE_EMPTY_CHOICE,
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
            ui.input_numeric(
                f"comp_gain_{slotIdx}",
                "",
                value=1.0,
                min=0.0,
                max=20.0,
                step=0.1,
                width="100%",
            ),
        ),
        class_="align-items-end gy-0 creator-slot-row",
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
            .controls-left .card {
                border: 1px solid #8f8f8f;
                box-shadow: 0 0.15rem 0.45rem rgba(0, 0, 0, 0.18);
            }
                      
            .pint-main-layout {
                display: flex;
                gap: 0.75rem;
                width: 100%;
                height: calc(100vh - 120px);
                min-height: 0;
            }

            .controls-left {
                flex: 0 0 500px;
                width: 500px;
                max-width: 500px;
                height: 100%;
                overflow-y: auto;
                padding-right: 0.25rem;
            }

            .controls-left .card-header {
                border-bottom: 1px solid #8f8f8f;
                background-color: #e9ecef;
            }          

            .controls-left hr {
                border: 0;
                border-top: 1px solid #8f8f8f;
                opacity: 1;
                margin: 0.6rem 0;
            }

            .viewer-navigator {
                flex: 0 0 auto;
                margin-bottom: 0.4rem;
            }
                      
            .viewer-main {
                flex: 1 1 auto;
                min-width: 0;
                height: 100%;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }

            .viewer-navigator .shiny-input-container {
                margin-bottom: 0 !important;
            }

            .viewer-navigator label {
                margin-bottom: 0.15rem;
                font-size: 0.85rem;
                font-weight: 600;
            }

            .viewer-navigator .btn {
                margin-bottom: 0 !important;
            }

            .viewer-plot-fill {
                flex: 1 1 auto;
                min-height: 0;
                display: flex;
                flex-direction: column;
            }
            
            .creator-header-row {
                padding-left: 0.15rem;
                padding-right: 0.15rem;
            }

            .creator-slot-row {
                margin-bottom: 0.15rem;
            }

            .creator-slot-row .shiny-input-container {
                margin-bottom: 0.2rem !important;
            }

            .creator-slot-row select,
            .creator-slot-row input {
                min-height: 32px;
            }
                      
            .viewer-plot-fill .shiny-plot-output {
                flex: 1 1 auto;
                height: 100% !important;
            }
                      
            /* Sidebar & parameter table */
            .sidebar-col {
                display: flex;
                flex-direction: column;
                height: 100%;
            }

            .param-table-wrap table {
                font-size: 12px;
                width: 100% !important;
                table-layout: auto;
                border-collapse: collapse;
            }

            .param-table-wrap td,
            .param-table-wrap th {
                padding: 2px 4px;
                white-space: nowrap;
                text-overflow: ellipsis;
                overflow: hidden;
                text-align: left;
            }

            .param-table-wrap th {
                font-weight: 750;
                text-align: left;
            }

            /* Make sure the sidebar overlays other content when open */
            .bslib-sidebar-layout > .bslib-sidebar {
                z-index: 1050;
            }

            .bslib-sidebar-layout .bslib-sidebar-toggle {
                z-index: 1060;
            }

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

            .mask-section-title {
                font-weight: 700;
                margin-top: 0.25rem;
                margin-bottom: 0.35rem;
            }

            .mask-divider {
                margin-top: 0.6rem;
                margin-bottom: 0.6rem;
            }

            .compact-stack p {
                margin-bottom: 0.2rem;
            }

            .compact-stack .shiny-input-container {
                margin-bottom: 0.4rem !important;
            }

            .compact-small-line {
                margin-bottom: 0.15rem;
                line-height: 1.2;
            }
        """)
    ),

    # This is the main content of the image handler and normalization settings tools
    ui.row(
        ui.column(
            12,
            ui.tags.div(
                # Toolbar + panels (so essentially everything but the table)
                ui.tags.div(
                    ui.navset_tab(
                        ui.nav_panel(
                        "PINT",
                            ui.tags.div(
                                # ============================================================
                                # LEFT CONTROL COLUMN
                                # ============================================================
                                ui.tags.div(
                                     # ----------------------------
                                    # Loading / selection / export
                                    # ----------------------------
                                    ui.card(
                                        ui.card_header("Image selection"),

                                        ui.input_text(
                                            "path",
                                            "Folder path",
                                            value="",
                                            width="100%",
                                        ),

                                        ui.input_action_button(
                                            "load",
                                            "Load images",
                                            class_="btn btn-primary w-100 mb-2",
                                        ),

                                        ui.tags.hr(class_="pint-divider"),

                                        ui.row(
                                            ui.column(
                                                6,
                                                ui.input_action_button(
                                                    "export_params",
                                                    "Export CSV",
                                                    class_="btn btn-secondary w-100",
                                                ),
                                            ),
                                            ui.column(
                                                6,
                                                ui.input_action_button(
                                                    "import_params",
                                                    "Import CSV",
                                                    class_="btn btn-secondary w-100",
                                                ),
                                            ),
                                            class_="gy-1 mb-2",
                                        ),

                                        ui.input_action_button(
                                            "perform_analysis",
                                            "Process Images",
                                            class_="btn btn-primary text-white w-100",
                                        ),

                                        class_="mb-2",
                                    ),

                                    # ----------------------------
                                    # Winsorization
                                    # ----------------------------
                                    ui.card(
                                        ui.card_header("Winsorization"),

                                        ui.row(
                                            ui.column(
                                                6,
                                                ui.input_numeric(
                                                    "winsor_low",
                                                    "Lower quantile",
                                                    value=0.00,
                                                    min=0.0,
                                                    max=1.0,
                                                    step=0.01,
                                                    width="100%",
                                                ),
                                            ),
                                            ui.column(
                                                6,
                                                ui.input_numeric(
                                                    "winsor_high",
                                                    "Upper quantile",
                                                    value=0.990,
                                                    min=0.9,
                                                    max=1.0,
                                                    step=0.001,
                                                    width="100%",
                                                ),
                                            ),
                                            class_="gy-1",
                                        ),

                                        ui.row(
                                            ui.column(
                                                6,
                                                ui.input_checkbox(
                                                    "doWinsor",
                                                    "Apply winsorization",
                                                    value=True,
                                                ),
                                            ),
                                            ui.column(
                                                6,
                                                ui.input_action_button(
                                                    "apply_one",
                                                    "Update channel",
                                                    class_="btn btn-primary w-100",
                                                ),
                                            ),
                                            class_="align-items-end gy-1",
                                        ),

                                        class_="mb-2",
                                    ),

                                    # ----------------------------
                                    # Thresholding
                                    # ----------------------------
                                    ui.card(
                                        ui.card_header("Thresholding"),

                                        ui.row(
                                            ui.column(
                                                7,
                                                ui.input_numeric(
                                                    "abs_threshold_val",
                                                    "Absolute threshold",
                                                    value=1,
                                                    min=0.0,
                                                    max=100.0,
                                                    step=0.1,
                                                    width="100%",
                                                ),
                                            ),
                                            ui.column(
                                                5,
                                                ui.input_checkbox(
                                                    "doAbsThreshold",
                                                    "Apply",
                                                    value=True,
                                                ),
                                            ),
                                            class_="align-items-end gy-1",
                                        ),

                                        ui.row(
                                            ui.column(
                                                7,
                                                ui.input_numeric(
                                                    "thr_fraction_val",
                                                    "Fraction of max",
                                                    value=0.1,
                                                    min=0.0,
                                                    max=1.0,
                                                    step=0.01,
                                                    width="100%",
                                                ),
                                            ),
                                            ui.column(
                                                5,
                                                ui.input_checkbox(
                                                    "doThreshold",
                                                    "Apply",
                                                    value=False,
                                                ),
                                            ),
                                            class_="align-items-end gy-1",
                                        ),

                                        ui.input_action_button(
                                            "apply_threshold",
                                            "Update channel",
                                            class_="btn btn-primary w-100",
                                        ),

                                        class_="mb-2",
                                    ),

                                    # ----------------------------
                                    # Sliding window noise removal
                                    # ----------------------------
                                    ui.card(
                                        ui.card_header("Sliding Window Noise Removal"),

                                        ui.row(
                                            ui.column(
                                                6,
                                                ui.input_numeric(
                                                    "noise_strength",
                                                    "Denoise strength",
                                                    value=0.1,
                                                    min=0.0,
                                                    max=1.0,
                                                    step=0.01,
                                                    width="100%",
                                                ),
                                            ),
                                            ui.column(
                                                6,
                                                ui.input_numeric(
                                                    "window_size",
                                                    "Window size",
                                                    value=3,
                                                    min=1,
                                                    step=2,
                                                    width="100%",
                                                ),
                                            ),
                                            class_="gy-1",
                                        ),

                                        ui.row(
                                            ui.column(
                                                6,
                                                ui.input_checkbox(
                                                    "doNoise",
                                                    "Apply noise removal",
                                                    value=True,
                                                ),
                                            ),
                                            ui.column(
                                                6,
                                                ui.input_action_button(
                                                    "apply_noise",
                                                    "Update channel",
                                                    class_="btn btn-primary w-100",
                                                ),
                                            ),
                                            class_="align-items-end gy-1",
                                        ),

                                        ui.output_ui("noise_tooltip"),

                                        class_="mb-2",
                                    ),

                                    # ----------------------------
                                    # Transformation and normalization
                                    # ----------------------------
                                    ui.card(
                                        ui.card_header("Transformation and normalization"),

                                        # First transform
                                        ui.row(
                                            ui.column(
                                                6,
                                                ui.input_checkbox(
                                                    "doAsinh",
                                                    "Arcsinh transform",
                                                    value=False,
                                                ),
                                            ),
                                            ui.column(
                                                6,
                                                ui.input_select(
                                                    "asinh_cofactor",
                                                    "Cofactor",
                                                    choices=[str(i) for i in range(2, 11)],
                                                    selected="5",
                                                    width="100%",
                                                ),
                                            ),
                                            class_="align-items-end gy-1",
                                        ),

                                        ui.tags.hr(class_="pint-divider"),

                                        # Then normalize
                                        ui.input_checkbox(
                                            "doNorm",
                                            "Normalize channel",
                                            value=True,
                                        ),

                                        ui.input_radio_buttons(
                                            "norm_scope",
                                            "Normalize using",
                                            choices={
                                                "page": "Per page",
                                                "global": "Global min/max",
                                            },
                                            selected="page",
                                            inline=True,
                                        ),

                                        ui.output_ui("norm_scope_hint"),

                                        ui.input_action_button(
                                            "apply_norm",
                                            "Apply transform/norm",
                                            class_="btn btn-primary w-100 mt-2",
                                        ),

                                        class_="mb-2",
                                    ),

                                    class_="controls-left",
                                ),

                                # ============================================================
                                # RIGHT VIEWER COLUMN
                                # ============================================================
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.row(
                                            # Sample selector: 3
                                            ui.column(
                                                3,
                                                ui.input_select(
                                                    "sample",
                                                    "Sample",
                                                    choices=[],
                                                    selected=None,
                                                    width="100%",
                                                ),
                                            ),

                                            # Previous sample: 1
                                            ui.column(
                                                1,
                                                ui.input_action_button(
                                                    "prev_sample",
                                                    "←",
                                                    class_="btn-sm w-100",
                                                ),
                                            ),

                                            # Next sample: 1
                                            ui.column(
                                                1,
                                                ui.input_action_button(
                                                    "next_sample",
                                                    "→",
                                                    class_="btn-sm w-100",
                                                ),
                                            ),

                                            # Empty spacer: 1
                                            ui.column(1),

                                            # Channel selector: 3
                                            ui.column(
                                                3,
                                                ui.input_select(
                                                    "channel",
                                                    "Channel",
                                                    choices=[],
                                                    selected=None,
                                                    width="100%",
                                                ),
                                            ),

                                            # Previous channel: 1
                                            ui.column(
                                                1,
                                                ui.input_action_button(
                                                    "prev_channel",
                                                    "←",
                                                    class_="btn-sm w-100",
                                                ),
                                            ),

                                            # Next channel: 1
                                            ui.column(
                                                1,
                                                ui.input_action_button(
                                                    "next_channel",
                                                    "→",
                                                    class_="btn-sm w-100",
                                                ),
                                            ),

                                            class_="align-items-end gy-0 gx-1 viewer-navigator-row",
                                        ),
                                        class_="viewer-navigator",
                                    ),

                                    ui.tags.div(
                                        ui.output_plot("pint_viewer", fill=True, height="100%"),
                                        class_="viewer-plot-fill",
                                    ),

                                    class_="viewer-main",
                                ),

                                class_="pint-main-layout",
                            ),

                            value="pint",
                        ),

                        ui.nav_panel(
                            "Image creator",
                            ui.tags.div(
                                ui.tags.div(
                                    ui.card(
                                        ui.card_header("Composite channels (processed images)"),

                                        ui.row(
                                            ui.column(7, ui.tags.strong("Channel")),
                                            ui.column(3, ui.tags.strong("Color")),
                                            ui.column(2, ui.tags.strong("Gain")),
                                            class_="mb-1 creator-header-row",
                                        ),

                                        *[make_composite_slot(i) for i in range(1, 9)],

                                        class_="mb-2",
                                    ),

                                    ui.card(
                                        ui.card_header("Composite export"),

                                        ui.input_action_button(
                                            "fill_composite_from_current",
                                            "Fill from first 8 channels",
                                            class_="btn btn-secondary w-100 mb-2",
                                        ),

                                        ui.input_action_button(
                                            "save_composite_tiff",
                                            "Save composite TIFF",
                                            class_="btn btn-primary w-100 mb-2",
                                        ),

                                        ui.input_action_button(
                                            "export_creator_composites_all",
                                            "Export composite TIFF for all images",
                                            class_="btn btn-secondary w-100",
                                        ),

                                        ui.br(),
                                        ui.br(),
                                        ui.output_ui("composite_summary"),

                                        class_="mb-2",
                                    ),

                                    class_="controls-left",
                                ),
                                ui.tags.div(
                                    ui.tags.div(
                                        ui.output_plot("creator_viewer", fill=True, height="100%"),
                                        class_="viewer-plot-fill",
                                    ),
                                    class_="viewer-main",
                                ),

                                class_="pint-main-layout",
                            ),
                            value="creator",
                        ),

                        ui.nav_panel(
                            "Mask visualization",
                            ui.row(
                                ui.column(
                                    3,
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
                                        ui.input_numeric("mask_scale_factor", "Scale factor", value=2, min=1, max=20, step=1),
                                    ),
                                    ui.br(),
                                    ui.card(
                                        ui.card_header("Matched masks"),
                                        ui.div(
                                            ui.tags.div("Selection / visualization", class_="mask-section-title"),
                                            ui.output_ui("mask_table_summary"),
                                            ui.output_ui("mask_match_summary"),
                                            ui.input_select("selected_mask_name", "Select mask", choices=[], selected=None, width="100%"),
                                            ui.output_ui("selected_mask_summary"),

                                            ui.hr(class_="mask-divider"),

                                            ui.tags.div("Manual matching for unmatched masks", class_="mask-section-title"),
                                            ui.output_ui("manual_mask_match_ui"),
                                            ui.input_action_button(
                                                "apply_manual_mask_matches",
                                                "Apply manual matches",
                                                class_="btn btn-secondary w-100 mt-2",
                                            ),
                                            ui.hr(class_="mask-divider"),
                                            ui.input_action_button(
                                                "export_all_mask_visualizations",
                                                "Export all matched mask visualizations",
                                                class_="btn btn-secondary w-100 mt-2",
                                            ),
                                            ui.hr(class_="mask-divider"),
                                            ui.input_action_button(
                                                "push_to_neighborhood",
                                                "Push to neighborhood",
                                                class_="btn btn-primary w-100 mt-2",
                                                ),
                                            ),
                                            class_="compact-stack",
                                        ),
                                ),
                                ui.column(
                                    9,
                                    ui.card(
                                        ui.card_header("Visualization"),
                                        ui.div(
                                            ui.output_plot("mask_viewer", fill=True, height="100%"),
                                            style="height: 75vh;",
                                        ),
                                    ),
                                ),
                            ),
                            value="mask",
                        ),

                        ui.nav_panel(
                            "Neigborhood Analysis",
                            ui.row(
                                ui.column(
                                    4,
                                    ui.card(
                                        ui.card_header("Neighborhood input"),
                                        ui.output_ui("neighborhood_input_summary"),
                                        ui.input_numeric("touching_n_perm", "N permutations", value=1000, min=0, step=100),
                                        ui.input_select(
                                            "analysis_unit_col",
                                            "Final comparison unit",
                                            choices=[],
                                            selected=None,
                                            width="100%",
                                        ),
                                        ui.input_checkbox("save_mask_copy_on_analysis", "Save matched mask TIFF copies", value=False),
                                        ui.input_action_button(
                                            "run_touching_analysis",
                                            "Run touching analysis",
                                            class_="btn btn-primary w-100 mt-2",
                                        ),
                                        ui.hr(),
                                        ui.tags.div("Fallback: kernel/radius-based analysis", class_="mask-section-title"),
                                        ui.tags.a(
                                            "Open legacy neighborhood analysis",
                                            href="/neighborhood/",
                                            target="_blank",
                                            role="button",
                                            class_="btn btn-secondary w-100 mt-2",
                                            style="pointer-events: auto;",
                                        ),
                                    )
                                ),
                                ui.column(
                                    8,
                                    ui.card(
                                        ui.card_header("Neighborhood status"),
                                        ui.output_text_verbatim("neighborhood_status_text"),
                                        ui.output_ui("neighborhood_touching_summary"),
                                    ),
                                ),
                            ),
                            ui.br(),
                            ui.row(
                                ui.column(
                                    4,
                                    ui.card(
                                        ui.card_header("PERMANOVA"),
                                        ui.output_ui("permanova_summary"),
                                    ),
                                ),
                                ui.column(
                                    8,
                                    ui.card(
                                        ui.card_header("Touching interaction results"),
                                        ui.output_data_frame("touching_results_preview"),
                                    ),
                                ),
                            ),
                            value="neighborhood",
                        ),
                        id="viewer_mode",
                    ),
                ),
            ),
        ),
    ),

    #IMPORTANT: This must be the last argument!
    position="right",
)



## <----------------> SERVER! <-------------------> ##
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
    manual_mask_match_df = reactive.Value(pd.DataFrame())
    final_mask_match_df = reactive.Value(pd.DataFrame())
    neighborhood_input_data = reactive.Value(None)
    neighborhood_status_msg = reactive.Value("No mask dataset pushed yet.")
    neighborhood_touching_edges = reactive.Value(pd.DataFrame())
    neighborhood_matched_cells = reactive.Value(pd.DataFrame())
    neighborhood_touching_results = reactive.Value(pd.DataFrame())
    neighborhood_sample_matrix = reactive.Value(pd.DataFrame())
    neighborhood_permanova_results = reactive.Value(pd.DataFrame())
        
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

            channelName = getattr(input, f"comp_channel_{slotIdx}")()
            colorName = getattr(input, f"comp_color_{slotIdx}")()
            gain = float(getattr(input, f"comp_gain_{slotIdx}")())

            if not channelName or channelName == COMPOSITE_EMPTY_CHOICE:
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

        choices = [COMPOSITE_EMPTY_CHOICE] + ordered

        # Default to the 3rd channel because the first 2 are usually Argon/Krypton.
        # If fewer than 3 channels exist, fall back to the first real channel.
        firstDefault = ordered[2] if len(ordered) >= 3 else (ordered[0] if ordered else COMPOSITE_EMPTY_CHOICE)

        for slotIdx in range(1, MAX_COMPOSITE_CHANNELS + 1):
            inputId = f"comp_channel_{slotIdx}"
            currentVal = getattr(input, inputId)()

            if slotIdx == 1:
                defaultVal = firstDefault
            else:
                defaultVal = COMPOSITE_EMPTY_CHOICE

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
            channelName = getattr(input, f"comp_channel_{slotIdx}")()

            if not channelName or channelName == COMPOSITE_EMPTY_CHOICE:
                continue

            colorName = getattr(input, f"comp_color_{slotIdx}")()
            gain = float(getattr(input, f"comp_gain_{slotIdx}")())
            selected.append((channelName, colorName, gain))

        return selected

    def _build_mask_visualization_figure(
        maskPath: str,
        maskName: str,
        matchingData: pd.DataFrame,
        clusterCol: str,
        xCol: str,
        yCol: str,
        scaleFactor: int,
    ):
        cellMask = read_mask_tiff(maskPath)

        matchedData = match_mask_centroids_to_cells(
            cellMask=cellMask,
            matchingData=matchingData,
            xCol=xCol,
            yCol=yCol,
        )

        plotData = make_mask_plot_data(
            cellMask=cellMask,
            matchedData=matchedData,
            clusterCol=clusterCol,
            scaleFactor=scaleFactor,
            background="white",
            borderColor="white",
            missingColor="#808080",
        )

        colorMat = plotData["colorMat"]
        clusterColors = plotData["clusterColors"]

        imgH, imgW = colorMat.shape[:2]
        imgRatio = imgW / imgH if imgH > 0 else 1.0

        # Keep image ratio intact while reserving a readable legend column.
        imagePanelHeight = 8.0
        imagePanelWidth = imagePanelHeight * imgRatio

        legendPanelWidth = 3.2
        figWidth = imagePanelWidth + legendPanelWidth
        figHeight = imagePanelHeight

        fig = plt.figure(figsize=(figWidth, figHeight), dpi=200)
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[imagePanelWidth, legendPanelWidth],
            wspace=0.02,
        )

        axImg = fig.add_subplot(gs[0, 0])
        axLeg = fig.add_subplot(gs[0, 1])

        axImg.imshow(colorMat, interpolation="nearest")
        axImg.set_axis_off()

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

        return fig, matchedData


    def _get_neighborhood_output_dir():
        obj = neighborhood_input_data.get()
        if obj is None:
            return None

        maskFolder = obj.get("mask_folder", None)
        if not maskFolder:
            return None

        outDir = Path(maskFolder) / "Neigborhood results"
        outDir.mkdir(parents=True, exist_ok=True)
        return outDir

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

    @reactive.Effect
    @reactive.event(input.export_all_mask_visualizations)
    def _confirm_export_all_mask_visualizations():
        finalDf = final_mask_match_df.get()

        if finalDf is None or finalDf.empty:
            print("⚠️ No matched masks available to export.")
            return

        nMasks = int(finalDf["MaskExists"].sum()) if "MaskExists" in finalDf.columns else len(finalDf)

        m = ui.modal(
            ui.p(
                f"You are about to export {nMasks:,} rendered mask visualizations as TIFF files."
            ),
            ui.p(
                "This can take substantial time and disk space because each output includes the image and a readable legend."
            ),
            ui.tags.small("Continue?"),
            title="Confirm export of all matched mask visualizations",
            easy_close=True,
            footer=ui.div(
                ui.modal_button("Cancel", class_="btn btn-secondary"),
                ui.input_action_button(
                    "fill_composite_from_current",
                    "Reset creator channels",
                    class_="btn btn-secondary w-100 mb-2",
                ),
            ),
        )
        ui.modal_show(m, session=session)

    @reactive.Effect
    @reactive.event(input.run_touching_analysis)
    def _run_touching_analysis():
        obj = neighborhood_input_data.get()

        if obj is None:
            neighborhood_status_msg.set("No pushed neighborhood dataset available.")
            neighborhood_matched_cells.set(pd.DataFrame())
            neighborhood_touching_edges.set(pd.DataFrame())
            neighborhood_touching_results.set(pd.DataFrame())
            neighborhood_sample_matrix.set(pd.DataFrame())
            neighborhood_permanova_results.set(pd.DataFrame())
            return

        matchTable = obj.get("mask_match_table", pd.DataFrame())
        nMasks = len(matchTable) if matchTable is not None else 0
        nPerm = int(input.touching_n_perm() or 0)

        sampleCol = obj["column_map"].get("sample_col", None)
        clusterCol = obj["column_map"].get("cluster_col", None)
        conditionCol = obj["column_map"].get("condition_col", None)

        if not sampleCol or not clusterCol:
            neighborhood_status_msg.set("Sample and cluster columns must be selected before running analysis.")
            return

        with ui.Progress(min=0, max=max(nMasks + 3, 1), session=session) as p:
            step = 0

            def progress(msg: str) -> None:
                nonlocal step
                if (
                    "Building touching graph for" in msg or
                    "Done " in msg or
                    "Summarizing observed touching interactions" in msg or
                    "Computing permutation expectation on fixed touching graph" in msg
                ):
                    step = min(step + 1, max(nMasks + 3, 1))
                p.set(step, message=msg)

            p.set(0, message="Starting touching analysis...")

            try:
                matchedData, touchingEdges = build_touching_edges_for_pushed_dataset(
                    obj,
                    progress=progress,
                )
            except Exception as e:
                neighborhood_matched_cells.set(pd.DataFrame())
                neighborhood_touching_edges.set(pd.DataFrame())
                neighborhood_touching_results.set(pd.DataFrame())
                neighborhood_sample_matrix.set(pd.DataFrame())
                neighborhood_permanova_results.set(pd.DataFrame())
                neighborhood_status_msg.set(f"Touching analysis failed: {e}")
                return

            neighborhood_matched_cells.set(matchedData)
            neighborhood_touching_edges.set(touchingEdges)

            # Optional TIFF copy export
            outDir = _get_neighborhood_output_dir()
            if bool(input.save_mask_copy_on_analysis()) and outDir is not None and matchTable is not None and not matchTable.empty:
                maskCopyDir = outDir / "matched_mask_tiff_copies"
                maskCopyDir.mkdir(parents=True, exist_ok=True)

                for _, row in matchTable.iterrows():
                    maskPath = row.get("MaskPath", None)
                    if maskPath and not pd.isna(maskPath):
                        src = Path(str(maskPath))
                        if src.exists():
                            dst = maskCopyDir / src.name
                            if not dst.exists():
                                dst.write_bytes(src.read_bytes())

            try:
                resultsDf = chance_correct_touching_interactions(
                    edgeDf=touchingEdges,
                    matchedDf=matchedData,
                    sample_col=sampleCol,
                    cluster_col=clusterCol,
                    n_perm=nPerm,
                    random_seed=1,
                    progress=progress,
                )
            except Exception as e:
                neighborhood_touching_results.set(pd.DataFrame())
                neighborhood_sample_matrix.set(pd.DataFrame())
                neighborhood_permanova_results.set(pd.DataFrame())
                neighborhood_status_msg.set(f"Touching graph built, but chance correction failed: {e}")
                return

            neighborhood_touching_results.set(resultsDf)

            sampleMatrixDf = make_sample_interaction_matrix(
                resultsDf,
                sample_col=sampleCol,
                value_col="ChanceCorrectedInteraction",
            )

            analysisUnitCol = input.analysis_unit_col()

            if not analysisUnitCol or analysisUnitCol not in matchedData.columns:
                neighborhood_touching_results.set(resultsDf)
                neighborhood_sample_matrix.set(pd.DataFrame())
                neighborhood_permanova_results.set(pd.DataFrame())
                neighborhood_status_msg.set(
                    f"Final comparison unit column '{analysisUnitCol}' is not available."
                )
                return

            if analysisUnitCol == sampleCol:
                finalMatrixDf = sampleMatrixDf.copy()
                finalMetaDf = matchedData[[sampleCol, conditionCol]].drop_duplicates().copy()
                permanovaSampleCol = sampleCol
            else:
                metaDf = matchedData[[sampleCol, analysisUnitCol, conditionCol]].drop_duplicates().copy()

                finalMatrixDf, finalMetaDf = aggregate_interaction_matrix(
                    matrixDf=sampleMatrixDf,
                    metadataDf=metaDf,
                    sample_col=sampleCol,
                    aggregate_col=analysisUnitCol,
                    group_col=conditionCol,
                )
                permanovaSampleCol = analysisUnitCol

            neighborhood_sample_matrix.set(finalMatrixDf)

            if conditionCol and conditionCol in finalMetaDf.columns and not finalMatrixDf.empty:
                permanovaDf = permanova_one_factor(
                    matrixDf=finalMatrixDf,
                    metadataDf=finalMetaDf,
                    sample_col=permanovaSampleCol,
                    group_col=conditionCol,
                    n_perm=999,
                    random_seed=1,
                )
                if not permanovaDf.empty:
                    permanovaDf["AnalysisUnit"] = str(analysisUnitCol)
                    permanovaDf["Status"] = "OK"
            else:
                permanovaDf = pd.DataFrame()

            neighborhood_permanova_results.set(permanovaDf)

            # Save outputs
            if outDir is not None:
                if not touchingEdges.empty:
                    touchingEdges.to_csv(outDir / "touching_edges_annotated.csv", index=False)
                if not matchedData.empty:
                    matchedData.to_csv(outDir / "matched_cells_for_touching_analysis.csv", index=False)
                if not resultsDf.empty:
                    resultsDf.to_csv(outDir / "touching_interactions_with_empirical_pvalues.csv", index=False)
                if not finalMatrixDf.empty:
                    finalMatrixDf.to_csv(outDir / "touching_interaction_final_matrix.csv", index=False)
                if not permanovaDf.empty:
                    permanovaDf.to_csv(outDir / "touching_interaction_permanova.csv", index=False)
                if not finalMetaDf.empty:
                    finalMetaDf.to_csv(outDir / "touching_interaction_final_metadata.csv", index=False)

            neighborhood_status_msg.set(
                f"Touching analysis complete: {len(touchingEdges):,} touching edges across {nMasks} masks. "
                f"Results saved to: {outDir if outDir is not None else 'no output folder'}"
            )
            p.set(max(nMasks + 3, 1), message="Touching analysis complete.")

    @reactive.Effect
    def _sync_analysis_unit_col_choices():
        obj = neighborhood_input_data.get()

        if obj is None:
            ui.update_select("analysis_unit_col", choices=[], selected=None, session=session)
            return

        df = obj.get("cell_table", pd.DataFrame())
        if df is None or df.empty:
            ui.update_select("analysis_unit_col", choices=[], selected=None, session=session)
            return

        colNames = list(df.columns)

        def pick_first(existingNames: list[str], fallback=None):
            for name in existingNames:
                if name in colNames:
                    return name
            return fallback

        default = pick_first(
            ["ROIName", "CellMaskName", "SampleNumber", "PatientID"],
            colNames[0] if colNames else None,
        )

        ui.update_select(
            "analysis_unit_col",
            choices=colNames,
            selected=default,
            session=session,
        )

    # <---------- plot ---------->
    ##Below is the actual img viewer where the image is rendered and all the different thresholding stept are visualized.
    @output
    @render.ui
    def neighborhood_input_summary():
        obj = neighborhood_input_data.get()

        if obj is None:
            return ui.tags.small("No dataset pushed from mask visualization yet.", class_="text-muted")

        df = obj["cell_table"]
        colMap = obj["column_map"]

        nClusters = 0
        clusterCol = colMap.get("cluster_col", None)
        if clusterCol and clusterCol in df.columns:
            nClusters = df[clusterCol].astype(str).nunique()

        return ui.div(
            ui.tags.div(f"Masks pushed: {obj['n_masks']}"),
            ui.tags.div(f"Rows: {len(df):,}"),
            ui.tags.div(f"Clusters: {nClusters}"),
            ui.tags.div(f"Mask folder: {obj['mask_folder'] or '—'}"),
            ui.tags.div(f"Pushed at: {obj['pushed_at']}"),
        )

    @output
    @render.text
    def neighborhood_status_text():
        return neighborhood_status_msg.get()

    @output
    @render.plot
    def creator_viewer():
        try:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

            rgb, used = _build_composite_rgb()

            if rgb is None:
                ax.text(
                    0.5,
                    0.5,
                    "No composite channels selected",
                    ha="center",
                    va="center",
                )
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                return fig

            ax.imshow(rgb)
            ax.set_axis_off()

            if used:
                labelLines = [
                    f"{ch} — {col} × {gain:.1f}"
                    for ch, col, gain in used
                ]

                fig.text(
                    0.01,
                    0.99,
                    "\n".join(labelLines),
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="white",
                    bbox=dict(
                        facecolor="black",
                        alpha=0.55,
                        edgecolor="none",
                        pad=4,
                    ),
                )

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

        except SilentException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()

            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

    @output
    @render.plot
    def pint_viewer():
        try:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

            imgs = images.get()
            s = input.sample()
            c = input.channel()

            if not imgs or not s or not c or s not in imgs:
                ax.text(0.5, 0.5, "No image", ha="center", va="center")
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                return fig

            arr = imgs[s]
            chlist = channels.get().get(s, [])

            if c not in chlist:
                ax.text(0.5, 0.5, f"Channel {c!r} not found", ha="center", va="center")
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
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

        except SilentException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()

            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
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
    
    @output
    @render.ui
    def neighborhood_touching_summary():
        edgeDf = neighborhood_touching_edges.get()
        matchedDf = neighborhood_matched_cells.get()
        resultsDf = neighborhood_touching_results.get()

        if edgeDf is None or edgeDf.empty:
            return ui.tags.small("No touching-edge results yet.", class_="text-muted")

        nCells = 0 if matchedDf is None or matchedDf.empty else len(matchedDf)

        nClusters = 0
        if "cell_cluster" in edgeDf.columns:
            nClusters = pd.unique(
                pd.concat(
                    [
                        edgeDf["cell_cluster"].astype(str),
                        edgeDf["neighbor_cluster"].astype(str),
                    ],
                    ignore_index=True,
                )
            ).size

        nInteractions = 0 if resultsDf is None or resultsDf.empty else len(resultsDf)

        return ui.div(
            ui.tags.div(f"Matched cells: {nCells:,}"),
            ui.tags.div(f"Touching edges: {len(edgeDf):,}"),
            ui.tags.div(f"Clusters observed: {nClusters:,}"),
            ui.tags.div(f"Chance-corrected interaction rows: {nInteractions:,}"),
        )
    
    @output
    @render.data_frame
    def touching_results_preview():
        df = neighborhood_touching_results.get()

        expectedCols = [
            "cell_cluster", "neighbor_cluster",
            "n_interactions", "n_cells",
            "normalized", "normalized_expected", "ChanceCorrectedInteraction",
            "observed", "expected", "perm_sd",
            "p_gt", "p_lt", "p_two_sided",
            "p_adj_gt", "p_adj_lt", "p_adj_two_sided",
            "Direction",
        ]

        if df is None or df.empty:
            empty = pd.DataFrame(columns=expectedCols)
            return render.DataGrid(empty, height="350px")

        show = df.copy()

        ## add missing columns if upstream result does not have them yet
        for col in expectedCols:
            if col not in show.columns:
                show[col] = np.nan

        ## order columns nicely, then keep any extras at the end
        extraCols = [c for c in show.columns if c not in expectedCols]
        show = show[expectedCols + extraCols]

        ## only show first 200 rows
        show = show.head(200).copy()

        ## round numeric columns
        numCols = show.select_dtypes(include=["number"]).columns
        show[numCols] = show[numCols].round(3)

        return render.DataGrid(show, height="350px")


    @output
    @render.ui
    def permanova_summary():
        df = neighborhood_permanova_results.get()

        if df is None or df.empty:
            return ui.tags.small("No PERMANOVA results available.", class_="text-muted")

        row = df.iloc[0]

        parts = []

        if "AnalysisUnit" in row.index:
            parts.append(ui.tags.div(f"Analysis unit: {row['AnalysisUnit']}"))

        if "Status" in row.index and pd.notna(row["Status"]):
            parts.append(ui.tags.div(f"Status: {row['Status']}"))

        if pd.notna(row.get("N", np.nan)):
            parts.append(ui.tags.div(f"N observations: {int(row['N'])}"))

        if pd.notna(row.get("Groups", np.nan)):
            parts.append(ui.tags.div(f"N groups: {int(row['Groups'])}"))

        if pd.notna(row.get("PseudoF", np.nan)):
            parts.append(ui.tags.div(f"Pseudo-F: {row['PseudoF']:.4f}"))
        else:
            parts.append(ui.tags.div("Pseudo-F: NA"))

        if pd.notna(row.get("PValue", np.nan)):
            parts.append(ui.tags.div(f"P-value: {row['PValue']:.4g}"))
        else:
            parts.append(ui.tags.div("P-value: NA"))

        if pd.notna(row.get("Permutations", np.nan)):
            parts.append(ui.tags.div(f"Permutations: {int(row['Permutations'])}"))

        return ui.div(*parts)

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

        final_df = match_df.loc[match_df["MaskExists"]].copy()
        final_mask_match_df.set(final_df)
        manual_mask_match_df.set(pd.DataFrame())

        selected_choices = list(final_df[mask_name_col].astype(str)) if not final_df.empty else []
        selected_default = selected_choices[0] if selected_choices else None

        ui.update_select(
            "selected_mask_name",
            choices=selected_choices,
            selected=selected_default,
            session=session,
        )

        print(f"✅ Mask matching complete: {len(matched_df)} matched, {len(missing_df)} missing.")

    @reactive.Effect
    @reactive.event(input.confirm_export_all_mask_visualizations)
    def _export_all_mask_visualizations():
        ui.modal_remove(session=session)

        finalDf = final_mask_match_df.get()
        inputDf = mask_input_df.get()

        if finalDf is None or finalDf.empty:
            print("⚠️ No matched masks available to export.")
            return

        if inputDf is None or inputDf.empty:
            print("⚠️ No cell table loaded.")
            return

        maskNameCol = input.mask_name_col() or "CellMaskName"
        xCol = input.mask_x_col()
        yCol = input.mask_y_col()
        clusterCol = input.mask_cluster_col()
        scaleFactor = int(input.mask_scale_factor() or 2)

        if not xCol or not yCol or not clusterCol or not maskNameCol:
            print("⚠️ Mask name, X, Y, and cluster columns must be selected.")
            return

        exportDf = finalDf.loc[finalDf["MaskExists"]].copy()
        if exportDf.empty:
            print("⚠️ No valid matched masks available to export.")
            return

        maskDir = (input.mask_path() or "").strip()
        if not maskDir:
            # fallback from first valid mask path
            if "MaskPath" in exportDf.columns and not exportDf["MaskPath"].isna().all():
                maskDir = str(Path(exportDf["MaskPath"].dropna().iloc[0]).parent)

        if not maskDir:
            print("⚠️ Could not determine mask folder.")
            return

        outDir = Path(maskDir) / "Exported mask visualizations"
        outDir.mkdir(parents=True, exist_ok=True)

        nMasks = len(exportDf)

        with ui.Progress(min=0, max=nMasks, session=session) as p:
            done = 0

            for _, row in exportDf.iterrows():
                maskName = str(row[maskNameCol])
                maskPath = row.get("MaskPath", None)

                if not maskPath or pd.isna(maskPath):
                    done += 1
                    p.set(done, message=f"Skipping {maskName}: no valid mask path")
                    continue

                try:
                    matchingData = get_cells_for_mask_name(
                        inputDf,
                        mask_name=maskName,
                        mask_name_col=maskNameCol,
                    )

                    if matchingData.empty:
                        done += 1
                        p.set(done, message=f"Skipping {maskName}: no rows in cell table")
                        continue

                    fig, matchedData = _build_mask_visualization_figure(
                        maskPath=str(maskPath),
                        maskName=maskName,
                        matchingData=matchingData,
                        clusterCol=clusterCol,
                        xCol=xCol,
                        yCol=yCol,
                        scaleFactor=scaleFactor,
                    )

                    safeName = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in maskName)
                    outPath = outDir / f"{safeName}.tiff"

                    fig.savefig(
                        outPath,
                        format="tiff",
                        dpi=200,
                        bbox_inches="tight",
                        pil_kwargs={"compression": "tiff_lzw"},
                    )
                    plt.close(fig)

                    done += 1
                    p.set(done, message=f"Exported {maskName} ({matchedData.shape[0]:,} cells)")

                except Exception as e:
                    try:
                        plt.close("all")
                    except Exception:
                        pass
                    done += 1
                    p.set(done, message=f"Failed {maskName}: {e}")

        print(f"✅ Exported {nMasks:,} matched mask visualizations to {outDir}")

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

    ##Push data from mask to neigborhood
    @reactive.Effect
    @reactive.event(input.push_to_neighborhood)
    def _push_to_neighborhood():
        try:
            pushed = _build_neighborhood_input_object()
            neighborhood_input_data.set(pushed)

            nRows = len(pushed["cell_table"])
            nMasks = int(pushed["n_masks"])
            neighborhood_status_msg.set(
                f"Pushed {nMasks} matched masks to neighborhood analysis ({nRows:,} rows)."
            )

            ui.update_navs("viewer_mode", selected="neighborhood", session=session)

        except Exception as e:
            neighborhood_status_msg.set(f"Failed to push dataset: {e}")

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

        matchType = "Manual" if bool(row.get("ManualMatch", False)) else "Automatic"

        return ui.div(
            ui.tags.div(f"Selected mask: {maskName}", class_="compact-small-line"),
            ui.tags.div(f"Mask file: {row.get('MaskFile', '—')}", class_="compact-small-line"),
            ui.tags.div(f"Match type: {matchType}", class_="compact-small-line"),
            ui.tags.div(f"Rows in cell table: {nCells}", class_="compact-small-line"),
            ui.tags.div(f"Clusters present: {nClusters}", class_="compact-small-line"),
        )

    @output
    @render.ui
    def mask_table_summary():
        df = mask_input_df.get()

        if df is None or df.empty:
            return ui.tags.small("No cell table loaded yet.", class_="text-muted")

        return ui.div(
            ui.tags.div(f"Rows in cell table: {len(df)}", class_="compact-small-line"),
            ui.tags.div(f"Columns in cell table: {len(df.columns)}", class_="compact-small-line"),
        )

    @output
    @render.ui
    def mask_match_summary():
        match_df = mask_match_df.get()
        final_df = final_mask_match_df.get()
        manual_df = manual_mask_match_df.get()

        if match_df is None or match_df.empty:
            return ui.tags.small("No mask matching performed yet.", class_="text-muted")

        autoMatched = int(match_df["MaskExists"].sum())
        total = len(match_df)
        manualMatched = 0 if manual_df is None or manual_df.empty else len(manual_df)
        finalMatched = 0 if final_df is None or final_df.empty else int(final_df["MaskExists"].sum())

        return ui.div(
            ui.tags.div(f"Unique mask names in table: {total}", class_="compact-small-line"),
            ui.tags.div(f"Automatic matches: {autoMatched}", class_="compact-small-line"),
            ui.tags.div(f"Manual matches: {manualMatched}", class_="compact-small-line"),
            ui.tags.div(f"Final matched masks: {finalMatched}", class_="compact-small-line"),
            ui.tags.div(f"Still unmatched: {total - finalMatched}", class_="compact-small-line"),
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
        match_df = final_mask_match_df.get()

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
    
    ##Handlers for unmatched masks
    @output
    @render.ui
    def manual_mask_match_ui():
        match_df = mask_match_df.get()
        files_df = mask_files_df.get()

        if match_df is None or match_df.empty:
            return ui.tags.small("No mask matching performed yet.", class_="text-muted")

        unmatched_df = match_df.loc[~match_df["MaskExists"]].copy()
        if unmatched_df.empty:
            return ui.tags.small("All masks matched automatically.", class_="text-muted")

        if files_df is None or files_df.empty:
            return ui.tags.small("No mask files available for manual matching.", class_="text-muted")

        fileChoices = [""] + list(files_df["MaskFile"].astype(str))

        rows = [
            ui.tags.small(
                "Only unmatched mask names are shown here. Select a file manually and click 'Apply manual matches'.",
                class_="text-muted"
            )
        ]
        for i, (_, row) in enumerate(unmatched_df.iterrows()):
            maskName = str(row[input.mask_name_col() or "CellMaskName"])
            rows.append(
                ui.input_select(
                    f"manual_mask_match_{i}",
                    label=maskName,
                    choices=fileChoices,
                    selected="",
                    width="100%",
                )
            )

        return ui.div(*rows)

    @reactive.Effect
    @reactive.event(input.apply_manual_mask_matches)
    def _apply_manual_mask_matches():
        match_df = mask_match_df.get()
        files_df = mask_files_df.get()

        if match_df is None or match_df.empty:
            print("⚠️ No automatic mask matching available.")
            manual_mask_match_df.set(pd.DataFrame())
            final_mask_match_df.set(pd.DataFrame())
            return

        unmatched_df = match_df.loc[~match_df["MaskExists"]].copy()
        if unmatched_df.empty:
            manual_mask_match_df.set(pd.DataFrame())
            final_mask_match_df.set(match_df.copy())
            print("✅ No manual matches needed.")
            return

        mask_name_col = input.mask_name_col() or "CellMaskName"
        manualRows = []

        for i, (_, row) in enumerate(unmatched_df.iterrows()):
            selectedFile = getattr(input, f"manual_mask_match_{i}")()
            if not selectedFile:
                continue

            fileRow = files_df.loc[files_df["MaskFile"].astype(str) == str(selectedFile)].copy()
            if fileRow.empty:
                continue

            fileRow = fileRow.iloc[0]

            manualRows.append(
                {
                    mask_name_col: row[mask_name_col],
                    "MaskFile": fileRow["MaskFile"],
                    "MaskPath": fileRow["MaskPath"],
                    "CellMaskName_file": fileRow["CellMaskName"],
                    "MaskExists": True,
                    "ManualMatch": True,
                }
            )

        manual_df = pd.DataFrame(manualRows)
        manual_mask_match_df.set(manual_df)

        auto_df = match_df.copy()
        auto_df["ManualMatch"] = False

        if not manual_df.empty:
            matched_names = set(manual_df[mask_name_col].astype(str))
            auto_df = auto_df.loc[~auto_df[mask_name_col].astype(str).isin(matched_names)].copy()

            final_df = pd.concat([auto_df, manual_df], ignore_index=True)
        else:
            final_df = auto_df

        final_mask_match_df.set(final_df)

        selected_choices = list(final_df.loc[final_df["MaskExists"], mask_name_col].astype(str)) if not final_df.empty else []
        current_selected = input.selected_mask_name()
        selected_default = current_selected if current_selected in selected_choices else (selected_choices[0] if selected_choices else None)

        ui.update_select(
            "selected_mask_name",
            choices=selected_choices,
            selected=selected_default,
            session=session,
        )

        nManual = len(manual_df)
        print(f"✅ Applied {nManual} manual mask match(es).")


##Helper to push the data to the neigborhood analysis tab
    def _build_neighborhood_input_object():
        matchDf = final_mask_match_df.get()
        inputDf = mask_input_df.get()

        if matchDf is None or matchDf.empty:
            raise ValueError("No matched masks available.")

        if inputDf is None or inputDf.empty:
            raise ValueError("No cell table loaded.")

        maskNameCol = input.mask_name_col() or "CellMaskName"
        xCol = input.mask_x_col()
        yCol = input.mask_y_col()
        clusterCol = input.mask_cluster_col()
        conditionCol = input.mask_condition_col()
        sampleCol = input.mask_sample_col()

        if not xCol or not yCol or not clusterCol or not maskNameCol:
            raise ValueError("Mask name, X, Y, and cluster columns must be selected.")

        validMatchDf = matchDf.loc[matchDf["MaskExists"]].copy()
        if validMatchDf.empty:
            raise ValueError("No valid matched masks available.")

        maskNames = set(validMatchDf[maskNameCol].astype(str))
        pushedCellTable = inputDf.loc[inputDf[maskNameCol].astype(str).isin(maskNames)].copy()

        if pushedCellTable.empty:
            raise ValueError("No cell-table rows found for the matched masks.")

        # Keep only the needed mask-match rows
        validMatchDf = validMatchDf.drop_duplicates(subset=[maskNameCol]).copy()

        # Attach mask path onto each cell-table row
        pushedCellTable = pushedCellTable.merge(
            validMatchDf[[maskNameCol, "MaskFile", "MaskPath", "MaskExists"]].copy(),
            on=maskNameCol,
            how="left",
        )

        maskFolder = None
        if "MaskPath" in validMatchDf.columns and not validMatchDf["MaskPath"].isna().all():
            firstPath = validMatchDf["MaskPath"].dropna().iloc[0]
            maskFolder = str(Path(firstPath).parent)

        return {
            "pushed_at": datetime.now().isoformat(timespec="seconds"),
            "n_masks": int(validMatchDf.shape[0]),
            "mask_match_table": validMatchDf.copy(),
            "cell_table": pushedCellTable.copy(),
            "mask_folder": maskFolder,
            "column_map": {
                "mask_name_col": maskNameCol,
                "x_col": xCol,
                "y_col": yCol,
                "cluster_col": clusterCol,
                "condition_col": conditionCol,
                "sample_col": sampleCol,
            },
        }

app = App(app_ui, server)