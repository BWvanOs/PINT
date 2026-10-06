from shiny import ui

from pint_app.core.mesmer_backend import DEFAULT_MESMER_ENV_NAME

def segmentation_panel():
    return ui.nav_panel(
        "Segmentation and quantification",

        ui.navset_tab(
            mesmer_segmentation_panel(),
            quantification_panel(),
            id="segmentation_quantification_mode",
        ),

        value="segmentation",
    )

def quantification_panel():
    return ui.nav_panel(
        "Quantification",

        ui.tags.div(

            # ============================================================
            # LEFT CONTROL COLUMN
            # ============================================================
            ui.tags.div(

                # --------------------------------------------------------
                # Quantification images
                # --------------------------------------------------------
                ui.card(
                    ui.card_header(
                        "Quantification images"
                    ),

                    ui.tags.div(
                        ui.tags.strong(
                            "Important"
                        ),
                        ui.tags.p(
                            "PINT quantifies the pixel values in the images "
                            "selected here. Load the preprocessed images that "
                            "you actually want to quantify. If you use raw "
                            "images, the resulting cell table will contain "
                            "raw intensities.",
                            class_="mb-0",
                        ),
                        class_="alert alert-info",
                    ),

                    ui.input_action_button(
                        "quant_use_pint_images",
                        "Use current PINT images",
                        class_="btn btn-primary w-100 mb-2",
                    ),

                    ui.input_action_button(
                        "quant_load_image_folder",
                        "Load image folder",
                        class_="btn btn-secondary w-100",
                    ),

                    ui.output_ui(
                        "quantification_image_summary"
                    ),

                    class_="mb-2",
                ),

                # --------------------------------------------------------
                # Segmentation masks
                # --------------------------------------------------------
                ui.card(
                    ui.card_header(
                        "Segmentation masks"
                    ),

                    ui.tags.p(
                        "Use masks created by Mesmer in this PINT session "
                        "or load an existing folder containing labelled "
                        "segmentation masks.",
                        class_="text-muted",
                    ),

                    ui.input_action_button(
                        "quant_use_mesmer_masks",
                        "Use current Mesmer masks",
                        class_="btn btn-primary w-100 mb-2",
                    ),

                    ui.input_action_button(
                        "quant_load_mask_folder",
                        "Load mask folder",
                        class_="btn btn-secondary w-100",
                    ),

                    ui.output_ui(
                        "quantification_mask_summary"
                    ),

                    class_="mb-2",
                ),

                # --------------------------------------------------------
                # Matching
                # --------------------------------------------------------
                ui.card(
                    ui.card_header(
                        "Image / mask matching"
                    ),

                    ui.input_action_button(
                        "quant_match_images_masks",
                        "Match images and masks",
                        class_="btn btn-primary w-100",
                    ),

                    ui.output_ui(
                        "quantification_match_summary"
                    ),

                    ui.hr(),

                    ui.input_select(
                        "quant_selected_roi",
                        "ROI / mask",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),

                    ui.row(
                        ui.column(
                            6,
                            ui.input_action_button(
                                "quant_prev_roi",
                                "← Previous",
                                class_="btn btn-secondary w-100",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_action_button(
                                "quant_next_roi",
                                "Next →",
                                class_="btn btn-secondary w-100",
                            ),
                        ),
                        class_="gx-2",
                    ),

                    class_="mb-2",
                ),

                # --------------------------------------------------------
                # Quantification controls
                # --------------------------------------------------------
                ui.card(
                    ui.card_header(
                        "Quantification"
                    ),

                    ui.input_select(
                        "seg_quantification_mode",
                        "Image values",
                        choices={
                            "pint": "PINT-processed images",
                            "raw": "Raw images (not recommended)",
                        },
                        selected="pint",
                        width="100%",
                    ),

                    ui.input_action_button(
                        "quantify_mesmer_masks",
                        "Create cell table",
                        class_="btn btn-primary w-100 mt-2",
                    ),

                    ui.input_action_button(
                        "push_mesmer_to_clustering",
                        "Push quantification to Clustering",
                        class_="btn btn-success w-100 mt-2",
                    ),

                    ui.input_action_button(
                        "push_mesmer_to_mask_visualization",
                        "Push results to Mask visualization",
                        class_="btn btn-secondary w-100 mt-2",
                    ),

                    ui.output_ui(
                        "segmentation_quantification_summary"
                    ),

                    class_="mb-2",
                ),

                class_="controls-left",
            ),

            # ============================================================
            # RIGHT VIEWER COLUMN
            # ============================================================
            ui.tags.div(

                ui.card(
                    ui.card_header(
                        "Segmentation mask overview"
                    ),

                    ui.output_plot(
                        "quantification_mask_preview",
                        width="100%",
                        height="700px",
                    ),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header(
                        "Matched datasets"
                    ),

                    ui.output_data_frame(
                        "quantification_match_preview"
                    ),
                ),

                class_="viewer-main",
            ),

            class_="pint-main-layout",
        ),

        value="quantification",
    )


def mesmer_segmentation_panel():
    return ui.nav_panel(
        "Segmentation",
        ui.tags.div(
            # ============================================================
            # LEFT CONTROL COLUMN
            # ============================================================
            ui.tags.div(
                ui.card(
                    ui.card_header("Segmentation module"),

                    ui.tags.div(
                        ui.tags.div("Beta function", class_="beta-warning-title"),
                        ui.tags.p(
                            "This segmentation module is in beta testing and should be functional. "
                            "Please inspect results carefully."
                        ),
                        class_="beta-warning-card p-3 mb-3",
                    ),

                    ui.input_text(
                        "mesmer_env_name",
                        "Mesmer conda environment",
                        value=DEFAULT_MESMER_ENV_NAME,
                        width="100%",
                    ),

                    ui.input_password(
                        "deepcell_access_token",
                        "DeepCell access token",
                        value="",
                        width="100%",
                    ),

                    ui.tags.small(
                        "Required by DeepCell/Mesmer to download/load model files. "
                        "The token is passed only to the external Mesmer process and is not saved by PINT.",
                        class_="text-muted",
                    ),

                    ui.br(),
                    ui.br(),

                    ui.row(
                        ui.column(
                            6,
                            ui.input_action_button(
                                "check_mesmer_backend",
                                "Check Mesmer installation",
                                class_="btn btn-primary w-100 mb-2",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_action_button(
                                "check_mesmer_gpu",
                                "Check TensorFlow GPU utilization",
                                class_="btn btn-secondary w-100 mb-2",
                            ),
                        ),
                        class_="gx-2",
                    ),

                    ui.row(
                        ui.column(
                            6,
                            ui.input_action_button(
                                "show_mesmer_install_commands",
                                "Show install commands",
                                class_="btn btn-secondary w-100",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_action_button(
                                "install_mesmer_backend",
                                "Install Mesmer backend",
                                class_="btn btn-warning w-100",
                            ),
                        ),
                        class_="gx-2",
                    ),

                    ui.br(),
                    ui.output_ui("mesmer_backend_summary"),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Pushed PINT images"),
                    ui.output_ui("segmentation_input_summary"),

                    ui.hr(),

                    ui.input_select(
                        "seg_sample",
                        "Sample / ROI",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),

                    ui.input_select(
                        "seg_nuclear_channel",
                        "Nuclear channel",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),

                    ui.tags.div(
                        ui.input_select(
                            "seg_boundary_channels",
                            "Boundary / cytoplasm channels",
                            choices=[],
                            selected=None,
                            multiple=True,
                            width="100%",
                        ),
                        class_="seg-boundary-select-wrap",
                    ),

                    ui.input_select(
                        "seg_preprocess_mode",
                        "Segmentation input preprocessing",
                        choices={
                            "soft": "Minimal preprocessing (recommended)",
                            "pint": "Current PINT settings (not recommended)",
                            "raw": "Raw values (not recommended)",
                        },
                        selected="soft",
                        width="100%",
                    ),

                    ui.tags.small(
                        "Soft preprocessing uses gentle percentile clipping and normalization. "
                        "It avoids hard thresholding so weak boundary information is preserved.",
                        class_="text-muted",
                    ),

                    ui.hr(),

                    ui.input_numeric(
                        "seg_image_mpp",
                        "Microns per pixel (mpp)",
                        value=1.0,
                        min=0.1,
                        max=5.0,
                        step=0.1,
                        width="100%",
                    ),

                    ui.input_select(
                        "seg_mesmer_compartment",
                        "Mesmer compartment",
                        choices={
                            "whole-cell": "Whole-cell",
                            "nuclear": "Nuclear",
                        },
                        selected="whole-cell",
                        width="100%",
                    ),

                    ui.input_action_button(
                        "run_mesmer_current",
                        "Run Mesmer on current ROI",
                        class_="btn btn-primary w-100 mt-2",
                    ),

                    ui.input_action_button(
                        "run_mesmer_all",
                        "Run Mesmer on all pushed ROIs",
                        class_="btn btn-secondary w-100 mt-2",
                    ),

                    ui.output_ui("segmentation_mesmer_result_summary"),

                    ui.hr(),

                    ui.input_action_button(
                        "push_mesmer_to_mask_visualization",
                        "Push Mesmer results to Mask visualization",
                        class_="btn btn-secondary w-100 mt-2",
                    ),

                    ui.output_ui("segmentation_quantification_summary"),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Planned workflow"),
                    ui.tags.div(
                        ui.tags.p("Future versions of this tab will:"),
                        ui.tags.ul(
                            ui.tags.li("create a nuclear input image"),
                            ui.tags.li("create a membrane / cytoplasm / boundary composite image"),
                            ui.tags.li("run Mesmer to generate masks"),
                            ui.tags.li("quantify all original PINT channels per cell"),
                            ui.tags.li("push masks and cell tables to Mask visualization"),
                        ),
                        class_="compact-stack",
                    ),
                    class_="mb-2",
                ),

                class_="controls-left",
            ),

            # ============================================================
            # RIGHT VIEWER COLUMN
            # ============================================================
            ui.tags.div(
                ui.card(
                    ui.tags.div(
                        ui.tags.div(
                            ui.tags.strong("Citation reminder"),
                            class_="seg-citation-title",
                        ),
                        ui.tags.div(
                            "If you use Mesmer/DeepCell segmentations, please cite the original Mesmer/TissueNet publication.",
                            class_="seg-citation-message",
                        ),
                        ui.tags.details(
                            ui.tags.summary("Copy citation"),
                            ui.tags.pre(
                                (
                                    "Greenwald, N. F. et al. (2022). Whole-cell segmentation of tissue images "
                                    "with human-level performance using large-scale data annotation and deep learning. "
                                    "Nature Biotechnology, 40, 555–565. https://doi.org/10.1038/s41587-021-01094-0"
                                ),
                                class_="seg-citation-box",
                            ),
                            ui.tags.pre(
                                (
                                    "@article{greenwald2022whole,\n"
                                    "  title={Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning},\n"
                                    "  author={Greenwald, Noah F. and Miller, Geneva and Moen, Erick and others},\n"
                                    "  journal={Nature Biotechnology},\n"
                                    "  volume={40},\n"
                                    "  pages={555--565},\n"
                                    "  year={2022},\n"
                                    "  doi={10.1038/s41587-021-01094-0}\n"
                                    "}"
                                ),
                                class_="seg-citation-box",
                            ),
                        ),
                        class_="seg-citation-content",
                    ),
                    class_="seg-citation-card mb-2",
                ),

                ui.card(
                    ui.card_header("Segmentation input preview"),
                    ui.navset_tab(
                        ui.nav_panel(
                            "Nuclear input",
                            ui.tags.div(
                                ui.output_plot(
                                    "segmentation_nuclear_preview",
                                    fill=True,
                                    height="100%",
                                ),
                                class_="seg-preview-fill",
                            ),
                        ),
                        ui.nav_panel(
                            "Boundary input",
                            ui.tags.div(
                                ui.output_plot(
                                    "segmentation_boundary_preview",
                                    fill=True,
                                    height="100%",
                                ),
                                class_="seg-preview-fill",
                            ),
                        ),
                        ui.nav_panel(
                            "Combined preview",
                            ui.tags.div(
                                ui.output_plot(
                                    "segmentation_combined_preview",
                                    fill=True,
                                    height="100%",
                                ),
                                class_="seg-preview-fill",
                            ),
                        ),

                        ui.nav_panel(
                            "Mesmer mask",
                            ui.tags.div(
                                ui.output_plot("segmentation_mesmer_mask_preview", fill=True, height="100%"),
                                class_="seg-preview-fill",
                            ),
                        ),

                        id="segmentation_preview_mode",
                    ),
                    class_="seg-preview-card",
                ),
                class_="viewer-main",
            ),

            class_="pint-main-layout",
        ),
        value="segmentation",
    )