from shiny import ui

from pint_app.core.mesmer_backend import DEFAULT_MESMER_ENV_NAME


def segmentation_panel():
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
                        ui.tags.div("Alpha function", class_="alpha-warning-title"),
                        ui.tags.p(
                            "This segmentation module is experimental and under active development."
                        ),
                        ui.tags.p(
                            "At this stage it manages the optional Mesmer/DeepCell backend "
                            "and prepares PINT images for segmentation preview."
                        ),
                        ui.tags.p(
                            "Do not treat this function as production-ready without manual QC."
                        ),
                        class_="alpha-warning-card p-3 mb-3",
                    ),

                    ui.input_text(
                        "mesmer_env_name",
                        "Mesmer conda environment",
                        value=DEFAULT_MESMER_ENV_NAME,
                        width="100%",
                    ),

                    ui.input_action_button(
                        "check_mesmer_backend",
                        "Check Mesmer installation",
                        class_="btn btn-primary w-100 mb-2",
                    ),

                    ui.input_action_button(
                        "show_mesmer_install_commands",
                        "Show install commands",
                        class_="btn btn-secondary w-100 mb-2",
                    ),

                    ui.input_checkbox(
                        "confirm_mesmer_alpha_install",
                        "I understand this is an experimental Alpha backend and will be installed in a separate conda environment.",
                        value=False,
                    ),

                    ui.input_action_button(
                        "install_mesmer_backend",
                        "Install Mesmer backend",
                        class_="btn btn-warning w-100 mt-2",
                    ),

                    ui.br(),
                    ui.br(),
                    ui.output_ui("mesmer_backend_summary"),

                    class_="mb-2",
                ),


                ui.card(
                    ui.card_header("Mesmer backend details"),
                    ui.output_text_verbatim("mesmer_backend_detail"),
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

                    ui.input_radio_buttons(
                        "seg_preprocess_mode",
                        "Segmentation input mode",
                        choices={
                            "soft": "Soft segmentation preprocessing — recommended",
                            "pint": "Use current PINT channel settings — not recommended",
                            "raw": "Use raw image values — not recommended",
                        },
                        selected="soft",
                    ),

                    ui.tags.small(
                        "Soft preprocessing uses gentle percentile clipping and normalization. "
                        "It avoids hard thresholding so weak boundary information is preserved.",
                        class_="text-muted",
                    ),

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