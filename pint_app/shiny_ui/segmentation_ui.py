from shiny import ui

from pint_app.core.mesmer_backend import DEFAULT_MESMER_ENV_NAME

def segmentation_panel():
    return ui.nav_panel(
        "Segmentation",
        ui.row(
            ui.column(
                4,
                ui.card(
                    ui.card_header("Segmentation module"),

                    ui.tags.div(
                        ui.tags.div(
                            "Alpha function",
                            class_="alpha-warning-title",
                        ),
                        ui.tags.p(
                            "This segmentation module is experimental and under active development."
                        ),
                        ui.tags.p(
                            "At this stage it only manages the optional Mesmer/DeepCell backend. "
                            "Actual segmentation, mask export, and quantification will be added later."
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
            ),

            ui.column(
                8,
                ui.card(
                    ui.card_header("Mesmer backend details"),
                    ui.output_text_verbatim("mesmer_backend_detail"),
                ),

                ui.br(),

                ui.card(
                    ui.card_header("Planned segmentation workflow"),
                    ui.tags.div(
                        ui.tags.p(
                            "Future versions of this tab will create two soft segmentation inputs:"
                        ),
                        ui.tags.ul(
                            ui.tags.li("nuclear input image"),
                            ui.tags.li("membrane / cytoplasm / boundary composite image"),
                        ),
                        ui.tags.p(
                            "Mesmer will generate masks only. PINT will then quantify all original channels per cell."
                        ),
                        ui.tags.p(
                            "The intended output will be a mask TIFF plus a single-cell intensity table "
                            "that can be pushed directly to Mask visualization."
                        ),
                        class_="compact-stack",
                    ),
                ),
            ),
        ),
        value="segmentation",
    )