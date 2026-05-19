from shiny import ui

from pint_app.core.composites import (
    COMPOSITE_COLOR_CHOICES,
    COMPOSITE_EMPTY_CHOICE,
    MAX_COMPOSITE_CHANNELS,
)


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


def creator_panel():
    return ui.nav_panel(
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
                    *[make_composite_slot(i) for i in range(1, MAX_COMPOSITE_CHANNELS + 1)],
                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Composite export"),
                    ui.input_action_button(
                        "fill_composite_from_current",
                        "Reset creator channels",
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
                    ui.row(
                        ui.column(
                            6,
                            ui.input_select(
                                "creator_sample_display",
                                "Sample",
                                choices=[],
                                selected=None,
                                width="100%",
                            ),
                        ),
                        ui.column(
                            3,
                            ui.input_action_button(
                                "creator_prev_sample",
                                "←",
                                class_="btn-sm w-100",
                            ),
                        ),
                        ui.column(
                            3,
                            ui.input_action_button(
                                "creator_next_sample",
                                "→",
                                class_="btn-sm w-100",
                            ),
                        ),
                        class_="align-items-end gy-0 gx-1 viewer-navigator-row",
                    ),
                    class_="viewer-navigator",
                ),

                ui.tags.div(
                    ui.output_plot("creator_viewer", fill=True, height="100%"),
                    class_="viewer-plot-fill",
                ),

                class_="viewer-main",
            ),

            class_="pint-main-layout",
        ),
        value="creator",
    )