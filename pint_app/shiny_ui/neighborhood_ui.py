from shiny import ui


def neighborhood_panel():
    return ui.nav_panel(
        "Neighborhood Analysis",
        ui.tags.div(
            ui.tags.div(
                ui.card(
                    ui.card_header("Neighborhood input"),

                    ui.output_ui("neighborhood_input_summary"),

                    ui.hr(class_="mask-divider"),

                    ui.input_numeric(
                        "touching_n_perm",
                        "N permutations",
                        value=1000,
                        min=0,
                        step=100,
                        width="100%",
                    ),

                    ui.input_select(
                        "analysis_unit_col",
                        "Final comparison unit",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),

                    ui.input_checkbox(
                        "save_mask_copy_on_analysis",
                        "Save matched mask TIFF copies",
                        value=False,
                    ),

                    ui.input_action_button(
                        "run_touching_analysis",
                        "Run touching analysis",
                        class_="btn btn-primary w-100 mt-2",
                    ),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Legacy analysis"),

                    ui.tags.div(
                        "Fallback: kernel/radius-based analysis",
                        class_="mask-section-title",
                    ),

                    ui.tags.small(
                        "Use this only if you need the older radius-based workflow.",
                        class_="text-muted",
                    ),

                    ui.tags.a(
                        "Open legacy neighborhood analysis",
                        href="/neighborhood/",
                        target="_blank",
                        role="button",
                        class_="btn btn-secondary w-100 mt-2",
                        style="pointer-events: auto;",
                    ),

                    class_="mb-2",
                ),

                class_="controls-left",
            ),

            ui.tags.div(
                ui.card(
                    ui.card_header("Neighborhood status"),

                    ui.output_text_verbatim("neighborhood_status_text"),

                    ui.hr(class_="mask-divider"),

                    ui.output_ui("neighborhood_touching_summary"),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("PERMANOVA"),

                    ui.output_ui("permanova_summary"),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Touching interaction results"),

                    ui.output_data_frame("touching_results_preview"),

                    class_="mb-2",
                ),

                class_="viewer-main",
            ),

            class_="pint-main-layout",
        ),
        value="neighborhood",
    )