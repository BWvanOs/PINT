from shiny import ui


def mask_visualization_panel():
    return ui.nav_panel(
        "Mask visualization",
        ui.row(
            ui.column(
                3,
                ui.card(
                    ui.card_header("Mask input"),

                    ui.input_text(
                        "mask_csv_path",
                        "Cell table CSV",
                        value="",
                        width="100%",
                    ),

                    ui.row(
                        ui.column(
                            6,
                            ui.input_action_button(
                                "browse_mask_csv",
                                "Browse CSV",
                                class_="btn btn-secondary w-100",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_action_button(
                                "load_mask_csv",
                                "Load cell table",
                                class_="btn btn-primary w-100",
                            ),
                        ),
                    ),

                    ui.br(),

                    ui.input_text(
                        "mask_path",
                        "Mask folder",
                        value="",
                        width="100%",
                    ),

                    ui.row(
                        ui.column(
                            6,
                            ui.input_action_button(
                                "browse_mask_path",
                                "Browse folder",
                                class_="btn btn-secondary w-100",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_action_button(
                                "match_masks",
                                "Match masks",
                                class_="btn btn-primary w-100",
                            ),
                        ),
                    ),

                    ui.br(),

                    ui.input_select(
                        "mask_cell_id_col",
                        "Cell ID column",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),
                    ui.input_select(
                        "mask_x_col",
                        "X coordinate column",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),
                    ui.input_select(
                        "mask_y_col",
                        "Y coordinate column",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),
                    ui.input_select(
                        "mask_name_col",
                        "Mask name column",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),
                    ui.input_select(
                        "mask_cluster_col",
                        "Cluster column",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),
                    ui.input_select(
                        "mask_condition_col",
                        "Condition column",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),
                    ui.input_select(
                        "mask_sample_col",
                        "SampleNumber column",
                        choices=[],
                        selected=None,
                        width="100%",
                    ),

                    ui.input_numeric(
                        "mask_scale_factor",
                        "Scale factor",
                        value=2,
                        min=1,
                        max=20,
                        step=1,
                    ),
                ),

                ui.br(),

                ui.card(
                    ui.card_header("Matched masks"),

                    ui.div(
                        ui.tags.div(
                            "Selection / visualization",
                            class_="mask-section-title",
                        ),

                        ui.output_ui("mask_table_summary"),
                        ui.output_ui("mask_match_summary"),

                        ui.input_select(
                            "selected_mask_name",
                            "Select mask",
                            choices=[],
                            selected=None,
                            width="100%",
                        ),

                        ui.output_ui("selected_mask_summary"),

                        ui.hr(class_="mask-divider"),

                        ui.tags.div(
                            "Manual matching for unmatched masks",
                            class_="mask-section-title",
                        ),

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

                        class_="compact-stack",
                    ),
                ),
            ),

            ui.column(
                9,
                ui.card(
                    ui.card_header("Visualization"),
                    ui.div(
                        ui.output_plot(
                            "mask_viewer",
                            fill=True,
                            height="100%",
                        ),
                        style="height: 75vh;",
                    ),
                ),
            ),
        ),
        value="mask",
    )