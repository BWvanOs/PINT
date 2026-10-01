from shiny import ui


def xenium_panel():
    return ui.nav_panel(
        "Xenium (not functional)",

        ui.layout_columns(
            # ------------------------------------------------------------
            # LEFT: controls
            # ------------------------------------------------------------
            ui.card(
                ui.card_header(
                    "Xenium dataset"
                ),

                ui.tags.p(
                    "Open an experiment.xenium file to inspect "
                    "the dataset manifest.",
                    class_="text-muted",
                ),

                ui.input_action_button(
                    "open_xenium_manifest",
                    "Open .xenium file",
                    class_="btn btn-primary mb-2",
                ),

                ui.output_ui(
                    "xenium_manifest_status"
                ),

                ui.input_action_button(
                    "load_xenium_data",
                    "Load Xenium data",
                    class_="btn btn-primary mb-2",
                ),

                ui.output_ui(
                    "xenium_data_status"
                ),

                col_widths=12,
            ),

            # ------------------------------------------------------------
            # RIGHT: manifest contents
            # ------------------------------------------------------------
            ui.card(
                ui.card_header(
                    "Xenium manifest"
                ),

                ui.tags.p(
                    "Contents of the selected experiment.xenium file.",
                    class_="text-muted",
                ),

                ui.navset_tab(
                    ui.nav_panel(
                        "Manifest",
                        ui.output_data_frame(
                            "xenium_manifest_preview"
                        ),
                    ),

                    ui.nav_panel(
                        "Cells Zarr",
                        ui.output_data_frame(
                            "xenium_cells_structure_preview"
                        ),
                    ),

                    ui.nav_panel(
                        "Count matrix Zarr",
                        ui.output_data_frame(
                            "xenium_matrix_structure_preview"
                        ),
                    ),

                    ui.nav_panel(
                        "Loaded data",
                        ui.output_data_frame(
                            "xenium_data_preview"
                        ),
                    ),

                    id="xenium_inspection_mode",
                ),

                col_widths=12,
            ),

            col_widths=(3, 9),
        ),

        value="xenium",
    )