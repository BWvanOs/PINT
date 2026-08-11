from shiny import ui


def image_loading_panel():
    return ui.nav_panel(
        "Image loading",

        ui.tags.div(
            ui.tags.div(
                ui.card(
                    ui.card_header("Normalize ChannelNames"),

                    ui.input_checkbox(
                        "standardize_channel_names",
                        "Standardize MCD channel names",
                        True,
                    ),
                ),
                ui.card(
                    ui.card_header("OME-TIFF folder"),

                    ui.input_text(
                        "path",
                        "Folder path",
                        value="",
                        width="100%",
                    ),

                    ui.input_action_button(
                        "load",
                        "Load OME-TIFF folder",
                        class_="btn btn-primary w-100",
                    ),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("MCD file"),

                    ui.input_action_button(
                        "open_mcd_file",
                        "Open MCD file(s)",
                        class_="btn btn-primary w-100 mb-2",
                    ),

                    ui.output_ui("mcd_file_status_ui"),

                    ui.tags.hr(),
                    ui.row(
                        ui.column(
                            6,
                            ui.input_action_button(
                                "select_all_mcd_rois",
                                "Select all ROIs",
                                class_="btn btn-secondary w-100",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_action_button(
                                "clear_mcd_roi_selection",
                                "Clear selection",
                                class_="btn btn-secondary w-100",
                            ),
                        ),
                        class_="mb-2",
                    ),
                    ui.input_action_button(
                        "load_selected_mcd_rois",
                        "Load selected ROIs into PINT",
                        class_="btn btn-success w-100 mb-2",
                    ),

                    ui.input_action_button(
                        "export_selected_mcd_rois",
                        "Export selected ROIs as OME-TIFF",
                        class_="btn btn-secondary w-100 mb-2",
                    ),

                    ui.input_checkbox(
                        "mcd_export_selected_slides_only",
                        "Only export panoramas from slides containing selected ROIs",
                        value=True,
                    ),

                    ui.input_action_button(
                        "export_mcd_panoramas",
                        "Export panoramas",
                        class_="btn btn-secondary w-100",
                    ),

                    class_="mb-2",
                ),

                class_="controls-left",
            ),

            ui.tags.div(
                ui.card(
                    ui.card_header("MCD acquisitions"),

                    ui.tags.p(
                        "Select one or more acquisitions to load or export.",
                        class_="text-muted",
                    ),

                    ui.output_data_frame("mcd_acquisition_table"),

                    class_="mb-2",
                ),

                class_="viewer-main",
            ),

            class_="pint-main-layout",
        ),

        value="image_loading",
    )