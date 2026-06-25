from shiny import ui


def pint_panel():
    return ui.nav_panel(
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

                    ui.input_action_button(
                        "push_to_segmentation",
                        "Push loaded images to Segmentation tab",
                        class_="btn btn-warning text-dark w-100 mt-2",
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
                    ui.tags.div(
                        ui.tags.div(
                            ui.input_select(
                                "sample",
                                "Sample",
                                choices=[],
                                selected=None,
                                width="100%",
                            ),
                        ),

                        ui.tags.div(
                            ui.input_action_button(
                                "prev_sample",
                                "←",
                                class_="btn-sm",
                            ),
                        ),

                        ui.tags.div(
                            ui.input_action_button(
                                "next_sample",
                                "→",
                                class_="btn-sm",
                            ),
                        ),

                        ui.tags.div(),

                        ui.tags.div(
                            ui.input_select(
                                "channel",
                                "Channel",
                                choices=[],
                                selected=None,
                                width="100%",
                            ),
                        ),

                        ui.tags.div(
                            ui.input_action_button(
                                "prev_channel",
                                "←",
                                class_="btn-sm",
                            ),
                        ),

                        ui.tags.div(
                            ui.input_action_button(
                                "next_channel",
                                "→",
                                class_="btn-sm",
                            ),
                        ),

                        class_="pint-navigator-grid",
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
    )