from shiny import ui


def thumbnail_panel():
    return ui.nav_panel(
        "Thumbnails",

        ui.tags.div(
            # ============================================================
            # LEFT CONTROL COLUMN
            # ============================================================
            ui.tags.div(
                ui.card(
                    ui.card_header("Thumbnail overview"),

                    ui.input_radio_buttons(
                        "thumbnail_view_mode",
                        "Display",
                        choices={
                            "sample": "All channels from one image",
                            "channel": "One channel from all images",
                        },
                        selected="sample",
                    ),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Thumbnail rendering"),

                    ui.input_radio_buttons(
                        "thumbnail_render_mode",
                        "Downscaling method",
                        choices={
                            "signal": "Signal preserving — maximum pooling",
                            "smooth": "Smooth overview — area averaging",
                        },
                        selected="signal",
                    ),

                    ui.tags.p(
                        "Signal preserving retains small bright structures. "
                        "Smooth overview produces a less noisy visual summary.",
                        class_="text-muted small",
                    ),

                    ui.input_action_button(
                        "generate_thumbnails",
                        "Generate thumbnails",
                        class_="btn btn-primary w-100 mb-2",
                    ),

                    ui.tags.p(
                        "Generates thumbnails required for the selected overview. "
                        "Existing valid cache entries are reused and not rerendered.",
                        class_="text-muted small",
                    ),

                    ui.input_action_button(
                        "generate_all_thumbnails",
                        "Generate all thumbnails",
                        class_="btn btn-outline-primary w-100 mb-2",
                    ),

                    ui.tags.p(
                        "Generates every channel for every loaded image.",
                        class_="text-muted small",
                    ),

                    ui.input_action_button(
                        "clear_thumbnail_cache",
                        "Clear thumbnail cache",
                        class_="btn btn-secondary w-100",
                    ),

                    class_="mb-2 thumbnail-render-controls",
                ),

                ui.card(
                    ui.card_header("Status"),
                    ui.output_ui("thumbnail_status_ui"),
                    class_="mb-2",
                ),

                class_="controls-left",
            ),

            # ============================================================
            # RIGHT THUMBNAIL COLUMN
            # ============================================================
            ui.tags.div(

                # --------------------------------------------------------
                # SAMPLE MODE NAVIGATOR
                # --------------------------------------------------------
                ui.panel_conditional(
                    "input.thumbnail_view_mode === 'sample'",

                    ui.tags.div(
                        ui.row(
                            ui.tags.div(
                                ui.input_select(
                                    "thumbnail_sample_display",
                                    "Image",
                                    choices=[],
                                    selected=None,
                                    width="100%",
                                ),
                                class_="navigator-select",
                            ),

                            ui.tags.div(
                                ui.input_action_button(
                                    "thumbnail_prev_sample",
                                    "←",
                                    class_="btn-sm w-100",
                                ),
                                class_="navigator-button",
                            ),

                            ui.tags.div(
                                ui.input_action_button(
                                    "thumbnail_next_sample",
                                    "→",
                                    class_="btn-sm w-100",
                                ),
                                class_="navigator-button",
                            ),

                            class_=(
                                "align-items-end gy-0 gx-1 "
                                "viewer-navigator-row "
                                "pint-navigator-half-fixed"
                            ),
                        ),
                        class_="viewer-navigator",
                    ),
                ),

                # --------------------------------------------------------
                # CHANNEL MODE NAVIGATOR
                # --------------------------------------------------------
                ui.panel_conditional(
                    "input.thumbnail_view_mode === 'channel'",

                    ui.tags.div(
                        ui.input_select(
                            "thumbnail_channel_display",
                            "Channel",
                            choices=[],
                            selected=None,
                            width="100%",
                        ),
                        class_="viewer-navigator",
                    ),
                ),

                ui.tags.div(
                    ui.output_ui("thumbnail_grid"),
                    class_="thumbnail-scroll-area",
                ),

                class_="viewer-main",
            ),

            class_="pint-main-layout",
        ),

        value="thumbnails",
    )