from shiny import ui


def advanced_settings_panel():
    return ui.nav_panel(
        "Advanced settings",
        ui.tags.div(
            ui.card(
                ui.card_header("Loaded objects / memory overview"),

                ui.tags.p(
                    "This diagnostic view shows which large PINT objects are currently loaded in memory. "
                    "It is read-only for now; delete/clear buttons can be added later.",
                    class_="text-muted",
                ),

                ui.tags.div(
                    ui.output_table("advanced_object_memory_table"),
                    class_="advanced-memory-table-wrap",
                ),

                class_="advanced-settings-card",
            ),

            ui.card(
                ui.card_header("Notes"),

                ui.tags.p(
                    "Sizes are estimates based on the Python objects currently held by the app. "
                    "Large NumPy arrays, pandas DataFrames, and dictionaries of image arrays are measured more accurately "
                    "than arbitrary Python objects.",
                    class_="text-muted",
                ),

                ui.tags.p(
                    "If RAM usage is much higher than the table suggests, the difference may come from cached plots, "
                    "temporary arrays created during processing, Python overhead, or memory not yet released back to the OS.",
                    class_="text-muted",
                ),

                class_="advanced-settings-card",
            ),

            class_="advanced-settings-wide compact-stack",
        ),
        value="advanced_settings",
    )