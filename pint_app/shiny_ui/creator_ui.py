from shiny import ui
import json

from pint_app.core.composites import (
    COMPOSITE_COLOR_CHOICES,
    COMPOSITE_EMPTY_CHOICE,
    MAX_COMPOSITE_CHANNELS,
    composite_color_to_hex,
)


def make_composite_slot(slotIdx: int):
    defaultColor = COMPOSITE_COLOR_CHOICES[(slotIdx - 1) % len(COMPOSITE_COLOR_CHOICES)]

    defaultHex = composite_color_to_hex(defaultColor)

    return ui.row(
        ui.column(
            5,
            ui.input_select(
                f"comp_channel_{slotIdx}",
                "",
                choices=[COMPOSITE_EMPTY_CHOICE],
                selected=COMPOSITE_EMPTY_CHOICE,
                width="100%",
            ),
        ),
        ui.column(
            2,
            ui.input_select(
                f"comp_color_{slotIdx}",
                "",
                choices=COMPOSITE_COLOR_CHOICES,
                selected=defaultColor,
                width="100%",
            ),
        ),
        ui.column(
            3,
            ui.tags.div(
                ui.tags.input(
                    type="color",
                    id=f"comp_color_picker_{slotIdx}",
                    value=defaultHex,
                    class_="creator-color-picker",
                    **{"data-target": f"comp_hex_{slotIdx}"},
                ),
                ui.input_text(
                    f"comp_hex_{slotIdx}",
                    "",
                    value="",
                    placeholder="#RRGGBB",
                    width="100%",
                ),
                class_="creator-color-custom-wrap",
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
    presetHexMap = {
    colorName: composite_color_to_hex(colorName)
    for colorName in COMPOSITE_COLOR_CHOICES
    }

    presetHexMapJson = json.dumps(presetHexMap)
    
    return ui.nav_panel(
        "Image creator",
        ui.tags.div(
            ui.tags.div(
                ui.card(
                    ui.card_header("Composite channels (processed images)"),
                    ui.row(
                        ui.column(5, ui.tags.strong("Channel")),
                        ui.column(2, ui.tags.strong("Preset")),
                        ui.column(3, ui.tags.strong("Custom")),
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

                class_="controls-left controls-left-wide",
            ),

            ui.tags.div(
                ui.tags.div(
                    ui.row(
                        ui.tags.div(
                            ui.input_select(
                                "creator_sample_display",
                                "Sample",
                                choices=[],
                                selected=None,
                                width="100%",
                            ),
                            class_="navigator-select",
                        ),

                        ui.tags.div(
                            ui.input_action_button(
                                "creator_prev_sample",
                                "←",
                                class_="btn-sm w-100",
                            ),
                            class_="navigator-button",
                        ),

                        ui.tags.div(
                            ui.input_action_button(
                                "creator_next_sample",
                                "→",
                                class_="btn-sm w-100",
                            ),
                            class_="navigator-button",
                        ),

                        class_="align-items-end gy-0 gx-1 viewer-navigator-row pint-navigator-half-fixed",
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
        ui.tags.script(
            f"""
            const creatorPresetHexMap = {presetHexMapJson};

            /*
            * Custom picker → custom text field.
            * Use "change" so Chromium does not flood Shiny while dragging.
            */
            document.addEventListener("change", function(event) {{
                if (!event.target.classList.contains("creator-color-picker")) {{
                    return;
                }}

                const targetId = event.target.dataset.target;

                if (!targetId) {{
                    return;
                }}

                const textInput = document.getElementById(targetId);

                if (!textInput) {{
                    return;
                }}

                textInput.value = event.target.value;
                textInput.dispatchEvent(
                    new Event("input", {{ bubbles: true }})
                );
                textInput.dispatchEvent(
                    new Event("change", {{ bubbles: true }})
                );
            }});

            /*
            * Custom text field → custom picker.
            */
            document.addEventListener("change", function(event) {{
                if (
                    !event.target.id ||
                    !event.target.id.startsWith("comp_hex_")
                ) {{
                    return;
                }}

                const value = event.target.value.trim();

                if (!/^#[0-9A-Fa-f]{{6}}$/.test(value)) {{
                    return;
                }}

                const pickerId = event.target.id.replace(
                    "comp_hex_",
                    "comp_color_picker_"
                );

                const picker = document.getElementById(pickerId);

                if (picker) {{
                    picker.value = value;
                }}
            }});

            /*
            * Preset selector → custom text field and picker.
            *
            * This deliberately overwrites an existing custom value because the
            * user explicitly selected a preset.
            */
            document.addEventListener("change", function(event) {{
                if (
                    !event.target.id ||
                    !event.target.id.startsWith("comp_color_") ||
                    event.target.id.startsWith("comp_color_picker_")
                ) {{
                    return;
                }}

                const presetName = event.target.value;
                const presetHex = creatorPresetHexMap[presetName];

                if (!presetHex) {{
                    return;
                }}

                const slotIndex = event.target.id.replace(
                    "comp_color_",
                    ""
                );

                const textInput = document.getElementById(
                    `comp_hex_${{slotIndex}}`
                );

                const picker = document.getElementById(
                    `comp_color_picker_${{slotIndex}}`
                );

                if (picker) {{
                    picker.value = presetHex;
                }}

                if (textInput) {{
                    textInput.value = presetHex;
                    textInput.dispatchEvent(
                        new Event("input", {{ bubbles: true }})
                    );
                    textInput.dispatchEvent(
                        new Event("change", {{ bubbles: true }})
                    );
                }}
            }});
            """
        ),

        value="creator",
    )
