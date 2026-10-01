from shiny import ui

from pint_app.shiny_ui.image_handler_ui import imc_panel
from pint_app.shiny_ui.xenium_ui import xenium_panel


def data_input_panel():
    return ui.nav_panel(
        "Data input and preprocessing",

        ui.navset_tab(
            imc_panel(),
            xenium_panel(),
            id="data_input_mode",
        ),

        value="data_input",
    )