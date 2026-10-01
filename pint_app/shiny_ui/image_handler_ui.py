from shiny import ui

from pint_app.shiny_ui.PINT_ui import pint_panel
from pint_app.shiny_ui.creator_ui import creator_panel
from pint_app.shiny_ui.thumbnail_ui import thumbnail_panel
from pint_app.shiny_ui.image_loading_ui import image_loading_panel


def imc_panel():
    return ui.nav_panel(
        "IMC",

        ui.navset_tab(
            image_loading_panel(),
            pint_panel(),
            creator_panel(),
            thumbnail_panel(),
            id="image_handler_mode",
        ),

        value="imc",
    )