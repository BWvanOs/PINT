from shiny import App, ui

app_ui = ui.page_fluid(
    ui.h2("Neighborhood analysis"),
    ui.p("Standalone for now. Later: Cellpose → clustering → neighborhood."),
)

def server(input, output, session):
    pass

app = App(app_ui, server)
