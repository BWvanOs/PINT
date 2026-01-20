from shiny import App, ui, render
from pint_app.neighborhood.schema import example_input_df, REQUIRED_COLUMNS

app_ui = ui.page_fluid(
    ui.h2("Neighborhood analysis"),

    ui.h4("Required columns"),
    ui.tags.ul(*[ui.tags.li(c) for c in REQUIRED_COLUMNS]),

    ui.h4("Example input format"),
    ui.output_data_frame("schema_example"),
)

def server(input, output, session):

    @render.data_frame
    def schema_example():
        # Shiny renders pandas DataFrames nicely in the browser
        return render.DataGrid(example_input_df(), height="500px")

app = App(app_ui, server)
