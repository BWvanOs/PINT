from shiny import App, ui, render, reactive
import pandas as pd
import tempfile
from datetime import datetime
from pint_app.neighborhood.params import example_input_df, REQUIRED_COLUMNS
from pint_app.neighborhood.csvloading import load_and_validate_csv
from pint_app.neighborhood.neighborhood_analysis import observed_neighbors, chance_corrected_interactions

app_ui = ui.page_fluid(
    ui.layout_columns(
        ui.div(
       
            ui.h2("Neighborhood analysis"),

            ui.h4("Required columns"),
            ui.tags.ul(*[ui.tags.li(c) for c in REQUIRED_COLUMNS]),

            ui.h4("Upload CSV data"),

            ui.input_file("csv_file",
                "Choose a csv file",
                accept=[".csv", ".txt", "text/csv"],
                multiple = False,
            ),


            ui.h4("Example input format"),
            ui.output_data_frame("schema_example"),

            style="border-right: 1px solid #ddd; padding-right: 1rem; height: 100%;",
        ),
        ui.div(
            ui.h2("Load status"),
            ui.output_text_verbatim("load_status"),

            ui.h4("Preview (validated)"),
            ui.output_data_frame("preview"),
            ui.h4("Parameters"),
            ui.input_numeric("radius", "Radius", 50, min=1, step=1),
            ui.input_numeric("n_perm", "N permutations", 1000, min=0, step=100),
            ui.input_action_button("analyze", "Analyze now"),
            ui.output_text_verbatim("analysis_status"),


            ui.hr(),
            ui.h2("Load previous results"),
            ui.h6("Save your results after running, or load previous results. Table below shows active results from neighborhood analysis"),
            ui.layout_columns(
                ui.div(
                    ui.download_button("save_results", "Save results (.csv)"),
                    ui.div(
                        ui.div(
                            ui.input_file(
                                "results_file",
                                None,
                                accept=[".csv", "text/csv"],
                                multiple=False,
                            ),
                            style="flex: 1; min-width: 320px; margin: 0; padding: 0; transform: translateY(15px);",
                        ),
                        ui.input_action_button("load_results", "Load"),
                        style="display: flex; gap: 0.5rem; align-items: center;",
                    ),
                    style="display: flex; gap: 1rem; align-items: center; justify-content: space-between; flex-wrap: wrap;",
                ),
            ),

            style="border-right: 1px solid #ddd; padding-right: 1rem; height: 100%;",
        ),
        ui.div(
            ui.h2("Results from neighborhood analysis"),
            ui.tags.hr(),
            
            ui.output_text_verbatim("results_io_status"),
            ui.output_data_frame("neighbors_preview"),
        ),

        col_widths=(3, 3, 6),
        
    )
)

def server(input, output, session):
    neighbors_val = reactive.Value(None)
    results_io_msg = reactive.Value("")
    analysisStatusVal = reactive.Value("Idle.")

    @render.data_frame
    def schema_example():
        # Shiny renders pandas DataFrames nicely in the browser
        return render.DataGrid(example_input_df(), height="500px")
    
    @reactive.calc
    def validated_df():
        info = input.csv_file()
        if not info:
            return None
        path = info[0]["datapath"]
        return load_and_validate_csv(path)
    
    @reactive.effect
    @reactive.event(input.analyze)
    def _run_analysis():
        df = validated_df()
        if df is None:
            neighbors_val.set(None)
            analysisStatusVal.set("No validated data loaded.")
            return

        # Rough max number of milestones = 4 per sample + a couple global steps
        n_samples = int(df["SampleNumber"].nunique())

        # 4 milestones per sample:
        # 1) sample start
        # 2) association done
        # 3) permutation start
        # 4) permutation done
        total_steps = 4 * n_samples
        step = 0

        with ui.Progress(min=0, max=total_steps, session=session) as p:
            def progress(msg: str) -> None:
                nonlocal step

                # Always update the message so the user sees what's happening
                # Only advance the progress bar on milestone messages.
                milestone = (
                    "Starting neighbor association" in msg
                    or "Neighbor association done" in msg
                    or "Starting permutations" in msg
                    or "Permutations done" in msg
                )

                if milestone and step < total_steps:
                    step += 1

                p.set(step, message=msg)

            p.set(0, message="Starting analysis...")

            res = chance_corrected_interactions(
                df,
                radius=float(input.radius()),
                n_perm=int(input.n_perm()),
                progress=progress,
            )

            neighbors_val.set(res)
            p.set(total_steps, message="Analysis complete.")

    @render.text
    def analysis_status():
            return analysisStatusVal.get()

    @render.text
    def load_status():
        info = input.csv_file()
        if not info:
            return "No file uploaded yet."

        try:
            df = validated_df()
            return f"Loaded and validated: {info[0]['name']} ({len(df):,} rows)"
        except Exception as e:
            return str(e)


    @render.data_frame
    def neighbors_preview():
        res = neighbors_val.get()
        if res is None:
            # show an empty table (or you can show example_input_df(n=0))
            return render.DataGrid(
                observed_neighbors(example_input_df(n=0), radius=1),  # empty structure
                height="500px",
            )
        
        show = res.head(200).copy()

        num_cols = show.select_dtypes(include=["number"]).columns
        show[num_cols] = show[num_cols].round(3)

        return render.DataGrid(show, height="500px")

    @render.data_frame
    def preview():
        df = validated_df()
        if df is None:
            # Return an empty grid so the output is still valid
            empty = example_input_df(n=0)
            return render.DataGrid(empty, height="750px")
        return render.DataGrid(df.head(200), height="750px")

    @render.download(
    filename=lambda: f"neighborhood_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    def save_results():
        res = neighbors_val.get()
        if res is None or (hasattr(res, "empty") and res.empty):
            # Create a tiny CSV so the download still works
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(b"")
            tmp.close()
            return tmp.name

        # Write to a temporary file and return its path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        res.to_csv(tmp.name, index=False)
        tmp.close()
        return tmp.name

    def _do_load_results():
        info = input.results_file()
        if not info:
            results_io_msg.set("No results file selected.")
            return

        path = info[0]["datapath"]
        name = info[0]["name"]

        try:
            loaded = pd.read_csv(path)
            neighbors_val.set(loaded)
            results_io_msg.set(f"Loaded results: {name} ({len(loaded):,} rows)")
        except Exception as e:
            results_io_msg.set(f"Failed to load results: {e}")


    @reactive.effect
    @reactive.event(input.load_results)
    def _confirm_before_load():
        info = input.results_file()
        if not info:
            results_io_msg.set("No results file selected.")
            return

        has_existing = neighbors_val.get() is not None

        if not has_existing:
            _do_load_results()
            return


        # Existing results -> show confirmation modal
        m = ui.modal(
            "Loading new results will delete current results, continue?",
            title="Confirm load",
            easy_close=False,
            fade=False,
            footer=ui.div(
                ui.input_action_button("confirm_load_results", "OK", class_="btn btn-danger"),
                ui.modal_button("Cancel"),
                style="display: flex; gap: 0.5rem; justify-content: flex-end;",
            ),
        )
        ui.modal_show(m)


    @reactive.effect
    @reactive.event(input.confirm_load_results)
    def _confirmed_load():
        ui.modal_remove()
        _do_load_results()

    @render.text
    def results_io_status():
        msg = results_io_msg.get()
        return msg if msg else ""        



app = App(app_ui, server)
