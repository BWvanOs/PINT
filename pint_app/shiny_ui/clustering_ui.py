from shiny import ui


def clustering_panel():
    return ui.nav_panel(
        "Clustering",
        ui.tags.div(
            # ============================================================
            # LEFT CONTROL COLUMN
            # ============================================================
            ui.tags.div(
                ui.card(
                    ui.card_header("Input data"),

                    ui.tags.p(
                        "Load a cell-level quantification table or receive one pushed from "
                        "the segmentation tab.",
                        class_="text-muted",
                    ),

                    ui.input_text(
                        "clustering_table_path",
                        "Clustering table path",
                        value="",
                        placeholder="Select or paste path to .csv, .tsv, or .txt file",
                    ),

                    ui.row(
                        ui.column(
                            6,
                            ui.input_action_button(
                                "browse_clustering_table",
                                "Browse table",
                                class_="btn btn-secondary w-100",
                            ),
                        ),
                        ui.column(
                            6,
                            ui.input_action_button(
                                "load_clustering_table",
                                "Load table",
                                class_="btn btn-primary w-100",
                            ),
                        ),
                    ),

                ui.tags.small(
                        "Browse table to open selection window, or add path directly. Press load to load the file",
                        class_="text-muted",
                    ),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Column selection for clustering"),

                    ui.card(
                        ui.card_header("Cell identity"),

                        ui.tags.p(
                            "Create a unique PINT_Cell_ID before clustering. "
                            "This ID links PCA, PaCMAP, and cluster labels back to the full master dataset.",
                            class_="text-muted",
                        ),

                        ui.input_select(
                            "clustering_cell_id_method",
                            "Cell ID method",
                            choices={
                                "existing": "Use existing unique column",
                                "combine": "Create from selected columns",
                                "row_order": "Create row-order fallback ID",
                            },
                            selected="combine",
                        ),

                        ui.output_ui("clustering_cell_id_controls"),

                        ui.input_action_button(
                            "create_clustering_cell_id",
                            "Create / validate PINT_Cell_ID",
                            class_="btn btn-primary w-100 mb-2",
                        ),

                        ui.output_ui("clustering_cell_id_summary"),

                        class_="mb-2",
                    ),

                    ui.tags.p(
                        "Define which columns are actual marker/channel data for clustering. "
                        "Export a template, edit it, then import it back into PINT.",
                        class_="text-muted",
                    ),

                    ui.tags.p(
                        "Expected columns: ChannelNamesForClustering and ChannelNameToDisplay.",
                        class_="text-muted",
                    ),

                    ui.input_action_button(
                        "export_clustering_column_map",
                        "Export column-names as template",
                        class_="btn btn-secondary w-100 mb-2",
                    ),

                    ui.input_action_button(
                        "import_clustering_column_map",
                        "Import edited column-name template",
                        class_="btn btn-primary w-100 mb-2",
                    ),

                    ui.output_ui("clustering_column_map_summary"),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Dimensionality reduction"),

                    ui.tags.p(
                        "Run PCA on selected marker/channel columns. Use all mapped features or provide "
                        "a smaller marker list for major population discovery.",
                        class_="text-muted",
                    ),

                    ui.input_select(
                        "clustering_pca_feature_mode",
                        "PCA feature mode",
                        choices={
                            "all_mapped": "Use all mapped clustering features",
                            "manual_subset": "Use manually entered feature list",
                        },
                        selected="all_mapped",
                    ),

                    ui.input_text_area(
                        "clustering_pca_feature_list",
                        "Manual PCA feature list",
                        value="",
                        placeholder="One feature per line, or comma-separated. May use source column names or display names.",
                        rows=5,
                    ),

                    ui.input_select(
                        "clustering_transform",
                        "Transformation",
                        choices={
                            "asinh": "arcsinh",
                            "log1p": "log1p",
                            "none": "none",
                        },
                        selected="asinh",
                    ),

                    ui.input_numeric(
                        "clustering_asinh_cofactor",
                        "arcsinh cofactor",
                        value=5,
                        min=0.1,
                        step=0.5,
                    ),

                    ui.input_checkbox(
                        "clustering_scale_data",
                        "Scale data before PCA",
                        value=True,
                    ),

                    ui.input_numeric(
                        "clustering_n_pcs",
                        "Number of PCs",
                        value=20,
                        min=2,
                        step=1,
                    ),

                    ui.input_numeric(
                        "clustering_random_seed",
                        "Random seed",
                        value=1,
                        min=1,
                        step=1,
                    ),

                    ui.input_action_button(
                        "run_clustering_pca",
                        "Run PCA",
                        class_="btn btn-primary w-100 mb-2",
                    ),

                    ui.hr(),

                    ui.input_numeric(
                        "clustering_loadings_n_pcs",
                        "PCs to include in loading plot",
                        value=20,
                        min=1,
                        step=1,
                    ),

                    ui.input_numeric(
                        "clustering_loadings_top_n",
                        "Top features per direction",
                        value=20,
                        min=1,
                        step=1,
                    ),

                    ui.input_action_button(
                        "export_pca_loadings_plot",
                        "Export PCA loading plot",
                        class_="btn btn-secondary w-100 mb-2",
                    ),

                    class_="mb-2",
                ),

                ui.card(
                    ui.card_header("Cluster finding"),

                    ui.tags.p(
                        "Build a k-nearest-neighbor graph from PCA scores and run Leiden clustering.",
                        class_="text-muted",
                    ),

                    ui.input_numeric(
                        "clustering_leiden_n_dims",
                        "PC dimensions for graph",
                        value=10,
                        min=2,
                        step=1,
                    ),

                    ui.input_numeric(
                        "clustering_leiden_n_neighbors",
                        "k-nearest neighbors",
                        value=15,
                        min=2,
                        step=1,
                    ),

                    ui.input_numeric(
                        "clustering_leiden_resolution",
                        "Leiden resolution",
                        value=1.0,
                        min=0.01,
                        step=0.1,
                    ),

                    ui.input_action_button(
                        "run_leiden_clustering",
                        "Run Leiden clustering",
                        class_="btn btn-primary w-100 mb-2",
                    ),

                    ui.hr(),

                    ui.input_select(
                        "clustering_scatter_palette",
                        "Scatter palette",
                        choices={
                            "viridis": "viridis",
                            "magma": "magma",
                            "plasma": "plasma",
                            "inferno": "inferno",
                            "cividis": "cividis",
                            "turbo": "turbo",
                        },
                        selected="viridis",
                    ),

                    ui.input_numeric(
                        "clustering_plot_point_size",
                        "Point size",
                        value=2,
                        min=0.1,
                        step=0.5,
                    ),

                    ui.input_numeric(
                        "clustering_plot_alpha",
                        "Point alpha",
                        value=0.7,
                        min=0.05,
                        max=1,
                        step=0.05,
                    ),

                    class_="mb-2",
                ),

                class_="controls-left",
            ),

            # ============================================================
            # RIGHT MAIN COLUMN
            # ============================================================
            ui.tags.div(
                ui.card(
                    ui.card_header("Clustering workspace"),

                    ui.navset_tab(
                        ui.nav_panel(
                            "Data preparation",
                            ui.tags.div(
                                ui.output_ui("clustering_data_summary"),

                                ui.hr(),

                                ui.tags.div(
                                    "Loaded master dataset preview",
                                    class_="mask-section-title",
                                ),

                                ui.tags.p(
                                    "This is the full loaded master dataset. PINT will only add columns to this table; "
                                    "it should not delete or destructively subset columns.",
                                    class_="text-muted",
                                ),

                                ui.output_data_frame("clustering_data_preview"),

                                ui.hr(),

                                ui.tags.div(
                                    "Selected dataset for clustering",
                                    class_="mask-section-title",
                                ),

                                ui.tags.p(
                                    "This preview shows PINT_Cell_ID plus the marker/channel columns selected by the "
                                    "clustering column map. The protected PINT_Cell_ID is used to map PCA, PaCMAP, "
                                    "and cluster labels back to the full master dataset. It is not used as a clustering feature.",
                                    class_="text-muted",
                                ),

                                ui.output_data_frame("selected_clustering_data_preview"),

                                class_="compact-stack",
                            ),
                        ),

                        ui.nav_panel(
                            "Clustering",
                            ui.tags.div(
                                ui.output_ui("clustering_analysis_summary"),

                                ui.hr(),

                                ui.tags.div(
                                    "PCA plot",
                                    class_="mask-section-title",
                                ),

                                ui.output_plot("clustering_pca_plot", height="500px"),

                                ui.hr(),

                                ui.row(
                                    ui.column(
                                        8,
                                        ui.tags.div(
                                            "PCA variance",
                                            class_="mask-section-title",
                                        ),
                                        ui.output_data_frame("clustering_pca_variance_preview"),
                                    ),
                                    ui.column(
                                        4,
                                        ui.tags.div(
                                            "Leiden cluster counts",
                                            class_="mask-section-title",
                                        ),
                                        ui.output_data_frame("clustering_leiden_counts_preview"),
                                    ),
                                ),

                                class_="compact-stack",
                            ),
                        ),

                        ui.nav_panel(
                            "Cluster names",
                            ui.tags.div(
                                ui.tags.p(
                                    "Cluster annotation and renaming tools will appear here.",
                                    class_="text-muted",
                                ),
                                class_="compact-stack",
                            ),
                        ),

                        id="clustering_workspace_mode",
                    ),

                    class_="seg-preview-card",
                ),

                class_="viewer-main",
            ),

            class_="pint-main-layout",
        ),
        value="clustering",
    )