from shiny import ui


VIRIDIS_CHOICES = {
    "viridis": "viridis",
    "magma": "magma",
    "plasma": "plasma",
    "inferno": "inferno",
    "cividis": "cividis",
    "turbo": "turbo",
}


def input_data_card():
    return ui.card(
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
            "Browse table to open a selection window, or paste a path directly. Press Load to load the file.",
            class_="text-muted",
        ),

        class_="mb-2",
    )


def cell_identity_card():
    return ui.card(
        ui.card_header("Cell identity"),

        ui.tags.p(
            "Create a unique PINT_Cell_ID before clustering. This ID links PCA, "
            "PaCMAP, and cluster labels back to the full master dataset.",
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
            class_="btn btn-primary compact-action-button mb-2",
        ),

        ui.output_ui("clustering_cell_id_summary"),

        class_="mb-2",
    )


def column_selection_card():
    return ui.card(
        ui.card_header("Column selection for clustering"),

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
            class_="btn btn-secondary compact-action-button mb-2",
        ),

        ui.input_action_button(
            "import_clustering_column_map",
            "Import edited column-name template",
            class_="btn btn-primary compact-action-button mb-2",
        ),

        ui.output_ui("clustering_column_map_summary"),

        class_="mb-2",
    )


def pca_controls_card():
    return ui.card(
        ui.card_header("PCA"),

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

        ui.row(
            ui.column(
                4,
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
            ),
            ui.column(
                4,
                ui.input_numeric(
                    "clustering_asinh_cofactor",
                    "arcsinh cofactor",
                    value=5,
                    min=0.1,
                    step=0.5,
                ),
            ),
            ui.column(
                4,
                ui.input_checkbox(
                    "clustering_scale_data",
                    "Scale data before PCA",
                    value=True,
                ),
            ),
        ),

        ui.row(
            ui.column(
                6,
                ui.input_numeric(
                    "clustering_n_pcs",
                    "Number of PCs",
                    value=20,
                    min=2,
                    step=1,
                ),
            ),
            ui.column(
                6,
                ui.input_numeric(
                    "clustering_random_seed",
                    "Random seed",
                    value=1,
                    min=1,
                    step=1,
                ),
            ),
        ),

        ui.input_action_button(
            "run_clustering_pca",
            "Run PCA",
            class_="btn btn-primary compact-action-button mb-2",
        ),

        class_="mb-2",
    )


def pca_loading_export_card():
    return ui.card(
        ui.card_header("PCA loading export"),

        ui.tags.p(
            "Export one image showing the strongest positive and negative marker loadings "
            "per PC.",
            class_="text-muted",
        ),

        ui.row(
            ui.column(
                6,
                ui.input_numeric(
                    "clustering_loadings_n_pcs",
                    "PCs to include",
                    value=20,
                    min=1,
                    step=1,
                ),
            ),
            ui.column(
                6,
                ui.input_numeric(
                    "clustering_loadings_top_n",
                    "Top features per direction",
                    value=20,
                    min=1,
                    step=1,
                ),
            ),
        ),

        ui.input_action_button(
            "export_pca_loadings_plot",
            "Export PCA loading plot",
            class_="btn btn-secondary w-100 mb-2",
        ),

        class_="mb-2",
    )


def leiden_controls_card():
    return ui.card(
        ui.card_header("Leiden clustering"),

        ui.tags.p(
            "Build a k-nearest-neighbor graph from PCA scores and run Leiden clustering.",
            class_="text-muted",
        ),

        ui.row(
            ui.column(
                4,
                ui.input_numeric(
                    "clustering_leiden_n_dims",
                    "PC dimensions",
                    value=10,
                    min=2,
                    step=1,
                ),
            ),
            ui.column(
                4,
                ui.input_numeric(
                    "clustering_leiden_n_neighbors",
                    "k-nearest neighbors",
                    value=15,
                    min=2,
                    step=1,
                ),
            ),
            ui.column(
                4,
                ui.input_numeric(
                    "clustering_leiden_resolution",
                    "Resolution",
                    value=1.0,
                    min=0.01,
                    step=0.1,
                ),
            ),
        ),

        ui.input_action_button(
            "run_leiden_clustering",
            "Run Leiden clustering",
            class_="btn btn-primary compact-action-button mb-2",
        ),

        class_="mb-2",
    )


def embedding_display_controls_card():
    return ui.card(
        ui.card_header("Embedding display"),

        ui.tags.p(
            "Controls shared by PCA and PaCMAP embedding plots.",
            class_="text-muted",
        ),

        ui.row(
            ui.column(
                4,
                ui.input_select(
                    "clustering_scatter_palette",
                    "PCA palette",
                    choices=VIRIDIS_CHOICES,
                    selected="viridis",
                ),
            ),
            ui.column(
                4,
                ui.input_numeric(
                    "clustering_embedding_plot_width",
                    "Plot width",
                    value=1200,
                    min=600,
                    max=1800,
                    step=100,
                ),
            ),
            ui.column(
                2,
                ui.input_numeric(
                    "clustering_plot_point_size",
                    "PCA point size",
                    value=2,
                    min=0.1,
                    step=0.5,
                ),
            ),
            ui.column(
                2,
                ui.input_numeric(
                    "clustering_plot_alpha",
                    "PCA alpha",
                    value=0.7,
                    min=0.05,
                    max=1,
                    step=0.05,
                ),
            ),
        ),

        class_="mb-2",
    )


def pacmap_controls_card():
    return ui.card(
        ui.card_header("PaCMAP"),

        ui.tags.p(
            "Run PaCMAP on PCA scores. This creates a 2D embedding for visual inspection "
            "of Leiden clusters.",
            class_="text-muted",
        ),

        ui.row(
            ui.column(
                3,
                ui.input_numeric(
                    "clustering_pacmap_n_dims",
                    "PC dimensions",
                    value=10,
                    min=2,
                    step=1,
                ),
            ),
            ui.column(
                3,
                ui.input_numeric(
                    "clustering_pacmap_n_neighbors",
                    "n_neighbors",
                    value=10,
                    min=2,
                    step=1,
                ),
            ),
            ui.column(
                3,
                ui.input_numeric(
                    "clustering_pacmap_mn_ratio",
                    "MN_ratio",
                    value=0.5,
                    min=0.0,
                    step=0.1,
                ),
            ),
            ui.column(
                3,
                ui.input_numeric(
                    "clustering_pacmap_fp_ratio",
                    "FP_ratio",
                    value=2.0,
                    min=0.1,
                    step=0.1,
                ),
            ),
        ),

        ui.input_action_button(
            "run_clustering_pacmap",
            "Run PaCMAP",
            class_="btn btn-primary compact-action-button mb-2",
        ),

        class_="mb-2",
    )


def pacmap_display_controls_card():
    return ui.card(
        ui.card_header("PaCMAP display"),

        ui.tags.p(
            "Cluster colors are controlled by the Cluster names and colors card.",
            class_="text-muted",
        ),

        ui.row(
            ui.column(
                6,
                ui.input_numeric(
                    "clustering_pacmap_point_size",
                    "Point size",
                    value=2,
                    min=0.1,
                    step=0.5,
                ),
            ),
            ui.column(
                6,
                ui.input_numeric(
                    "clustering_pacmap_alpha",
                    "Point alpha",
                    value=0.7,
                    min=0.05,
                    max=1,
                    step=0.05,
                ),
            ),
        ),

        class_="mb-2",
    )


def heatmap_controls_card():
    return ui.card(
        ui.card_header("Cluster heatmap export"),

        ui.tags.p(
            "Export cluster × marker heatmaps from Leiden clusters. Use absolute expression "
            "to inspect true marker abundance, or z-scores to emphasize cluster-specific patterns.",
            class_="text-muted",
        ),

        ui.row(
            ui.column(
                4,
                ui.input_select(
                    "clustering_heatmap_aggregation",
                    "Cluster expression summary",
                    choices={
                        "mean": "Mean expression",
                        "median": "Median expression",
                    },
                    selected="mean",
                ),
            ),
            ui.column(
                4,
                ui.input_select(
                    "clustering_heatmap_mode",
                    "Heatmap value mode",
                    choices={
                        "absolute": "Absolute expression",
                        "zscore": "Z-score per marker across clusters",
                    },
                    selected="absolute",
                ),
            ),
            ui.column(
                4,
                ui.input_select(
                    "clustering_heatmap_palette",
                    "Heatmap palette",
                    choices=VIRIDIS_CHOICES,
                    selected="viridis",
                ),
            ),
        ),

        ui.row(
            ui.column(
                6,
                ui.input_select(
                    "clustering_heatmap_feature_mode",
                    "Heatmap features",
                    choices={
                        "all_mapped": "Use all mapped clustering features",
                        "manual_subset": "Use manually entered feature list",
                    },
                    selected="all_mapped",
                ),
            ),
            ui.column(
                6,
                ui.input_numeric(
                    "clustering_heatmap_z_clip",
                    "Z-score clip",
                    value=2,
                    min=0.1,
                    step=0.25,
                ),
            ),
        ),

        ui.input_text_area(
            "clustering_heatmap_feature_list",
            "Manual heatmap feature list",
            value="",
            placeholder="One marker per line, or comma-separated. May use source column names or display names.",
            rows=5,
        ),

        ui.input_action_button(
            "export_cluster_heatmap",
            "Export cluster heatmap",
            class_="btn btn-secondary compact-action-button mb-2",
        ),

        class_="mb-2",
    )

def cluster_annotation_controls_card():
    return ui.card(
        ui.card_header("Cluster names and colors"),

        ui.tags.p(
            "Import a cluster-name CSV with columns OldClusterName and NewClusterName. "
            "OldClusterName should match the PINT Leiden cluster names, for example Cluster_0.",
            class_="text-muted",
        ),

        ui.input_action_button(
            "export_cluster_name_template",
            "Export cluster-name template",
            class_="btn btn-secondary compact-action-button mb-2",
        ),

        ui.input_action_button(
            "import_cluster_name_map",
            "Import cluster names",
            class_="btn btn-primary compact-action-button mb-2",
        ),

        ui.hr(),

        ui.input_select(
            "annotation_cluster_color_palette",
            "Cluster color palette",
            choices={
                "viridis": "viridis",
                "magma": "magma",
                "plasma": "plasma",
                "inferno": "inferno",
                "cividis": "cividis",
                "turbo": "turbo",
                "custom": "custom colors",
            },
            selected="viridis",
        ),

        ui.output_ui("annotation_cluster_custom_color_ui"),

        class_="mb-2",
    )

def annotation_export_controls_card():
    return ui.card(
        ui.card_header("Export annotation results"),

        ui.tags.p(
            "Export publication-quality PaCMAP figures and a full annotation table for plotting "
            "in R, Python, Illustrator, or other tools.",
            class_="text-muted",
        ),

        ui.input_numeric(
            "annotation_export_dpi",
            "Figure export DPI",
            value=1000,
            min=100,
            max=2000,
            step=100,
        ),

        ui.input_numeric(
            "annotation_export_width",
            "Figure width in inches",
            value=8,
            min=2,
            max=20,
            step=0.5,
        ),

        ui.input_action_button(
            "export_pacmap_figure",
            "Export PaCMAP PNG + SVG",
            class_="btn btn-secondary compact-action-button mb-2",
        ),

        ui.input_action_button(
            "export_annotation_csv",
            "Export annotation CSV",
            class_="btn btn-secondary compact-action-button mb-2",
        ),

        class_="mb-2",
    )

def clustering_panel():
    return ui.nav_panel(
        "Clustering",
        ui.tags.div(
            # ============================================================
            # LEFT CONTROL COLUMN: only global / setup controls
            # ============================================================
            ui.tags.div(
                input_data_card(),
                cell_identity_card(),
                column_selection_card(),
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

                                ui.row(
                                    # Main output column
                                    ui.column(
                                        8,
                                        ui.tags.div(
                                            "PCA plot",
                                            class_="mask-section-title",
                                        ),

                                        ui.output_ui("clustering_pca_plot_ui"),

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

                                        class_="clustering-main-column",
                                    ),

                                    # Right control column
                                    ui.column(
                                        4,
                                        ui.tags.div(
                                            pca_controls_card(),
                                            leiden_controls_card(),
                                            embedding_display_controls_card(),
                                            pca_loading_export_card(),
                                            class_="clustering-control-column ms-auto",
                                        ),
                                    ),
                                ),

                                class_="compact-stack",
                            ),
                        ),

                        ui.nav_panel(
                            "Annotation",
                            ui.tags.div(
                                ui.output_ui("clustering_annotation_summary"),

                                ui.hr(),

                                ui.row(
                                    # Main output column
                                    ui.column(
                                        8,
                                        ui.tags.div(
                                            "Cluster annotations",
                                            class_="mask-section-title",
                                        ),

                                        ui.output_data_frame("clustering_cluster_name_map_preview"),

                                        ui.hr(),

                                        ui.tags.div(
                                            "PaCMAP embedding",
                                            class_="mask-section-title",
                                        ),

                                        ui.tags.p(
                                            "PaCMAP is calculated from PCA scores and colored by the current Leiden clusters.",
                                            class_="text-muted",
                                        ),

                                        ui.output_ui("clustering_pacmap_plot_ui"),

                                        ui.hr(),

                                        ui.tags.div(
                                            "Last exported heatmap matrix",
                                            class_="mask-section-title",
                                        ),

                                        ui.tags.p(
                                            "This table shows the most recently exported cluster × marker heatmap matrix.",
                                            class_="text-muted",
                                        ),

                                        ui.output_data_frame("clustering_marker_summary_preview"),

                                        class_="clustering-main-column",
                                    ),

                                    # Right control column
                                    ui.column(
                                        4,
                                        ui.tags.div(
                                            cluster_annotation_controls_card(),
                                            pacmap_controls_card(),
                                            pacmap_display_controls_card(),
                                            heatmap_controls_card(),
                                            annotation_export_controls_card(),
                                            class_="clustering-control-column ms-auto",
                                        ),
                                    ),
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