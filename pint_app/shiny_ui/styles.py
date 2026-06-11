from shiny import ui


def app_styles():
    return ui.head_content(
        ui.tags.style("""
            .controls-left .card {
                border: 1px solid #8f8f8f;
                box-shadow: 0 0.15rem 0.45rem rgba(0, 0, 0, 0.18);
            }

            .pint-main-layout {
                display: flex;
                gap: 0.75rem;
                width: 100%;
                height: calc(100vh - 120px);
                min-height: 0;
            }

            .controls-left {
                flex: 0 0 500px;
                width: 500px;
                max-width: 500px;
                height: 100%;
                overflow-y: auto;
                padding-right: 0.25rem;
            }

            .controls-left .card-header {
                border-bottom: 1px solid #8f8f8f;
                background-color: #e9ecef;
            }

            .controls-left hr {
                border: 0;
                border-top: 1px solid #8f8f8f;
                opacity: 1;
                margin: 0.6rem 0;
            }

            .viewer-navigator {
                flex: 0 0 auto;
                margin-bottom: 0.4rem;
            }

            .viewer-main {
                flex: 1 1 auto;
                min-width: 0;
                height: 100%;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }

            .viewer-navigator .shiny-input-container {
                margin-bottom: 0 !important;
            }

            .viewer-navigator label {
                margin-bottom: 0.15rem;
                font-size: 0.85rem;
                font-weight: 600;
            }

            .viewer-navigator .btn {
                margin-bottom: 0 !important;
            }

            .viewer-plot-fill {
                flex: 1 1 auto;
                min-height: 0;
                display: flex;
                flex-direction: column;
            }

            .viewer-plot-fill .shiny-plot-output {
                flex: 1 1 auto;
                height: 100% !important;
            }

            .creator-header-row {
                padding-left: 0.15rem;
                padding-right: 0.15rem;
            }

            .creator-slot-row {
                margin-bottom: 0.15rem;
            }

            .creator-slot-row .shiny-input-container {
                margin-bottom: 0.2rem !important;
            }

            .creator-slot-row select,
            .creator-slot-row input {
                min-height: 32px;
            }
                      
            /* Sidebar & parameter table */
            .sidebar-col {
                display: flex;
                flex-direction: column;
                height: 100%;
            }

            .param-table-wrap table {
                font-size: 12px;
                width: 100% !important;
                table-layout: auto;
                border-collapse: collapse;
            }

            .param-table-wrap td,
            .param-table-wrap th {
                padding: 2px 4px;
                white-space: nowrap;
                text-overflow: ellipsis;
                overflow: hidden;
                text-align: left;
            }

            .param-table-wrap th {
                font-weight: 750;
                text-align: left;
            }

            /* Make sure the sidebar overlays other content when open */
            .bslib-sidebar-layout > .bslib-sidebar {
                z-index: 1050;
            }

            .bslib-sidebar-layout .bslib-sidebar-toggle {
                z-index: 1060;
            }

            .nav-tabs {
                margin-bottom: 10px;
            }

            .nav-tabs .nav-link {
                font-weight: 600;
            }

            .nav-tabs .nav-link.active {
                background-color: #f8f9fa;
                border-color: #dee2e6 #dee2e6 #fff;
            }
                      
            .seg-citation-card {
                flex: 0 0 auto;
            }
                      
            .seg-citation-title {
                font-size: 1.5rem;
                font-weight: 800;
                line-height: 1.2;
                margin-bottom: 0.15rem;
            }

            .seg-citation-message {
                font-size: 0.92rem;
                line-height: 1.25;
            }

            .seg-citation-card .card-body {
                padding: 0.45rem 0.65rem !important;
            }

            .seg-citation-content {
                font-size: 0.86rem;
                line-height: 1.25;
            }

            .seg-citation-content details {
                margin-top: 0.25rem;
            }

            .seg-citation-content summary {
                cursor: pointer;
                font-weight: 650;
                font-size: 0.88rem;
            }

            .seg-citation-box {
                background: #f8f9fa;
                border: 1px solid #d0d0d0;
                border-radius: 0.35rem;
                padding: 0.45rem;
                white-space: pre-wrap;
                font-size: 0.76rem;
                line-height: 1.2;
                margin-top: 0.35rem;
                margin-bottom: 0.35rem;
                max-height: 160px;
                overflow-y: auto;
            }

            .seg-preview-card {
                flex: 1 1 auto;
                min-height: 0;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .seg-preview-card .card-body {
                flex: 1 1 auto;
                min-height: 0;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .seg-preview-card .tab-content {
                flex: 1 1 auto;
                min-height: 0;
                height: 100%;
                overflow: hidden;
            }

            .seg-preview-card .tab-pane {
                height: 100%;
                min-height: 0;
                overflow: hidden;
            }

            .seg-preview-fill {
                height: 100%;
                min-height: 0;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .seg-preview-fill .shiny-plot-output {
                flex: 1 1 auto;
                height: 100% !important;
                min-height: 0;
            }

            /* Segmentation boundary channel picker */
            .seg-boundary-select-wrap select {
                min-height: 220px !important;
                height: 220px !important;
                max-height: 260px !important;
                overflow-y: auto !important;
            }

            .seg-boundary-select-wrap .selectize-control {
                margin-bottom: 0.4rem !important;
            }

            .seg-boundary-select-wrap .selectize-input {
                min-height: 220px !important;
                max-height: 260px !important;
                overflow-y: auto !important;
                align-content: flex-start;
            }
                      



            .mask-section-title {
                font-weight: 700;
                margin-top: 0.25rem;
                margin-bottom: 0.35rem;
            }

            .mask-divider {
                margin-top: 0.6rem;
                margin-bottom: 0.6rem;
            }

            .compact-stack p {
                margin-bottom: 0.2rem;
            }

            .compact-stack .shiny-input-container {
                margin-bottom: 0.4rem !important;
            }

            .compact-small-line {
                margin-bottom: 0.15rem;
                line-height: 1.2;
            }

            .beta-warning-card {
                border: 1px solid #6ea8fe;
                background-color: #e7f1ff;
                color: #084298;
                border-radius: 0.35rem;
            }

            .beta-warning-title {
                font-weight: 750;
                margin-bottom: 0.35rem;
            }
                      
        """)
    )