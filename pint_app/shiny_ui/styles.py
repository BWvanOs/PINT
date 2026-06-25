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
                      
            /* Fixed-width navigator controls.
            Prevents sample/channel selectors and arrow buttons from stretching on ultrawide screens. */
            .pint-navigator-fixed {
                display: flex !important;
                flex-wrap: nowrap !important;
                align-items: flex-end;
                width: 950px;
                max-width: 100%;
            }

            .pint-navigator-fixed .navigator-select {
                flex: 0 0 375px;
                width: 375px;
                max-width: 375px;
            }

            .pint-navigator-fixed .navigator-button {
                flex: 0 0 50px;
                width: 50px;
                max-width: 50px;
            }

            .pint-navigator-fixed .navigator-spacer {
                flex: 0 0 50px;
                width: 50px;
                max-width: 50px;
            }

            .pint-navigator-fixed .navigator-button .btn {
                width: 100%;
                min-width: 0;
                padding-left: 0;
                padding-right: 0;
            }
                                
            .pint-navigator-half-fixed {
                display: flex !important;
                flex-wrap: nowrap !important;
                align-items: flex-end;
                width: 550px;
                max-width: 100%;
            }
                      
            .pint-navigator-half-fixed .navigator-select {
                flex: 0 0 425px;
                max-width: 425px;
            }

            .pint-navigator-half-fixed .navigator-button {
                flex: 0 0 50px;
                max-width: 50px;
            }

            .pint-navigator-half-fixed .navigator-button .btn {
                width: 100%;
                min-width: 0;
                padding-left: 0;
                padding-right: 0;
            }
                      
            /* PINT tab: fixed pixel navigator, independent of screen width */
            .pint-navigator-grid {
                display: grid !important;
                grid-template-columns: 375px 50px 50px 50px 375px 50px 50px;
                column-gap: 0.25rem;
                align-items: end;
                width: max-content;
                max-width: 100%;
            }

            .pint-navigator-grid .shiny-input-container {
                width: 100% !important;
                margin-bottom: 0 !important;
            }

            .pint-navigator-grid .btn {
                width: 100% !important;
                min-width: 0 !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
                margin-bottom: 0 !important;
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
                      
            /* Remove numeric spinner arrows from creator gain fields */
            .creator-slot-row input[type="number"] {
                appearance: textfield;
                -moz-appearance: textfield;
            }

            /* Chrome, Edge, Safari */
            .creator-slot-row input[type="number"]::-webkit-outer-spin-button,
            .creator-slot-row input[type="number"]::-webkit-inner-spin-button {
                -webkit-appearance: none;
                margin: 0;
            }
            
            .controls-left.controls-left-wide {
                flex-basis: 600px;
                width: 600px;
                max-width: 600px;
            }
                                
            .creator-color-custom-wrap {
                display: flex;
                align-items: center;
                gap: 0.25rem;
            }

            .creator-color-picker {
                width: 34px;
                height: 34px;
                min-width: 34px;
                padding: 0;
                border: 1px solid #8f8f8f;
                border-radius: 0.25rem;
                background: transparent;
                cursor: pointer;
            }

            .creator-color-custom-wrap .shiny-input-container {
                flex: 1 1 auto;
                margin-bottom: 0 !important;
            }

            .creator-color-custom-wrap input[type="text"] {
                height: 34px;
                font-size: 0.8rem;
                padding-left: 0.35rem;
                padding-right: 0.35rem;
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
                      
            /* Keep Shiny progress/notification UI above PINT sidebars */
            #shiny-notification-panel,
            .shiny-notification,
            .shiny-progress-container,
            .shiny-progress {
                z-index: 3000 !important;
            }
                      
        """)
    )