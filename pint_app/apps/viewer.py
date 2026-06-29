from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from shiny import App, ui, render, reactive
from shiny.types import SilentException

from scipy.ndimage import grey_opening, uniform_filter, median_filter
from scipy import sparse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except Exception:
    ig = None
    leidenalg = None
    LEIDEN_AVAILABLE = False


import os, sys, subprocess
import shutil
import warnings
from datetime import datetime

from tifffile import imwrite, imread

from pint_app.core.load_tiffs import load_tiffs_raw
from pint_app.core.formatting import fmt1
from pint_app.core.mask_neighbors import build_touching_edges_for_pushed_dataset
from pint_app.core.dialogs import (
    pick_folder_dialog,
    pick_open_csv_dialog,
    pick_save_csv_dialog,
    pick_save_png_dialog,
    pick_save_tiff_dialog,
)
from pint_app.core.processing import (
    clamp01,
    strength_to_percentile,
    global_minmax_for_channel as compute_global_minmax_for_channel,
    global_winsor_range_for_channel as compute_global_winsor_range_for_channel,
    image_winsor_range as compute_image_winsor_range,
    process_image_pipeline,
)

from pint_app.core.selection import cycle_list, order_by_canonical

from pint_app.core.params import (
    PARAM_COLUMNS,
    make_params_df,
    update_channel_row,
    format_for_display,
    validate_and_normalize_import,
)

from pint_app.core.load_masks import (
    validate_mask_input_table,
    list_mask_files,
    match_cellmask_names_to_files,
    split_mask_matches,
    get_cells_for_mask_name,
)

from pint_app.core.mask_render_cache import (
    MaskRenderCache,
    make_mask_render_cache_key,
)

from pint_app.core.mask_viz import (
    read_mask_tiff,
    match_mask_centroids_to_cells,
    make_mask_plot_data,
)

from pint_app.core.mask_colors import (
    get_sorted_cluster_names,
    make_palette_color_map,
    make_custom_color_map,
)

from pint_app.core.mask_neighbors_stats import (
    chance_correct_touching_interactions,
    make_sample_interaction_matrix,
    permanova_one_factor,
    aggregate_interaction_matrix,
)

from pint_app.core.composites import (
    MAX_COMPOSITE_CHANNELS,
    COMPOSITE_EMPTY_CHOICE,
    screen_blend_layer,
)

from pint_app.core.mesmer_backend import (
    DEFAULT_MESMER_ENV_NAME,
    check_mesmer_backend,
    get_mesmer_install_commands,
    install_mesmer_backend,
    check_mesmer_gpu,
    run_mesmer_backend,
)

from pint_app.core.segmentation_quantification import quantify_mesmer_masks_for_dataset

##Import of the UI modules
##CSS UI module
from pint_app.shiny_ui.styles import app_styles
##Rest of the ui panels
from pint_app.shiny_ui.creator_ui import creator_panel
from pint_app.shiny_ui.PINT_ui import pint_panel
from pint_app.shiny_ui.segmentation_ui import segmentation_panel
from pint_app.shiny_ui.clustering_ui import clustering_panel
from pint_app.shiny_ui.mask_visualization_ui import mask_visualization_panel
from pint_app.shiny_ui.neighborhood_ui import neighborhood_panel
from pint_app.shiny_ui.advanced_settings_ui import advanced_settings_panel

warnings.filterwarnings("ignore", category=UserWarning, message=".*Tight layout not applied.*")

app_ui = ui.page_sidebar(
    # This is the right side of the window that collapses when opening the sidebar.
    ui.sidebar(
        ui.h4("Current Parameters"),
        ui.tags.div(ui.output_table("param_table"), class_="param-table-wrap"),
        open="closed",          # collapsed by default
        id="sidebar",
        class_="sidebar-col",
        width="850px",          # tweak as needed
    ),

    # This is CSS to fix scaling issues with the viewer. Not that this is 100% made by codex, so edit at your own risk.
    app_styles(),

    # This is the main content of the image handler and normalization settings tools
    ui.row(
        ui.column(
            12,
            ui.tags.div(
                # Toolbar + panels (so essentially everything but the table)
                ui.tags.div(
                    ui.navset_tab(
                        ##------->PINT panel of the shiny app<------##
                        pint_panel(),

                        ##------->Creator panel of the shiny app<------##
                        creator_panel(),

                        ##------->MESMER segmentation panel of the shiny app<------##
                        segmentation_panel(),
                        
                        ##------->Clustering and annotation panel of the shiny app<------##
                        clustering_panel(),

                        ##------->Mask visulaization and niegborhood preperation panel of the shiny app<------##
                        mask_visualization_panel(),

                        ##------->Touching neigborhood panel of the shiny app<------##
                        neighborhood_panel(),
  
                        ##------->Advanced settings and memory diagnostics panel<------##
                        advanced_settings_panel(),
                        id="viewer_mode",
                    ),
                ),
            ),
        ),
    ),

    #IMPORTANT: This must be the last argument!
    position="right",
)



## <----------------> SERVER! <-------------------> ##
def server(input, output, session):
    #Make all the reactive states of the app, will be filled later
    images = reactive.Value({})                  #Dictionary of images, will be filled by tiffloader
    channels = reactive.Value({})                #Dic of channel names
    canonical_channels = reactive.Value([])      #Reference list fr the tabel
    ##Below is the table for all the values the user can adjust . This table is loaded into the collapsable side bar
    params_df = reactive.Value(pd.DataFrame(columns=PARAM_COLUMNS))

    loading = reactive.Value(False)
    setting_selects = reactive.Value(False)
    syncing_controls = reactive.Value(False)
    data_loaded = reactive.Value(False)
    last_loaded_folder = reactive.Value("")  #This stores the last path used to load images so saving throws them into the same folder update here If you want this to cahnge
    
    segmentation_mesmer_mask = reactive.Value(None)
    segmentation_mesmer_result = reactive.Value(None)
    segmentation_mesmer_mask_path = reactive.Value("")
    segmentation_mesmer_batch_results = reactive.Value(pd.DataFrame())
    segmentation_cell_table = reactive.Value(pd.DataFrame())
    segmentation_mask_table = reactive.Value(pd.DataFrame())
    segmentation_cell_table_path = reactive.Value("")
    segmentation_mask_table_path = reactive.Value("")
    segmentation_quantification_status = reactive.Value("No Mesmer mask quantification run yet.")

    # Shared clustering dataset state.
    # This is intentionally one central object used by all clustering sub-tabs:
    # Data preparation, Clustering, and Cluster names.
    #
    # For now this stores a pandas DataFrame from segmentation quantification.
    # Later this will probably become a mini AnnData-like structure to include xenium:
    #   {
    #       "obs": cell metadata,
    #       "X": sparse feature matrix,
    #       "var_names": feature/channel names,
    #       "obsm": PCA/UMAP/PaCMAP embeddings,
    #       "source": input source,
    #   }
    clustering_data = reactive.Value(pd.DataFrame())
    clustering_source = reactive.Value("")
    clustering_status = reactive.Value("No clustering dataset loaded yet.")
    clustering_feature_columns = reactive.Value([])
    clustering_metadata_columns = reactive.Value([])
    clustering_excluded_columns = reactive.Value([])
    ##User defined column to input cluster parameters and names to plot
    clustering_column_map = reactive.Value(pd.DataFrame(columns=["ChannelNamesForClustering", "ChannelNameToDisplay"]))
    # PCA / clustering analysis state.
    clustering_analysis_status = reactive.Value("No PCA or clustering run yet.")
    clustering_feature_matrix = reactive.Value(None)
    clustering_feature_source_columns = reactive.Value([])
    clustering_feature_display_columns = reactive.Value([])
    clustering_pca_scores = reactive.Value(pd.DataFrame())
    clustering_pca_loadings = reactive.Value(pd.DataFrame())
    clustering_pca_variance = reactive.Value(pd.DataFrame())
    clustering_leiden_labels = reactive.Value(pd.DataFrame())
    clustering_marker_summary = reactive.Value(pd.DataFrame())
    # Future subclustering hook.
    # For now None means use all cells.
    # Later this can hold selected PINT_Cell_ID values for major-cluster subclustering. But this is for a later implementation
    clustering_active_cell_ids = reactive.Value(None)

    mask_input_df = reactive.Value(pd.DataFrame())
    mask_files_df = reactive.Value(pd.DataFrame())
    mask_match_df = reactive.Value(pd.DataFrame())
    matched_masks_df = reactive.Value(pd.DataFrame())
    missing_masks_df = reactive.Value(pd.DataFrame())
    selected_mask_match = reactive.Value(pd.DataFrame())
    manual_mask_match_df = reactive.Value(pd.DataFrame())
    final_mask_match_df = reactive.Value(pd.DataFrame())
    maskRenderCache = MaskRenderCache(max_items=25)
    maskRenderDataVersion = reactive.Value(0)

    neighborhood_input_data = reactive.Value(None)
    neighborhood_status_msg = reactive.Value("No mask dataset pushed yet.")
    neighborhood_touching_edges = reactive.Value(pd.DataFrame())
    neighborhood_matched_cells = reactive.Value(pd.DataFrame())
    neighborhood_touching_results = reactive.Value(pd.DataFrame())
    neighborhood_sample_matrix = reactive.Value(pd.DataFrame())
    neighborhood_permanova_results = reactive.Value(pd.DataFrame())

    mesmer_backend_status = reactive.Value(None)
    mesmer_backend_detail_text = reactive.Value(
        "Mesmer backend has not been checked yet.\n\n"
        "This Alpha tab currently only manages optional Mesmer/DeepCell installation."
    ) 
    segmentation_input_data = reactive.Value(None) 
    
    ##Reserved column for the clustering data table
    PINT_CELL_ID_COL = "PINT_Cell_ID"
    RESERVED_CLUSTERING_COLUMNS = {PINT_CELL_ID_COL}



    ## <----------------> Helper functions <-------------------> ##
    def _get_winsor_settings():
        """
        Reads the UI winsor settings, which is clamped to [0,1]. 
        Output is input for winsor_quantiles
        """
        do_w = bool(input.doWinsor())
        lo = clamp01(input.winsor_low())
        hi = clamp01(input.winsor_high())
        return do_w, lo, hi

    #Stores the values from the winzorization so you don't have to recompute every time you return to this channel.
    _global_range_cache = reactive.Value({})  ##(channel, kind, lo, hi) -> (gmin,gmax). eg ("CD45", "winsor", 0.01, 0.99): (12.3, 845.7)
    def _global_range_for_channel(images_dict: dict, channels_dict: dict,
                                channel_name: str, do_winsor: bool, lo: float, hi: float):
        """
        If winsor is enabled and hi>lo*: global range is min of per-image q_lo, max of per-image q_hi. This return the global high and low for a channel
        Otherwise: fallback (so if doWindor is FALSE) is to retturn the raw global min/max.
        *There is no check on this in the interface
        """
        ##key construct a cache key. If there is an exact match (so same channel, values and wins) in the global cache it'll pull it from there. If not it will compute it from there
        key = (channel_name, "winsor" if (do_winsor and hi > lo) else "raw",
            round(lo, 6), round(hi, 6))
        cache = _global_range_cache.get()
        if key in cache:
            return cache[key]
        ##If not in the cache, run the winsorization on the array:
        ##This is the CPU heavy part
        if do_winsor and hi > lo:
            # Compute global winsor range (min of per-sample q_lo, max of per-sample q_hi)
            result = compute_global_winsor_range_for_channel(images_dict, channels_dict, channel_name, lo, hi)
        ##Fallback if winsorization is off
        else:
            result = _global_minmax_for_channel(images_dict, channels_dict, channel_name)
        ##Stores it in the global cache and also return the results
        cache[key] = result
        _global_range_cache.set(cache)
        return result


    def _image_range_for_channel(images_dict: dict, channels_dict: dict,
                                channel_name: str, sample: str,
                                do_winsor: bool, lo: float, hi: float):
        """Per-image range for current sample; winsorized if enabled (and hi>lo), else return raw values."""
        ##Same of the previous one, but then per channel. No cache is stored as the per image winsor is not that heavy.
        if not sample or sample not in images_dict:
            return None
        chlist = channels_dict.get(sample, [])
        if channel_name not in chlist:
            return None
        idx = chlist.index(channel_name)
        arr = images_dict[sample][idx]
        if do_winsor and hi > lo:
            return compute_image_winsor_range(images_dict, channels_dict, channel_name, sample, lo, hi)
        # raw fallback
        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not (np.isfinite(mn) and np.isfinite(mx)):
            return None
        return mn, mx


    # File/folder dialogs were moved to HelperFiles.dialogs

    def _prefill_params(first_chlist: list[str]) -> None:
        """
        This is loaded after the _do_load() to prepare the dataframe with parameters
        """
        # Read the current UI values once and apply them to all channels
        s = float(input.noise_strength())
        p = strength_to_percentile(s)

        df = make_params_df(
            first_chlist,
            do_winsor=bool(input.doWinsor()),
            low=float(input.winsor_low()),
            high=float(input.winsor_high()),
            do_threshold=bool(input.doThreshold()),
            thr_val=float(input.thr_fraction_val()),
            do_abs_threshold=bool(input.doAbsThreshold()),
            abs_thr_val=float(input.abs_threshold_val()),
            do_noise=bool(input.doNoise()),
            noise_strength=s,
            noise_percentile=p,
            window_size=int(input.window_size()),
            do_norm=bool(input.doNorm()),
            norm_scope=(input.norm_scope() or "page"),
            do_asinh=bool(input.doAsinh()),
            cofactor=int(float(input.asinh_cofactor() or 5)),
        )
        params_df.set(df)

    def _sync_controls_from_table(channel: str) -> None:
        """
        This handles the defaults for the empty dataframe when loading the parameters
        """
        if not channel:
            return
        df = params_df.get()
        if df.empty:
            return
        m = (df["Channel"] == channel)
        if not m.any():
            return
        row = df.loc[m].iloc[0]
        syncing_controls.set(True)
        ##This is how each value is determines. Eg, doWinsor -> read out current value and return it. Fallback is FALSE
        try:
            session.send_input_message("doWinsor", {"value": bool(row.get("DoWinsor", False))})
            session.send_input_message("winsor_low", {"value": float(row.get("Low", 0.0))})
            session.send_input_message("winsor_high", {"value": float(row.get("High", 1.0))})

            session.send_input_message("doThreshold", {"value": bool(row.get("DoThr", False))})
            session.send_input_message("thr_fraction_val", {"value": float(row.get("ThrVal", 0.0))})

            session.send_input_message("doAbsThreshold", {"value": bool(row.get("DoAbsThr", False))})
            session.send_input_message("abs_threshold_val", {"value": float(row.get("AbsThrVal", 0.0))})

            session.send_input_message("doNoise", {"value": bool(row.get("Noise", False))})
            winsz = int(row.get("WinSz", int(input.window_size())))
            session.send_input_message("noise_strength", {"value": float(row.get("NStr", 1.0))})
            session.send_input_message("window_size", {"value": winsz})

            # Normalize checkbox (if present)
            session.send_input_message("doNorm", {"value": bool(row.get("DoNorm", True))})
            session.send_input_message("norm_scope", {"value": str(row.get("NormScope", "page"))})

            ##arcsinh controls
            session.send_input_message("doAsinh", {"value": bool(row.get("DoAsinh", False))})
            session.send_input_message("asinh_cofactor", {"value": str(int(row.get("Cofac", 5)))})
        finally:
            syncing_controls.set(False)
    
    # The following helpers have been moved to HelperFiles/*.
    # We keep these thin wrappers to minimize churn in the rest of viewer.py.
    def _cycle(lst, current, step):
        return cycle_list(lst, current, step)

    WINSOR_MIN_UPPER_BOUND = 5.0

    def _strength_to_percentile(s: float, eps: float = 0.005) -> float:
        return strength_to_percentile(s, eps=eps)

    ##Cache for global min/max per channel (invalidated on load)
    ##Also stored previous values so moving back and forth is cached
    _global_minmax_cache = reactive.Value({})   ##looks like {channel_name: (gmin, gmax)}

    def _invalidate_global_cache():
        _global_minmax_cache.set({})
        _global_range_cache.set({})


    def _global_minmax_for_channel(images_dict: dict, channels_dict: dict, channel_name: str):
        """Return (gmin, gmax) across all samples for the given channel name.
        Uses & updates a reactive cache. Skips samples that lack this channel, 
        so it is compatible with samples with a different channel layout to harmonize samples
        """
        ##Checks cache first. This is invalidated on loading. This solved to problem that the cache has old values when loading images
        ##Invalid channel will return 0. Only needs 1 image in the channel to return something to return something relevant
        ##entry examples:
        ##  "CD45": (12.3, 845.7), 
        ##  "PanCK": (0.0, 501.2),
        ##  "FOXP3": None,
        cache = _global_minmax_cache.get()
        if channel_name in cache:
            return cache[channel_name]

        result = compute_global_minmax_for_channel(images_dict, channels_dict, channel_name)
        cache[channel_name] = result
        _global_minmax_cache.set(cache)
        return result

    def _get_channel_param_row(channelName: str):
        df = params_df.get()
        if df.empty or not channelName:
            return None
        m = (df["Channel"] == channelName)
        if not m.any():
            return None
        return df.loc[m].iloc[0]

    def _process_channel_from_table(sampleName: str, channelName: str) -> np.ndarray | None:
        imgs = images.get()
        chs = channels.get()

        if not sampleName or not channelName or sampleName not in imgs:
            return None

        chlist = chs.get(sampleName, [])
        if channelName not in chlist:
            return None

        idx = chlist.index(channelName)
        img = imgs[sampleName][idx, :, :].astype(np.float32)

        row = _get_channel_param_row(channelName)
        if row is None:
            return None

        doWin = bool(row.get("DoWinsor", False))
        low = float(row.get("Low", 0.0))
        high = float(row.get("High", 1.0))

        doAbsThr = bool(row.get("DoAbsThr", False))
        absThrVal = float(row.get("AbsThrVal", 0.0))

        doThr = bool(row.get("DoThr", False))
        thrVal = float(row.get("ThrVal", 0.0))

        doNoise = bool(row.get("Noise", False))
        noiseStrength = float(row.get("NStr", 0.0))
        winSz = int(row.get("WinSz", 3))

        doNorm = bool(row.get("DoNorm", True))
        normScope = str(row.get("NormScope", "page")).strip().lower()

        doAsinh = bool(row.get("DoAsinh", False))
        cofac = int(row.get("Cofac", 5))

        normVmin = None
        normVmax = None
        if doNorm and normScope == "global":
            gpair = _global_range_for_channel(
                images.get(),
                channels.get(),
                channelName,
                doWin,
                low,
                high,
            )
            if gpair is not None:
                normVmin, normVmax = gpair

        return process_image_pipeline(
            img,
            do_winsor=doWin,
            winsor_low=low,
            winsor_high=high,
            do_abs_threshold=doAbsThr,
            abs_threshold=absThrVal,
            do_fraction_threshold=doThr,
            thr_fraction=thrVal,
            do_noise=doNoise,
            noise_strength=noiseStrength,
            window_size=winSz,
            do_asinh=doAsinh,
            asinh_cofactor=cofac,
            do_norm=doNorm,
            norm_vmin=normVmin,
            norm_vmax=normVmax,
            winsor_min_upper_bound=WINSOR_MIN_UPPER_BOUND,
        )
    

    def _build_composite_rgb(sampleName: str | None = None):
        if sampleName is None:
            sampleName = input.sample()
        if not sampleName:
            return None, []

        rgb = None
        used = []

        for slotIdx in range(1, MAX_COMPOSITE_CHANNELS + 1):

            channelName = getattr(input, f"comp_channel_{slotIdx}")()
            colorName = getattr(input, f"comp_color_{slotIdx}")()
            customHex = getattr(input, f"comp_hex_{slotIdx}")()
            gain = float(getattr(input, f"comp_gain_{slotIdx}")())

            if customHex:
                customHex = str(customHex).strip()
                if customHex.startswith("#") and len(customHex) == 7:
                    colorName = customHex

            if not channelName or channelName == COMPOSITE_EMPTY_CHOICE:
                continue

            proc = _process_channel_from_table(sampleName, channelName)
            if proc is None:
                continue

            if rgb is None:
                h, w = proc.shape
                rgb = np.zeros((h, w, 3), dtype=np.float32)

            rgb = screen_blend_layer(rgb, proc, colorName, gain)

            used.append((channelName, colorName, gain))

        if rgb is None or len(used) == 0:
            return None, []

        rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
        return rgb, used

    def _sync_composite_channel_choices(sampleName: str, overwriteDefaults: bool = False) -> None:
        if not sampleName:
            return

        chlistCurrent = channels.get().get(sampleName, [])
        canon = canonical_channels.get() or []
        ordered = order_by_canonical(canon, chlistCurrent)

        choices = [COMPOSITE_EMPTY_CHOICE] + ordered

        # Default to the 3rd channel because the first 2 are usually Argon/Krypton.
        # If fewer than 3 channels exist, fall back to the first real channel.
        firstDefault = ordered[2] if len(ordered) >= 3 else (ordered[0] if ordered else COMPOSITE_EMPTY_CHOICE)

        for slotIdx in range(1, MAX_COMPOSITE_CHANNELS + 1):
            inputId = f"comp_channel_{slotIdx}"
            currentVal = getattr(input, inputId)()

            if slotIdx == 1:
                defaultVal = firstDefault
            else:
                defaultVal = COMPOSITE_EMPTY_CHOICE

            if overwriteDefaults:
                selectedVal = defaultVal
            else:
                selectedVal = currentVal if currentVal in ordered else defaultVal

            ui.update_select(
                inputId,
                choices=choices,
                selected=selectedVal,
                session=session,
            )

    def _get_selected_creator_channels():
        selected = []
        for slotIdx in range(1, MAX_COMPOSITE_CHANNELS + 1):
            channelName = getattr(input, f"comp_channel_{slotIdx}")()

            if not channelName or channelName == COMPOSITE_EMPTY_CHOICE:
                continue

            colorName = getattr(input, f"comp_color_{slotIdx}")()
            gain = float(getattr(input, f"comp_gain_{slotIdx}")())
            selected.append((channelName, colorName, gain))

        return selected


    def _pick_default_nuclear_channel(chlist: list[str]) -> str | None:
        if not chlist:
            return None

        nuclearHints = [
            "dna",
            "iridium",
            "histone",
            "hoechst",
            "dapi",
        ]

        for hint in nuclearHints:
            for ch in chlist:
                if hint in ch.lower():
                    return ch

        return chlist[0]


    def _pick_default_boundary_channels(chlist: list[str]) -> list[str]:
        if not chlist:
            return []

        boundaryHints = [
            "ecad",
            "e-cad",
            "cadherin",
            "panck",
            "cytokeratin",
            "keratin",
            "cd45",
            "vimentin",
            "vim",
            "sma",
            "actin",
            "cd31",
            "cd3",
            "cd20",
            "cd68",
            "cd90",
        ]

        selected = []

        for hint in boundaryHints:
            for ch in chlist:
                if ch in selected:
                    continue
                if hint in ch.lower():
                    selected.append(ch)

        return selected[:8]


    def _sync_segmentation_channel_choices(sampleName: str | None = None) -> None:
        obj = segmentation_input_data.get()

        if obj is None:
            ui.update_select("seg_sample", choices=[], selected=None, session=session)
            ui.update_select("seg_nuclear_channel", choices=[], selected=None, session=session)
            ui.update_select("seg_boundary_channels", choices=[], selected=None, session=session)
            return

        imgs = obj.get("images", {})
        chs = obj.get("channels", {})
        canon = obj.get("canonical_channels", [])

        samples = list(imgs.keys())
        if not samples:
            return

        if sampleName is None or sampleName not in samples:
            sampleName = samples[0]

        chlistCurrent = chs.get(sampleName, [])
        ordered = order_by_canonical(canon, chlistCurrent)

        nuclearDefault = _pick_default_nuclear_channel(ordered)
        boundaryDefault = _pick_default_boundary_channels(ordered)

        ui.update_select(
            "seg_sample",
            choices=samples,
            selected=sampleName,
            session=session,
        )

        ui.update_select(
            "seg_nuclear_channel",
            choices=ordered,
            selected=nuclearDefault,
            session=session,
        )

        ui.update_select(
            "seg_boundary_channels",
            choices=ordered,
            selected=boundaryDefault,
            session=session,
        )


    def _robust_normalize_for_segmentation(
        img: np.ndarray,
        lowQuantile: float = 0.005,
        highQuantile: float = 0.998,
    ) -> np.ndarray:
        """
        Gentle segmentation-oriented normalization.

        This is intentionally softer than the visualization pipeline:
        - percentile clipping
        - no hard biological threshold
        - no aggressive background deletion
        - output clipped to 0..1
        """
        arr = img.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.size == 0:
            return arr

        try:
            lo = float(np.nanquantile(arr, lowQuantile))
            hi = float(np.nanquantile(arr, highQuantile))
        except Exception:
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                return np.zeros_like(arr, dtype=np.float32)
            lo, hi = mn, mx

        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)

        return arr


    def _get_raw_channel_image_from_segmentation_push(
        sampleName: str,
        channelName: str,
    ) -> np.ndarray | None:
        obj = segmentation_input_data.get()

        if obj is None:
            return None

        imgs = obj.get("images", {})
        chs = obj.get("channels", {})

        if sampleName not in imgs:
            return None

        chlist = chs.get(sampleName, [])

        if channelName not in chlist:
            return None

        idx = chlist.index(channelName)
        return imgs[sampleName][idx, :, :].astype(np.float32)


    def _process_channel_for_segmentation(
        sampleName: str,
        channelName: str,
        mode: str,
    ) -> np.ndarray | None:
        raw = _get_raw_channel_image_from_segmentation_push(sampleName, channelName)

        if raw is None:
            return None

        mode = (mode or "soft").strip().lower()

        if mode == "raw":
            # Still normalize for display / model input scaling, but do not use PINT settings.
            return _robust_normalize_for_segmentation(
                raw,
                lowQuantile=0.0,
                highQuantile=1.0,
            )

        if mode == "pint":
            # Uses current PINT channel settings from params_df.
            # This is intentionally marked as not recommended because it may include hard thresholding.
            proc = _process_channel_from_table(sampleName, channelName)
            if proc is None:
                return None
            return np.clip(
                np.nan_to_num(proc.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )

        # Recommended soft segmentation preprocessing.
        return _robust_normalize_for_segmentation(
            raw,
            lowQuantile=0.005,
            highQuantile=0.998,
        )


    def _get_selected_segmentation_channels():
        sampleName = input.seg_sample()
        nuclearChannel = input.seg_nuclear_channel()
        boundaryChannels = input.seg_boundary_channels()
        mode = input.seg_preprocess_mode() or "soft"

        if isinstance(boundaryChannels, str):
            boundaryChannels = [boundaryChannels]
        elif boundaryChannels is None:
            boundaryChannels = []
        else:
            boundaryChannels = list(boundaryChannels)

        return sampleName, nuclearChannel, boundaryChannels, mode


    def _build_segmentation_preview_images():
        obj = segmentation_input_data.get()

        if obj is None:
            return None, None, None, "No images pushed from PINT."

        sampleName, nuclearChannel, boundaryChannels, mode = _get_selected_segmentation_channels()

        if not sampleName or not nuclearChannel:
            return None, None, None, "Select a sample and nuclear channel."

        nuclearImg = _process_channel_for_segmentation(
            sampleName=sampleName,
            channelName=nuclearChannel,
            mode=mode,
        )

        if nuclearImg is None:
            return None, None, None, "Could not build nuclear input."

        boundaryImgs = []
        for ch in boundaryChannels:
            img = _process_channel_for_segmentation(
                sampleName=sampleName,
                channelName=ch,
                mode=mode,
            )
            if img is not None:
                boundaryImgs.append(img)

        if boundaryImgs:
            boundaryStack = np.stack(boundaryImgs, axis=0)
            boundaryImg = np.nanmax(boundaryStack, axis=0).astype(np.float32)
        else:
            boundaryImg = np.zeros_like(nuclearImg, dtype=np.float32)

        combinedRgb = np.zeros((*nuclearImg.shape, 3), dtype=np.float32)
        combinedRgb[..., 0] = boundaryImg
        combinedRgb[..., 1] = nuclearImg
        combinedRgb[..., 2] = nuclearImg
        combinedRgb = np.clip(combinedRgb, 0.0, 1.0)

        return nuclearImg, boundaryImg, combinedRgb, None


    def _make_single_image_preview_figure(
        img,
        title: str,
        cmap: str | None = "gray",
        errorMessage: str | None = None,
    ):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

        if errorMessage:
            ax.text(
                0.5,
                0.5,
                errorMessage,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

        if img is None:
            ax.text(
                0.5,
                0.5,
                "No preview available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

        if cmap is None:
            ax.imshow(img, interpolation="nearest")
        else:
            ax.imshow(img, cmap=cmap, interpolation="nearest")

        ax.set_axis_off()

        # Small overlay label instead of matplotlib title, to avoid extra whitespace.
        if title:
            ax.text(
                0.01,
                0.99,
                title,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="white",
                bbox=dict(
                    facecolor="black",
                    alpha=0.55,
                    edgecolor="none",
                    pad=4,
                ),
            )

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig

    def _safe_file_stem(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(name))


    def _get_segmentation_output_dir() -> Path:
        obj = segmentation_input_data.get()

        folder = None
        if obj is not None:
            folder = obj.get("folder", None)

        if folder and os.path.isdir(folder):
            outDir = Path(folder) / "segmentation alpha"
        else:
            outDir = Path.cwd() / "segmentation alpha"

        outDir.mkdir(parents=True, exist_ok=True)
        return outDir


    def _save_current_mesmer_inputs():
        sampleName = input.seg_sample()
        return _save_mesmer_inputs_for_sample(sampleName)

    def _save_mesmer_inputs_for_sample(sampleName: str):
        nuclearChannel = input.seg_nuclear_channel()
        boundaryChannels = input.seg_boundary_channels()
        mode = input.seg_preprocess_mode() or "soft"

        if isinstance(boundaryChannels, str):
            boundaryChannels = [boundaryChannels]
        elif boundaryChannels is None:
            boundaryChannels = []
        else:
            boundaryChannels = list(boundaryChannels)

        if not sampleName:
            raise ValueError("No sample selected.")

        if not nuclearChannel:
            raise ValueError("No nuclear channel selected.")

        nuclearImg = _process_channel_for_segmentation(
            sampleName=sampleName,
            channelName=nuclearChannel,
            mode=mode,
        )

        if nuclearImg is None:
            raise ValueError(f"Could not build nuclear input for {sampleName}.")

        boundaryImgs = []

        for ch in boundaryChannels:
            img = _process_channel_for_segmentation(
                sampleName=sampleName,
                channelName=ch,
                mode=mode,
            )
            if img is not None:
                boundaryImgs.append(img)

        if boundaryImgs:
            boundaryStack = np.stack(boundaryImgs, axis=0)
            boundaryImg = np.nanmax(boundaryStack, axis=0).astype(np.float32)
        else:
            boundaryImg = np.zeros_like(nuclearImg, dtype=np.float32)

        outDir = _get_segmentation_output_dir()
        stem = _safe_file_stem(sampleName)

        nuclearPath = outDir / f"{stem}_mesmer_nuclear_input.tiff"
        boundaryPath = outDir / f"{stem}_mesmer_boundary_input.tiff"
        inputStackPath = outDir / f"{stem}_mesmer_input_stack.tiff"
        maskPath = outDir / f"{stem}_mesmer_mask_uint32.tiff"
        jsonPath = outDir / f"{stem}_mesmer_summary.json"

        nuclearU16 = np.clip(np.round(nuclearImg * 65535.0), 0, 65535).astype(np.uint16)
        boundaryU16 = np.clip(np.round(boundaryImg * 65535.0), 0, 65535).astype(np.uint16)

        imwrite(nuclearPath, nuclearU16, dtype=np.uint16)
        imwrite(boundaryPath, boundaryU16, dtype=np.uint16)

        inputStack = np.stack([nuclearU16, boundaryU16], axis=0)
        imwrite(inputStackPath, inputStack, dtype=np.uint16)

        meta = {
            "SampleName": sampleName,
            "NuclearChannel": nuclearChannel,
            "BoundaryChannels": ";".join(boundaryChannels),
            "PreprocessMode": mode,
            "NuclearInput": str(nuclearPath),
            "BoundaryInput": str(boundaryPath),
            "InputStack": str(inputStackPath),
            "MaskOutput": str(maskPath),
            "SummaryJson": str(jsonPath),
        }

        return nuclearPath, boundaryPath, maskPath, jsonPath, meta    

    def _build_pint_processed_stack_for_sample(sampleName: str) -> tuple[np.ndarray, list[str]]:
        imgs = images.get()
        chs = channels.get()

        if sampleName not in imgs:
            raise ValueError(f"Sample not found: {sampleName}")

        channelNames = chs.get(sampleName, [])
        processedChannels = []

        for channelName in channelNames:
            proc = _process_channel_from_table(sampleName, channelName)

            if proc is None:
                raise ValueError(
                    f"Could not process channel '{channelName}' for sample '{sampleName}'."
                )

            processedChannels.append(proc.astype(np.float32))

        stack = np.stack(processedChannels, axis=0)

        return stack, channelNames
    
    def _build_pint_processed_dataset_for_quantification() -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        obj = segmentation_input_data.get()

        if obj is None:
            raise ValueError("No images pushed to Segmentation.")

        imgs = obj.get("images", {})
        processedImages = {}
        processedChannels = {}

        for sampleName in imgs.keys():
            stack, channelNames = _build_pint_processed_stack_for_sample(sampleName)
            processedImages[sampleName] = stack
            processedChannels[sampleName] = channelNames

        return processedImages, processedChannels
    
    def _make_default_clustering_column_map(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a default clustering column map from the currently loaded table.

        Reserved columns such as PINT_Cell_ID are not exported as clustering features.
        """
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["ChannelNamesForClustering", "ChannelNameToDisplay"]
            )

        exportCols = [
            c for c in df.columns
            if c not in RESERVED_CLUSTERING_COLUMNS
        ]

        return pd.DataFrame(
            {
                "ChannelNamesForClustering": exportCols,
                "ChannelNameToDisplay": exportCols,
            }
        )

    def _parse_feature_list_text(text: str) -> list[str]:
        """
        Parse a user-provided feature list.

        Accepts comma-separated, semicolon-separated, or newline-separated values.
        """
        text = str(text or "").replace(";", "\n").replace(",", "\n")
        values = [x.strip() for x in text.splitlines()]
        return [x for x in values if x]


    def _get_pca_feature_map() -> pd.DataFrame:
        """
        Return the feature map used for PCA/clustering.

        Starts from the validated clustering column map.
        Optionally filters to a manually entered feature subset.
        The manual list may contain either source column names or display names.
        """
        validMap = _get_valid_clustering_column_map()

        expectedCols = ["ChannelNamesForClustering", "ChannelNameToDisplay"]

        if validMap is None or validMap.empty:
            return pd.DataFrame(columns=expectedCols)

        featureMode = input.clustering_pca_feature_mode() or "all_mapped"

        if featureMode == "manual_subset":
            requested = _parse_feature_list_text(input.clustering_pca_feature_list())

            if not requested:
                return pd.DataFrame(columns=expectedCols)

            requestedSet = set(requested)

            validMap = validMap.loc[
                validMap["ChannelNamesForClustering"].isin(requestedSet)
                | validMap["ChannelNameToDisplay"].isin(requestedSet)
            ].copy()

        return validMap.reset_index(drop=True)

    def _prepare_clustering_feature_matrix() -> tuple[np.ndarray, pd.DataFrame, list[str], list[str]]:
        """
        Build the numeric feature matrix for PCA/clustering.

        Returns:
        - X: numeric matrix, cells × selected features
        - obs: dataframe with PINT_Cell_ID
        - sourceCols: original dataframe column names
        - displayCols: display names used in plots
        """
        df = clustering_data.get()

        if df is None or df.empty:
            raise ValueError("No clustering dataset loaded.")

        if PINT_CELL_ID_COL not in df.columns:
            raise ValueError("Create/validate PINT_Cell_ID before running PCA.")

        featureMap = _get_pca_feature_map()

        if featureMap is None or featureMap.empty:
            raise ValueError("No valid feature columns selected for PCA/clustering.")

        sourceCols = featureMap["ChannelNamesForClustering"].tolist()
        displayCols = featureMap["ChannelNameToDisplay"].tolist()

        missingCols = [c for c in sourceCols if c not in df.columns]
        if missingCols:
            raise ValueError(
                "Selected feature columns are missing from clustering_data: "
                + ", ".join(missingCols)
            )

        # Optional future subclustering.
        activeIds = clustering_active_cell_ids.get()

        if activeIds is not None:
            activeIds = set(map(str, activeIds))
            useDf = df.loc[df[PINT_CELL_ID_COL].astype(str).isin(activeIds)].copy()
        else:
            useDf = df

        if useDf.empty:
            raise ValueError("No cells available for the current clustering run.")

        obs = useDf[[PINT_CELL_ID_COL]].copy()

        Xdf = useDf.loc[:, sourceCols].copy()
        Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
        Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

        # Drop features that are entirely missing.
        allMissing = Xdf.columns[Xdf.isna().all()].tolist()
        if allMissing:
            keepMask = ~Xdf.columns.isin(allMissing)
            Xdf = Xdf.loc[:, keepMask].copy()

            featureMap = featureMap.loc[keepMask].copy()
            sourceCols = featureMap["ChannelNamesForClustering"].tolist()
            displayCols = featureMap["ChannelNameToDisplay"].tolist()

        if Xdf.shape[1] == 0:
            raise ValueError("No numeric feature columns remain after filtering.")

        # Fill missing values by marker median.
        medians = Xdf.median(axis=0, numeric_only=True)
        Xdf = Xdf.fillna(medians).fillna(0)

        transform = input.clustering_transform() or "asinh"
        cofactor = float(input.clustering_asinh_cofactor() or 5)

        X = Xdf.to_numpy(dtype=np.float32, copy=True)

        if transform == "asinh":
            if cofactor <= 0:
                raise ValueError("asinh cofactor must be > 0.")
            X = np.arcsinh(X / cofactor).astype(np.float32, copy=False)

        elif transform == "log1p":
            X = np.clip(X, a_min=0, a_max=None)
            X = np.log1p(X).astype(np.float32, copy=False)

        elif transform == "none":
            pass

        else:
            raise ValueError(f"Unknown transform: {transform}")

        if bool(input.clustering_scale_data()):
            scaler = StandardScaler(copy=True)
            X = scaler.fit_transform(X).astype(np.float32, copy=False)

        return X, obs, sourceCols, displayCols


    def _make_pca_loadings_grid_figure(
        loadingsDf: pd.DataFrame,
        *,
        n_pcs: int = 20,
        top_n: int = 20,
        n_cols: int = 5,
    ):
        """
        Build one multi-panel PCA loading figure.

        Each panel shows the top positive and top negative feature loadings
        for one PC, similar in spirit to Seurat's PCA loading plots.
        """
        if loadingsDf is None or loadingsDf.empty:
            raise ValueError("No PCA loadings available. Run PCA first.")

        if "Feature" not in loadingsDf.columns:
            raise ValueError("PCA loadings table does not contain a Feature column.")

        pcCols = [c for c in loadingsDf.columns if c.startswith("PC_")]

        if not pcCols:
            raise ValueError("No PC loading columns found.")

        n_pcs = max(1, min(int(n_pcs), len(pcCols)))
        top_n = max(1, int(top_n))
        n_cols = max(1, int(n_cols))

        usePcCols = pcCols[:n_pcs]

        n_rows = int(np.ceil(len(usePcCols) / n_cols))

        # Width/height chosen to keep feature labels readable.
        fig_width = n_cols * 4.2
        fig_height = n_rows * 6.0

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            dpi=160,
            squeeze=False,
        )

        for ax in axes.ravel():
            ax.set_axis_off()

        for i, pcCol in enumerate(usePcCols):
            ax = axes.ravel()[i]
            ax.set_axis_on()

            temp = loadingsDf[["Feature", pcCol]].copy()
            temp = temp.rename(columns={pcCol: "Loading"})
            temp["Loading"] = pd.to_numeric(temp["Loading"], errors="coerce")
            temp = temp.replace([np.inf, -np.inf], np.nan).dropna(subset=["Loading"])

            if temp.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"{pcCol}\nNo valid loadings",
                    ha="center",
                    va="center",
                )
                ax.set_axis_off()
                continue

            top_pos = (
                temp
                .sort_values("Loading", ascending=False)
                .head(top_n)
            )

            top_neg = (
                temp
                .sort_values("Loading", ascending=True)
                .head(top_n)
            )

            plotDf = pd.concat([top_neg, top_pos], ignore_index=True)
            plotDf = plotDf.drop_duplicates(subset=["Feature"], keep="first")

            # Sort for a clean horizontal bar plot.
            plotDf = plotDf.sort_values("Loading", ascending=True).copy()

            y = np.arange(len(plotDf))

            colors = [
                "#3B528B" if v < 0 else "#8B0000"
                for v in plotDf["Loading"]
            ]

            ax.barh(
                y,
                plotDf["Loading"],
                color=colors,
                edgecolor="none",
            )

            ax.axvline(0, color="black", linewidth=0.7)

            ax.set_yticks(y)
            ax.set_yticklabels(plotDf["Feature"], fontsize=6)

            ax.tick_params(axis="x", labelsize=7)
            ax.set_xlabel("Loading", fontsize=8)
            ax.set_title(pcCol, fontsize=10, fontweight="bold")

            # Reduce clutter.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            f"PCA loadings | top {top_n} positive and top {top_n} negative features per PC",
            fontsize=14,
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.985])

        return fig


    def _build_mask_visualization_figure(
            maskPath: str,
            maskName: str,
            matchingData: pd.DataFrame,
            clusterCol: str,
            xCol: str,
            yCol: str,
            scaleFactor: int,
            clusterColorMap: dict[str, str] | None = None,
        ):
        cellMask = read_mask_tiff(maskPath)

        matchedData = match_mask_centroids_to_cells(
            cellMask=cellMask,
            matchingData=matchingData,
            xCol=xCol,
            yCol=yCol,
        )

        plotData = make_mask_plot_data(
            cellMask=cellMask,
            matchedData=matchedData,
            clusterCol=clusterCol,
            scaleFactor=scaleFactor,
            background="white",
            borderColor="white",
            missingColor="#808080",
            clusterColorMap=clusterColorMap,
        )

        colorMat = plotData["colorMat"]
        clusterColors = plotData["clusterColors"]

        imgH, imgW = colorMat.shape[:2]
        imgRatio = imgW / imgH if imgH > 0 else 1.0

        # Keep image ratio intact while reserving a readable legend column.
        imagePanelHeight = 8.0
        imagePanelWidth = imagePanelHeight * imgRatio

        legendPanelWidth = 3.2
        figWidth = imagePanelWidth + legendPanelWidth
        figHeight = imagePanelHeight

        fig = plt.figure(figsize=(figWidth, figHeight), dpi=200)
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[imagePanelWidth, legendPanelWidth],
            wspace=0.02,
        )

        axImg = fig.add_subplot(gs[0, 0])
        axLeg = fig.add_subplot(gs[0, 1])

        axImg.imshow(colorMat, interpolation="nearest")
        axImg.set_axis_off()

        handles = [
            Patch(facecolor=clusterColors[name], edgecolor="none", label=name)
            for name in sorted(clusterColors.keys(), key=lambda x: x.lower())
        ]

        axLeg.set_axis_off()

        if len(handles) > 0:
            legendFontSize = 8
            if len(handles) <= 12:
                legendFontSize = 10
            elif len(handles) >= 30:
                legendFontSize = 6

            axLeg.legend(
                handles=handles,
                loc="upper left",
                frameon=False,
                fontsize=legendFontSize,
                borderaxespad=0.0,
            )

        fig.suptitle(
            f"{maskName} | {matchedData.shape[0]} masks matched",
            fontsize=10,
            y=0.995,
        )
        plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)

        return fig, matchedData
    
    def _get_cached_mask_plot_data(
        maskPath: str,
        maskName: str,
        matchingData: pd.DataFrame,
        clusterCol: str,
        xCol: str,
        yCol: str,
        scaleFactor: int,
        clusterColorMap: dict[str, str] | None = None,
        colorPalette: str = "viridis",
    ):
        """
        Return cached mask plot data if available; otherwise compute and cache it.

        The cache intentionally does not depend on browser size or browser zoom.
        It does depend on color settings, because color changes alter the rendered image.
        """
        cacheKey = make_mask_render_cache_key(
            mask_path=maskPath,
            data_version=maskRenderDataVersion.get(),
            render_settings={
                "mask_name": str(maskName),
                "cluster_col": str(clusterCol),
                "x_col": str(xCol),
                "y_col": str(yCol),
                "scale_factor": int(scaleFactor),
                "background": "white",
                "border_color": "white",
                "missing_color": "#808080",
                "color_palette": str(colorPalette),
                "cluster_color_map": clusterColorMap or {},
            },
        )

        cached = maskRenderCache.get(cacheKey)
        if cached is not None:
            print(f"✅ Using cached mask visualization for {maskName}", flush=True)
            return cached

        print(f"▶️ Rendering mask visualization for {maskName}", flush=True)

        cellMask = read_mask_tiff(maskPath)

        matchedData = match_mask_centroids_to_cells(
            cellMask=cellMask,
            matchingData=matchingData,
            xCol=xCol,
            yCol=yCol,
        )

        plotData = make_mask_plot_data(
            cellMask=cellMask,
            matchedData=matchedData,
            clusterCol=clusterCol,
            scaleFactor=scaleFactor,
            background="white",
            borderColor="white",
            missingColor="#808080",
            clusterColorMap=clusterColorMap,
        )

        cachedValue = {
            "plotData": plotData,
            "matchedData": matchedData,
        }

        maskRenderCache.set(cacheKey, cachedValue)
        return cachedValue
    
    
    def clear_mask_render_cache(reason: str = "") -> None:
        maskRenderCache.clear()
        maskRenderDataVersion.set(maskRenderDataVersion.get() + 1)

        msg = f"Mask render cache cleared"
        if reason:
            msg += f": {reason}"

        print(msg, flush=True)

    def _get_final_mask_name_col() -> str:
        return input.mask_name_col() or "CellMaskName"


    def _get_valid_mask_choices() -> list[str]:
        finalDf = final_mask_match_df.get()

        if finalDf is None or finalDf.empty:
            return []

        maskNameCol = _get_final_mask_name_col()

        if maskNameCol not in finalDf.columns:
            return []

        if "MaskExists" in finalDf.columns:
            finalDf = finalDf.loc[finalDf["MaskExists"]].copy()

        return _deduplicate_mask_choices(finalDf[maskNameCol].tolist())


    def _set_selected_mask_match(selectedName: str | None) -> None:
        finalDf = final_mask_match_df.get()

        if finalDf is None or finalDf.empty or not selectedName:
            selected_mask_match.set(pd.DataFrame())
            return

        maskNameCol = _get_final_mask_name_col()

        if maskNameCol not in finalDf.columns:
            selected_mask_match.set(pd.DataFrame())
            return

        sel = finalDf.loc[
            finalDf[maskNameCol].astype(str) == str(selectedName)
        ].copy()

        selected_mask_match.set(sel)


    def _deduplicate_mask_choices(values) -> list[str]:
        seen = set()
        out = []

        for value in values or []:
            if value is None or pd.isna(value):
                continue

            value = str(value)

            if value in seen:
                continue

            seen.add(value)
            out.append(value)

        return out


    def _update_mask_select_controls(
        choices: list[str] | None = None,
        selected: str | None = None,
    ) -> None:
        if choices is None:
            choices = _get_valid_mask_choices()

        choices = _deduplicate_mask_choices(choices)

        if selected is not None:
            selected = str(selected)

        if selected not in choices:
            selected = choices[0] if choices else None

        ui.update_select(
            "selected_mask_name",
            choices=choices,
            selected=selected,
            session=session,
        )

        ui.update_select(
            "selected_mask_name_display",
            choices=choices,
            selected=selected,
            session=session,
        )

        _set_selected_mask_match(selected)


    def _clear_mask_selection_controls() -> None:
        ui.update_select(
            "selected_mask_name",
            choices=[],
            selected=None,
            session=session,
        )

        ui.update_select(
            "selected_mask_name_display",
            choices=[],
            selected=None,
            session=session,
        )

        selected_mask_match.set(pd.DataFrame())


    def _select_mask_by_name(selectedName: str | None) -> None:
        choices = _get_valid_mask_choices()

        if not selectedName or str(selectedName) not in choices:
            selectedName = choices[0] if choices else None

        _update_mask_select_controls(
            choices=choices,
            selected=selectedName,
        )


    def _cycle_mask_selection(step: int) -> None:
        choices = _get_valid_mask_choices()

        if not choices:
            _clear_mask_selection_controls()
            return

        current = input.selected_mask_name_display() or input.selected_mask_name()

        if current not in choices:
            selected = choices[0]
        else:
            idx = choices.index(current)
            selected = choices[(idx + step) % len(choices)]

        _select_mask_by_name(selected)

    def _get_neighborhood_output_dir():
        obj = neighborhood_input_data.get()
        if obj is None:
            return None

        maskFolder = obj.get("mask_folder", None)
        if not maskFolder:
            return None

        outDir = Path(maskFolder) / "Neighborhood results"
        outDir.mkdir(parents=True, exist_ok=True)
        return outDir


    @reactive.Effect
    @reactive.event(input.push_to_segmentation)
    def _push_to_segmentation():
        imgs = images.get()
        chs = channels.get()

        if not imgs:
            print("⚠️ No images loaded. Load images in the PINT tab before pushing to Segmentation.")
            segmentation_input_data.set(None)
            return

        obj = {
            "pushed_at": datetime.now().isoformat(timespec="seconds"),
            "images": imgs,
            "channels": chs,
            "canonical_channels": list(canonical_channels.get() or []),
            "folder": last_loaded_folder.get() or (input.path() or "").strip(),
            "n_images": len(imgs),
        }

        segmentation_input_data.set(obj)

        firstSample = list(imgs.keys())[0]
        _sync_segmentation_channel_choices(firstSample)

        ui.update_navs("viewer_mode", selected="segmentation", session=session)

        print(
            f"✅ Pushed {len(imgs):,} loaded image(s) to Segmentation. "
            f"Folder: {obj['folder'] or '—'}"
        )


    @reactive.Effect
    @reactive.event(input.seg_sample)
    def _on_seg_sample_change():
        obj = segmentation_input_data.get()

        if obj is None:
            return

        sampleName = input.seg_sample()
        if not sampleName:
            return

        _sync_segmentation_channel_choices(sampleName)


    @reactive.Effect
    @reactive.event(input.save_composite_tiff)
    def _save_composite_tiff():
        rgb, used = _build_composite_rgb()
        if rgb is None or len(used) == 0:
            print("⚠️ No composite available to save.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        sampleName = input.sample() or "sample"
        defaultName = f"{sampleName} Composite.tiff"

        savePath = pick_save_tiff_dialog(
            initialdir=initdir,
            initialfile=defaultName,
        )

        if not savePath:
            print("🛑 Save canceled.")
            return

        try:
            rgb_u16 = np.clip(np.round(rgb * 65535.0), 0, 65535).astype(np.uint16)
            imwrite(
                savePath,
                rgb_u16,
                dtype=np.uint16,
                photometric="rgb",
            )
            print(f"✅ Saved composite TIFF → {savePath}")
        except Exception as e:
            print(f"❌ Failed to save composite TIFF: {e}")

    @reactive.Effect
    @reactive.event(input.export_creator_composites_all)
    def _export_creator_composites_all():
        imgs = images.get()
        if not imgs:
            print("⚠️ No images loaded.")
            return

        selected = _get_selected_creator_channels()
        if len(selected) == 0:
            print("⚠️ No creator channels selected.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        if not folder or not os.path.isdir(folder):
            print("⚠️ Invalid input folder.")
            return

        out_dir = os.path.join(folder, "creator composite exports")
        os.makedirs(out_dir, exist_ok=True)

        # Save creator selection metadata for reproducibility
        try:
            creator_meta = pd.DataFrame(selected, columns=["Channel", "Color", "Gain"])
            creator_meta.to_csv(os.path.join(out_dir, "creator_selection.csv"), index=False)
        except Exception as e:
            print(f"⚠️ Could not save creator_selection.csv: {e}")

        for sampleName in imgs.keys():
            rgb, used = _build_composite_rgb(sampleName=sampleName)

            if rgb is None or len(used) == 0:
                print(f"⚠️ Skipped {sampleName}: no selected creator channels were present.")
                continue

            out_path = os.path.join(out_dir, f"{sampleName} Composite.tiff")

            try:
                rgb_u16 = np.clip(np.round(rgb * 65535.0), 0, 65535).astype(np.uint16)
                imwrite(
                    out_path,
                    rgb_u16,
                    dtype=np.uint16,
                    photometric="rgb",
                )
                print(f"✅ Exported composite TIFF → {out_path}")
            except Exception as e:
                print(f"❌ Failed for {sampleName}: {e}")

        print("✅ Finished exporting composite TIFFs for all images.")

    @reactive.Effect
    @reactive.event(input.fill_composite_from_current)
    def _fill_composite_from_current():
        s = input.sample()
        if not s:
            return
        _sync_composite_channel_choices(s, overwriteDefaults=True)

    @output
    @render.ui
    def composite_summary():
        rgb, used = _build_composite_rgb()
        if not used:
            return ui.tags.small("No channels selected for the composite.", class_="text-muted")

        lines = [
            ui.tags.li(f"{ch} — {col} × {gain:.1f}")
            for ch, col, gain in used
        ]
        return ui.div(
            ui.tags.small("Composite uses the stored per-channel processing settings.", class_="text-muted"),
            ui.tags.ul(*lines, class_="mt-2 mb-0"),
        )

    @reactive.Effect
    @reactive.event(input.save_composite_png)
    def _save_composite_png():
        rgb, used = _build_composite_rgb()
        if rgb is None or len(used) == 0:
            print("⚠️ No composite available to save.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        sampleName = input.sample() or "sample"
        defaultName = f"{sampleName} Composite.png"

        savePath = pick_save_png_dialog(
            initialdir=initdir,
            initialfile=defaultName,
        )

        if not savePath:
            print("🛑 Save canceled.")
            return

        try:
            plt.imsave(savePath, rgb)
            print(f"✅ Saved composite PNG → {savePath}")
        except Exception as e:
            print(f"❌ Failed to save composite PNG: {e}")

    ##<--------load images module ---------->
    @reactive.Effect
    @reactive.event(input.load)
    def _do_load():
        ##If loading is already in progress don't try to start again!
        if loading.get():
            return
        loading.set(True) 
        try:
            ##Try the text box first; if empty, open the dialog. 
            ##So if you paste the path into the box it will not open the window ad just start loading
            folder = (input.path() or "").strip()
            if not folder:
                ##So if the box is empty or not a folder
                try:
                    folder = pick_folder_dialog()
                except Exception:
                    folder = ""
            ##If the loader also isn't valid, return this:
            if not folder:
                print("🛑 Load canceled (no folder selected).")
                return

            ##Purely for visibilty, the actual folder path used is put into the text box
            session.send_input_message("path", {"value": folder})
            print(">>> Loading triggered with folder:", folder)
            ##Loads the OME.tiff, also checks if channels are consistent and throws an error if one of the images if different from the first one
            ##Error from load_tiffs_raw is passed on to the interface
            try:
                imgs, chs = load_tiffs_raw(folder)  # ideally load_tiffs_raw(folder, validate_consistent=True)
            except ValueError as e:
                ui.modal_show(
                    ui.modal(
                        ui.h4("Channel mismatch..."),
                        ui.pre(str(e)),
                        easy_close=True,
                        footer=ui.modal_button("OK"),
                    ),
                    session=session,
                )
                return
            ##If the folder didn't contain any images:
            if not imgs:
                print("⚠️ No images found in selected folder.")
                return

            ##Store images and channels
            images.set(imgs)
            channels.set(chs)
            ##Clear the cache. this prevends old settings from similar channal names to remain in memory and prevent from actually processing the images for the viewer
            _invalidate_global_cache()
            ##Picks the first images as the default display.
            samples = list(imgs.keys())
            first_sample = samples[0]
            first_chlist = chs[first_sample]
            first_channel = first_chlist[0] if first_chlist else None

            #set the channel list for the left sided table and fill it will the channels. Each chanel gets a row.
            canonical_channels.set(list(first_chlist))
            _prefill_params(first_chlist)

            ##update selects under guard, selects first image
            setting_selects.set(True)
            try:
                ui.update_select("sample", choices=samples, selected=first_sample, session=session)

                ui.update_select(
                    "creator_sample_display",
                    choices=samples,
                    selected=first_sample,
                    session=session,
                )
                if first_channel:
                    ui.update_select("channel", choices=first_chlist, selected=first_channel, session=session)
            finally:
                setting_selects.set(False)

            _sync_composite_channel_choices(first_sample, overwriteDefaults=True)

            if first_channel:
                _sync_controls_from_table(first_channel)

            ##Tell user that everything is loaded and save the foldername for further use
            data_loaded.set(True)
            last_loaded_folder.set(folder)

        finally:
            loading.set(False)

    # ---------- react to sample change ----------
    @reactive.Effect
    @reactive.event(input.next_sample)
    def _next_sample():
        ##if loading in progress or images are not set, abort.
        if loading.get() or not images.get():
            return
        samples = list(images.get().keys())
        cur = input.sample() or (samples[0] if samples else None)
        ##Do samplenr++
        nxt = _cycle(samples, cur, +1)
        if nxt:
            ##Don’t set setting_selects here; we want _on_sample_change to run
            ui.update_select("sample", choices=samples, selected=nxt, session=session)

    @reactive.Effect
    @reactive.event(input.prev_sample)
    def _prev_sample():
        ##if loading in progress or images are not set, abort.
        if loading.get() or not images.get():
            return
        ##get list of samples
        samples = list(images.get().keys())
        ##current sample select (or first of none selected)
        cur = input.sample() or (samples[0] if samples else None)
        ##previous sample, uses cycler to wrap around
        prv = _cycle(samples, cur, -1)
        ##If valid selected, update the sample dropdown menu with current sample
        if prv:
            ui.update_select("sample", choices=samples, selected=prv, session=session)

    @reactive.Effect
    @reactive.event(input.creator_next_sample)
    def _creator_next_sample():
        if loading.get() or not images.get():
            return

        samples = list(images.get().keys())
        cur = input.sample() or input.creator_sample_display() or (samples[0] if samples else None)
        nxt = _cycle(samples, cur, +1)

        if nxt:
            ui.update_select("sample", choices=samples, selected=nxt, session=session)
            ui.update_select("creator_sample_display", choices=samples, selected=nxt, session=session)


    @reactive.Effect
    @reactive.event(input.creator_prev_sample)
    def _creator_prev_sample():
        if loading.get() or not images.get():
            return

        samples = list(images.get().keys())
        cur = input.sample() or input.creator_sample_display() or (samples[0] if samples else None)
        prv = _cycle(samples, cur, -1)

        if prv:
            ui.update_select("sample", choices=samples, selected=prv, session=session)
            ui.update_select("creator_sample_display", choices=samples, selected=prv, session=session)

    @reactive.Effect
    @reactive.event(input.creator_sample_display)
    def _on_creator_sample_display_change():
        if loading.get() or setting_selects.get():
            return

        s = input.creator_sample_display()
        if not s:
            return

        if s != input.sample():
            ui.update_select("sample", selected=s, session=session)

    @reactive.Effect
    @reactive.event(input.sample)
    def _on_sample_change():
        if loading.get() or setting_selects.get():
            return

        s = input.sample()
        if not s:
            return

        chlist_current = channels.get().get(s, [])
        if not chlist_current:
            return

        canon = canonical_channels.get()
        ordered = order_by_canonical(canon, chlist_current)

        sel = input.channel()
        if sel not in ordered:
            sel = ordered[0]

        setting_selects.set(True)
        try:
            ui.update_select("channel", choices=ordered, selected=sel, session=session)

            ui.update_select(
                "creator_sample_display",
                choices=list(images.get().keys()),
                selected=s,
                session=session,
            )
        finally:
            setting_selects.set(False)

        _sync_controls_from_table(sel)
        _sync_composite_channel_choices(s, overwriteDefaults=False)

    # ---------- react to channel change ----------
    ##Below reactive effect work the sample as their sample counterpart, check comments there on how they work
    @reactive.Effect
    @reactive.event(input.next_channel)
    def _next_channel():
        if loading.get():
            return
        s = input.sample()
        if not s:
            return
        chlist_current = channels.get().get(s, [])
        if not chlist_current:
            return
        canon = canonical_channels.get() or []
        ordered = order_by_canonical(canon, chlist_current)
        cur = input.channel() or ordered[0]
        nxt = _cycle(ordered, cur, +1)
        if nxt:
            ui.update_select("channel", choices=ordered, selected=nxt, session=session)

    @reactive.Effect
    @reactive.event(input.prev_channel)
    def _prev_channel():
        if loading.get():
            return
        s = input.sample()
        if not s:
            return
        chlist_current = channels.get().get(s, [])
        if not chlist_current:
            return
        canon = canonical_channels.get() or []
        ordered = order_by_canonical(canon, chlist_current)
        cur = input.channel() or ordered[0]
        prv = _cycle(ordered, cur, -1)
        if prv:
            ui.update_select("channel", choices=ordered, selected=prv, session=session)

    @reactive.Effect
    @reactive.event(input.channel)
    def _on_channel_change():
        if loading.get() or setting_selects.get():
            return
        c = input.channel()
        if not c:
            return
        _sync_controls_from_table(c)

    # <---------- update parameter table ---------->
    ##Below are the effects that tie the buttons to the parameter dataframe
    ##Important not is that i switched from immediately updating the parametern dataframe to making a copy and then applying
    ##This solved a bug where the image viewer seemed to respond to a direct change in the parameter table?
    ##Incidently, this will also prevent future functions that are tied to the parameter table from introducing weird behaviour 
    ##associated with reading the table and mutating the original later. 
    ##All buttons work generally the same way. 
    ##Button trigger reading of the input channel, followed by a pull of the parameter table which is the copied
    ##The user input is read, written to the copied dataframe, which in turn is written back to the params_df

    ##Winsorization
    @reactive.Effect
    @reactive.event(input.apply_one)
    def _apply_one_channel():
        ##Prevents feedback between application of change and the change of the UI
        if syncing_controls.get():
            return
        ##read input channel, if non exit
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        params_df.set(
            update_channel_row(
                df,
                c,
                {
                    "DoWinsor": bool(input.doWinsor()),
                    "Low": float(input.winsor_low()),
                    "High": float(input.winsor_high()),
                },
            )
        )

    ##Thresholding
    @reactive.Effect
    @reactive.event(input.apply_threshold)
    ##This connects the update channel threshold value button to the parameter table
    def _apply_threshold_channel():
        ##This prevents feedbackloop between the update table and "UI changed" action
        if syncing_controls.get():
            return
        #Select input channel, if none, exit
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        # Fraction-of-max threshold (0..1)
        try:
            thr_val = float(input.thr_fraction_val())
        except Exception:
            thr_val = 0.0
        thr_val = clamp01(thr_val)

        # Absolute threshold (counts)
        try:
            abs_thr_val = float(input.abs_threshold_val())
        except Exception:
            abs_thr_val = 0.0

        params_df.set(
            update_channel_row(
                df,
                c,
                {
                    "DoAbsThr": bool(input.doAbsThreshold()),
                    "AbsThrVal": float(abs_thr_val),
                    "DoThr": bool(input.doThreshold()),
                    "ThrVal": float(thr_val),
                },
            )
        )

    ##Noise removal
    @reactive.Effect
    @reactive.event(input.apply_noise)
    def _apply_noise_channel():
        ##Same thing as the previous ones
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        s = float(input.noise_strength())
        p = strength_to_percentile(s)
        params_df.set(
            update_channel_row(
                df,
                c,
                {
                    "Noise": bool(input.doNoise()),
                    "NStr": s,
                    "NPrctl": p,
                    "WinSz": int(input.window_size()),
                },
            )
        )

    ##Normalization and transformation
    @reactive.Effect
    @reactive.event(input.apply_norm)
    def _apply_options_channel():
        if syncing_controls.get():
            return
        c = input.channel()
        if not c:
            return
        df = params_df.get()
        try:
            cofac = int(float(input.asinh_cofactor()))
        except Exception:
            cofac = 5
        cofac = max(2, min(10, cofac))
        params_df.set(
            update_channel_row(
                df,
                c,
                {
                    "DoNorm": bool(input.doNorm()),
                    "NormScope": (input.norm_scope() or "page"),
                    "DoAsinh": bool(input.doAsinh()),
                    "Cofac": cofac,
                },
            )
        )

    def _get_mesmer_env_name() -> str:
        envName = (input.mesmer_env_name() or "").strip()
        return envName or DEFAULT_MESMER_ENV_NAME


    @reactive.Effect
    @reactive.event(input.check_mesmer_backend)
    def _check_mesmer_backend():
        envName = _get_mesmer_env_name()

        print(f"🔎 Checking Mesmer backend in conda environment: {envName}")

        status = check_mesmer_backend(env_name=envName)
        mesmer_backend_status.set(status)
        mesmer_backend_detail_text.set(status.detail)

        if status.ok:
            print(f"✅ Mesmer backend available in '{envName}'.")
        else:
            print(f"⚠️ Mesmer backend not available in '{envName}': {status.status}")


    @reactive.Effect
    @reactive.event(input.show_mesmer_install_commands)
    def _show_mesmer_install_commands():
        envName = _get_mesmer_env_name()
        commands = get_mesmer_install_commands(env_name=envName)

        text = (
            "Manual Mesmer/DeepCell backend installation commands:\n\n"
            + "\n".join(commands)
            + "\n\n"
            "After running these in a terminal, return to PINT and click "
            "'Check Mesmer installation'."
        )

        mesmer_backend_detail_text.set(text)

        ui.modal_show(
            ui.modal(
                ui.h4("Install Mesmer backend"),
                ui.p(
                    "Run these commands in a terminal. This installs Mesmer/DeepCell "
                    "into a separate conda environment and does not modify the main PINT environment."
                ),
                ui.tags.pre(
                    "\n".join(commands),
                    class_="seg-command-box",
                ),
                ui.p(
                    "After installation, click 'Check Mesmer installation' in the Segmentation tab."
                ),
                easy_close=True,
                footer=ui.modal_button("OK"),
                size="l",
            ),
            session=session,
        )


    @reactive.Effect
    @reactive.event(input.install_mesmer_backend)
    def _install_mesmer_backend():
        envName = _get_mesmer_env_name()

        ui.modal_show(
            ui.modal(
                ui.h4("Install Mesmer backend?"),
                ui.p(
                    f"This will install Mesmer/DeepCell into a separate conda environment called '{envName}'."
                ),
                ui.p(
                    "The main PINT environment will not be modified. Installation can take several minutes "
                    "and may fail depending on TensorFlow/DeepCell dependency resolution."
                ),
                ui.tags.small(
                    "This is an Alpha function. Use only for testing at this stage.",
                    class_="text-muted",
                ),
                easy_close=True,
                footer=ui.div(
                    ui.modal_button("Cancel", class_="btn btn-secondary"),
                    ui.input_action_button(
                        "confirm_install_mesmer_backend",
                        "Install",
                        class_="btn btn-warning ms-2",
                    ),
                ),
            ),
            session=session,
        )


    @reactive.Effect
    @reactive.event(input.confirm_install_mesmer_backend)
    def _confirm_install_mesmer_backend():
        ui.modal_remove(session=session)

        envName = _get_mesmer_env_name()

        mesmer_backend_detail_text.set(
            f"Installing Mesmer/DeepCell backend into conda environment '{envName}'...\n\n"
            "This may take a while. Progress is also printed to the terminal."
        )

        print(f"⚠️ Installing Alpha Mesmer backend into environment: {envName}")

        with ui.Progress(min=0, max=3, session=session) as p:
            p.set(value=0, message="Starting Mesmer backend installation...")

            status = install_mesmer_backend(env_name=envName)

            p.set(value=3, message="Mesmer backend installation finished.")

        mesmer_backend_status.set(status)
        mesmer_backend_detail_text.set(status.detail)

        if status.ok:
            print(f"✅ Mesmer backend installed and available in '{envName}'.")
        else:
            print(f"❌ Mesmer backend installation failed or is unavailable: {status.status}")

    @reactive.Effect
    @reactive.event(input.check_mesmer_gpu)
    def _check_mesmer_gpu():
        envName = _get_mesmer_env_name()

        print(f"🔎 Checking TensorFlow GPU in Mesmer environment: {envName}")

        status = check_mesmer_gpu(env_name=envName)
        mesmer_backend_status.set(status)
        mesmer_backend_detail_text.set(status.detail)

        if status.ok:
            print(f"✅ TensorFlow GPU detected in '{envName}'.")
        else:
            print(f"⚠️ TensorFlow GPU not detected in '{envName}'. Mesmer will likely run on CPU.")

    @output
    @render.ui
    def segmentation_input_summary():
        obj = segmentation_input_data.get()

        if obj is None:
            return ui.div(
                ui.tags.div("No images pushed from PINT yet.", class_="seg-status-bad"),
                ui.tags.small(
                    "Load images in the PINT tab, then click 'Push loaded images to Segmentation'.",
                    class_="text-muted",
                ),
            )

        folder = obj.get("folder", "") or "—"
        pushedAt = obj.get("pushed_at", "—")
        nImages = obj.get("n_images", 0)

        return ui.div(
            ui.tags.div("Images available for segmentation", class_="seg-status-ok"),
            ui.tags.div(f"Images: {nImages:,}"),
            ui.tags.div(f"Pushed at: {pushedAt}"),
            ui.tags.div(f"Folder: {folder}"),
        )

    @reactive.Effect
    @reactive.event(input.run_mesmer_all)
    def _run_mesmer_all():
        obj = segmentation_input_data.get()

        if obj is None:
            print("⚠️ No images pushed to Segmentation.")
            return

        imgs = obj.get("images", {})
        if not imgs:
            print("⚠️ No pushed images available for Mesmer batch run.")
            return

        envName = _get_mesmer_env_name()

        status = check_mesmer_backend(env_name=envName)
        if not status.ok:
            segmentation_mesmer_result.set(status)
            mesmer_backend_status.set(status)
            mesmer_backend_detail_text.set(status.detail)
            print("❌ Mesmer backend is not available. Check installation first.")
            return

        deepcellToken = (input.deepcell_access_token() or "").strip()
        if not deepcellToken:
            print("⚠️ No DeepCell access token entered. Mesmer may fail if model files are not already cached.")

        imageMpp = float(input.seg_image_mpp() or 1.0)
        compartment = input.seg_mesmer_compartment() or "whole-cell"

        sampleNames = list(imgs.keys())
        outRows = []

        outDir = _get_segmentation_output_dir()
        batchSummaryPath = outDir / "mesmer_batch_summary.csv"

        print(f"▶️ Running Mesmer batch on {len(sampleNames):,} ROI(s).")
        print(f"   Output folder: {outDir}")

        with ui.Progress(min=0, max=len(sampleNames), session=session) as p:
            for i, sampleName in enumerate(sampleNames, start=1):
                p.set(value=i - 1, message=f"Preparing {sampleName} ({i}/{len(sampleNames)})")

                row = {
                    "SampleName": sampleName,
                    "Status": "Not started",
                    "Ok": False,
                    "MaskPath": "",
                    "NLabels": np.nan,
                    "Error": "",
                }

                try:
                    nuclearPath, boundaryPath, maskPath, jsonPath, meta = _save_mesmer_inputs_for_sample(sampleName)

                    row.update(meta)
                    row["Status"] = "Running"

                    p.set(value=i - 1, message=f"Running Mesmer: {sampleName} ({i}/{len(sampleNames)})")

                    result = run_mesmer_backend(
                        nuclear_path=nuclearPath,
                        boundary_path=boundaryPath,
                        out_mask_path=maskPath,
                        out_json_path=jsonPath,
                        image_mpp=imageMpp,
                        compartment=compartment,
                        env_name=envName,
                        deepcell_access_token=deepcellToken if deepcellToken else None,
                    )

                    row["Status"] = result.status
                    row["Ok"] = bool(result.ok)
                    row["MaskPath"] = str(maskPath) if Path(maskPath).exists() else ""
                    row["Error"] = "" if result.ok else result.detail

                    if result.ok and Path(maskPath).exists():
                        try:
                            mask = imread(str(maskPath))
                            row["NLabels"] = int(np.nanmax(mask)) if mask.size > 0 else 0

                            # Keep last successful mask in preview.
                            segmentation_mesmer_mask.set(mask)
                            segmentation_mesmer_mask_path.set(str(maskPath))
                            segmentation_mesmer_result.set(result)

                        except Exception as e:
                            row["Ok"] = False
                            row["Status"] = "Mask read failed"
                            row["Error"] = str(e)

                    mesmer_backend_detail_text.set(result.detail)

                    if result.ok:
                        print(f"✅ {sampleName}: {result.status}")
                    else:
                        print(f"❌ {sampleName}: {result.status}")

                except Exception as e:
                    row["Status"] = "Failed before Mesmer"
                    row["Error"] = str(e)
                    print(f"❌ {sampleName}: {e}")

                outRows.append(row)

                # Write partial results after every ROI.
                try:
                    pd.DataFrame(outRows).to_csv(batchSummaryPath, index=False)
                except Exception as e:
                    print(f"⚠️ Could not write batch summary: {e}")

                p.set(value=i, message=f"Finished {sampleName} ({i}/{len(sampleNames)})")

        batchDf = pd.DataFrame(outRows)
        segmentation_mesmer_batch_results.set(batchDf)

        nOk = int(batchDf["Ok"].sum()) if "Ok" in batchDf.columns and not batchDf.empty else 0

        print(
            f"✅ Mesmer batch finished: {nOk}/{len(sampleNames)} ROI(s) completed. "
            f"Summary saved to: {batchSummaryPath}"
        )    


    @output
    @render.plot
    def segmentation_nuclear_preview():
        try:
            nuclearImg, boundaryImg, combinedRgb, errorMessage = _build_segmentation_preview_images()

            sampleName, nuclearChannel, boundaryChannels, mode = _get_selected_segmentation_channels()
            title = f"Nuclear input: {nuclearChannel or '—'} | mode: {mode}"

            return _make_single_image_preview_figure(
                nuclearImg,
                title=title,
                cmap="gray",
                errorMessage=errorMessage,
            )

        except SilentException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()
            return _make_single_image_preview_figure(
                None,
                title="Nuclear input",
                errorMessage=f"Nuclear preview error:\n{e}",
            )


    @output
    @render.plot
    def segmentation_boundary_preview():
        try:
            nuclearImg, boundaryImg, combinedRgb, errorMessage = _build_segmentation_preview_images()

            sampleName, nuclearChannel, boundaryChannels, mode = _get_selected_segmentation_channels()
            nBoundary = 0 if boundaryChannels is None else len(boundaryChannels)
            title = f"Boundary input: {nBoundary} channel(s) | mode: {mode}"

            return _make_single_image_preview_figure(
                boundaryImg,
                title=title,
                cmap="gray",
                errorMessage=errorMessage,
            )

        except SilentException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()
            return _make_single_image_preview_figure(
                None,
                title="Boundary input",
                errorMessage=f"Boundary preview error:\n{e}",
            )


    @output
    @render.plot
    def segmentation_combined_preview():
        try:
            nuclearImg, boundaryImg, combinedRgb, errorMessage = _build_segmentation_preview_images()

            sampleName, nuclearChannel, boundaryChannels, mode = _get_selected_segmentation_channels()
            title = f"Combined Mesmer input preview | mode: {mode}"

            return _make_single_image_preview_figure(
                combinedRgb,
                title=title,
                cmap=None,
                errorMessage=errorMessage,
            )

        except SilentException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()
            return _make_single_image_preview_figure(
                None,
                title="Combined preview",
                errorMessage=f"Combined preview error:\n{e}",
            )

    @reactive.Effect
    @reactive.event(input.run_mesmer_current)
    def _run_mesmer_current():
        status = check_mesmer_backend(env_name=_get_mesmer_env_name())

        if not status.ok:
            segmentation_mesmer_result.set(status)
            mesmer_backend_status.set(status)
            mesmer_backend_detail_text.set(status.detail)
            print("❌ Mesmer backend is not available. Check installation first.")
            return

        try:
            nuclearPath, boundaryPath, maskPath, jsonPath, meta = _save_current_mesmer_inputs()
        except Exception as e:
            msg = f"Could not prepare Mesmer input: {e}"
            segmentation_mesmer_mask.set(None)
            segmentation_mesmer_result.set(None)
            print(f"❌ {msg}")
            mesmer_backend_detail_text.set(msg)
            return

        sampleName = meta.get("SampleName", input.seg_sample() or "unknown")

        envName = _get_mesmer_env_name()
        imageMpp = float(input.seg_image_mpp() or 1.0)
        compartment = input.seg_mesmer_compartment() or "whole-cell"
        deepcellToken = (input.deepcell_access_token() or "").strip()

        if not deepcellToken:
            print("ℹ️ No DeepCell access token entered. This is fine if Mesmer model files are already cached.")

        print(f"▶️ Running Mesmer on current ROI: {sampleName}")
        print(f"   Nuclear: {nuclearPath}")
        print(f"   Boundary: {boundaryPath}")
        print(f"   Output mask: {maskPath}")

        with ui.Progress(min=0, max=3, session=session) as p:
            p.set(value=0, message="Preparing Mesmer input...")
            p.set(value=1, message="Running Mesmer in external environment...")

            deepcellToken = (input.deepcell_access_token() or "").strip()

            result = run_mesmer_backend(
                nuclear_path=nuclearPath,
                boundary_path=boundaryPath,
                out_mask_path=maskPath,
                out_json_path=jsonPath,
                image_mpp=imageMpp,
                compartment=compartment,
                env_name=envName,
                deepcell_access_token=deepcellToken if deepcellToken else None,
            )

            p.set(value=2, message="Reading Mesmer output...")

            if result.ok and Path(maskPath).exists():
                mask = imread(str(maskPath))
                segmentation_mesmer_mask.set(mask)
                segmentation_mesmer_mask_path.set(str(maskPath))
            else:
                segmentation_mesmer_mask.set(None)
                segmentation_mesmer_mask_path.set("")

            p.set(value=3, message="Mesmer run complete.")

        segmentation_mesmer_result.set(result)
        mesmer_backend_detail_text.set(result.detail)

        if result.ok:
            print(f"✅ {result.status}")
        else:
            print(f"❌ {result.status}")


    @output
    @render.ui
    def segmentation_mesmer_result_summary():
        result = segmentation_mesmer_result.get()
        maskPath = segmentation_mesmer_mask_path.get()
        batchDf = segmentation_mesmer_batch_results.get()

        parts = []

        if result is None:
            parts.append(ui.tags.small("No single-ROI Mesmer run yet.", class_="text-muted"))
        elif not result.ok:
            parts.append(
                ui.div(
                    ui.tags.div(f"Last Mesmer status: {result.status}", class_="seg-status-bad"),
                    ui.tags.small("See backend details for the full error.", class_="text-muted"),
                )
            )
        else:
            mask = segmentation_mesmer_mask.get()
            nLabels = int(np.nanmax(mask)) if mask is not None and np.size(mask) > 0 else 0

            parts.append(
                ui.div(
                    ui.tags.div(f"Last Mesmer status: {result.status}", class_="seg-status-ok"),
                    ui.tags.div(f"Detected labels: {nLabels:,}"),
                    ui.tags.div(f"Mask: {maskPath or '—'}"),
                )
            )

        if batchDf is not None and not batchDf.empty:
            nTotal = len(batchDf)
            nOk = int(batchDf["Ok"].sum()) if "Ok" in batchDf.columns else 0

            parts.append(ui.hr())
            parts.append(
                ui.div(
                    ui.tags.div("Batch run summary", class_="mask-section-title"),
                    ui.tags.div(f"Completed: {nOk}/{nTotal} ROI(s)"),
                )
            )

        return ui.div(*parts)


    @output
    @render.ui
    def segmentation_quantification_summary():
        status = segmentation_quantification_status.get()
        cellDf = segmentation_cell_table.get()
        maskDf = segmentation_mask_table.get()
        cellPath = segmentation_cell_table_path.get()

        parts = [
            ui.tags.div(status, class_="compact-small-line"),
        ]

        if maskDf is not None and not maskDf.empty:
            nMasks = int(maskDf["MaskExists"].sum()) if "MaskExists" in maskDf.columns else 0
            nOk = int((maskDf["Status"] == "OK").sum()) if "Status" in maskDf.columns else nMasks

            parts.append(ui.tags.div(f"Masks found: {nMasks:,}", class_="compact-small-line"))
            parts.append(ui.tags.div(f"Masks quantified: {nOk:,}", class_="compact-small-line"))

        if cellDf is not None and not cellDf.empty:
            parts.append(ui.tags.div(f"Cells quantified: {len(cellDf):,}", class_="compact-small-line"))
            parts.append(ui.tags.div(f"Columns: {len(cellDf.columns):,}", class_="compact-small-line"))

        if cellPath:
            parts.append(ui.tags.div(f"Cell table: {cellPath}", class_="compact-small-line"))

        return ui.div(*parts)

    @reactive.Effect
    @reactive.event(input.load_clustering_table)
    def _load_clustering_table():
        table_path = (input.clustering_table_path() or "").strip()

        print(f"▶️ Clustering table load requested: {table_path}", flush=True)

        try:
            if not table_path or not os.path.isfile(table_path):
                raise ValueError(f"Invalid clustering table path: {table_path!r}")

            file_size_mb = os.path.getsize(table_path) / 1024 / 1024

            with ui.Progress(min=0, max=3, session=session) as p:
                p.set(value=0, message=f"Reading clustering table ({file_size_mb:.1f} MB)...")

                t0 = datetime.now()
                df, original_name = _read_clustering_table_path(table_path)
                t1 = datetime.now()

                print(
                    f"✅ pandas read complete in {(t1 - t0).total_seconds():.1f} sec. "
                    f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns.",
                    flush=True,
                )

                if df is None or df.empty:
                    raise ValueError("Loaded file is empty.")

                p.set(value=1, message="Storing clustering table...")

                clustering_data.set(df)
                clustering_source.set("file_path")
                clustering_column_map.set(_make_default_clustering_column_map(df))

                clustering_feature_columns.set([])
                clustering_metadata_columns.set([])
                clustering_excluded_columns.set([])

                p.set(value=2, message="Updating Data preparation tab...")

                msg = (
                    f"Loaded clustering table '{original_name}' into Data preparation: "
                    f"{len(df):,} rows, {len(df.columns):,} columns."
                )

                clustering_status.set(msg)

                p.set(value=3, message="Done.")

            ui.update_navs("viewer_mode", selected="clustering", session=session)
            ui.update_navs("clustering_workspace_mode", selected="Data preparation", session=session)

            print(f"✅ {msg}", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()

            msg = f"Could not load clustering table: {e}"
            clustering_status.set(msg)
            print(f"⚠️ {msg}", flush=True)

    @reactive.Effect
    @reactive.event(input.export_clustering_column_map)
    def _export_clustering_column_map():
        df = clustering_data.get()

        if df is None or df.empty:
            print("⚠️ No clustering dataset loaded. Load a table first.")
            return

        colMap = clustering_column_map.get()

        if colMap is None or colMap.empty:
            colMap = _make_default_clustering_column_map(df)
            clustering_column_map.set(colMap)

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        savePath = pick_save_csv_dialog(
            initialdir=initdir,
            initialfile="clustering_column_map.csv",
        )

        if not savePath:
            print("🛑 Export clustering column map canceled.")
            return

        try:
            colMap.to_csv(savePath, index=False)
            print(
                f"✅ Exported clustering column map → {savePath}. "
                f"Protected columns such as {PINT_CELL_ID_COL} were excluded."
            )
        except Exception as e:
            print(f"❌ Failed to export clustering column map: {e}")


    @reactive.Effect
    @reactive.event(input.import_clustering_column_map)
    def _import_clustering_column_map():
        df = clustering_data.get()

        if df is None or df.empty:
            print("⚠️ No clustering dataset loaded. Load a table before importing a column map.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        csvPath = pick_open_csv_dialog(initialdir=initdir)

        if not csvPath:
            print("🛑 Import clustering column map canceled.")
            return

        try:
            colMap = pd.read_csv(csvPath)
        except Exception as e:
            print(f"❌ Failed to read clustering column map: {e}")
            return

        requiredCols = ["ChannelNamesForClustering", "ChannelNameToDisplay"]
        missing = [c for c in requiredCols if c not in colMap.columns]

        if missing:
            print(
                "❌ Clustering column map is missing required column(s): "
                + ", ".join(missing)
            )
            return

        colMap = colMap[requiredCols].copy()
        clustering_column_map.set(colMap)

        validMap = _get_valid_clustering_column_map()

        print(
            f"✅ Imported clustering column map → {csvPath}. "
            f"{len(validMap):,} valid selected columns found in current dataset."
        )

    @reactive.Effect
    @reactive.event(input.run_clustering_pca)
    def _run_clustering_pca():
        try:
            X, obs, sourceCols, displayCols = _prepare_clustering_feature_matrix()

            nCells, nFeatures = X.shape
            requestedPcs = int(input.clustering_n_pcs() or 20)
            nPcs = max(2, min(requestedPcs, nFeatures, nCells - 1))

            if nPcs < 2:
                raise ValueError("Need at least 2 PCs. Check number of cells/features.")

            clustering_analysis_status.set(
                f"Running PCA on {nCells:,} cells × {nFeatures:,} features..."
            )

            print(
                f"▶️ Running PCA: {nCells:,} cells × {nFeatures:,} features, "
                f"{nPcs} PCs.",
                flush=True,
            )

            pca = PCA(n_components=nPcs, random_state=int(input.clustering_random_seed() or 1))
            scores = pca.fit_transform(X)

            scoreCols = [f"PC_{i}" for i in range(1, nPcs + 1)]

            scoresDf = obs.copy()
            for i, col in enumerate(scoreCols):
                scoresDf[col] = scores[:, i].astype(np.float32)

            loadingsDf = pd.DataFrame(
                pca.components_.T,
                columns=scoreCols,
            )
            loadingsDf.insert(0, "Feature", displayCols)
            loadingsDf.insert(0, "SourceColumn", sourceCols)

            varianceDf = pd.DataFrame(
                {
                    "PC": scoreCols,
                    "ExplainedVarianceRatio": pca.explained_variance_ratio_,
                    "ExplainedVariancePercent": pca.explained_variance_ratio_ * 100,
                    "CumulativeVariancePercent": np.cumsum(pca.explained_variance_ratio_) * 100,
                }
            )

            clustering_feature_matrix.set(X)
            clustering_feature_source_columns.set(sourceCols)
            clustering_feature_display_columns.set(displayCols)

            clustering_pca_scores.set(scoresDf)
            clustering_pca_loadings.set(loadingsDf)
            clustering_pca_variance.set(varianceDf)

            clustering_leiden_labels.set(pd.DataFrame())
            clustering_marker_summary.set(pd.DataFrame())

            msg = (
                f"PCA complete: {nCells:,} cells × {nFeatures:,} features, "
                f"{nPcs:,} PCs."
            )

            clustering_analysis_status.set(msg)
            print(f"✅ {msg}", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()

            msg = f"PCA failed: {e}"
            clustering_analysis_status.set(msg)
            print(f"❌ {msg}", flush=True)

    @reactive.Effect
    @reactive.event(input.export_pca_loadings_plot)
    def _export_pca_loadings_plot():
        loadingsDf = clustering_pca_loadings.get()

        if loadingsDf is None or loadingsDf.empty:
            msg = "No PCA loadings available. Run PCA first."
            clustering_analysis_status.set(msg)
            print(f"⚠️ {msg}")
            return

        try:
            nPcs = int(input.clustering_loadings_n_pcs() or 20)
            topN = int(input.clustering_loadings_top_n() or 20)

            outDir = pick_folder_dialog()

            if not outDir:
                print("🛑 PCA loading export canceled.")
                return

            outDir = Path(outDir)
            outDir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outPath = outDir / f"PINT_PCA_loadings_top{topN}_{nPcs}PCs_{timestamp}.png"

            fig = _make_pca_loadings_grid_figure(
                loadingsDf,
                n_pcs=nPcs,
                top_n=topN,
                n_cols=5,
            )

            fig.savefig(
                outPath,
                dpi=600,
                bbox_inches="tight",
            )
            plt.close(fig)

            csvPath = outDir / f"PINT_PCA_loadings_table_{timestamp}.csv"
            loadingsDf.to_csv(csvPath, index=False)

            msg = f"Exported PCA loading plot to: {outPath}"
            clustering_analysis_status.set(msg)
            print(f"✅ {msg}", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()

            msg = f"PCA loading export failed: {e}"
            clustering_analysis_status.set(msg)
            print(f"❌ {msg}", flush=True)


    @reactive.Effect
    @reactive.event(input.run_leiden_clustering)
    def _run_leiden_clustering():
        try:
            if not LEIDEN_AVAILABLE:
                raise ImportError(
                    "Leiden clustering requires python-igraph and leidenalg. "
                    "Install with: pip install python-igraph leidenalg"
                )

            pcaDf = clustering_pca_scores.get()

            if pcaDf is None or pcaDf.empty:
                raise ValueError("Run PCA before Leiden clustering.")

            nDims = int(input.clustering_leiden_n_dims() or 10)
            nNeighbors = int(input.clustering_leiden_n_neighbors() or 15)
            resolution = float(input.clustering_leiden_resolution() or 1.0)
            seed = int(input.clustering_random_seed() or 1)

            pcCols = [c for c in pcaDf.columns if c.startswith("PC_")]

            if len(pcCols) < 2:
                raise ValueError("PCA scores do not contain enough PC columns.")

            usePcCols = pcCols[: min(nDims, len(pcCols))]
            Xpca = pcaDf.loc[:, usePcCols].to_numpy(dtype=np.float32, copy=True)

            nCells = Xpca.shape[0]

            if nCells < 3:
                raise ValueError("Need at least 3 cells for graph clustering.")

            nNeighbors = max(2, min(nNeighbors, nCells - 1))

            clustering_analysis_status.set(
                f"Building kNN graph using {len(usePcCols)} PCs and {nNeighbors} neighbors..."
            )

            print(
                f"▶️ Leiden clustering: {nCells:,} cells, "
                f"{len(usePcCols)} PCs, k={nNeighbors}, resolution={resolution}",
                flush=True,
            )

            nn = NearestNeighbors(
                n_neighbors=nNeighbors + 1,
                metric="euclidean",
                algorithm="auto",
            )
            nn.fit(Xpca)

            distances, indices = nn.kneighbors(Xpca)

            edges = []
            weights = []

            for i in range(nCells):
                # Skip first neighbor because it is usually the cell itself.
                for j, dist in zip(indices[i, 1:], distances[i, 1:]):
                    if i == j:
                        continue

                    a = int(i)
                    b = int(j)

                    if a < b:
                        edges.append((a, b))
                    else:
                        edges.append((b, a))

                    # Convert distance to a simple positive similarity weight.
                    weights.append(float(1.0 / (1.0 + dist)))

            if not edges:
                raise ValueError("No graph edges were created.")

            edgeDf = pd.DataFrame(edges, columns=["source", "target"])
            edgeDf["weight"] = weights

            # Collapse duplicate undirected edges.
            edgeDf = (
                edgeDf
                .groupby(["source", "target"], as_index=False)["weight"]
                .max()
            )

            graph = ig.Graph(n=nCells, edges=list(map(tuple, edgeDf[["source", "target"]].to_numpy())))
            graph.es["weight"] = edgeDf["weight"].tolist()

            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights=graph.es["weight"],
                resolution_parameter=resolution,
                seed=seed,
            )

            labels = np.array(partition.membership, dtype=int)

            labelsDf = pcaDf[[PINT_CELL_ID_COL]].copy()
            labelsDf["PINT_Leiden_cluster"] = [
                f"Cluster_{x}" for x in labels
            ]

            clustering_leiden_labels.set(labelsDf)

            # Add cluster labels back to the master dataset by PINT_Cell_ID.
            master = clustering_data.get().copy()

            if "PINT_Leiden_cluster" in master.columns:
                master = master.drop(columns=["PINT_Leiden_cluster"])

            master = master.merge(
                labelsDf,
                on=PINT_CELL_ID_COL,
                how="left",
            )

            clustering_data.set(master)

            nClusters = labelsDf["PINT_Leiden_cluster"].nunique()

            msg = (
                f"Leiden clustering complete: {nCells:,} cells, "
                f"{nClusters:,} clusters."
            )

            clustering_analysis_status.set(msg)
            print(f"✅ {msg}", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()

            msg = f"Leiden clustering failed: {e}"
            clustering_analysis_status.set(msg)
            print(f"❌ {msg}", flush=True)

    @output
    @render.data_frame
    def clustering_pca_variance_preview():
        df = clustering_pca_variance.get()

        if df is None or df.empty:
            empty = pd.DataFrame({"Message": ["No PCA variance available yet."]})
            return render.DataGrid(empty, height="250px", filters=False)

        show = df.copy()
        numCols = show.select_dtypes(include=["number"]).columns
        show[numCols] = show[numCols].round(3)

        return render.DataGrid(show, height="300px", filters=False)


    @output
    @render.data_frame
    def clustering_leiden_counts_preview():
        labelsDf = clustering_leiden_labels.get()

        if labelsDf is None or labelsDf.empty:
            empty = pd.DataFrame({"Message": ["No Leiden clusters available yet."]})
            return render.DataGrid(empty, height="250px", filters=False)

        out = (
            labelsDf
            .groupby("PINT_Leiden_cluster", sort=False)
            .size()
            .reset_index(name="n_cells")
            .sort_values("PINT_Leiden_cluster")
        )

        return render.DataGrid(out, height="300px", filters=False)


    @output
    @render.plot
    def clustering_pca_plot():
        pcaDf = clustering_pca_scores.get()
        labelsDf = clustering_leiden_labels.get()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=140)

        if pcaDf is None or pcaDf.empty:
            ax.text(0.5, 0.5, "Run PCA to show PC plot", ha="center", va="center")
            ax.set_axis_off()
            return fig

        plotDf = pcaDf.copy()

        if labelsDf is not None and not labelsDf.empty:
            plotDf = plotDf.merge(labelsDf, on=PINT_CELL_ID_COL, how="left")
        else:
            plotDf["PINT_Leiden_cluster"] = "Unclustered"

        if "PC_1" not in plotDf.columns or "PC_2" not in plotDf.columns:
            ax.text(0.5, 0.5, "PC_1 / PC_2 not available", ha="center", va="center")
            ax.set_axis_off()
            return fig

        clusters = plotDf["PINT_Leiden_cluster"].fillna("Unclustered").astype(str)
        clusterCodes, clusterNames = pd.factorize(clusters)

        ax.scatter(
            plotDf["PC_1"],
            plotDf["PC_2"],
            c=clusterCodes,
            s=float(input.clustering_plot_point_size() or 2),
            alpha=float(input.clustering_plot_alpha() or 0.7),
            cmap=input.clustering_scatter_palette() or "viridis",
            linewidths=0,
        )

        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("PCA")

        return fig


    @reactive.Effect
    @reactive.event(input.push_mesmer_to_mask_visualization)
    def _push_mesmer_to_mask_visualization():
        cellDf = segmentation_cell_table.get()
        maskDf = segmentation_mask_table.get()

        if cellDf is None or cellDf.empty:
            print("⚠️ No Mesmer cell table available. Run quantification first.")
            return

        if maskDf is None or maskDf.empty:
            print("⚠️ No Mesmer mask table available. Run quantification first.")
            return

        if "MaskExists" in maskDf.columns:
            validMasks = maskDf.loc[maskDf["MaskExists"]].copy()
        else:
            validMasks = maskDf.copy()

        if validMasks.empty:
            print("⚠️ No valid Mesmer masks available to push.")
            return

        outDir = _get_segmentation_output_dir()
        cellPath = segmentation_cell_table_path.get()

        # Push generated cell table into the existing Mask visualization state.
        mask_input_df.set(cellDf.copy())

        session.send_input_message("mask_csv_path", {"value": cellPath or ""})
        session.send_input_message("mask_path", {"value": str(outDir)})

        _sync_mask_column_choices(cellDf)

        colNames = list(cellDf.columns)

        ui.update_select(
            "mask_cell_id_col",
            choices=colNames,
            selected="ObjectNumber",
            session=session,
        )

        ui.update_select(
            "mask_x_col",
            choices=colNames,
            selected="Location_Center_X",
            session=session,
        )

        ui.update_select(
            "mask_y_col",
            choices=colNames,
            selected="Location_Center_Y",
            session=session,
        )

        ui.update_select(
            "mask_name_col",
            choices=colNames,
            selected="CellMaskName",
            session=session,
        )

        ui.update_select(
            "mask_cluster_col",
            choices=colNames,
            selected="Cluster" if "Cluster" in colNames else "ObjectNumber",
            session=session,
        )

        ui.update_select(
            "mask_condition_col",
            choices=colNames,
            selected="Condition" if "Condition" in colNames else "SampleName",
            session=session,
        )

        ui.update_select(
            "mask_sample_col",
            choices=colNames,
            selected="SampleName",
            session=session,
        )

        # Build direct match table. This avoids asking the user to run manual mask matching.
        finalMaskDf = validMasks.copy()

        if "CellMaskName" not in finalMaskDf.columns:
            finalMaskDf["CellMaskName"] = finalMaskDf["SampleName"]

        finalMaskDf["ManualMatch"] = False

        # Keep columns expected by downstream mask tools.
        if "MaskFile" not in finalMaskDf.columns:
            finalMaskDf["MaskFile"] = finalMaskDf["MaskPath"].map(lambda x: Path(str(x)).name)

        if "MaskExists" not in finalMaskDf.columns:
            finalMaskDf["MaskExists"] = True

        filesDf = finalMaskDf[["MaskFile", "MaskPath", "CellMaskName"]].copy()

        mask_files_df.set(filesDf)
        mask_match_df.set(finalMaskDf.copy())
        matched_masks_df.set(finalMaskDf.copy())
        missing_masks_df.set(pd.DataFrame())
        final_mask_match_df.set(finalMaskDf.copy())
        manual_mask_match_df.set(pd.DataFrame())

        clear_mask_render_cache("Mesmer results pushed to mask visualization")

        selectedChoices = list(finalMaskDf["CellMaskName"].astype(str))
        selectedDefault = selectedChoices[0] if selectedChoices else None

        _update_mask_select_controls(
            choices=selectedChoices,
            selected=selectedDefault,
        )

        ui.update_navs("viewer_mode", selected="mask", session=session)

        print(
            f"✅ Pushed Mesmer results to Mask visualization: "
            f"{len(cellDf):,} cells across {len(finalMaskDf):,} masks."
        )

    @reactive.Effect
    @reactive.event(input.push_mesmer_to_clustering)
    def _push_mesmer_to_clustering():
        cellDf = segmentation_cell_table.get()

        if cellDf is None or cellDf.empty:
            msg = "No segmentation quantification available. Run 'Create cell table from Mesmer masks' first."
            clustering_status.set(msg)
            print(f"⚠️ {msg}")
            return

        # Do not copy here.
        # This avoids duplicating a potentially large segmentation quantification table.
        clustering_data.set(cellDf)
        clustering_source.set("segmentation_quantification")
        clustering_column_map.set(_make_default_clustering_column_map(cellDf))

        # Clear column-role guesses for now.
        clustering_feature_columns.set([])
        clustering_metadata_columns.set([])
        clustering_excluded_columns.set([])

        sourcePath = segmentation_cell_table_path.get()

        msg = (
            f"Loaded segmentation quantification into Clustering: "
            f"{len(cellDf):,} cells, {len(cellDf.columns):,} columns."
        )

        if sourcePath:
            msg += f" Source table: {sourcePath}"

        clustering_status.set(msg)

        ui.update_navs("viewer_mode", selected="clustering", session=session)
        ui.update_navs("clustering_workspace_mode", selected="Data preparation", session=session)

        print(f"✅ {msg}")

    
    def _read_clustering_table_path(path: str):
        """
        Read a clustering table directly from disk.

        This avoids Shiny/browser upload overhead and is much better for large
        cell-by-marker tables.
        """
        path = str(path or "").strip()

        if not path:
            raise ValueError("No clustering table path provided.")

        if not os.path.isfile(path):
            raise ValueError(f"File does not exist: {path}")

        name_lower = path.lower()
        original_name = os.path.basename(path)

        if name_lower.endswith(".csv"):
            df = pd.read_csv(path, low_memory=False)
        elif name_lower.endswith(".tsv") or name_lower.endswith(".txt"):
            df = pd.read_csv(path, sep="\t", low_memory=False)
        else:
            df = pd.read_csv(path, sep=None, engine="python")

        return df, original_name


    def _get_valid_clustering_column_map() -> pd.DataFrame:
        """
        Return a cleaned column map containing only columns that exist in clustering_data.
        """
        df = clustering_data.get()
        colMap = clustering_column_map.get()

        expectedCols = ["ChannelNamesForClustering", "ChannelNameToDisplay"]

        if df is None or df.empty:
            return pd.DataFrame(columns=expectedCols)

        if colMap is None or colMap.empty:
            return pd.DataFrame(columns=expectedCols)

        missing = [c for c in expectedCols if c not in colMap.columns]
        if missing:
            return pd.DataFrame(columns=expectedCols)

        out = colMap[expectedCols].copy()

        out["ChannelNamesForClustering"] = (
            out["ChannelNamesForClustering"]
            .astype(str)
            .str.strip()
        )

        out["ChannelNameToDisplay"] = (
            out["ChannelNameToDisplay"]
            .astype(str)
            .str.strip()
        )

        # Remove empty rows.
        out = out.loc[out["ChannelNamesForClustering"] != ""].copy()

        # Keep only columns actually present in the loaded table.
        # Keep only columns actually present in the loaded table.
        availableCols = set(df.columns)
        out = out.loc[out["ChannelNamesForClustering"].isin(availableCols)].copy()

        # Never allow protected structural columns to become clustering features.
        out = out.loc[
            ~out["ChannelNamesForClustering"].isin(RESERVED_CLUSTERING_COLUMNS)
        ].copy()

        # If display name is empty, fall back to the source column name.
        emptyDisplay = out["ChannelNameToDisplay"] == ""
        out.loc[emptyDisplay, "ChannelNameToDisplay"] = out.loc[
            emptyDisplay,
            "ChannelNamesForClustering"
        ]

        # Avoid duplicate source columns.
        out = out.drop_duplicates(subset=["ChannelNamesForClustering"], keep="first")

        return out.reset_index(drop=True)


    def _build_selected_clustering_dataset() -> pd.DataFrame:
        """
        Build the selected marker/channel table for clustering preview.

        The protected PINT_Cell_ID is shown first in the preview but is not part
        of the clustering feature map.
        """
        df = clustering_data.get()
        colMap = _get_valid_clustering_column_map()

        if df is None or df.empty or colMap.empty:
            return pd.DataFrame()

        if PINT_CELL_ID_COL not in df.columns:
            return pd.DataFrame()

        sourceCols = colMap["ChannelNamesForClustering"].tolist()
        displayCols = colMap["ChannelNameToDisplay"].tolist()

        selected = df.loc[:, [PINT_CELL_ID_COL] + sourceCols].copy()

        # Red marker is visual only. The real master-data column remains PINT_Cell_ID.
        selected.columns = ["🔴 PINT_Cell_ID"] + displayCols

        return selected



    @output
    @render.plot
    def segmentation_mesmer_mask_preview():
        try:
            mask = segmentation_mesmer_mask.get()

            if mask is None:
                return _make_single_image_preview_figure(
                    None,
                    title="Mesmer mask",
                    errorMessage="No Mesmer mask available yet.\nRun Mesmer on the current ROI first.",
                )

            mask = np.asarray(mask)
            display = mask.astype(np.float32)

            if display.max() > 0:
                display = display / display.max()

            return _make_single_image_preview_figure(
                display,
                title=f"Mesmer mask | labels: {int(mask.max()):,}",
                cmap="nipy_spectral",
                errorMessage=None,
            )

        except SilentException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()
            return _make_single_image_preview_figure(
                None,
                title="Mesmer mask",
                errorMessage=f"Mesmer mask preview error:\n{e}",
            )


    @reactive.Effect
    @reactive.event(input.quantify_mesmer_masks)
    def _quantify_mesmer_masks():
        obj = segmentation_input_data.get()

        if obj is None:
            segmentation_quantification_status.set("No images pushed to Segmentation.")
            print("⚠️ No images pushed to Segmentation.")
            return

        quantMode = input.seg_quantification_mode() or "raw"

        try:
            if quantMode == "pint":
                imgs, chs = _build_pint_processed_dataset_for_quantification()
                quantModeLabel = "current PINT-processed images"
            else:
                imgs = obj.get("images", {})
                chs = obj.get("channels", {})
                quantModeLabel = "raw pushed images"

        except Exception as e:
            segmentation_cell_table.set(pd.DataFrame())
            segmentation_mask_table.set(pd.DataFrame())
            segmentation_cell_table_path.set("")
            segmentation_mask_table_path.set("")
            segmentation_quantification_status.set(f"Could not prepare quantification images: {e}")
            print(f"❌ Could not prepare quantification images: {e}")
            return

        if not imgs:
            segmentation_quantification_status.set("No pushed images available.")
            print("⚠️ No pushed images available.")
            return

        outDir = _get_segmentation_output_dir()
        cellTablePath = outDir / "mesmer_cell_table.csv"
        maskTablePath = outDir / "mesmer_mask_table.csv"

        sampleNames = list(imgs.keys())

        print(f"▶️ Quantifying Mesmer masks for {len(sampleNames):,} ROI(s).")
        print(f"   Quantification mode: {quantModeLabel}")
        print(f"   Mask folder: {outDir}")

        with ui.Progress(min=0, max=max(len(sampleNames), 1), session=session) as p:
            step = 0

            def progress(msg: str):
                nonlocal step
                step = min(step + 1, len(sampleNames))
                p.set(value=step, message=msg)
                print(msg, flush=True)

            p.set(value=0, message="Starting Mesmer mask quantification...")

            try:
                cellDf, maskDf = quantify_mesmer_masks_for_dataset(
                    images=imgs,
                    channels=chs,
                    mask_folder=outDir,
                    mask_suffix="_mesmer_mask_uint32.tiff",
                    progress=progress,
                )
            except Exception as e:
                segmentation_cell_table.set(pd.DataFrame())
                segmentation_mask_table.set(pd.DataFrame())
                segmentation_cell_table_path.set("")
                segmentation_mask_table_path.set("")
                segmentation_quantification_status.set(f"Quantification failed: {e}")
                print(f"❌ Quantification failed: {e}")
                return

        segmentation_cell_table.set(cellDf)
        segmentation_mask_table.set(maskDf)

        try:
            cellDf.to_csv(cellTablePath, index=False)
            maskDf.to_csv(maskTablePath, index=False)

            segmentation_cell_table_path.set(str(cellTablePath))
            segmentation_mask_table_path.set(str(maskTablePath))

        except Exception as e:
            segmentation_quantification_status.set(f"Quantified masks, but saving failed: {e}")
            print(f"⚠️ Quantified masks, but saving failed: {e}")
            return

        nMasks = int(maskDf["MaskExists"].sum()) if not maskDf.empty and "MaskExists" in maskDf.columns else 0
        nCells = len(cellDf)

        msg = (
            f"Quantification complete: {nCells:,} cells from {nMasks:,} mask(s). "
            f"Mode: {quantModeLabel}. "
            f"Saved cell table to: {cellTablePath}"
        )

        segmentation_quantification_status.set(msg)
        print(f"✅ {msg}")


    @reactive.Effect
    @reactive.event(input.export_all_mask_visualizations)
    def _confirm_export_all_mask_visualizations():
        finalDf = final_mask_match_df.get()

        if finalDf is None or finalDf.empty:
            print("⚠️ No matched masks available to export.")
            return

        nMasks = int(finalDf["MaskExists"].sum()) if "MaskExists" in finalDf.columns else len(finalDf)

        m = ui.modal(
            ui.p(
                f"You are about to export {nMasks:,} rendered mask visualizations as TIFF files."
            ),
            ui.p(
                "This can take substantial time and disk space because each output includes the image and a readable legend."
            ),
            ui.tags.small("Continue?"),
            title="Confirm export of all matched mask visualizations",
            easy_close=True,
            footer=ui.div(
                ui.modal_button("Cancel", class_="btn btn-secondary"),
                ui.input_action_button(
                    "confirm_export_all_mask_visualizations",
                    "Export",
                    class_="btn btn-primary ms-2",
                ),
            ),
        )
        ui.modal_show(m, session=session)

    @reactive.Effect
    @reactive.event(input.run_touching_analysis)
    def _run_touching_analysis():
        obj = neighborhood_input_data.get()

        if obj is None:
            neighborhood_status_msg.set("No pushed neighborhood dataset available.")
            neighborhood_matched_cells.set(pd.DataFrame())
            neighborhood_touching_edges.set(pd.DataFrame())
            neighborhood_touching_results.set(pd.DataFrame())
            neighborhood_sample_matrix.set(pd.DataFrame())
            neighborhood_permanova_results.set(pd.DataFrame())
            return

        matchTable = obj.get("mask_match_table", pd.DataFrame())
        nMasks = len(matchTable) if matchTable is not None else 0
        nPerm = int(input.touching_n_perm() or 0)

        outDir = _get_neighborhood_output_dir()

        sampleCol = obj["column_map"].get("sample_col", None)
        clusterCol = obj["column_map"].get("cluster_col", None)
        conditionCol = obj["column_map"].get("condition_col", None)

        if not sampleCol or not clusterCol or not conditionCol:
            neighborhood_status_msg.set(
                "Sample, cluster, and condition columns must be selected before running analysis."
            )
            return
        
        class NeighborhoodProgress:
            def __init__(self, p, status_setter, total=100):
                self.p = p
                self.status_setter = status_setter
                self.total = total
                self.value = 0

            def set(self, value, msg):
                self.value = max(0, min(int(value), self.total))
                msg = str(msg)

                # Bottom-right Shiny progress
                self.p.set(value=self.value, message=msg)

                # Persistent in-tab status
                self.status_setter(msg)

                # Terminal/debug output
                print(msg, flush=True)

            def message(self, msg):
                # Update message without artificially advancing too much
                self.set(self.value, msg)

            def bump(self, amount, msg):
                self.set(self.value + amount, msg)

        with ui.Progress(min=0, max=100, session=session) as p:
            prog = NeighborhoodProgress(
                p=p,
                status_setter=neighborhood_status_msg.set,
                total=100,
            )

            def progress(msg: str) -> None:
                prog.message(msg)

            prog.set(0, "Starting touching neighborhood analysis...")
            prog.set(3, "Validating neighborhood input data...")

            try:
                prog.set(
                    10,
                    f"Building touching graph for {nMasks:,} matched masks..."
                )

                matchedData, touchingEdges = build_touching_edges_for_pushed_dataset(
                    obj,
                    progress=progress,
                )

                prog.set(
                    45,
                    f"Touching graph built: {len(matchedData):,} matched cells, "
                    f"{len(touchingEdges):,} touching edges."
                )

            except Exception as e:
                neighborhood_matched_cells.set(pd.DataFrame())
                neighborhood_touching_edges.set(pd.DataFrame())
                neighborhood_touching_results.set(pd.DataFrame())
                neighborhood_sample_matrix.set(pd.DataFrame())
                neighborhood_permanova_results.set(pd.DataFrame())
                prog.set(100, f"Touching graph construction failed: {e}")
                return

            neighborhood_matched_cells.set(matchedData)
            neighborhood_touching_edges.set(touchingEdges)

            try:
                prog.set(
                    50,
                    f"Running chance correction with {nPerm:,} permutations per sample..."
                )

                resultsDf = chance_correct_touching_interactions(
                    edgeDf=touchingEdges,
                    matchedDf=matchedData,
                    sample_col=sampleCol,
                    cluster_col=clusterCol,
                    n_perm=nPerm,
                    random_seed=1,
                    progress=progress,
                )

                prog.set(
                    85,
                    f"Chance correction complete: {len(resultsDf):,} interaction rows."
                )

            except Exception as e:
                neighborhood_touching_results.set(pd.DataFrame())
                neighborhood_sample_matrix.set(pd.DataFrame())
                neighborhood_permanova_results.set(pd.DataFrame())
                prog.set(100, f"Touching graph built, but chance correction failed: {e}")
                return

            neighborhood_touching_results.set(resultsDf)

            prog.set(88, "Building sample-by-interaction matrix...")

            sampleMatrixDf = make_sample_interaction_matrix(
                resultsDf,
                sample_col=sampleCol,
                value_col="ChanceCorrectedInteraction",
            )

            analysisUnitCol = input.analysis_unit_col()

            if not analysisUnitCol or analysisUnitCol not in matchedData.columns:
                neighborhood_sample_matrix.set(pd.DataFrame())
                neighborhood_permanova_results.set(pd.DataFrame())
                prog.set(
                    100,
                    f"Final comparison unit column '{analysisUnitCol}' is not available."
                )
                return

            prog.set(92, f"Aggregating results by analysis unit: {analysisUnitCol}...")

            if analysisUnitCol == sampleCol:
                finalMatrixDf = sampleMatrixDf.copy()
                finalMetaDf = matchedData[[sampleCol, conditionCol]].drop_duplicates().copy()
                permanovaSampleCol = sampleCol
            else:
                metaDf = matchedData[[sampleCol, analysisUnitCol, conditionCol]].drop_duplicates().copy()
                finalMatrixDf, finalMetaDf = aggregate_interaction_matrix(
                    matrixDf=sampleMatrixDf,
                    metadataDf=metaDf,
                    sample_col=sampleCol,
                    aggregate_col=analysisUnitCol,
                    group_col=conditionCol,
                )
                permanovaSampleCol = analysisUnitCol

            neighborhood_sample_matrix.set(finalMatrixDf)

            prog.set(95, "Running PERMANOVA on final interaction matrix...")

            if conditionCol and conditionCol in finalMetaDf.columns and not finalMatrixDf.empty:
                permanovaDf = permanova_one_factor(
                    matrixDf=finalMatrixDf,
                    metadataDf=finalMetaDf,
                    sample_col=permanovaSampleCol,
                    group_col=conditionCol,
                    n_perm=999,
                    random_seed=1,
                )

                if not permanovaDf.empty:
                    permanovaDf["AnalysisUnit"] = str(analysisUnitCol)
                    permanovaDf["Status"] = "OK"
            else:
                permanovaDf = pd.DataFrame()

            neighborhood_permanova_results.set(permanovaDf)

            prog.set(97, "Saving neighborhood analysis output files...")

            # Save outputs
            if outDir is not None:
                if not touchingEdges.empty:
                    touchingEdges.to_csv(outDir / "touching_edges_annotated.csv", index=False)
                if not matchedData.empty:
                    matchedData.to_csv(outDir / "matched_cells_for_touching_analysis.csv", index=False)
                if not resultsDf.empty:
                    resultsDf.to_csv(outDir / "touching_interactions_with_empirical_pvalues.csv", index=False)
                if not finalMatrixDf.empty:
                    finalMatrixDf.to_csv(outDir / "touching_interaction_final_matrix.csv", index=False)
                if not permanovaDf.empty:
                    permanovaDf.to_csv(outDir / "touching_interaction_permanova.csv", index=False)
                if not finalMetaDf.empty:
                    finalMetaDf.to_csv(outDir / "touching_interaction_final_metadata.csv", index=False)

            neighborhood_status_msg.set(
                f"Touching analysis complete: {len(touchingEdges):,} touching edges across {nMasks} masks. "
                f"Results saved to: {outDir if outDir is not None else 'no output folder'}"
            )
            prog.set(
                100,
                f"Touching analysis complete: {len(touchingEdges):,} touching edges, "
                f"{len(resultsDf):,} interaction rows. "
                f"Results saved to: {outDir if outDir is not None else 'no output folder'}"
            )


    @reactive.Effect
    def _sync_analysis_unit_col_choices():
        obj = neighborhood_input_data.get()

        if obj is None:
            ui.update_select("analysis_unit_col", choices=[], selected=None, session=session)
            return

        df = obj.get("cell_table", pd.DataFrame())
        if df is None or df.empty:
            ui.update_select("analysis_unit_col", choices=[], selected=None, session=session)
            return

        colNames = list(df.columns)

        def pick_first(existingNames: list[str], fallback=None):
            for name in existingNames:
                if name in colNames:
                    return name
            return fallback

        default = pick_first(
            ["ROIName", "CellMaskName", "SampleNumber", "PatientID"],
            colNames[0] if colNames else None,
        )

        ui.update_select(
            "analysis_unit_col",
            choices=colNames,
            selected=default,
            session=session,
        )

    # <---------- plot ---------->
    ##Below is the actual img viewer where the image is rendered and all the different thresholding stept are visualized.
    @output
    @render.ui
    def neighborhood_input_summary():
        obj = neighborhood_input_data.get()

        if obj is None:
            return ui.tags.small("No dataset pushed from mask visualization yet.", class_="text-muted")

        df = obj["cell_table"]
        colMap = obj["column_map"]

        nClusters = 0
        clusterCol = colMap.get("cluster_col", None)
        if clusterCol and clusterCol in df.columns:
            nClusters = df[clusterCol].astype(str).nunique()

        return ui.div(
            ui.tags.div(f"Masks pushed: {obj['n_masks']}"),
            ui.tags.div(f"Rows: {len(df):,}"),
            ui.tags.div(f"Clusters: {nClusters}"),
            ui.tags.div(f"Mask folder: {obj['mask_folder'] or '—'}"),
            ui.tags.div(f"Pushed at: {obj['pushed_at']}"),
        )

    @output
    @render.text
    def neighborhood_status_text():
        return neighborhood_status_msg.get()

    @output
    @render.plot
    def creator_viewer():
        try:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

            rgb, used = _build_composite_rgb()

            if rgb is None:
                ax.text(
                    0.5,
                    0.5,
                    "No composite channels selected",
                    ha="center",
                    va="center",
                )
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                return fig

            ax.imshow(rgb)
            ax.set_axis_off()

            if used:
                labelLines = [
                    f"{ch} — {col} × {gain:.1f}"
                    for ch, col, gain in used
                ]

                fig.text(
                    0.01,
                    0.99,
                    "\n".join(labelLines),
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="white",
                    bbox=dict(
                        facecolor="black",
                        alpha=0.55,
                        edgecolor="none",
                        pad=4,
                    ),
                )

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

        except SilentException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()

            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

    @output
    @render.plot
    def pint_viewer():
        try:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

            imgs = images.get()
            s = input.sample()
            c = input.channel()

            if not imgs or not s or not c or s not in imgs:
                ax.text(0.5, 0.5, "No image", ha="center", va="center")
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                return fig

            arr = imgs[s]
            chlist = channels.get().get(s, [])

            if c not in chlist:
                ax.text(0.5, 0.5, f"Channel {c!r} not found", ha="center", va="center")
                ax.set_axis_off()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                return fig

            idx = chlist.index(c)
            img = arr[idx, :, :].astype(np.float32)

            normVmin = None
            normVmax = None

            if bool(input.doNorm()) and (input.norm_scope() or "page") == "global":
                doW, lo, hi = _get_winsor_settings()
                gpair = _global_range_for_channel(
                    images.get(),
                    channels.get(),
                    c,
                    doW,
                    lo,
                    hi,
                )
                if gpair is not None:
                    normVmin, normVmax = gpair

            img = process_image_pipeline(
                img,
                do_winsor=bool(input.doWinsor()),
                winsor_low=float(input.winsor_low()),
                winsor_high=float(input.winsor_high()),
                do_abs_threshold=bool(input.doAbsThreshold()),
                abs_threshold=float(input.abs_threshold_val()),
                do_fraction_threshold=bool(input.doThreshold()),
                thr_fraction=float(input.thr_fraction_val()),
                do_noise=bool(input.doNoise()),
                noise_strength=float(input.noise_strength()),
                window_size=int(input.window_size()),
                do_asinh=bool(input.doAsinh()),
                asinh_cofactor=int(float(input.asinh_cofactor() or 5)),
                do_norm=bool(input.doNorm()),
                norm_vmin=normVmin,
                norm_vmax=normVmax,
                winsor_min_upper_bound=WINSOR_MIN_UPPER_BOUND,
            )

            ax.imshow(img, cmap="gray")
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

        except SilentException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()

            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            return fig

    def _sync_mask_column_choices(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        colNames = list(df.columns)

        def pick_first(existingNames: list[str], fallback=None):
            for name in existingNames:
                if name in colNames:
                    return name
            return fallback

        default = colNames[0] if colNames else None

        ui.update_select(
            "mask_cell_id_col",
            choices=colNames,
            selected=pick_first(["CellName", "ObjectNumber", "Identifier"], default),
            session=session,
        )
        ui.update_select(
            "mask_x_col",
            choices=colNames,
            selected=pick_first(["Location_Center_X", "Center_X", "X"], default),
            session=session,
        )
        ui.update_select(
            "mask_y_col",
            choices=colNames,
            selected=pick_first(["Location_Center_Y", "Center_Y", "Y"], default),
            session=session,
        )
        ui.update_select(
            "mask_name_col",
            choices=colNames,
            selected=pick_first(["CellMaskName", "ROIName", "MaskName"], default),
            session=session,
        )
        ui.update_select(
            "mask_cluster_col",
            choices=colNames,
            selected=pick_first(
                ["MergedSublineage", "MergedSubsetNames", "CellClusterNames", "Cluster", "ClusterName"],
                default,
            ),
            session=session,
        )
        ui.update_select(
            "mask_condition_col",
            choices=colNames,
            selected=pick_first(["Condition"], default),
            session=session,
        )
        ui.update_select(
            "mask_sample_col",
            choices=colNames,
            selected=pick_first(["SampleNumber", "SampleName"], default),
            session=session,
        )

    @reactive.calc
    def active_mask_cluster_color_map() -> dict[str, str]:
        df = mask_input_df.get()

        if df is None or df.empty:
            return {}

        clusterCol = input.mask_cluster_col()

        if not clusterCol or clusterCol not in df.columns:
            return {}

        clusterNames = get_sorted_cluster_names(df[clusterCol])
        paletteName = input.mask_color_palette() or "viridis"

        if paletteName == "custom":
            return make_custom_color_map(
                clusterNames,
                input,
                default_color="#808080",
            )

        return make_palette_color_map(
            clusterNames,
            paletteName,
        )

    @output
    @render.ui
    def mask_custom_color_ui():
        df = mask_input_df.get()

        if df is None or df.empty:
            return ui.tags.small(
                "Load a cell table first.",
                class_="text-muted",
            )

        clusterCol = input.mask_cluster_col()

        if not clusterCol or clusterCol not in df.columns:
            return ui.tags.small(
                "Select a cluster column first.",
                class_="text-muted",
            )

        clusterNames = get_sorted_cluster_names(df[clusterCol])

        if len(clusterNames) == 0:
            return ui.tags.small(
                "No cluster names found.",
                class_="text-muted",
            )

        rows = [
            ui.tags.small(
                "Custom colors are used when the color palette is set to custom.",
                class_="text-muted",
            ),
            ui.tags.div(
                f"Clusters found: {len(clusterNames):,}",
                class_="compact-small-line mb-2",
            ),
        ]

        for i, clusterName in enumerate(clusterNames):
            inputId = f"mask_cluster_color_{i}"

            # Preserve existing color if the UI is rebuilt.
            try:
                currentValue = getattr(input, inputId)()
            except Exception:
                currentValue = None

            if not currentValue:
                currentValue = "#808080"

            rows.append(
                ui.row(
                    ui.column(
                        7,
                        ui.tags.div(
                            clusterName,
                            class_="small",
                            title=clusterName,
                            style=(
                                "white-space: nowrap; "
                                "overflow: hidden; "
                                "text-overflow: ellipsis;"
                            ),
                        ),
                    ),
                    ui.column(
                        5,
                        ui.tags.input(
                            id=inputId,
                            type="color",
                            value=currentValue,
                            oninput=(
                                f"Shiny.setInputValue('{inputId}', this.value, "
                                "{priority: 'event'});"
                            ),
                            onchange=(
                                f"Shiny.setInputValue('{inputId}', this.value, "
                                "{priority: 'event'});"
                            ),
                            style="width: 100%; height: 32px;",
                        ),
                    ),
                    class_="gx-1 gy-1 align-items-center",
                )
            )

        return ui.tags.div(
            *rows,
            class_="compact-stack",
            style="max-height: 350px; overflow-y: auto;",
        )

    @output
    @render.plot
    def mask_viewer():
        fig = plt.figure(figsize=(12, 8), dpi=120)

        try:
            sel = selected_mask_match.get()
            inputDf = mask_input_df.get()

            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.02)
            axImg = fig.add_subplot(gs[0, 0])
            axLeg = fig.add_subplot(gs[0, 1])

            if sel is None or sel.empty:
                axImg.text(0.5, 0.5, "No matched mask selected", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            if inputDf is None or inputDf.empty:
                axImg.text(0.5, 0.5, "No cell table loaded", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            maskNameCol = input.mask_name_col() or "CellMaskName"
            xCol = input.mask_x_col()
            yCol = input.mask_y_col()
            clusterCol = input.mask_cluster_col()
            scaleFactor = int(input.mask_scale_factor() or 5)

            if not xCol or not yCol or not clusterCol or not maskNameCol:
                axImg.text(0.5, 0.5, "Please select mask, X, Y and cluster columns", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            if clusterCol not in inputDf.columns:
                axImg.text(0.5, 0.5, f"Cluster column '{clusterCol}' not found", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            if maskNameCol not in sel.columns:
                axImg.text(
                    0.5,
                    0.5,
                    f"Selected mask table does not contain column '{maskNameCol}'",
                    ha="center",
                    va="center",
                )
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            row = sel.iloc[0]
            maskName = str(row[maskNameCol])
            maskPath = row.get("MaskPath", None)

            if not maskPath or pd.isna(maskPath):
                axImg.text(0.5, 0.5, "Selected mask has no valid file path", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            matchingData = get_cells_for_mask_name(
                inputDf,
                mask_name=maskName,
                mask_name_col=maskNameCol,
            )

            if matchingData.empty:
                axImg.text(0.5, 0.5, "No cell rows found for selected mask", ha="center", va="center")
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            clusterColorMap = active_mask_cluster_color_map()
            colorPalette = input.mask_color_palette() or "viridis"

            try:
                cachedMask = _get_cached_mask_plot_data(
                    maskPath=str(maskPath),
                    maskName=maskName,
                    matchingData=matchingData,
                    clusterCol=clusterCol,
                    xCol=xCol,
                    yCol=yCol,
                    scaleFactor=scaleFactor,
                    clusterColorMap=clusterColorMap,
                    colorPalette=colorPalette,
                )
                plotData = cachedMask["plotData"]
                matchedData = cachedMask["matchedData"]

            except Exception as e:
                axImg.text(
                    0.5,
                    0.5,
                    f"Mask visualization failed:\n{e}",
                    ha="center",
                    va="center",
                )
                axImg.set_axis_off()
                axLeg.set_axis_off()
                return fig

            axImg.imshow(plotData["colorMat"], interpolation="nearest")
            axImg.set_axis_off()

            clusterColors = plotData["clusterColors"]
            handles = [
                Patch(facecolor=clusterColors[name], edgecolor="none", label=name)
                for name in sorted(clusterColors.keys(), key=lambda x: x.lower())
            ]

            axLeg.set_axis_off()

            if len(handles) > 0:
                legendFontSize = 8
                if len(handles) <= 12:
                    legendFontSize = 10
                elif len(handles) >= 30:
                    legendFontSize = 6

                axLeg.legend(
                    handles=handles,
                    loc="upper left",
                    frameon=False,
                    fontsize=legendFontSize,
                    borderaxespad=0.0,
                )

            fig.suptitle(
                f"{maskName} | {matchedData.shape[0]} masks matched",
                fontsize=10,
                y=0.995,
            )
            plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
            return fig

        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.01, 0.98, f"Plot error: {e}", ha="left", va="top")
            ax.set_axis_off()
            return fig


    @output
    @render.ui
    ##This is the small tooltip that gived information about how the normalization actually workd
    def noise_tooltip():
        s = float(input.noise_strength())
        p = strength_to_percentile(s)
        p_txt = f"P{p*100:.1f}"
        long = (
            f"Pixels above their local {p_txt} are marked ‘bright.’ "
            "A pixel is removed only if ≤ 2 of its 8 neighbors (not scaled with window size) are also bright. "
            "Increase sensitivity to flag more pixels as bright (more aggressive denoising). "
            "Decrease to preserve detail in bright patches."
        )
        ##Small, muted line with a hover tooltip and a compact readout
        return ui.tags.small(
            ui.tags.span("ℹ️ ", class_="me-1"),
            f"Local cutoff ≈ {p_txt}",
            title=long,
            class_="text-muted"
        )

    @output
    @render.table
    ##This is the table that is actually used to render the big parameter table on the left side
    def param_table():
        return format_for_display(params_df.get())
    

    @output
    @render.ui
    def mesmer_backend_summary():
        status = mesmer_backend_status.get()

        if status is None:
            return ui.div(
                ui.tags.div("Status: not checked yet", class_="seg-status-bad"),
                ui.tags.small(
                    "Click 'Check Mesmer installation' to test the optional backend.",
                    class_="text-muted",
                ),
            )

        if status.ok:
            return ui.div(
                ui.tags.div(f"Status: {status.status}", class_="seg-status-ok"),
                ui.tags.div(f"Environment: {status.env_name}"),
                ui.tags.div(f"Conda executable: {status.conda_executable or '—'}"),
            )

        return ui.div(
            ui.tags.div(f"Status: {status.status}", class_="seg-status-bad"),
            ui.tags.div(f"Environment: {status.env_name}"),
            ui.tags.div(f"Conda executable: {status.conda_executable or '—'}"),
            ui.tags.small(
                "Use 'Show install commands' for manual setup instructions.",
                class_="text-muted",
            ),
        )

    @output
    @render.text
    def mesmer_backend_detail():
        return mesmer_backend_detail_text.get()


    @reactive.Effect
    @reactive.event(input.browse_clustering_table)
    def _browse_clustering_table():
        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        csv_path = pick_open_csv_dialog(initialdir=initdir)

        if not csv_path:
            print("🛑 Clustering table selection canceled.")
            return

        session.send_input_message("clustering_table_path", {"value": csv_path})


    @output
    @render.ui
    def clustering_data_summary():
        df = clustering_data.get()
        status = clustering_status.get()
        source = clustering_source.get()

        parts = [
            ui.tags.div(status, class_="compact-small-line"),
        ]

        if source:
            parts.append(
                ui.tags.div(
                    f"Source: {source}",
                    class_="compact-small-line",
                )
            )

        if df is None or df.empty:
            parts.append(
                ui.tags.div(
                    "No shared clustering dataset available yet.",
                    class_="compact-small-line text-muted",
                )
            )
            return ui.div(*parts)

        parts.extend(
            [
                ui.tags.div(f"Rows/cells: {len(df):,}", class_="compact-small-line"),
                ui.tags.div(f"Columns: {len(df.columns):,}", class_="compact-small-line"),
            ]
        )

        return ui.div(*parts)

    @output
    @render.data_frame
    def clustering_data_preview():
        df = clustering_data.get()

        if df is None or df.empty:
            empty = pd.DataFrame(
                {
                    "Message": [
                        "No clustering dataset loaded yet. Load a table or push segmentation quantification to Clustering."
                    ]
                }
            )
            return render.DataGrid(empty, height="400px")

        # Show only a small preview. This avoids rendering a huge table in the browser.
        show = df.head(100).copy()

        # Round numeric values in the preview only.
        # This does NOT modify clustering_data.
        numCols = show.select_dtypes(include=["number"]).columns

        if len(numCols) > 0:
            show[numCols] = show[numCols].round(4)

        return render.DataGrid(
            show,
            height="500px",
            filters=False,
        )
    
    @output
    @render.ui
    def clustering_cell_id_controls():
        df = clustering_data.get()

        if df is None or df.empty:
            return ui.tags.small(
                "Load a clustering dataset first.",
                class_="text-muted",
            )

        colNames = list(df.columns)
        method = input.clustering_cell_id_method() or "combine"

        if method == "existing":
            return ui.div(
                ui.input_select(
                    "clustering_existing_cell_id_col",
                    "Existing unique ID column",
                    choices=colNames,
                    selected=colNames[0] if colNames else None,
                )
            )

        if method == "combine":
            defaultCols = [
                c for c in ["SampleName", "CellMaskName", "ObjectNumber"]
                if c in colNames
            ]

            if not defaultCols:
                defaultCols = colNames[:2]

            return ui.div(
                ui.input_selectize(
                    "clustering_cell_id_component_cols",
                    "Columns to combine into PINT_Cell_ID",
                    choices=colNames,
                    selected=defaultCols,
                    multiple=True,
                ),

                ui.input_text(
                    "clustering_cell_id_separator",
                    "Separator",
                    value="__",
                ),
            )

        return ui.div(
            ui.tags.small(
                "PINT will create IDs from row order. This is useful for testing, "
                "but less stable than using biological/sample/cell columns.",
                class_="text-muted",
            )
        )
    
    def _validate_pint_cell_id_column(df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate that PINT_Cell_ID exists and uniquely identifies every row.
        """
        if df is None or df.empty:
            return False, "No clustering dataset loaded."

        if PINT_CELL_ID_COL not in df.columns:
            return False, f"{PINT_CELL_ID_COL} does not exist."

        ids_raw = df[PINT_CELL_ID_COL]
        ids = ids_raw.astype(str).str.strip()

        n_rows = len(ids)
        n_missing = int(ids_raw.isna().sum())
        n_empty = int((ids == "").sum())
        n_unique = int(ids.nunique(dropna=False))
        n_duplicate_rows = int(n_rows - n_unique)

        if n_missing > 0:
            return False, f"{PINT_CELL_ID_COL} contains {n_missing:,} missing value(s)."

        if n_empty > 0:
            return False, f"{PINT_CELL_ID_COL} contains {n_empty:,} empty value(s)."

        if n_unique != n_rows:
            return (
                False,
                f"{PINT_CELL_ID_COL} is not unique: "
                f"{n_unique:,} unique IDs for {n_rows:,} rows "
                f"({n_duplicate_rows:,} duplicate row(s))."
            )

        return True, f"{PINT_CELL_ID_COL} is valid for {n_rows:,} rows."


    @reactive.Effect
    @reactive.event(input.create_clustering_cell_id)
    def _create_clustering_cell_id():
        df = clustering_data.get()

        if df is None or df.empty:
            msg = "No clustering dataset loaded. Load a table first."
            print(f"⚠️ {msg}")
            clustering_status.set(msg)
            return

        method = input.clustering_cell_id_method() or "combine"

        try:
            df_new = df.copy()

            if method == "existing":
                col = input.clustering_existing_cell_id_col()

                if not col or col not in df_new.columns:
                    raise ValueError("Select a valid existing ID column.")

                ids = df_new[col].astype(str).str.strip()

            elif method == "combine":
                cols = input.clustering_cell_id_component_cols()

                if isinstance(cols, str):
                    cols = [cols]
                elif cols is None:
                    cols = []
                else:
                    cols = list(cols)

                if len(cols) < 2:
                    raise ValueError("Select at least two columns to create a combined cell ID.")

                missing_cols = [c for c in cols if c not in df_new.columns]
                if missing_cols:
                    raise ValueError(
                        "Selected ID component column(s) are missing: "
                        + ", ".join(missing_cols)
                    )

                sep = input.clustering_cell_id_separator() or "__"

                component_df = df_new.loc[:, cols].copy()

                if component_df.isna().any().any():
                    missing_counts = component_df.isna().sum()
                    missing_counts = missing_counts.loc[missing_counts > 0]
                    detail = ", ".join(
                        f"{col}: {int(n):,}"
                        for col, n in missing_counts.items()
                    )
                    raise ValueError(
                        "Cannot create combined PINT_Cell_ID because selected columns "
                        f"contain missing values: {detail}"
                    )

                ids = component_df.astype(str).apply(
                    lambda row: sep.join(v.strip() for v in row),
                    axis=1,
                )

            elif method == "row_order":
                n = len(df_new)
                width = max(9, len(str(n)))

                ids = pd.Series(
                    [f"PINT_Cell_{i:0{width}d}" for i in range(1, n + 1)],
                    index=df_new.index,
                )

            else:
                raise ValueError(f"Unknown Cell ID method: {method}")

            df_new[PINT_CELL_ID_COL] = ids.astype(str)

            ok, msg = _validate_pint_cell_id_column(df_new)

            if not ok:
                raise ValueError(msg)

            # Add/update only the protected ID column.
            # All original columns are retained.
            clustering_data.set(df_new)

            # Refresh the default map after creating PINT_Cell_ID.
            # The map helper should exclude PINT_Cell_ID.
            clustering_column_map.set(_make_default_clustering_column_map(df_new))

            clustering_status.set(msg)

            print(f"✅ {msg}", flush=True)

        except Exception as e:
            msg = f"Could not create PINT_Cell_ID: {e}"
            clustering_status.set(msg)
            print(f"❌ {msg}", flush=True)

    @output
    @render.ui
    def clustering_cell_id_summary():
        df = clustering_data.get()

        if df is None or df.empty:
            return ui.tags.small(
                "No clustering dataset loaded yet.",
                class_="text-muted",
            )

        if PINT_CELL_ID_COL not in df.columns:
            return ui.div(
                ui.tags.div(
                    "PINT_Cell_ID has not been created yet.",
                    class_="compact-small-line text-muted",
                ),
                ui.tags.small(
                    "Create a unique cell identifier before selecting columns for PCA, PaCMAP, or clustering.",
                    class_="text-muted",
                ),
            )

        ids = df[PINT_CELL_ID_COL].astype(str)

        nRows = len(ids)
        nUnique = ids.nunique(dropna=False)
        nMissing = int(df[PINT_CELL_ID_COL].isna().sum())
        nEmpty = int((ids.str.strip() == "").sum())
        nDuplicateRows = int(nRows - nUnique)

        isValid = (
            nRows > 0
            and nMissing == 0
            and nEmpty == 0
            and nUnique == nRows
        )

        if isValid:
            return ui.div(
                ui.tags.div(
                    "PINT_Cell_ID status: valid",
                    class_="compact-small-line text-success",
                ),
                ui.tags.div(
                    f"Rows: {nRows:,}",
                    class_="compact-small-line",
                ),
                ui.tags.div(
                    f"Unique IDs: {nUnique:,}",
                    class_="compact-small-line",
                ),
                ui.tags.div(
                    "Missing/empty IDs: 0",
                    class_="compact-small-line",
                ),
            )

        return ui.div(
            ui.tags.div(
                "PINT_Cell_ID status: invalid",
                class_="compact-small-line text-danger",
            ),
            ui.tags.div(
                f"Rows: {nRows:,}",
                class_="compact-small-line",
            ),
            ui.tags.div(
                f"Unique IDs: {nUnique:,}",
                class_="compact-small-line",
            ),
            ui.tags.div(
                f"Missing IDs: {nMissing:,}",
                class_="compact-small-line",
            ),
            ui.tags.div(
                f"Empty IDs: {nEmpty:,}",
                class_="compact-small-line",
            ),
            ui.tags.div(
                f"Duplicate rows: {nDuplicateRows:,}",
                class_="compact-small-line",
            ),
            ui.tags.small(
                "Use an existing unique column, or combine columns such as SampleName + ObjectNumber.",
                class_="text-muted",
            ),
        )

    @output
    @render.ui
    def clustering_column_map_summary():
        df = clustering_data.get()
        colMap = clustering_column_map.get()
        validMap = _get_valid_clustering_column_map()

        if df is None or df.empty:
            return ui.tags.small(
                "No clustering dataset loaded yet.",
                class_="text-muted",
            )

        nMapped = 0 if colMap is None or colMap.empty else len(colMap)
        nValid = len(validMap)
        nTotal = len(df.columns)

        return ui.div(
            ui.tags.div(
                f"Current dataset columns: {nTotal:,}",
                class_="compact-small-line",
            ),
            ui.tags.div(
                f"Rows in column map: {nMapped:,}",
                class_="compact-small-line",
            ),
            ui.tags.div(
                f"Valid selected columns for clustering: {nValid:,}",
                class_="compact-small-line",
            ),
            ui.tags.small(
                "Only columns listed in the imported column map and present in the loaded table "
                "will be used for clustering.",
                class_="text-muted",
            ),
        )


    @output
    @render.data_frame
    def selected_clustering_data_preview():
        selected = _build_selected_clustering_dataset()

        if selected is None or selected.empty:
            empty = pd.DataFrame(
                {
                    "Message": [
                        "No selected clustering dataset yet. Export, edit, and import a clustering column map first."
                    ]
                }
            )
            return render.DataGrid(empty, height="300px", filters=False)

        show = selected.head(100).copy()

        numCols = show.select_dtypes(include=["number"]).columns
        if len(numCols) > 0:
            show[numCols] = show[numCols].round(4)

        return render.DataGrid(
            show,
            height="350px",
            filters=False,
        )


    
    @output
    @render.ui
    def neighborhood_touching_summary():
        edgeDf = neighborhood_touching_edges.get()
        matchedDf = neighborhood_matched_cells.get()
        resultsDf = neighborhood_touching_results.get()

        if edgeDf is None or edgeDf.empty:
            return ui.tags.small("No touching-edge results yet.", class_="text-muted")

        nCells = 0 if matchedDf is None or matchedDf.empty else len(matchedDf)

        nClusters = 0
        if "cell_cluster" in edgeDf.columns:
            nClusters = pd.unique(
                pd.concat(
                    [
                        edgeDf["cell_cluster"].astype(str),
                        edgeDf["neighbor_cluster"].astype(str),
                    ],
                    ignore_index=True,
                )
            ).size

        nInteractions = 0 if resultsDf is None or resultsDf.empty else len(resultsDf)

        return ui.div(
            ui.tags.div(f"Matched cells: {nCells:,}"),
            ui.tags.div(f"Touching edges: {len(edgeDf):,}"),
            ui.tags.div(f"Clusters observed: {nClusters:,}"),
            ui.tags.div(f"Chance-corrected interaction rows: {nInteractions:,}"),
        )
    
    @output
    @render.data_frame
    def touching_results_preview():
        df = neighborhood_touching_results.get()

        expectedCols = [
            "cell_cluster",
            "neighbor_cluster",
            "observed",
            "expected",
            "n_cells",
            "observed_per_cell",
            "expected_per_cell",
            "ChanceCorrectedInteraction",
            "Direction",
            "perm_sd",
            "p_enriched",
            "p_depleted",
            "p_two_sided",
            "p_adj_enriched",
            "p_adj_depleted",
            "p_adj_two_sided",
        ]

        if df is None or df.empty:
            empty = pd.DataFrame(columns=expectedCols)
            return render.DataGrid(empty, height="350px")

        show = df.copy()

        # Add missing expected columns only for forward/backward compatibility.
        # Do not include removed legacy columns here.
        for col in expectedCols:
            if col not in show.columns:
                show[col] = np.nan

        # Order columns nicely, then keep any extras at the end.
        extraCols = [c for c in show.columns if c not in expectedCols]
        show = show[expectedCols + extraCols]

        # Only show first 200 rows.
        show = show.head(200).copy()

        # Round numeric columns.
        numCols = show.select_dtypes(include=["number"]).columns
        show[numCols] = show[numCols].round(3)

        return render.DataGrid(show, height="350px")


    @output
    @render.ui
    def permanova_summary():
        df = neighborhood_permanova_results.get()

        if df is None or df.empty:
            return ui.tags.small("No PERMANOVA results available.", class_="text-muted")

        row = df.iloc[0]

        parts = []

        if "AnalysisUnit" in row.index:
            parts.append(ui.tags.div(f"Analysis unit: {row['AnalysisUnit']}"))

        if "Status" in row.index and pd.notna(row["Status"]):
            parts.append(ui.tags.div(f"Status: {row['Status']}"))

        if pd.notna(row.get("N", np.nan)):
            parts.append(ui.tags.div(f"N observations: {int(row['N'])}"))

        if pd.notna(row.get("Groups", np.nan)):
            parts.append(ui.tags.div(f"N groups: {int(row['Groups'])}"))

        if pd.notna(row.get("PseudoF", np.nan)):
            parts.append(ui.tags.div(f"Pseudo-F: {row['PseudoF']:.4f}"))
        else:
            parts.append(ui.tags.div("Pseudo-F: NA"))

        if pd.notna(row.get("PValue", np.nan)):
            parts.append(ui.tags.div(f"P-value: {row['PValue']:.4g}"))
        else:
            parts.append(ui.tags.div("P-value: NA"))

        if pd.notna(row.get("Permutations", np.nan)):
            parts.append(ui.tags.div(f"Permutations: {int(row['Permutations'])}"))

        return ui.div(*parts)

    @reactive.Effect
    @reactive.event(input.perform_analysis)
    ##The moment you hit "analyze" this will popup to give the user a chance to cancel if it was by accident
    def _confirm_start_modal():
        folder = (input.path() or "").strip()
        msg = ui.div(
            ui.p("Start image processing?"),
            ui.tags.small(f"Folder: {folder or '— (no folder chosen)'}")
        )
        m = ui.modal(
            msg,
            title="Confirm",
            easy_close=True,  # clicking outside = Cancel
            footer=ui.div(
                ui.modal_button("Cancel", class_="btn btn-secondary"),
                ui.input_action_button("confirm_start", "Start", class_="btn btn-primary ms-2"),
            ),
            size="m",
        )
        ui.modal_show(m, session=session)


    @reactive.Effect
    @reactive.event(input.confirm_start)
    ##This will run the batch of images through he analysis.py pipeline
    def _run_batch_analysis():
        #Close the modal (the yes i'm sure button), see above block
        ui.modal_remove(session=session)
        ##reads the folder path, uses the input folder, checks if its non-empty and it exist. Strips all the white space.
        ##Throws an exception if not
        folder = (input.path() or "").strip()
        if not folder or not os.path.isdir(folder):
            print(f"⚠️ Invalid folder: {folder!r}")
            return
        ##Create new folder inside of the input one.
        out_dir = os.path.join(folder, "normalized images")
        os.makedirs(out_dir, exist_ok=True)
        ##Makes an output path for the parameter file to save it
        ##If the table is empty it will return a messege to tell the user
        params_path = os.path.join(out_dir, "parameter_table.csv")
        df = params_df.get().copy()
        if df.empty:
            print("⚠️ Parameter table is empty — nothing to analyze.")
            return
        df.to_csv(params_path, index=False)
        print(f"✅ Saved parameter table → {params_path}")
        ##Tries to open de analysis.py file from the same folder as the viewer.
        cmd = [
            sys.executable, "-m", "pint_app.core.analysis",
            "--input-dir", folder,
            "--params-csv", params_path,
            "--output-dir", out_dir,
        ]   
        print("▶️ Running analysis:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print("✅ Analysis finished.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis script failed: {e}")
    
    @output
    @render.ui
    def norm_scope_hint():
        ##If nothing loaded, exit
        if not images.get() or not input.channel():
            return ui.tags.small(" ")
        ##If not empty, but not set to global, also return nothing
        if (input.norm_scope() or "page") != "global":
            return ui.tags.small(" ")
        ##Reads the current setting winsor quantile etc
        ch = input.channel()
        do_w = bool(input.doWinsor())
        lo   = max(0.0, min(1.0, float(input.winsor_low())))
        hi   = max(0.0, min(1.0, float(input.winsor_high())))
        ##looks across all samples to get thehighest and the lowest numbers. This is the global max/min for per image winsorization
        gpair = _global_range_for_channel(images.get(), channels.get(), ch, do_w, lo, hi)
        if not gpair:
            return ui.tags.small("Global range: NO VALID RANGE")
        gmin, gmax = gpair

        try:
            s = input.sample()
        except Exception:
            s = None
        ##This is the image winsorization settings 
        ipair = _image_range_for_channel(images.get(), channels.get(), ch, s, do_w, lo, hi)

        parts = [
            ui.tags.small(
                f'Global range: “{ch}” (winsor {"on" if (do_w and hi>lo) else "off"}): '
                f'[{fmt1(gmin)}, {fmt1(gmax)}]'
            )
        ]
        ##Note that the range is the min/max range per channel but the winsorization is per image. If winsorization if turned off there it's just glabal min max
        if ipair:
            imin, imax = ipair
            label = f'Image range: “{s}” range' if s else "Image range"
            parts += [ui.br(), ui.tags.small(f"{label}: [{fmt1(imin)}, {fmt1(imax)}]")]
        return ui.div(*parts)

    @reactive.Effect
    @reactive.event(input.export_params)
    ##Export function for the parameter file
    def _export_params():
        ##copy of the parameter table, check if empty. If so, warn the user
        df = params_df.get().copy()
        if df.empty:
            print("⚠️ No parameters to export.")
            return
        ##The input folder is used as default folder in the save as thats called below it
        folder = (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()
        save_path = pick_save_csv_dialog(initialdir=initdir, initialfile="parameter_table.csv")
        ##Save or not saved warnings
        if not save_path:
            print("🛑 Export canceled.")
            return

        try:
            df.to_csv(save_path, index=False)
            print(f"✅ Exported parameters → {save_path}")
                 
        except Exception as e:
            print(f"❌ Failed to write CSV: {e}")

 
    @reactive.Effect
    @reactive.event(input.import_params)
    def _import_params():
        if not data_loaded.get():
            print("⚠️ Load images before importing parameters.")
            return

        folder = last_loaded_folder.get() or (input.path() or "").strip()
        initdir = folder if folder and os.path.isdir(folder) else os.getcwd()

        csv_path = pick_open_csv_dialog(initialdir=initdir)
        if not csv_path:
            print("🛑 Import canceled.")
            return

        try:
            df_in = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Failed to read CSV: {e}")
            return

        if "Channel" not in df_in.columns:
            print("❌ CSV missing required 'Channel' column.")
            return

        canon = list(canonical_channels.get() or [])
        try:
            df_in = validate_and_normalize_import(df_in, canon)
        except ValueError as e:
            print(f"❌ {e}")
            return

        params_df.set(df_in.reset_index(drop=True))

        sel = input.channel()
        if sel:
            _sync_controls_from_table(sel)

        s = input.sample()
        if s:
            _sync_composite_channel_choices(s, overwriteDefaults=False)

        print(f"✅ Imported parameters from {csv_path}")

    @reactive.Effect
    @reactive.event(input.browse_mask_path)
    def _browse_mask_path():
        try:
            folder = pick_folder_dialog()
        except Exception:
            folder = ""

        if not folder:
            return

        session.send_input_message("mask_path", {"value": folder})

    @reactive.Effect
    @reactive.event(input.match_masks)
    def _match_masks():
        df = mask_input_df.get()

        if df is None or df.empty:
            print("⚠️ No cell table available for mask matching. Load a mask CSV first.")
            mask_files_df.set(pd.DataFrame())
            mask_match_df.set(pd.DataFrame())
            matched_masks_df.set(pd.DataFrame())
            missing_masks_df.set(pd.DataFrame())
            final_mask_match_df.set(pd.DataFrame())
            manual_mask_match_df.set(pd.DataFrame())
            selected_mask_match.set(pd.DataFrame())
            _clear_mask_selection_controls()
            clear_mask_render_cache("mask matching cleared: no cell table")
            return

        mask_dir = (input.mask_path() or "").strip()

        if not mask_dir or not os.path.isdir(mask_dir):
            print("⚠️ Invalid mask folder.")
            mask_files_df.set(pd.DataFrame())
            mask_match_df.set(pd.DataFrame())
            matched_masks_df.set(pd.DataFrame())
            missing_masks_df.set(pd.DataFrame())
            final_mask_match_df.set(pd.DataFrame())
            manual_mask_match_df.set(pd.DataFrame())
            selected_mask_match.set(pd.DataFrame())
            _clear_mask_selection_controls()
            clear_mask_render_cache("mask matching cleared: invalid mask folder")
            return

        cell_id_col = input.mask_cell_id_col()
        x_col = input.mask_x_col()
        y_col = input.mask_y_col()
        mask_name_col = input.mask_name_col()

        try:
            df_valid = validate_mask_input_table(
                df,
                cell_id_col=cell_id_col,
                x_col=x_col,
                y_col=y_col,
                mask_name_col=mask_name_col,
            )

            files_df = list_mask_files(mask_dir)

            match_df = match_cellmask_names_to_files(
                df_valid,
                files_df,
                mask_name_col=mask_name_col,
            )

            matched_df, missing_df = split_mask_matches(match_df)

        except Exception as e:
            print(f"❌ Mask matching failed: {e}")
            mask_files_df.set(pd.DataFrame())
            mask_match_df.set(pd.DataFrame())
            matched_masks_df.set(pd.DataFrame())
            missing_masks_df.set(pd.DataFrame())
            final_mask_match_df.set(pd.DataFrame())
            manual_mask_match_df.set(pd.DataFrame())
            selected_mask_match.set(pd.DataFrame())
            _clear_mask_selection_controls()
            clear_mask_render_cache("mask matching failed or cleared")
            return

        mask_input_df.set(df_valid)
        mask_files_df.set(files_df)
        mask_match_df.set(match_df)
        matched_masks_df.set(matched_df)
        missing_masks_df.set(missing_df)

        final_df = match_df.loc[match_df["MaskExists"]].copy()
        final_mask_match_df.set(final_df)
        manual_mask_match_df.set(pd.DataFrame())

        clear_mask_render_cache("mask matching updated")

        selected_choices = (
            list(final_df[mask_name_col].astype(str))
            if not final_df.empty
            else []
        )
        selected_default = selected_choices[0] if selected_choices else None

        _update_mask_select_controls(
            choices=selected_choices,
            selected=selected_default,
        )

        print(f"✅ Mask matching complete: {len(matched_df)} matched, {len(missing_df)} missing.")

    @reactive.Effect
    @reactive.event(input.confirm_export_all_mask_visualizations)
    def _export_all_mask_visualizations():
        ui.modal_remove(session=session)

        finalDf = final_mask_match_df.get()
        inputDf = mask_input_df.get()

        if finalDf is None or finalDf.empty:
            print("⚠️ No matched masks available to export.")
            return

        if inputDf is None or inputDf.empty:
            print("⚠️ No cell table loaded.")
            return

        maskNameCol = input.mask_name_col() or "CellMaskName"
        xCol = input.mask_x_col()
        yCol = input.mask_y_col()
        clusterCol = input.mask_cluster_col()
        scaleFactor = int(input.mask_scale_factor() or 2)

        if not xCol or not yCol or not clusterCol or not maskNameCol:
            print("⚠️ Mask name, X, Y, and cluster columns must be selected.")
            return

        exportDf = finalDf.loc[finalDf["MaskExists"]].copy()
        if exportDf.empty:
            print("⚠️ No valid matched masks available to export.")
            return

        maskDir = (input.mask_path() or "").strip()
        if not maskDir:
            # fallback from first valid mask path
            if "MaskPath" in exportDf.columns and not exportDf["MaskPath"].isna().all():
                maskDir = str(Path(exportDf["MaskPath"].dropna().iloc[0]).parent)

        if not maskDir:
            print("⚠️ Could not determine mask folder.")
            return

        outDir = Path(maskDir) / "Exported mask visualizations"
        outDir.mkdir(parents=True, exist_ok=True)

        nMasks = len(exportDf)

        clusterColorMap = active_mask_cluster_color_map()

        with ui.Progress(min=0, max=nMasks, session=session) as p:
            done = 0

            for _, row in exportDf.iterrows():
                maskName = str(row[maskNameCol])
                maskPath = row.get("MaskPath", None)

                if not maskPath or pd.isna(maskPath):
                    done += 1
                    p.set(value=done, message=f"Skipping {maskName}: no valid mask path")
                    continue

                try:
                    matchingData = get_cells_for_mask_name(
                        inputDf,
                        mask_name=maskName,
                        mask_name_col=maskNameCol,
                    )

                    if matchingData.empty:
                        done += 1
                        p.set(value=done, message=f"Skipping {maskName}: no rows in cell table")
                        continue

                    fig, matchedData = _build_mask_visualization_figure(
                        maskPath=str(maskPath),
                        maskName=maskName,
                        matchingData=matchingData,
                        clusterCol=clusterCol,
                        xCol=xCol,
                        yCol=yCol,
                        scaleFactor=scaleFactor,
                        clusterColorMap=clusterColorMap,
                    )

                    safeName = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in maskName)
                    outPath = outDir / f"{safeName}.tiff"

                    fig.savefig(
                        outPath,
                        format="tiff",
                        dpi=200,
                        bbox_inches="tight",
                        pil_kwargs={"compression": "tiff_lzw"},
                    )
                    plt.close(fig)

                    done += 1
                    p.set(value=done, message=f"Exported {maskName} ({matchedData.shape[0]:,} cells)")

                except Exception as e:
                    try:
                        plt.close("all")
                    except Exception:
                        pass
                    done += 1
                    p.set(value=done, message=f"Failed {maskName}: {e}")

        print(f"✅ Exported {nMasks:,} matched mask visualizations to {outDir}")


    @reactive.Effect
    @reactive.event(input.browse_mask_csv)
    def _browse_mask_csv():
        csv_path = pick_open_csv_dialog(initialdir=os.getcwd())
        if not csv_path:
            return
        session.send_input_message("mask_csv_path", {"value": csv_path})


    @reactive.Effect
    @reactive.event(input.load_mask_csv)
    def _load_mask_csv():
        csv_path = (input.mask_csv_path() or "").strip()

        def clear_mask_state(reason: str) -> None:
            mask_input_df.set(pd.DataFrame())
            mask_files_df.set(pd.DataFrame())
            mask_match_df.set(pd.DataFrame())
            matched_masks_df.set(pd.DataFrame())
            missing_masks_df.set(pd.DataFrame())
            final_mask_match_df.set(pd.DataFrame())
            manual_mask_match_df.set(pd.DataFrame())
            selected_mask_match.set(pd.DataFrame())
            _clear_mask_selection_controls()
            clear_mask_render_cache(reason)

        if not csv_path or not os.path.isfile(csv_path):
            print("⚠️ Invalid mask CSV path.")
            clear_mask_state("invalid mask CSV path")
            return

        try:
            df_mask = pd.read_csv(csv_path, index_col=0)
        except Exception as e:
            print(f"❌ Failed to read mask CSV: {e}")
            clear_mask_state("failed to read mask CSV")
            return

        if df_mask.empty:
            print("⚠️ Mask CSV is empty.")
            clear_mask_state("empty mask CSV")
            return

        mask_input_df.set(df_mask)

        # New source cell table means all previous mask renders are stale.
        mask_files_df.set(pd.DataFrame())
        mask_match_df.set(pd.DataFrame())
        matched_masks_df.set(pd.DataFrame())
        missing_masks_df.set(pd.DataFrame())
        final_mask_match_df.set(pd.DataFrame())
        manual_mask_match_df.set(pd.DataFrame())
        selected_mask_match.set(pd.DataFrame())
        _clear_mask_selection_controls()

        clear_mask_render_cache("new mask cell table loaded")
        _sync_mask_column_choices(df_mask)

        print(f"✅ Loaded mask cell table → {csv_path}")


    ##Push data from mask to neigborhood
    @reactive.Effect
    @reactive.event(input.push_to_neighborhood)
    def _push_to_neighborhood():
        try:
            pushed = _build_neighborhood_input_object()
            neighborhood_input_data.set(pushed)

            nRows = len(pushed["cell_table"])
            nMasks = int(pushed["n_masks"])
            neighborhood_status_msg.set(
                f"Pushed {nMasks} matched masks to neighborhood analysis ({nRows:,} rows)."
            )

            ui.update_navs("viewer_mode", selected="neighborhood", session=session)

        except Exception as e:
            neighborhood_status_msg.set(f"Failed to push dataset: {e}")

    @output
    @render.ui
    def selected_mask_summary():
        sel = selected_mask_match.get()
        df = mask_input_df.get()

        if sel is None or sel.empty:
            return ui.tags.small("No matched mask selected.", class_="text-muted")

        row = sel.iloc[0]
        maskNameCol = input.mask_name_col() or "CellMaskName"
        clusterCol = input.mask_cluster_col()

        maskName = str(row[maskNameCol])
        nCells = 0
        nClusters = 0

        if df is not None and not df.empty and maskNameCol in df.columns:
            tempDf = df.loc[df[maskNameCol].astype(str) == maskName].copy()
            nCells = tempDf.shape[0]
            if clusterCol and clusterCol in tempDf.columns:
                nClusters = tempDf[clusterCol].astype(str).nunique()

        matchType = "Manual" if bool(row.get("ManualMatch", False)) else "Automatic"

        return ui.div(
            ui.tags.div(f"Selected mask: {maskName}", class_="compact-small-line"),
            ui.tags.div(f"Mask file: {row.get('MaskFile', '—')}", class_="compact-small-line"),
            ui.tags.div(f"Match type: {matchType}", class_="compact-small-line"),
            ui.tags.div(f"Rows in cell table: {nCells}", class_="compact-small-line"),
            ui.tags.div(f"Clusters present: {nClusters}", class_="compact-small-line"),
        )

    @output
    @render.ui
    def mask_table_summary():
        df = mask_input_df.get()

        if df is None or df.empty:
            return ui.tags.small("No cell table loaded yet.", class_="text-muted")

        return ui.div(
            ui.tags.div(f"Rows in cell table: {len(df)}", class_="compact-small-line"),
            ui.tags.div(f"Columns in cell table: {len(df.columns)}", class_="compact-small-line"),
        )

    @output
    @render.ui
    def mask_match_summary():
        match_df = mask_match_df.get()
        final_df = final_mask_match_df.get()
        manual_df = manual_mask_match_df.get()

        if match_df is None or match_df.empty:
            return ui.tags.small("No mask matching performed yet.", class_="text-muted")

        autoMatched = int(match_df["MaskExists"].sum())
        total = len(match_df)
        manualMatched = 0 if manual_df is None or manual_df.empty else len(manual_df)
        finalMatched = 0 if final_df is None or final_df.empty else int(final_df["MaskExists"].sum())

        return ui.div(
            ui.tags.div(f"Unique mask names in table: {total}", class_="compact-small-line"),
            ui.tags.div(f"Automatic matches: {autoMatched}", class_="compact-small-line"),
            ui.tags.div(f"Manual matches: {manualMatched}", class_="compact-small-line"),
            ui.tags.div(f"Final matched masks: {finalMatched}", class_="compact-small-line"),
            ui.tags.div(f"Still unmatched: {total - finalMatched}", class_="compact-small-line"),
        )
    
    @output
    @render.table
    def mask_match_table():
        df = mask_match_df.get()

        if df is None or df.empty:
            return pd.DataFrame(columns=["CellMaskName", "MaskFile", "MaskExists"])

        return df
    
    @reactive.Effect
    @reactive.event(input.selected_mask_name)
    def _on_selected_mask_name():
        selectedName = input.selected_mask_name()

        if not selectedName:
            selected_mask_match.set(pd.DataFrame())
            return

        if selectedName != input.selected_mask_name_display():
            ui.update_select(
                "selected_mask_name_display",
                selected=selectedName,
                session=session,
            )

        _set_selected_mask_match(selectedName)

    @reactive.Effect
    @reactive.event(input.selected_mask_name_display)
    def _on_selected_mask_name_display():
        selectedName = input.selected_mask_name_display()

        if not selectedName:
            return

        if selectedName == input.selected_mask_name():
            return

        _select_mask_by_name(selectedName)


    @reactive.Effect
    @reactive.event(input.mask_prev)
    def _mask_prev():
        _cycle_mask_selection(step=-1)


    @reactive.Effect
    @reactive.event(input.mask_next)
    def _mask_next():
        _cycle_mask_selection(step=1)       



    @output
    @render.ui
    def mask_visualization_placeholder():
        sel = selected_mask_match.get()

        if sel is None or sel.empty:
            return ui.tags.div(
                ui.tags.p("No matched mask selected."),
                class_="text-muted"
            )

        row = sel.iloc[0]
        return ui.tags.div(
            ui.tags.p("Visualization area reserved for mask rendering."),
            ui.tags.p(f"Selected file: {row.get('MaskFile', '—')}"),
            class_="text-muted"
        )
    
    ##Handlers for unmatched masks
    @output
    @render.ui
    def manual_mask_match_ui():
        match_df = mask_match_df.get()
        files_df = mask_files_df.get()

        if match_df is None or match_df.empty:
            return ui.tags.small("No mask matching performed yet.", class_="text-muted")

        unmatched_df = match_df.loc[~match_df["MaskExists"]].copy()
        if unmatched_df.empty:
            return ui.tags.small("All masks matched automatically.", class_="text-muted")

        if files_df is None or files_df.empty:
            return ui.tags.small("No mask files available for manual matching.", class_="text-muted")

        fileChoices = [""] + list(files_df["MaskFile"].astype(str))

        rows = [
            ui.tags.small(
                "Only unmatched mask names are shown here. Select a file manually and click 'Apply manual matches'.",
                class_="text-muted"
            )
        ]
        for i, (_, row) in enumerate(unmatched_df.iterrows()):
            maskName = str(row[input.mask_name_col() or "CellMaskName"])
            rows.append(
                ui.input_select(
                    f"manual_mask_match_{i}",
                    label=maskName,
                    choices=fileChoices,
                    selected="",
                    width="100%",
                )
            )

        return ui.div(*rows)

    @reactive.Effect
    @reactive.event(input.apply_manual_mask_matches)
    def _apply_manual_mask_matches():
        match_df = mask_match_df.get()
        files_df = mask_files_df.get()

        if match_df is None or match_df.empty:
            print("⚠️ No automatic mask matching available.")
            manual_mask_match_df.set(pd.DataFrame())
            final_mask_match_df.set(pd.DataFrame())
            _clear_mask_selection_controls()
            return

        mask_name_col = input.mask_name_col() or "CellMaskName"

        if "MaskExists" not in match_df.columns:
            print("⚠️ Mask match table does not contain 'MaskExists'.")
            manual_mask_match_df.set(pd.DataFrame())
            final_mask_match_df.set(pd.DataFrame())
            _clear_mask_selection_controls()
            return

        unmatched_df = match_df.loc[~match_df["MaskExists"]].copy()

        if unmatched_df.empty:
            manual_mask_match_df.set(pd.DataFrame())

            final_df = match_df.copy()
            final_mask_match_df.set(final_df)

            selected_choices = (
                list(final_df.loc[final_df["MaskExists"], mask_name_col].astype(str))
                if not final_df.empty and mask_name_col in final_df.columns
                else []
            )

            current_selected = input.selected_mask_name()
            selected_default = (
                current_selected
                if current_selected in selected_choices
                else (selected_choices[0] if selected_choices else None)
            )

            _update_mask_select_controls(
                choices=selected_choices,
                selected=selected_default,
            )

            print("✅ No manual matches needed.")
            return

        if files_df is None or files_df.empty:
            print("⚠️ No mask files available for manual matching.")
            return

        manualRows = []

        for i, (_, row) in enumerate(unmatched_df.iterrows()):
            selectedFile = getattr(input, f"manual_mask_match_{i}")()

            if not selectedFile:
                continue

            fileRow = files_df.loc[
                files_df["MaskFile"].astype(str) == str(selectedFile)
            ].copy()

            if fileRow.empty:
                continue

            fileRow = fileRow.iloc[0]

            manualRows.append(
                {
                    mask_name_col: row[mask_name_col],
                    "MaskFile": fileRow["MaskFile"],
                    "MaskPath": fileRow["MaskPath"],
                    "CellMaskName_file": fileRow["CellMaskName"],
                    "MaskExists": True,
                    "ManualMatch": True,
                }
            )

        manual_df = pd.DataFrame(manualRows)
        manual_mask_match_df.set(manual_df)

        auto_df = match_df.copy()
        auto_df["ManualMatch"] = False

        if not manual_df.empty:
            matched_names = set(manual_df[mask_name_col].astype(str))

            auto_df = auto_df.loc[
                ~auto_df[mask_name_col].astype(str).isin(matched_names)
            ].copy()

            final_df = pd.concat([auto_df, manual_df], ignore_index=True)
        else:
            final_df = auto_df

        final_mask_match_df.set(final_df)
        clear_mask_render_cache("manual mask matching updated")

        selected_choices = (
            list(final_df.loc[final_df["MaskExists"], mask_name_col].astype(str))
            if not final_df.empty and "MaskExists" in final_df.columns
            else []
        )

        current_selected = input.selected_mask_name()
        selected_default = (
            current_selected
            if current_selected in selected_choices
            else (selected_choices[0] if selected_choices else None)
        )

        _update_mask_select_controls(
            choices=selected_choices,
            selected=selected_default,
        )

        nManual = len(manual_df)
        print(f"✅ Applied {nManual} manual mask match(es).")


    ##Helper to push the data to the neigborhood analysis tab
    def _build_neighborhood_input_object():
        matchDf = final_mask_match_df.get()
        inputDf = mask_input_df.get()

        if matchDf is None or matchDf.empty:
            raise ValueError("No matched masks available.")

        if inputDf is None or inputDf.empty:
            raise ValueError("No cell table loaded.")

        maskNameCol = input.mask_name_col() or "CellMaskName"
        xCol = input.mask_x_col()
        yCol = input.mask_y_col()
        clusterCol = input.mask_cluster_col()
        conditionCol = input.mask_condition_col()
        sampleCol = input.mask_sample_col()

        if not xCol or not yCol or not clusterCol or not maskNameCol:
            raise ValueError("Mask name, X, Y, and cluster columns must be selected.")

        validMatchDf = matchDf.loc[matchDf["MaskExists"]].copy()
        if validMatchDf.empty:
            raise ValueError("No valid matched masks available.")

        maskNames = set(validMatchDf[maskNameCol].astype(str))
        pushedCellTable = inputDf.loc[inputDf[maskNameCol].astype(str).isin(maskNames)].copy()

        if pushedCellTable.empty:
            raise ValueError("No cell-table rows found for the matched masks.")

        # Keep only the needed mask-match rows
        validMatchDf = validMatchDf.drop_duplicates(subset=[maskNameCol]).copy()

        # Attach mask path onto each cell-table row
        pushedCellTable = pushedCellTable.merge(
            validMatchDf[[maskNameCol, "MaskFile", "MaskPath", "MaskExists"]].copy(),
            on=maskNameCol,
            how="left",
        )

        maskFolder = None
        if "MaskPath" in validMatchDf.columns and not validMatchDf["MaskPath"].isna().all():
            firstPath = validMatchDf["MaskPath"].dropna().iloc[0]
            maskFolder = str(Path(firstPath).parent)

        return {
            "pushed_at": datetime.now().isoformat(timespec="seconds"),
            "n_masks": int(validMatchDf.shape[0]),
            "mask_match_table": validMatchDf.copy(),
            "cell_table": pushedCellTable.copy(),
            "mask_folder": maskFolder,
            "column_map": {
                "mask_name_col": maskNameCol,
                "x_col": xCol,
                "y_col": yCol,
                "cluster_col": clusterCol,
                "condition_col": conditionCol,
                "sample_col": sampleCol,
            },
        }

    ## <----------------> Advanced settings / memory helpers <-------------------> ##
    def _format_bytes(n_bytes):
        """
        Convert a byte count into a readable string.
        """
        if n_bytes is None:
            return "unknown"

        try:
            n_bytes = float(n_bytes)
        except Exception:
            return "unknown"

        units = ["B", "KB", "MB", "GB", "TB"]
        size = n_bytes

        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{size:.0f} {unit}"
                return f"{size:.2f} {unit}"
            size /= 1024

        return f"{size:.2f} TB"


    def _estimate_object_size(obj, seen=None):
        """
        Estimate memory used by common PINT objects.

        This is intentionally conservative and avoids expensive deep inspection
        where possible.

        Handles:
        - pandas DataFrame / Series
        - NumPy arrays
        - scipy sparse matrices
        - dictionaries/lists/tuples containing those objects
        - generic Python objects via sys.getsizeof fallback
        """
        if obj is None:
            return 0

        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        # pandas tables
        if isinstance(obj, pd.DataFrame):
            return int(obj.memory_usage(index=True, deep=True).sum())

        if isinstance(obj, pd.Series):
            return int(obj.memory_usage(index=True, deep=True))

        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)

        # Sparse matrices, relevant for future Xenium-style workflows
        if sparse.issparse(obj):
            total = 0
            for attr in ("data", "indices", "indptr"):
                arr = getattr(obj, attr, None)
                if arr is not None:
                    total += int(arr.nbytes)
            return total

        # Dictionaries, for example the loaded image library:
        # {image_name: numpy array}
        if isinstance(obj, dict):
            total = sys.getsizeof(obj)
            for key, value in obj.items():
                total += _estimate_object_size(key, seen)
                total += _estimate_object_size(value, seen)
            return int(total)

        # Lists/tuples/sets
        if isinstance(obj, (list, tuple, set)):
            total = sys.getsizeof(obj)
            for item in obj:
                total += _estimate_object_size(item, seen)
            return int(total)

        # Fallback for other Python objects
        try:
            return int(sys.getsizeof(obj))
        except Exception:
            return 0


    def _describe_object(obj):
        """
        Return a compact human-readable description of an object.
        """
        if obj is None:
            return "None"

        if isinstance(obj, pd.DataFrame):
            return f"DataFrame: {len(obj):,} rows × {len(obj.columns):,} columns"

        if isinstance(obj, pd.Series):
            return f"Series: {len(obj):,} values"

        if isinstance(obj, np.ndarray):
            return f"NumPy array: shape={obj.shape}, dtype={obj.dtype}"

        if sparse.issparse(obj):
            return f"Sparse matrix: shape={obj.shape}, nnz={obj.nnz:,}, dtype={obj.dtype}"

        if isinstance(obj, dict):
            return f"dict: {len(obj):,} entries"

        if isinstance(obj, (list, tuple, set)):
            return f"{type(obj).__name__}: {len(obj):,} entries"

        return type(obj).__name__


    def _memory_row(name, obj, note=""):
        """
        Build one row for the advanced memory table.
        """
        size_bytes = _estimate_object_size(obj)

        return {
            "Object": name,
            "Loaded": "Yes" if obj is not None and not (
                isinstance(obj, pd.DataFrame) and obj.empty
            ) and not (
                isinstance(obj, dict) and len(obj) == 0
            ) else "No",
            "Estimated size": _format_bytes(size_bytes),
            "Bytes": size_bytes,
            "Description": _describe_object(obj),
            "Note": note,
        }

    @output
    @render.table
    def advanced_object_memory_table():
        rows = []

        rows.append(
            _memory_row(
                "Loaded image library",
                images.get(),
                "Main dictionary of loaded IMC image arrays.",
            )
        )

        rows.append(
            _memory_row(
                "Channel names",
                channels.get(),
                "Dictionary linking images to channel names.",
            )
        )

        rows.append(
            _memory_row(
                "Parameter table",
                params_df.get(),
                "Current per-channel processing parameters shown in the sidebar.",
            )
        )

        rows.append(
            _memory_row(
                "Segmentation Mesmer mask",
                segmentation_mesmer_mask.get(),
                "Most recent Mesmer mask array loaded/generated in the segmentation tab.",
            )
        )

        rows.append(
            _memory_row(
                "Segmentation Mesmer result",
                segmentation_mesmer_result.get(),
                "Full Mesmer result object, if stored.",
            )
        )

        rows.append(
            _memory_row(
                "Segmentation batch results",
                segmentation_mesmer_batch_results.get(),
                "Table of batch Mesmer segmentation results.",
            )
        )

        rows.append(
            _memory_row(
                "Segmentation quantification dataframe",
                segmentation_cell_table.get(),
                "Cell-level quantification table generated from segmentation masks.",
            )
        )

        rows.append(
            _memory_row(
                "Segmentation statistics / mask table",
                segmentation_mask_table.get(),
                "Mask-level statistics table generated during quantification.",
            )
        )

        rows.append(
            _memory_row(
                "Mask input table",
                mask_input_df.get(),
                "Loaded cell table for mask visualization.",
            )
        )

        rows.append(
            _memory_row(
                "Mask files table",
                mask_files_df.get(),
                "Detected mask files.",
            )
        )

        rows.append(
            _memory_row(
                "Final mask match table",
                final_mask_match_df.get(),
                "Final mapping between cell tables and mask files.",
            )
        )

        rows.append(
            _memory_row(
                "Neighborhood input data",
                neighborhood_input_data.get(),
                "Dataset pushed into the neighborhood analysis workflow.",
            )
        )

        rows.append(
            _memory_row(
                "Neighborhood touching edges",
                neighborhood_touching_edges.get(),
                "Cell-cell touching edge table.",
            )
        )

        rows.append(
            _memory_row(
                "Neighborhood matched cells",
                neighborhood_matched_cells.get(),
                "Cells matched to touching/neighborhood data.",
            )
        )

        rows.append(
            _memory_row(
                "Neighborhood touching results",
                neighborhood_touching_results.get(),
                "Chance-corrected neighborhood interaction results.",
            )
        )

        rows.append(
            _memory_row(
                "Neighborhood sample matrix",
                neighborhood_sample_matrix.get(),
                "Sample-level interaction matrix.",
            )
        )

        rows.append(
            _memory_row(
                "Neighborhood PERMANOVA results",
                neighborhood_permanova_results.get(),
                "PERMANOVA output table.",
            )
        )


        rows.append(
            _memory_row(
                "Clustering dataset",
                clustering_data.get(),
                "Shared clustering dataset used by Data preparation, Clustering, and Cluster names.",
            )
        )

        out = pd.DataFrame(rows)

        # Sort largest objects to the top.
        out = out.sort_values("Bytes", ascending=False)

        # Keep raw bytes for sorting/debugging but put it near the end.
        out = out[
            [
                "Object",
                "Loaded",
                "Estimated size",
                "Description",
                "Note",
                "Bytes",
            ]
        ]

        return out

app = App(app_ui, server)