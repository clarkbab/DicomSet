import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Literal, Tuple, Union

from ..config import get_orientation
from ..dicom.series import DicomSeries
from ..nifti.series import NiftiSeries
from ..typing import AffineMatrix, AffineMatrix2D, AffineMatrix3D, BatchBox, BatchBox2D, BatchBox3D, BatchLabelImage, BatchLabelImage2D, BatchLabelImage3D, BatchPoints, BatchPoints2D, BatchPoints3D, BatchVoxelBox, Box, Box2D, Box3D, Image, Image2D, Image3D, LabelImage, LabelImage2D, LabelImage3D, Landmark, Landmark2D, Landmark3D, LandmarkID, Landmarks, Landmarks2D, Landmarks3D, Number, Orientation2D, Orientation3D, Pixel, PixelBox, Point, Point2D, Point3D, Points, Points2D, Points3D, RegionID, Size, View, Voxel, VoxelBox, Window
from . import logging
from .args import alias_kwargs, arg_default, arg_to_list, assert_2d, assert_3d
from .assertions import assert_orientation
from .conversion import to_numpy
from .geometry import affine_origin, affine_spacing, centre_of_mass, foreground_fov, foreground_fov_centre, to_image_coords
from .landmarks import landmarks_to_points
from .logging import logger
from .transforms import crop, hist_eq as hist_eq_fn

VIEWS = ['Sagittal', 'Coronal', 'Axial']

# CT windowing presets: (width, level).
WINDOW_PRESETS = {
    'bone': (1800, 400),
    'brain': (80, 40),
    'liver': (150, 30),
    'lung': (1500, -600),
    'mediastinum': (350, 50),
    'tissue': (400, 50),
}

def __get_origin_2d(
    orientation: Orientation2D,
    ) -> Tuple[Literal['lower', 'upper'], Literal['lower', 'upper']]:
    assert_orientation(orientation, 2)
    origin_x = 'lower' if orientation[0] == 'L' else 'upper'
    origin_y = 'lower' if orientation[1] == 'S' else 'upper'
    return (origin_x, origin_y)

def __get_view_aspect(
    view: View,
    affine: AffineMatrix3D | None,
    ) -> float | None:
    if affine is None:
        return None
    spacing = affine_spacing(affine)
    axes = [i for i in range(3) if i != view]
    aspect = float(np.abs(spacing[axes[1]] / spacing[axes[0]]))
    return aspect

def __get_view_origin(
    view: View,
    orientation: Orientation3D,
    ) -> Tuple[Literal['lower', 'upper'], Literal['lower', 'upper']]:
    assert_orientation(orientation, 3)
    if view == 0:
        origin_x = 'lower' if orientation[1] == 'P' else 'upper'
        origin_y = 'lower' if orientation[2] == 'S' else 'upper'
    elif view == 1:
        origin_x = 'lower' if orientation[0] == 'L' else 'upper'
        origin_y = 'lower' if orientation[2] == 'S' else 'upper'
    else:
        origin_x = 'lower' if orientation[0] == 'L' else 'upper'
        origin_y = 'upper' if orientation[1] == 'P' else 'lower'

    return (origin_x, origin_y)

def __get_view_slice(
    view: View,
    data: np.ndarray,
    idx: int,
    ) -> np.ndarray:
    slicing: list[int | slice] = [slice(None)] * 3
    slicing[view] = idx
    return data[tuple(slicing)]

def __get_view_xy(
    view: View,
    values: tuple | np.ndarray,
    ) -> tuple:
    axes = [i for i in range(3) if i != view]
    return values[axes[0]], values[axes[1]]

def plot_dataframe(
    data: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    ax: mpl.axes.Axes | None = None,
    axis_title: Union[str, List[str]] = None,
    axis_title_fontsize: Number | None = None,
    axis_title_fontweight: str = 'normal',
    axis_title_rotation: Union[str, List[str]] = 'horizontal',
    axis_title_x: Union[Number, List[Number]] = 0.5,
    axis_title_y: Union[Number, List[Number]] = 0.98,
    dpi: float = 300,
    exclude_x: str | List[str] | None = None,
    figsize: Tuple[Number, Number] = (16, 6),
    figsize_pixels: Tuple[int, int] | None = None,
    filt: Dict[str, Any] | None = {},
    fontsize: float = 6,
    fontsize_legend: float | None = None,
    fontsize_stats: float | None = None,
    fontsize_tick_label: float | None = None,
    fontsize_title: float | None = None,
    hue_connections_index: str | List[str] | None = None,
    hue_hatch: str | List[str] | None = None,
    hue_label: str | List[str] | None = None,
    hue_order: List[str] | None = None,
    hspace: float | None = None,
    include_x: str | List[str] | None = None,
    legend_bbox: Tuple[float, float] = (1.0, 1.0),
    # legend_borderaxespad: float = 1.0,
    legend_borderpad: float = 0.1,
    legend_loc: str = 'upper left',
    linecolour: str = 'black',
    line_width: float = 0.25,
    major_tick_freq: float | None = None,
    marker_size: float = 4,
    minor_tick_freq: float | None = None,
    n_cols: int | None = None,
    n_rows: int | None = None,
    outlier_legend_loc: str = 'upper left',
    palette: str = 'colorblind',
    savepath: str | None = None,
    save_pad: float = 0.03,
    share_y: bool = True,
    show_boxes: bool = True,
    show_hue_connections: bool = False,
    show_hue_connections_inliers: bool = False,
    show_legend: Union[bool, List[bool]] = True,
    show_n: bool = True,
    show_points: bool = True,
    show_stats: bool = False,
    show_x_tick_labels: bool = True,
    stats_alt: Literal['greater', 'less', 'two-sided'] = 'two-sided',
    stats_bar_alg_use_lowest_level: bool = True,
    stats_bar_alpha: float = 0.5,
    # Stats bars must sit at least this high above data points.
    stats_bar_grid_offset: float = 0.015,           # Proportion of window height.
    # This value is important! Without this number, the first grid line
    # will only have one bar, which means our bars will be stacked much higher. This allows
    # a little wiggle room.
    stats_bar_grid_offset_wiggle: float = 0.01,     # Proportion of window height.
    stats_bar_grid_spacing: float = 0.04,          # Proportion of window height.
    stats_bar_height: float = 0.007,                # Proportion of window height.
    stats_bar_show_direction: bool = False,
    stats_bar_text_offset: float = 0.008,            # Proportion of window height.
    stats_boot_df: pd.DataFrame | None = None,
    stats_boot_df_cols: List[str] | None = None,
    stats_exclude: List[str] = [],
    stats_exclude_left: List[str] = [],
    stats_exclude_right: List[str] = [],
    stats_paired: bool = True,
    stats_paired_by: str | List[str] | None = None,
    stats_sig: List[float] = [0.05, 0.01, 0.001],
    style: Literal['box', 'violin'] | None = 'box',
    tick_length: float = 1.0,
    title: str | None = None,
    x_label: str | None = None,
    x_label_fontweight: str = 'normal',
    x_label_pad: float = 0,
    x_label_x: float = 0.5,
    x_label_y: float = 0.02,
    x_lim: Tuple[float | None, float | None] | None = (None, None),
    x_order: List[str] | None = None,
    x_width: float = 0.8,
    x_tick_label: List[str] | None = None,
    x_label_fontsize: float | None = None,
    x_tick_label_pad: float = 0,
    x_tick_label_rot: float = 0,
    y_label: str | None = None,
    y_label_fontsize: float | None = None,
    y_label_fontweight: str = 'normal',
    y_label_pad: float = 0,
    y_label_x: float = 0,
    y_label_y: float = 0.5,
    y_lim: Tuple[float | None, float | None] | None = (None, None),
    y_tick_label_pad: float = 1.0,
    ) -> None:
    if data is None:
        raise ValueError("'data' is None.")
    elif len(data) == 0:
        raise ValueError("DataFrame is empty.")
    hue_hatches = arg_to_list(hue_hatch, str)
    hue_labels = arg_to_list(hue_label, str)
    include_xs = arg_to_list(include_x, str)
    exclude_xs = arg_to_list(exclude_x, str)
    if show_hue_connections and hue_connections_index is None:
        raise ValueError(f"Please set 'hue_connections_index' to allow matching points between hues.")
    if show_stats and stats_paired_by is None:
        raise ValueError(f"Please set 'stats_paired_by' to determine sample pairing for Wilcoxon test.")
    x_tick_labels = arg_to_list(x_tick_label, str)

    # Set default fontsizes.
    axis_title_fontsize = axis_title_fontsize if axis_title_fontsize is not None else fontsize
    x_label_fontsize = x_label_fontsize if x_label_fontsize is not None else fontsize
    y_label_fontsize = y_label_fontsize if y_label_fontsize is not None else fontsize
    if fontsize_legend is None:
        fontsize_legend = fontsize
    if fontsize_stats is None:
        fontsize_stats = fontsize
    if fontsize_tick_label is None:
        fontsize_tick_label = fontsize
    if fontsize_title is None:
        fontsize_title = fontsize

    # Filter data.
    for k, v in filt.items():
        data = data[data[k] == v]
    if len(data) == 0:
        raise ValueError(f"DataFrame is empty after applying filters: {filt}.")
        
    # Include/exclude.
    if include_xs is not None:
        data = data[data[x].isin(include_xs)]
    if exclude_xs is not None:
        data = data[~data[x].isin(exclude_xs)]

    # Add outlier data.
    data = __add_outlier_info(data, x, y, hue)

    # Calculate global min/max values for when sharing y-limits across
    # rows (share_y=True).
    global_min_y = data[y].min()
    global_max_y = data[y].max()

    # Get x values.
    if x_order is None:
        x_order = list(sorted(data[x].unique()))

    # Determine x labels.
    groupby = x if hue is None else [x, hue]
    count_map = data.groupby(groupby)[y].count()
    if x_tick_labels is None:
        x_tick_labels = []
        for x_val in x_order:
            count = count_map.loc[x_val]
            if hue is not None:
                ns = count.values
                # Use a single number, e.g. (n=99) if all hues have the same number of points.
                if len(np.unique(ns)) == 1:
                    ns = ns[:1]
            else:
                ns = [count]
            label = f"{x_val}\n(n={','.join([str(n) for n in ns])})" if show_n else x_val
            x_tick_labels.append(label)

    # Create subplots if required.
    figsize = figsize_to_inches(figsize)
    if figsize_pixels is not None:
        figsize = tuple(np.array(figsize_pixels) / dpi)
    if n_cols is None:
        n_cols = len(x_order)
    if n_rows is None:
        n_rows = int(np.ceil(len(x_order) / n_cols))
    if ax is not None:
        # Figure created externally - figsize handled.
        assert n_rows == 1
        axs = [ax]
        show_figure = False
    elif n_rows > 1:
        # Expand figsize height to match number of rows.
        if figsize_pixels is None:
            figsize = (figsize[0], n_rows * figsize[1])

        # Use gridspec, this is required so we can create a final row that is smaller
        # than the other rows to preserve scale across rows. Sharing the x-axes doesn't work
        # here because the last one has smaller x-lim and will affect all rows - unless we use
        # inset axes? But this might be the reason our plotting was breaking.
        fig = plt.figure(dpi=dpi, figsize=figsize)
        hr = [1] * n_rows
        n_cols_final_row = len(x_order) % n_cols if len(x_order) % n_cols != 0 else n_cols
        width_ratios = [n_cols_final_row, n_cols - n_cols_final_row]
        gs = mpl.gridspec.GridSpec(n_rows, height_ratios=hr, hspace=hspace, ncols=2, width_ratios=width_ratios)
        axs = []
        for r in range(n_rows):
            for c in range(2):  # since you used ncols=2
                ax = fig.add_subplot(gs[r, c])
                axs.append(ax)
        show_figure = True
    else:
        fig = plt.figure(dpi=dpi, figsize=figsize)
        axs = [plt.gca()]
        show_figure = True

    # Get hue order/colour/labels.
    if hue is not None:
        if hue_order is None:
            hue_order = list(sorted(data[hue].unique()))

        # Calculate x width for each hue.
        hue_width = x_width / len(hue_order) 

        # Create map from hue to colour.
        hue_palette = sns.color_palette(palette, n_colors=len(hue_order))
        hue_colours = dict((h, hue_palette[i]) for i, h in enumerate(hue_order))

        if hue_labels is not None:
            if len(hue_labels) != len(hue_order):
                raise ValueError(f"Length of 'hue_labels' ({hue_labels}) should match hues ({hue_order}).")
    
    # Expand args to match number of rows.
    axis_titles = arg_to_list(axis_title, (None, str), broadcast=n_rows)
    axis_title_rotations = arg_to_list(axis_title_rotation, str, broadcast=n_rows)
    axis_title_xs = arg_to_list(axis_title_x, (int, float), broadcast=n_rows)
    axis_title_ys = arg_to_list(axis_title_y, (int, float), broadcast=n_rows)
    if isinstance(show_legend, bool):
        show_legends = [show_legend] * n_rows
    else: 
        if len(show_legend) != n_rows:
            raise ValueError(f"Length of 'show_legend' ({len(show_legend)}) should match number of rows ({n_rows}).")
        else:
            show_legends = show_legend

    # Plot rows.
    for i, show_legend in zip(range(n_rows), show_legends):
        # Split data.
        row_x_order = x_order[i * n_cols:(i + 1) * n_cols]
        row_x_tick_labels = x_tick_labels[i * n_cols:(i + 1) * n_cols]

        # Get x colours.
        if hue is None:
            row_palette = sns.color_palette(palette, n_colors=len(row_x_order))
            x_colours = dict((x, row_palette[i]) for i, x in enumerate(row_x_order))

        # Get row data.
        row_df = data[data[x].isin(row_x_order)].copy()

        # Get x-axis limits.
        x_lim_row = list(x_lim)
        n_cols_row = len(row_x_order)
        if x_lim_row[0] is None:
            x_lim_row[0] = -0.5
        if x_lim_row[1] is None:
            x_lim_row[1] = n_cols_row - 0.5

        # Get y-axis limits.
        y_margin = 0.05
        row_min_y = row_df[y].min()
        row_max_y = row_df[y].max()
        min_y = global_min_y if share_y else row_min_y
        max_y = global_max_y if share_y else row_max_y
        y_lim_row = list(y_lim)
        if y_lim_row[0] is None:
            if y_lim_row[1] is None:
                width = max_y - min_y
                y_lim_row[0] = min_y - y_margin * width
                y_lim_row[1] = max_y + y_margin * width
            else:
                width = y_lim_row[1] - min_y
                y_lim_row[0] = min_y - y_margin * width
        else:
            if y_lim_row[1] is None:
                width = max_y - y_lim_row[0]
                y_lim_row[1] = max_y + y_margin * width

        # Get axis object.
        if n_rows > 1:
            # Create axis on gridspec.
            if i != n_rows - 1:
                ax = fig.add_subplot(gs[i, :])
            else:
                ax = fig.add_subplot(gs[i, 0])
        else:
            ax = axs[i]

        # Set axis limits.
        # This has to be done twice - once to set parent axes, and once to set child (inset) axes.
        # I can't remember why we made inset axes...?
        # Make this a function, as we need to call when stats bars exceed the y-axis limit.
        inset_ax = __set_axes_limits(ax, x_lim_row, y_lim_row)

        # Keep track of legend items.
        hue_artists = {}

        for j, x_val in enumerate(row_x_order):
            # Add x positions.
            if hue is not None:
                for k, hue_name in enumerate(hue_order):
                    x_pos = j - 0.5 * x_width + (k + 0.5) * hue_width
                    row_df.loc[(row_df[x] == x_val) & (row_df[hue] == hue_name), 'x_pos'] = x_pos
            else:
                x_pos = j
                row_df.loc[row_df[x] == x_val, 'x_pos'] = x_pos
                
            # Plot boxes.
            if show_boxes:
                if hue is not None:
                    for k, hue_name in enumerate(hue_order):
                        # Get hue data and pos.
                        hue_df = row_df[(row_df[x] == x_val) & (row_df[hue] == hue_name)]
                        if len(hue_df) == 0:
                            continue
                        hue_pos = hue_df.iloc[0]['x_pos']

                        # Get hue 'label' - allows us to use names more display-friendly than the data values.
                        hue_label = hue_name if hue_labels is None else hue_labels[k]

                        hatch = hue_hatches[k] if hue_hatches is not None else None
                        if style == 'box':
                            # Plot box.
                            res = inset_ax.boxplot(hue_df[y].dropna(), boxprops=dict(color=linecolour, facecolor=hue_colours[hue_name], linewidth=line_width), capprops=dict(color=linecolour, linewidth=line_width), flierprops=dict(color=linecolour, linewidth=line_width, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=line_width), patch_artist=True, positions=[hue_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=line_width), widths=hue_width)
                            if hatch is not None:
                                mpl.rcParams['hatch.linewidth'] = line_width
                                res['boxes'][0].set_hatch(hatch)
                                # res['boxes'][0].set(hatch=hatch)
                                # res['boxes'][0].set_edgecolor('white')
                                # res['boxes'][0].set(facecolor='white')

                            # Save reference to plot for legend.
                            if not hue_label in hue_artists:
                                hue_artists[hue_label] = res['boxes'][0]
                        elif style == 'violin':
                            # Plot violin.
                            res = inset_ax.violinplot(hue_df[y], positions=[hue_pos], widths=hue_width)

                            # Save reference to plot for legend.
                            if not hue_label in hue_artists:
                                hue_artists[hue_label] = res['boxes'][0]
                else:
                    # Plot box.
                    x_df = row_df[row_df[x] == x_val]
                    if len(x_df) == 0:
                        continue
                    x_pos = x_df.iloc[0]['x_pos']
                    if style == 'box':
                        inset_ax.boxplot(x_df[y], boxprops=dict(color=linecolour, facecolor=x_colours[x_val], linewidth=line_width), capprops=dict(color=linecolour, linewidth=line_width), flierprops=dict(color=linecolour, linewidth=line_width, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=line_width), patch_artist=True, positions=[x_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=line_width))
                    elif style == 'violin':
                        inset_ax.violinplot(x_df[y], positions=[x_pos])

            # Plot points.
            if show_points:
                if hue is not None:
                    for j, hue_name in enumerate(hue_order):
                        hue_df = row_df[(row_df[x] == x_val) & (row_df[hue] == hue_name)]
                        if len(hue_df) == 0:
                            continue
                        res = inset_ax.scatter(hue_df['x_pos'], hue_df[y], color=hue_colours[hue_name], edgecolors=linecolour, linewidth=line_width, s=marker_size, zorder=100)
                        if not hue_label in hue_artists:
                            hue_artists[hue_label] = res
                else:
                    x_df = row_df[row_df[x] == x_val]
                    inset_ax.scatter(x_df['x_pos'], x_df[y], color=x_colours[x_val], edgecolors=linecolour, linewidth=line_width, s=marker_size, zorder=100)

            # Identify connections between hues.
            if hue is not None and show_hue_connections:
                # Get column/value pairs to group across hue levels.
                # line_ids = row_df[(row_df[x] == x_val) & row_df['outlier']][outlier_cols]
                x_df = row_df[(row_df[x] == x_val)]
                if not show_hue_connections_inliers:
                    line_ids = x_df[x_df['outlier']][hue_connections_index]
                else:
                    line_ids = x_df[hue_connections_index]

                # Drop duplicates.
                line_ids = line_ids.drop_duplicates()

                # Get palette.
                line_palette = sns.color_palette('husl', n_colors=len(line_ids))

                # Plot lines.
                artists = []
                labels = []
                for j, (_, line_id) in enumerate(line_ids.iterrows()):
                    # Get line data.
                    x_df = row_df[(row_df[x] == x_val)]
                    for k, v in zip(line_ids.columns, line_id):
                        x_df = x_df[x_df[k] == v]
                    x_df = x_df.sort_values('x_pos')
                    x_pos = x_df['x_pos'].tolist()
                    y_data = x_df[y].tolist()

                    # Plot line.
                    lines = inset_ax.plot(x_pos, y_data, color=line_palette[j])

                    # Save line/label for legend.
                    artists.append(lines[0])
                    label = ':'.join(line_id.tolist())
                    labels.append(label)

                # Annotate outlier legend.
                if show_legend:
                    # Save main legend.
                    main_legend = inset_ax.get_legend()

                    # Show outlier legend.
                    inset_ax.legend(artists, labels, bbox_to_anchor=legend_bbox, borderaxespad=legend_borderaxespad, borderpad=legend_borderpad, fontsize=fontsize_legend, loc=legend_loc)

                    # Re-add main legend.
                    inset_ax.add_artist(main_legend)

        # Show legend.
        if hue is not None:
            if show_legend:
                # Filter 'hue_labels' based on hue 'artists'. Some hues may not be present in this row,
                # and 'hue_labels' is a global (across all rows) tracker.
                hue_labels = hue_order if hue_labels is None else hue_labels
                labels, artists = list(zip(*[(h, hue_artists[h]) for h in hue_labels if h in hue_artists]))

                # Show legend.
                # legend = inset_ax.legend(artists, labels, borderaxespad=legend_borderaxespad, borderpad=legend_borderpad, bbox_to_anchor=legend_bbox, fontsize=fontsize_legend, loc=legend_loc)
                # Calculate anchor point for 'upper left' of legend.
                legend = inset_ax.legend(artists, labels, bbox_to_anchor=legend_bbox, borderpad=legend_borderpad, fontsize=fontsize_legend, loc=legend_loc)
                frame = legend.get_frame()
                frame.set_boxstyle('square')
                frame.set_edgecolor('black')
                frame.set_linewidth(line_width)

        # Get pairs for stats tests.
        if show_stats:
            if hue is None:
                # Create pairs of 'x' values.
                if n_rows != 1:
                    raise ValueError(f"Can't show stats between 'x' values with multiple rows - not handled.")

                pairs = []
                max_skips = len(x_order) - 1
                for skip in range(1, max_skips + 1):
                    for j, x_val in enumerate(x_order):
                        other_x_idx = j + skip
                        if other_x_idx < len(x_order):
                            pair = (x_val, x_order[other_x_idx])
                            pairs.append(pair)
            else:
                # Create pairs of 'hue' values.
                pairs = []
                for x_val in row_x_order:
                    # Create pairs - start at lower numbers of skips as this will result in a 
                    # condensed plot.
                    hue_pairs = []
                    max_skips = len(hue_order) - 1
                    for skip in range(1, max_skips + 1):
                        for j, hue_val in enumerate(hue_order):
                            other_hue_index = j + skip
                            if other_hue_index < len(hue_order):
                                pair = (hue_val, hue_order[other_hue_index])
                                hue_pairs.append(pair)
                    pairs.append(hue_pairs)

        # Get p-values for each pair.
        if show_stats:
            nonsig_pairs = []
            nonsig_p_vals = []
            sig_pairs = []
            sig_p_vals = []

            if hue is None:
                for x_l, x_r in pairs:
                    row_pivot_df = row_df.pivot(columns=x, index=stats_paired_by, values=y).reset_index()
                    if x_l in row_pivot_df.columns and x_r in row_pivot_df.columns:
                        vals_l = row_pivot_df[x_l]
                        vals_r = row_pivot_df[x_r]

                        p_val = __calculate_p_val(vals_l, vals_r, stats_alt, stats_paired, stats_sig)

                        if p_val < stats_sig[0]:
                            sig_pairs.append((x_l, x_r))
                            sig_p_vals.append(p_val)
                        else:
                            nonsig_pairs.append((x_l, x_r))
                            nonsig_p_vals.append(p_val)
            else:
                for x_val, hue_pairs in zip(row_x_order, pairs):
                    x_df = row_df[row_df[x] == x_val]

                    hue_nonsig_pairs = []
                    hue_nonsig_p_vals = []
                    hue_sig_pairs = []
                    hue_sig_p_vals = []
                    for hue_l, hue_r in hue_pairs:
                        if stats_boot_df is not None:
                            # Load p-values from 'stats_boot_df'.
                            x_pivot_df = x_df.pivot(columns=[hue], index=stats_paired_by, values=[y]).reset_index()
                            if (y, hue_l) in x_pivot_df.columns and (y, hue_r) in x_pivot_df.columns:
                                vals_l = x_pivot_df[y][hue_l]
                                vals_r = x_pivot_df[y][hue_r]

                                # Don't add stats pair if main data is empty.
                                if len(vals_l) == 0 or len(vals_r) == 0:
                                    continue
                            else:
                                # Don't add stats pair if main data is empty.
                                continue
                            
                            # Get ('*', '<direction>') from dataframe. We have 'x_val' which is our region.
                            x_boot_df = stats_boot_df[stats_boot_df[x] == x_val]
                            boot_hue_l, boot_hue_r, boot_p_val = stats_boot_df_cols
                            x_pair_boot_df = x_boot_df[(x_boot_df[boot_hue_l] == hue_l) & (x_boot_df[boot_hue_r] == hue_r)]
                            if len(x_pair_boot_df) == 0:
                                raise ValueError(f"No matches found in 'stats_boot_df' for '{x}' ('{x_val}') '{boot_hue_l}' ('{hue_l}') and '{boot_hue_r}' ('{hue_r}').")
                            if len(x_pair_boot_df) > 1:
                                raise ValueError(f"Found multiple matches in 'stats_boot_df' for '{x}' ('{x_val}') '{boot_hue_l}' ('{hue_l}') and '{boot_hue_r}' ('{hue_r}').")
                            p_val = x_pair_boot_df.iloc[0][boot_p_val]

                            if p_val != '':
                                hue_sig_pairs.append((hue_l, hue_r))
                                hue_sig_p_vals.append(p_val)
                            else:
                                hue_nonsig_pairs.append((hue_l, hue_r))
                                hue_nonsig_p_vals.append(p_val)
                    else:
                        # Calculate p-values using stats tests.
                        for hue_l, hue_r in hue_pairs:
                            x_pivot_df = x_df.pivot(columns=[hue], index=stats_paired_by, values=[y]).reset_index()
                            if (y, hue_l) in x_pivot_df.columns and (y, hue_r) in x_pivot_df.columns:
                                vals_l = x_pivot_df[y][hue_l]
                                vals_r = x_pivot_df[y][hue_r]

                                p_val = __calculate_p_val(vals_l, vals_r, stats_alt, stats_paired, stats_sig)

                                if p_val < stats_sig[0]:
                                    hue_sig_pairs.append((hue_l, hue_r))
                                    hue_sig_p_vals.append(p_val)
                                else:
                                    hue_nonsig_pairs.append((hue_l, hue_r))
                                    hue_nonsig_p_vals.append(p_val)
                
                    nonsig_pairs.append(hue_nonsig_pairs)
                    nonsig_p_vals.append(hue_nonsig_p_vals)
                    sig_pairs.append(hue_sig_pairs)
                    sig_p_vals.append(hue_sig_p_vals)

        # Format p-values.
        if show_stats:
            if hue is None:
                sig_p_vals = __format_p_vals(sig_p_vals, stats_sig)
            else:
                sig_p_vals = [__format_p_vals(p, stats_sig) for p in sig_p_vals]

        # Remove 'excluded' pairs.
        if show_stats:
            filt_pairs = []
            filt_p_vals = []
            if hue is None:
                for (x_l, x_r), p_val in zip(sig_pairs, sig_p_vals):
                    if (x_l in stats_exclude or x_r in stats_exclude) or (x_l in stats_exclude_left) or (x_r in stats_exclude_right):
                        continue
                    filt_pairs.append((x_l, x_r))
                    filt_p_vals.append(p_val)
            else:
                for ps, p_vals in zip(sig_pairs, sig_p_vals):
                    hue_filt_pairs = []
                    hue_filt_p_vals = []
                    for (hue_l, hue_r), p_val in zip(ps, p_vals):
                        if (hue_l in stats_exclude or hue_r in stats_exclude) or (hue_l in stats_exclude_left) or (hue_r in stats_exclude_right):
                            continue
                        hue_filt_pairs.append((hue_l, hue_r))
                        hue_filt_p_vals.append(p_val)
                    filt_pairs.append(hue_filt_pairs)
                    filt_p_vals.append(hue_filt_p_vals)

        # Display stats bars.
        # To display stats bars, we fit a vertical grid over the plot and place stats bars
        # on the grid lines - so they look nice.
        if show_stats:
            # Calculate heights based on window height.
            y_height = max_y - min_y
            stats_bar_height = y_height * stats_bar_height
            stats_bar_grid_spacing = y_height * stats_bar_grid_spacing
            stats_bar_grid_offset = y_height * stats_bar_grid_offset
            stats_bar_grid_offset_wiggle = y_height * stats_bar_grid_offset_wiggle
            stats_bar_text_offset = y_height * stats_bar_text_offset
                
            if hue is None:
                # Calculate 'y_grid_offset' - the bottom of the grid.
                # For each pair, we calculate the max value of the data, as the bar should
                # lie above this. Then we find the smallest of these max values across all pairs.
                # 'stats_bar_grid_offset' is added to give spacing between the data and the bar.
                y_grid_offset = np.inf
                min_skip = None
                for x_l, x_r in filt_pairs:
                    if stats_bar_alg_use_lowest_level:
                        skip = x_order.index(x_r) - x_order.index(x_l) - 1
                        if min_skip is None:
                            min_skip = skip
                        elif skip > min_skip:
                            continue

                    x_l_df = row_df[row_df[x] == x_l]
                    x_r_df = row_df[row_df[x] == x_r]
                    y_max = max(x_l_df[y].max(), x_r_df[y].max())
                    y_max = y_max + stats_bar_grid_offset
                    if y_max < y_grid_offset:
                        y_grid_offset = y_max

                # Add data offset.
                y_grid_offset = y_grid_offset + stats_bar_grid_offset_wiggle

                # Annotate figure.
                # We keep track of bars we've plotted using 'y_idxs'.
                # This is a mapping from the hue to the grid positions that have already
                # been used for either a left or right hand side of a bar.
                y_idxs: Dict[str, List[int]] = {}
                for (x_l, x_r), p_val in zip(filt_pairs, filt_p_vals):
                    # Get plot 'x_pos' for each x value.
                    x_l_df = row_df[row_df[x] == x_l]
                    x_r_df = row_df[row_df[x] == x_r]
                    x_left = x_l_df['x_pos'].iloc[0]
                    x_right = x_r_df['x_pos'].iloc[0]

                    # Get 'y_idx_min' (e.g. 0, 1, 2,...) which tells us the lowest grid line
                    # we can use based on our data points.
                    # We calculate this by finding the max data value for the pair, and also
                    # the max values for any hues between the pair values - as our bar should
                    # not collide with these 'middle' hues. 
                    y_data_maxes = [x_end_df[y].max() for x_end_df in [x_l_df, x_r_df]]
                    n_mid_xs = x_order.index(x_r) - x_order.index(x_l) - 1
                    for j in range(n_mid_xs):
                        x_mid = x_order[x_order.index(x_l) + j + 1]
                        x_mid_df = row_df[row_df[x] == x_mid]
                        y_data_max = x_mid_df[y].max()
                        y_data_maxes.append(y_data_max)
                    y_data_max = max(y_data_maxes) + stats_bar_grid_offset
                    y_idx_min = int(np.ceil((y_data_max - y_grid_offset) / stats_bar_grid_spacing))

                    # We don't want our new stats bar to collide with any existing bars.
                    # Get the y positions for all stats bar that have already been plotted
                    # and that have their left or right end at one of the 'middle' hues for
                    # our current pair.
                    n_mid_xs = x_order.index(x_r) - x_order.index(x_l) - 1
                    y_idxs_mid_xs = []
                    for j in range(n_mid_xs):
                        x_mid = x_order[x_order.index(x_l) + j + 1]
                        if x_mid in y_idxs:
                            y_idxs_mid_xs += y_idxs[x_mid]

                    # Get the next free position that doesn't collide with any existing bars.
                    y_idx_max = 100
                    for y_idx in range(y_idx_min, y_idx_max):
                        if y_idx not in y_idxs_mid_xs:
                            break

                    # Plot bar.
                    y_min = y_grid_offset + y_idx * stats_bar_grid_spacing
                    y_max = y_min + stats_bar_height
                    inset_ax.plot([x_left, x_left, x_right, x_right], [y_min, y_max, y_max, y_min], alpha=stats_bar_alpha, color=linecolour, linewidth=line_width)    

                    # Adjust y-axis limits if bar would be plotted outside of window.
                    # Unless y_lim is set manually.
                    y_lim_top = y_max + 1.5 * stats_bar_grid_spacing
                    if y_lim[1] is None and y_lim_top > y_lim_row[1]:
                        y_lim_row[1] = y_lim_top
                        inset_ax = __set_axes_limits(axs[i], x_lim_row, y_lim_row, inset_ax=inset_ax)

                    # Plot p-value.
                    x_text = (x_left + x_right) / 2
                    y_text = y_max + stats_bar_text_offset
                    inset_ax.text(x_text, y_text, p_val, alpha=stats_bar_alpha, fontsize=fontsize_stats, horizontalalignment='center', verticalalignment='center')

                    # Save position of plotted stats bar.
                    if not x_l in y_idxs:
                        y_idxs[x_l] = [y_idx]
                    elif y_idx not in y_idxs[x_l]:
                        y_idxs[x_l] = list(sorted(y_idxs[x_l] + [y_idx]))
                    if not x_r in y_idxs:
                        y_idxs[x_r] = [y_idx]
                    elif y_idx not in y_idxs[x_r]:
                        y_idxs[x_r] = list(sorted(y_idxs[x_r] + [y_idx]))
            else:
                for hue_pairs, hue_p_vals in zip(filt_pairs, filt_p_vals):
                    # Calculate 'y_grid_offset' - the bottom of the grid.
                    # For each pair, we calculate the max value of the data, as the bar should
                    # lie above this. Then we find the smallest of these max values across all pairs.
                    # 'stats_bar_grid_offset' is added to give spacing between the data and the bar.
                    y_grid_offset = np.inf
                    min_skip = None
                    for hue_l, hue_r in hue_pairs:
                        if stats_bar_alg_use_lowest_level:
                            skip = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                            if min_skip is None:
                                min_skip = skip
                            elif skip > min_skip:
                                continue

                        hue_l_df = x_df[x_df[hue] == hue_l]
                        hue_r_df = x_df[x_df[hue] == hue_r]
                        y_max = max(hue_l_df[y].max(), hue_r_df[y].max())
                        y_max = y_max + stats_bar_grid_offset
                        if y_max < y_grid_offset:
                            y_grid_offset = y_max

                    # Add data offset.
                    y_grid_offset = y_grid_offset + stats_bar_grid_offset_wiggle

                    # Annotate figure.
                    # We keep track of bars we've plotted using 'y_idxs'.
                    # This is a mapping from the hue to the grid positions that have already
                    # been used for either a left or right hand side of a bar.
                    y_idxs: Dict[str, List[int]] = {}
                    for (hue_l, hue_r), p_val in zip(hue_pairs, hue_p_vals):
                        # Get plot 'x_pos' for each hue.
                        hue_l_df = x_df[x_df[hue] == hue_l]
                        hue_r_df = x_df[x_df[hue] == hue_r]
                        if len(hue_l_df) == 0 or len(hue_r_df) == 0:
                            continue
                        x_left = hue_l_df['x_pos'].iloc[0]
                        x_right = hue_r_df['x_pos'].iloc[0]

                        # Get 'y_idx_min' (e.g. 0, 1, 2,...) which tells us the lowest grid line
                        # we can use based on our data points.
                        # We calculate this by finding the max data value for the pair, and also
                        # the max values for any hues between the pair values - as our bar should
                        # not collide with these 'middle' hues. 
                        y_data_maxes = [hue_df[y].max() for hue_df in [hue_l_df, hue_r_df]]
                        n_mid_hues = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                        for j in range(n_mid_hues):
                            hue_mid = hue_order[hue_order.index(hue_l) + j + 1]
                            hue_mid_df = x_df[x_df[hue] == hue_mid]
                            y_data_max = hue_mid_df[y].max()
                            y_data_maxes.append(y_data_max)
                        y_data_max = max(y_data_maxes) + stats_bar_grid_offset
                        y_idx_min = int(np.ceil((y_data_max - y_grid_offset) / stats_bar_grid_spacing))

                        # We don't want our new stats bar to collide with any existing bars.
                        # Get the y positions for all stats bar that have already been plotted
                        # and that have their left or right end at one of the 'middle' hues for
                        # our current pair.
                        n_mid_hues = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                        y_idxs_mid_hues = []
                        for j in range(n_mid_hues):
                            hue_mid = hue_order[hue_order.index(hue_l) + j + 1]
                            if hue_mid in y_idxs:
                                y_idxs_mid_hues += y_idxs[hue_mid]

                        # Get the next free position that doesn't collide with any existing bars.
                        y_idx_max = 100
                        for y_idx in range(y_idx_min, y_idx_max):
                            if y_idx not in y_idxs_mid_hues:
                                break

                        # Plot bar.
                        y_min = y_grid_offset + y_idx * stats_bar_grid_spacing
                        y_max = y_min + stats_bar_height
                        inset_ax.plot([x_left, x_left, x_right, x_right], [y_min, y_max, y_max, y_min], alpha=stats_bar_alpha, color=linecolour, linewidth=line_width)    

                        # Adjust y-axis limits if bar would be plotted outside of window.
                        # Unless y_lim is set manually.
                        y_lim_top = y_max + 1.5 * stats_bar_grid_spacing
                        if y_lim[1] is None and y_lim_top > y_lim_row[1]:
                            y_lim_row[1] = y_lim_top
                            inset_ax = __set_axes_limits(axs[i], x_lim_row, y_lim_row, inset_ax=inset_ax)

                        # Plot p-value.
                        x_text = (x_left + x_right) / 2
                        y_text = y_max + stats_bar_text_offset
                        inset_ax.text(x_text, y_text, p_val, alpha=stats_bar_alpha, fontsize=fontsize_stats, horizontalalignment='center', verticalalignment='center')

                        # Save position of plotted stats bar.
                        if not hue_l in y_idxs:
                            y_idxs[hue_l] = [y_idx]
                        elif y_idx not in y_idxs[hue_l]:
                            y_idxs[hue_l] = list(sorted(y_idxs[hue_l] + [y_idx]))
                        if not hue_r in y_idxs:
                            y_idxs[hue_r] = [y_idx]
                        elif y_idx not in y_idxs[hue_r]:
                            y_idxs[hue_r] = list(sorted(y_idxs[hue_r] + [y_idx]))
          
        # Set axis title and labels.
        x_label = x_label if x_label is not None else ''
        y_label = y_label if y_label is not None else ''
        inset_ax.set_xlabel(x_label, fontsize=x_label_fontsize, fontweight=x_label_fontweight, labelpad=x_label_pad, x=x_label_x, y=x_label_y)
        inset_ax.set_ylabel(y_label, fontsize=y_label_fontsize, fontweight=y_label_fontweight, labelpad=y_label_pad, x=y_label_x, y=y_label_y)
        if axis_titles[i] is not None:
            inset_ax.set_title(axis_titles[i], fontsize=axis_title_fontsize, fontweight=axis_title_fontweight, rotation=axis_title_rotations[i], x=axis_title_xs[i], y=axis_title_ys[i])
                
        # Set axis tick labels.
        inset_ax.set_xticks(list(range(len(row_x_tick_labels))))
        if show_x_tick_labels:
            inset_ax.set_xticklabels(row_x_tick_labels, fontsize=fontsize_tick_label, rotation=x_tick_label_rot)
        else:
            inset_ax.set_xticklabels([])

        # Set tick label font size and padding.
        inset_ax.tick_params(axis='x', labelsize=fontsize_tick_label, pad=x_tick_label_pad, which='major')
        inset_ax.tick_params(axis='y', labelsize=fontsize_tick_label, pad=y_tick_label_pad, which='major')

        # Set y axis major ticks.
        if major_tick_freq is not None:
            major_tick_min = y_lim[0]
            if major_tick_min is None:
                major_tick_min = inset_ax.get_ylim()[0]
            major_tick_max = y_lim[1]
            if major_tick_max is None:
                major_tick_max = inset_ax.get_ylim()[1]
            
            # Round range to nearest multiple of 'major_tick_freq'.
            major_tick_min = np.ceil(major_tick_min / major_tick_freq) * major_tick_freq
            major_tick_max = np.floor(major_tick_max / major_tick_freq) * major_tick_freq
            n_major_ticks = int((major_tick_max - major_tick_min) / major_tick_freq) + 1
            major_ticks = np.linspace(major_tick_min, major_tick_max, n_major_ticks)
            major_tick_labels = [str(round(t, 3)) for t in major_ticks]     # Some weird str() conversion without rounding.
            inset_ax.set_yticks(major_ticks)
            inset_ax.set_yticklabels(major_tick_labels)

        # Set y axis minor ticks.
        if minor_tick_freq is not None:
            minor_tick_min = y_lim[0]
            if minor_tick_min is None:
                minor_tick_min = inset_ax.get_ylim()[0]
            minor_tick_max = y_lim[1]
            if minor_tick_max is None:
                minor_tick_max = inset_ax.get_ylim()[1]
            
            # Round range to nearest multiple of 'minor_tick_freq'.
            minor_tick_min = np.ceil(minor_tick_min / minor_tick_freq) * minor_tick_freq
            minor_tick_max = np.floor(minor_tick_max / minor_tick_freq) * minor_tick_freq
            n_minor_ticks = int((minor_tick_max - minor_tick_min) / minor_tick_freq) + 1
            minor_ticks = np.linspace(minor_tick_min, minor_tick_max, n_minor_ticks)
            inset_ax.set_yticks(minor_ticks, minor=True)

        # Set y grid lines.
        inset_ax.grid(alpha=0.1, axis='y', color='grey', linewidth=line_width)
        inset_ax.set_axisbelow(True)

        # Set axis spine/tick linewidths and tick lengths.
        spines = ['top', 'bottom','left','right']
        for spine in spines:
            inset_ax.spines[spine].set_linewidth(line_width)
        inset_ax.tick_params(length=tick_length, which='both', width=line_width)

    # Set title.
    okwargs = dict(
        fontsize=fontsize_title,
        style='italic',
    )
    fig.suptitle(title, **okwargs)

    # Save plot to disk.
    if savepath is not None:
        savepath = escape_filepath(savepath)
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        fig.savefig(savepath, bbox_inches='tight', pad_inches=save_pad)
        logging.info(f"Saved plot to '{savepath}'.")

    # Show plot.
    if show_figure:
        fig.show()

def plot_hist(
    data: Image,
    ax: mpl.axes.Axes | None = None,
    bins: int = 50,
    log_scale: bool = False,
    min: Number | None = None,
    max: Number | None = None,
    return_axis: bool = False,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> mpl.axes.Axes | None:
    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False
    okwargs = {}
    if min is not None and max is not None:
        okwargs['range'] = (min, max)
    ax.hist(data.flatten(), bins=bins, color='gray', **okwargs)
    if log_scale:
        ax.set_yscale('log')

    # Add text.
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if show:
        plt.show()

    if return_axis:
        return ax

@alias_kwargs(
    ('a', 'affine'),
    ('b', 'box'),
    ('c', 'crop'),
    ('cm', 'crop_margin'),
    ('l', 'labels'),
    ('ln', 'label_names'),
    ('o', 'orientation'),
    ('p', 'points'),
    ('sl', 'show_labels'),
    ('spn', 'show_point_names'),
    ('w', 'window'),
)
def plot_slice(
    data: Image2D | None,
    affine: AffineMatrix2D | None = None,
    alpha: Number = 0.3,
    aspect: Number | None = None,
    ax: mpl.axes.Axes | None = None,
    box: Box2D | BatchBox2D | RegionID | List[RegionID] | None = None,
    cmap: str = 'gray',
    crop: Box2D | Point2D | RegionID | None = None,
    crop_margin: Number = 100.0,
    figsize: Tuple[Number, Number] = (8, 8),
    hist_eq: bool = False,
    labels: LabelImage2D | BatchLabelImage2D | None = None,
    label_names: RegionID | List[RegionID] | None = None,
    orientation: Orientation2D | None = None,
    points: Point2D | Points2D | BatchPoints2D | List[Points2D] | Landmark2D | Landmarks2D | None = None,
    points_colour: str = 'yellow',
    point_names: LandmarkID | List[LandmarkID] | None = None,
    return_axis: bool = False,
    show_labels: bool = True,
    show_point_idxs: bool = False,
    show_point_names: bool = False,
    title: str | None = None,
    title_fontsize: float = 10,
    use_image_coords: bool = False,
    vmin: Number | None = None,
    vmax: Number | None = None,
    window: Window | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ) -> mpl.axes.Axes | None:
    if data is None:
        assert labels is not None, "Labels must be provided if data is None."
        data = np.zeros(labels.shape[-2:])
    data = to_numpy(data)
    data = data.squeeze()  # Remove any singleton dimensions.
    assert_2d(data)
    affine = to_numpy(affine)
    orientation = arg_default(orientation, orientation, 'LS')
    origin_x, origin_y = __get_origin_2d(orientation)

    # Resolve labels.
    labels = __resolve_labels(labels, dim=2)

    # Resolve window to vmin/vmax.
    vmin, vmax = __resolve_window(window, affine=affine, data=data, labels=labels, vmax=vmax, vmin=vmin)

    # Resolve points and point names.
    points, point_names = __resolve_points(points, affine=affine, point_names=point_names)

    # Resolve crop.
    crop_box = __resolve_crop(crop, crop_margin, data.shape, affine=affine, label_names=None, labels=labels, point_names=point_names, points=points)

    # Resolve boxes for overlay.
    boxes = __resolve_boxes(box, data.shape, affine=affine, label_names=label_names, labels=labels, point_names=point_names, points=points)

    if ax is None:
        _, axs = plt.subplots(1, 1, figsize=figsize)
        ax = axs
        show = True
    else:
        show = False

    # Get aspect ratio.
    if aspect is None and affine is not None:
        spacing = affine_spacing(affine)
        aspect = float(spacing[1] / spacing[0])

    # Plot the image.
    if crop_box is not None:
        data = data[crop_box[0, 0]:crop_box[1, 0] + 1, crop_box[0, 1]:crop_box[1, 1] + 1]
    if hist_eq:
        data = hist_eq_fn(data)
    ax.imshow(data.T, aspect=aspect, cmap=cmap, origin=origin_y, vmax=vmax, vmin=vmin)
    if origin_x == 'upper':
        ax.invert_xaxis()

    # Plot labels.
    if show_labels and labels is not None:
        label_palette = sns.color_palette('colorblind', len(labels))
        for i, l in enumerate(labels):
            if crop_box is not None:
                l = l[crop_box[0, 0]:crop_box[1, 0] + 1, crop_box[0, 1]:crop_box[1, 1] + 1]
            l_bin = (l > 0).astype(float)
            cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), label_palette[i]))
            ax.imshow(l_bin.T, alpha=alpha, cmap=cmap_label, origin=origin_y)
            ax.contour(l_bin.T, colors=[label_palette[i]], levels=[0.5], linestyles='solid')

    # Plot points.
    if points is not None:
        n_batches = len(points)
        batch_palette = sns.color_palette('bright', n_batches)
        for bi, (batch, batch_names) in enumerate(zip(points, point_names)):
            batch_colour = batch_palette[bi]
            if points_colour == 'gradient' and len(batch) > 1:
                r, g, b = batch_colour[:3]
                light = (r + (1 - r) * 0.7, g + (1 - g) * 0.7, b + (1 - b) * 0.7)
                points_cmap = mpl.colors.LinearSegmentedColormap.from_list('batch_grad', [light, batch_colour])
                p_colours = [points_cmap(i / (len(batch) - 1)) for i in range(len(batch))]
            elif points_colour == 'batch':
                p_colours = [batch_colour] * len(batch)
            else:
                p_colours = [points_colour] * len(batch)
            for pi, p in enumerate(batch):
                if crop_box is not None:
                    p = p - crop_box[0]
                ax.scatter(p[0], p[1], c=[p_colours[pi]], marker='o', s=20, zorder=5)
                if show_point_names and batch_names is not None:
                    ax.annotate(batch_names[pi], (p[0], p[1]),
                        color=p_colours[pi], fontsize=8,
                        textcoords='offset points', xytext=(5, 5), zorder=5)
                elif show_point_idxs:
                    ax.annotate(str(pi), (p[0], p[1]),
                        color=p_colours[pi], fontsize=8,
                        textcoords='offset points', xytext=(5, 5), zorder=5)

    # Box overlays.
    if boxes is not None:
        # Get 2D boxes.
        box_palette = sns.color_palette('colorblind', len(boxes))
        for bi, b in enumerate(boxes):
            # Draw rectangle from min/max corners.
            width = b[1][0] - b[0][0]
            height = b[1][1] - b[0][1]
            rect = mpl.patches.Rectangle((b[0][0], b[0][1]), width, height, edgecolor=box_palette[bi % len(box_palette)], facecolor='none', linestyle='dotted', linewidth=2, zorder=10)
            ax.add_patch(rect)

    # Get tick positions in image coords.
    size = data.shape
    x_tick_spacing = np.unique(np.diff(ax.get_xticks()))[0]
    x_ticks = np.arange(0, size[0], x_tick_spacing)
    y_tick_spacing = np.unique(np.diff(ax.get_yticks()))[0]
    y_ticks = np.arange(0, size[1], y_tick_spacing)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Create tick labels.
    # Apply crop. Only the labels get the offset applied,
    # positions in image coords don't change.
    if crop_box is not None:
        x_ticks += crop_box[0, 0]
        y_ticks += crop_box[0, 1]
    # Convert tick labels to world coords.
    if not use_image_coords and affine is not None:
        sx, sy = affine_spacing(affine)
        ox, oy = affine_origin(affine)
        x_ticks = (x_ticks * sx + ox)
        y_ticks = (y_ticks * sy + oy)
    ax.set_xticklabels([f'{t:.1f}' for t in x_ticks])
    ax.set_yticklabels([f'{t:.1f}' for t in y_ticks])

    # Hide axis spines.
    for p in ['right', 'top', 'bottom', 'left']:
        ax.spines[p].set_visible(False)

    # Add text.
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if show:
        plt.show()

    if return_axis:
        return axs[0]

@alias_kwargs(
    ('a', 'affine'),
    ('b', 'box'),
    ('c', 'crop'),
    ('ch', 'crosshairs'),
    ('cm', 'crop_margin'),
    ('d', 'dose'),
    ('i', 'idx'),
    ('l', 'labels'),
    ('ln', 'label_names'),
    ('p', 'points'),
    ('scc', 'show_crosshairs_coords'),
    ('sl', 'show_labels'),
    ('spn', 'show_point_names'),
)
def plot_volume(
    data: Image3D | DicomSeries | NiftiSeries | None,
    affine: AffineMatrix3D | None = None,
    ax: mpl.axes.Axes | List[mpl.axes.Axes] | None = None,
    box: Box3D | BatchBox3D | RegionID | List[RegionID] | None = None,
    cmap: str = 'gray',
    crop: Box3D | Point3D | RegionID | None = None,
    crop_margin: Number = 100.0,
    dose: Image3D | None = None,
    dose_alpha_min: Number = 0.3,
    dose_alpha_max: Number = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: Number = 0.15,
    figsize: Tuple[Number, Number] = (16, 6),
    hist_eq: bool = False,
    idx: int | float | str | Point3D | None = None,
    labels: LabelImage3D | BatchLabelImage3D | None = None,
    label_names: RegionID | List[RegionID] | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    orientation: Orientation3D | None = None,
    label_alpha: Number = 0.3,
    points: Point3D | Points3D | BatchPoints3D | List[Points3D] | Landmark3D | Landmarks3D | None = None,
    points_colour: str = 'yellow',
    point_names: LandmarkID | List[LandmarkID] | None = None,
    crosshairs: Point3D | str | None = None,
    crosshairs_colour: str = 'yellow',
    return_axis: bool = False,
    show_crosshairs_coords: bool = True,
    show_labels: bool = True,
    show_point_idxs: bool = False,
    show_points: bool = True,
    show_point_names: bool = False,
    show_title: bool = True,
    title_fontsize: float = 10,
    use_image_coords: bool = False,
    view: View | List[View] | Literal['all'] = 'all',
    vmin: Number | None = None,
    vmax: Number | None = None,
    window: Window | None = None,
    ) -> np.ndarray | None:
    if data is None:
        assert labels is not None, "Labels must be provided if data is None."
        data = np.zeros(labels.shape[-3:])
    elif isinstance(data, (DicomSeries, NiftiSeries)):
        # This saves passing affine - just pass the image series.
        series = data
        data = series.data
        affine = series.affine
    data = to_numpy(data)
    data = data.squeeze()  # Remove any singleton dimensions.
    assert_3d(data)
    affine = to_numpy(affine)
    orientation = arg_default(orientation, orientation, get_orientation(3))

    # Resolve labels.
    labels = __resolve_labels(labels, dim=3)

    # Change the affine and any data so that the world coordinates always increase
    # to the right of and top of the image.
    if affine is not None:
        affine = affine.copy()
        for i in range(3):
            if affine[i, i] < 0:
                n = data.shape[i]
                affine[i, -1] += (n - 1) * affine[i, i]
                affine[i, i] = -affine[i, i]
                data = np.flip(data, axis=i)
                if dose is not None:
                    dose = np.flip(dose, axis=i)
                if labels is not None:
                    labels = np.flip(labels, axis=i + 1)


    # Resolve window to vmin/vmax.
    vmin, vmax = __resolve_window(window, affine=affine, data=data, labels=labels, vmax=vmax, vmin=vmin)

    # Histogram equalisation (applied to full volume for consistent slices across views).
    if hist_eq:
        data = hist_eq_fn(data)

    # Resolve points and point names.
    points, point_names = __resolve_points(points, affine=affine, point_names=point_names)

    # Resolve views.
    views = list(range(3)) if view == 'all' else (view if isinstance(view, list) else [view])

    # Resolve idx to a 3D voxel point.
    idx_vox = __resolve_point(idx, data.shape, affine=affine, centre_method=centre_method, label_names=label_names, labels=labels, point_names=point_names, points=points)

    # Resolve crosshairs to image coords.
    crosshairs_vox = __resolve_crosshairs(crosshairs, data.shape, affine=affine, centre_method=centre_method, label_names=label_names, labels=labels, point_names=point_names, points=points)

    # Resolve crop to a voxel bounding box.
    crop_box = __resolve_crop(crop, crop_margin, data.shape, affine=affine, label_names=label_names, labels=labels, point_names=point_names, points=points)

    # Resolve boxes for overlay.
    boxes = __resolve_boxes(box, data.shape, affine=affine, label_names=label_names, labels=labels, point_names=point_names, points=points)

    if ax is not None:
        axs = arg_to_list(ax, mpl.axes.Axes)
        assert len(axs) == len(views), f"Expected {len(views)} axes but got {len(axs)}."
        show = False
    else:
        _, axs_grid = plt.subplots(1, len(views), figsize=figsize, squeeze=False)
        axs = list(axs_grid[0])
        show = True

    for col_ax, v in zip(axs, views):
        view_idx = idx_vox[v]

        # Compute the view crop box.
        x_axis, y_axis = __get_view_xy(v, list(range(3)))
        view_crop_box = crop_box[:, [x_axis, y_axis]] if crop_box is not None else None

        image = __get_view_slice(v, data, view_idx)
        if view_crop_box is not None:
            image = image[view_crop_box[0, 0]:view_crop_box[1, 0] + 1, view_crop_box[0, 1]:view_crop_box[1, 1] + 1]
        aspect = __get_view_aspect(v, affine)
        origin_x, origin_y = __get_view_origin(v, orientation)

        # The two non-view axes: first is displayed on x, second on y.
        col_ax.imshow(image.T, aspect=aspect, cmap=cmap, origin=origin_y, vmax=vmax, vmin=vmin)
        if origin_x == 'upper':
            col_ax.invert_xaxis()

        # Box overlays.
        if boxes is not None:
            # Get 2D boxes.
            x_axis, y_axis = __get_view_xy(v, list(range(3)))
            boxes2d = boxes[:, :, [x_axis, y_axis]]
            box_palette = sns.color_palette('colorblind', len(boxes2d))
            for bi, b in enumerate(boxes2d):
                # Draw rectangle from min/max corners.
                width = b[1][0] - b[0][0]
                height = b[1][1] - b[0][1]
                rect = mpl.patches.Rectangle((b[0][0], b[0][1]), width, height, edgecolor=box_palette[bi % len(box_palette)], facecolor='none', linestyle='dotted', linewidth=2, zorder=10)
                col_ax.add_patch(rect)

        # Dose overlay.
        if dose is not None:
            dose_slice = __get_view_slice(v, dose, view_idx)
            if view_crop_box is not None:
                dose_slice = dose_slice[view_crop_box[0, 0]:view_crop_box[1, 0] + 1, view_crop_box[0, 1]:view_crop_box[1, 1] + 1]
            base_cmap = plt.get_cmap(dose_cmap)
            trunc_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                f'{base_cmap.name}_truncated',
                base_cmap(np.linspace(dose_cmap_trunc, 1.0, 256)),
            )
            colours = trunc_cmap(np.arange(trunc_cmap.N))
            colours[0, -1] = 0
            colours[1:, -1] = np.linspace(dose_alpha_min, dose_alpha_max, trunc_cmap.N - 1)
            alpha_cmap = mpl.colors.ListedColormap(colours)
            col_ax.imshow(dose_slice.T, aspect=aspect, cmap=alpha_cmap, origin=origin_y)

        # Label overlays.
        if show_labels and labels is not None:
            label_palette = sns.color_palette('colorblind', len(labels))
            label_names_list = arg_to_list(label_names, str) if label_names is not None else None
            for j, lab in enumerate(labels):
                label_slice = __get_view_slice(v, lab, view_idx)
                if view_crop_box is not None:
                    label_slice = label_slice[view_crop_box[0, 0]:view_crop_box[1, 0] + 1, view_crop_box[0, 1]:view_crop_box[1, 1] + 1]
                label_bin = (label_slice > 0).astype(float)
                cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), label_palette[j]))
                col_ax.imshow(label_bin.T, alpha=label_alpha, aspect=aspect, cmap=cmap_label, origin=origin_y)
                col_ax.contour(label_bin.T, colors=[label_palette[j]], levels=[0.5], linestyles='solid')

            # Add legend on first view only.
            if label_names_list is not None and v == views[0]:
                handles = [mpl.patches.Patch(facecolor=label_palette[j], label=label_names_list[j]) for j in range(len(labels)) if j < len(label_names_list)]
                col_ax.legend(fontsize='small', framealpha=0.7, handles=handles, loc='upper right')

        # Point overlays.
        if show_points and points is not None:
            n_batches = len(points)
            batch_palette = sns.color_palette('colorblind', n_batches)
            for bi, (batch, batch_names) in enumerate(zip(points, point_names)):
                batch_colour = batch_palette[bi]
                if points_colour == 'gradient' and len(batch) > 1:
                    r, g, b = batch_colour[:3]
                    light = (r + (1 - r) * 0.7, g + (1 - g) * 0.7, b + (1 - b) * 0.7)
                    points_cmap = mpl.colors.LinearSegmentedColormap.from_list('batch_grad', [light, batch_colour])
                    p_colours = [points_cmap(i / (len(batch) - 1)) for i in range(len(batch))]
                elif points_colour == 'batch':
                    p_colours = [batch_colour] * len(batch)
                else:
                    p_colours = [points_colour] * len(batch)
                for pi, p in enumerate(batch):
                    p_x, p_y = __get_view_xy(v, p)
                    if view_crop_box is not None:
                        p_x -= view_crop_box[0, 0]
                        p_y -= view_crop_box[0, 1]
                    col_ax.scatter(p_x, p_y, c=[p_colours[pi]], marker='o', s=20, zorder=5)
                    if show_point_names and batch_names is not None:
                        col_ax.annotate(batch_names[pi], (p_x, p_y),
                            color=p_colours[pi], fontsize=8,
                            textcoords='offset points', xytext=(5, 5), zorder=5)
                    elif show_point_idxs:
                        col_ax.annotate(str(pi), (p_x, p_y),
                            color=p_colours[pi], fontsize=8,
                            textcoords='offset points', xytext=(5, 5), zorder=5)

        # Crosshairs.
        if crosshairs_vox is not None:
            ch_x, ch_y = __get_view_xy(v, crosshairs_vox)
            if view_crop_box is not None:
                ch_x -= view_crop_box[0, 0]
                ch_y -= view_crop_box[0, 1]
            col_ax.axvline(color=crosshairs_colour, linestyle='dashed', linewidth=0.5, x=ch_x)
            col_ax.axhline(color=crosshairs_colour, linestyle='dashed', linewidth=0.5, y=ch_y)

            if show_crosshairs_coords:
                # Convert point back to world coords for label if necesssary.
                if not use_image_coords and affine is not None:
                    s = affine_spacing(affine)
                    o = affine_origin(affine)
                    ch_x_world, ch_y_world = __get_view_xy(v, crosshairs_vox * s + o)
                    label = f'({ch_x_world:.1f}, {ch_y_world:.1f})'
                else:
                    label = f'({ch_x}, {ch_y})'
                col_ax.text(ch_x + 10, ch_y - 10, label, color=crosshairs_colour, fontsize=8)

        # Get tick positions in image coords.
        size_x, size_y = image.shape
        x_tick_spacing = np.unique(np.diff(col_ax.get_xticks()))[0]
        x_ticks = np.arange(0, size_x, x_tick_spacing)
        y_tick_spacing = np.unique(np.diff(col_ax.get_yticks()))[0]
        y_ticks = np.arange(0, size_y, y_tick_spacing)
        col_ax.set_xticks(x_ticks)
        col_ax.set_yticks(y_ticks)

        # Create tick labels.
        # Apply crop. Only the labels get the offset applied,
        # positions in image coords don't change.
        if view_crop_box is not None:
            x_ticks += view_crop_box[0, 0]
            y_ticks += view_crop_box[0, 1]
        # Convert tick labels to world coords.
        if not use_image_coords and affine is not None:
            s = affine_spacing(affine)
            o = affine_origin(affine)
            sx, sy = __get_view_xy(v, s)
            ox, oy = __get_view_xy(v, o)
            x_ticks = (x_ticks * sx + ox)
            y_ticks = (y_ticks * sy + oy)
        col_ax.set_xticklabels([f'{t:.1f}' for t in x_ticks])
        col_ax.set_yticklabels([f'{t:.1f}' for t in y_ticks])

        # Hide spines.
        for p in ['right', 'top', 'bottom', 'left']:
            col_ax.spines[p].set_visible(False)

        # Title.
        if show_title:
            title = f'{VIEWS[v]}, slice {view_idx}'
            if affine is not None:
                s = affine_spacing(affine)
                o = affine_origin(affine)
                world_pos = view_idx * s[v] + o[v]
                title += f' ({world_pos:.1f}mm)'
            col_ax.set_title(title, fontsize=title_fontsize)

    if show:
        plt.tight_layout()
        plt.show()

    if return_axis:
        return axs

def __resolve_boxes(
    box: Box | BatchBox | RegionID | List[RegionID] | None,
    size: Size,
    affine: AffineMatrix | None = None,
    labels: LabelImage | BatchLabelImage | None = None,
    label_names: List[RegionID] | None = None,
    points: Point | Points | BatchPoints | None = None,
    point_names: List[LandmarkID] | List[List[LandmarkID]] | None = None,
    ) -> BatchVoxelBox | None:
    if box is None:
        return None
    box = box.copy()    # Otherwise it modifies box passed to the plotting method.

    # Resolve foreground fov for region IDs.
    if isinstance(box, (str, list)):
        if labels is None:
            raise ValueError("If box is a region ID, labels must be provided.")
        if label_names is not None:
            label_names_list = arg_to_list(label_names, str)
        else:
            label_names_list = None
        region_ids = arg_to_list(box, str)
        boxes = []
        for r in region_ids:
            if label_names_list is None:
                raise ValueError("Label names must be provided for string region IDs.")
            if r not in label_names_list:
                raise ValueError(f"Region name '{r}' not found in label_names: {label_names_list}.")
            idx = label_names_list.index(r)
            fov = foreground_fov(labels[idx])
            if fov is None:
                continue
            boxes.append(fov)
        if not boxes:
            return None
        return np.stack(boxes)

    # Convert single box to batch.
    if box.ndim == 2:
        boxes = box[None]

    # Convert to image coords.
    if affine is not None:
        for i in range(len(boxes)):
            boxes[i] = to_image_coords(boxes[i], affine)
            boxes[i] = np.clip(boxes[i], 0, np.array(size) - 1)

    return boxes

def __resolve_crop(
    crop: Box | Point | str | None,
    crop_margin: int,
    size: Size,
    affine: AffineMatrix | None = None,
    labels: BatchLabelImage | None = None,
    label_names: List[RegionID] | None = None,
    points: List[np.ndarray] | None = None,
    point_names: List[List[str]] | None = None,
    ) -> PixelBox | VoxelBox | None:
    if crop is None:
        return None

    size = to_numpy(size)
    dim = len(size)

    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        margin_vox = np.full(dim, crop_margin, dtype=np.float32) / spacing
    else:
        margin_vox = np.full(dim, crop_margin, dtype=np.float32)

    apply_margin = True

    if isinstance(crop, str):
        source, value = crop.split(':', 1)

        if source in ('l', 'label', 'labels'):
            if labels is None:
                raise ValueError(f"crop='{crop}' but no labels were provided.")
            if value.lstrip('-').isdigit():
                label_idx = int(value)
            else:
                if label_names is None:
                    raise ValueError(f"crop='{crop}' uses a label name but no 'label_names' were provided.")
                label_names_list = arg_to_list(label_names, str)
                if value not in label_names_list:
                    raise ValueError(f"Label name '{value}' not found in label_names: {label_names_list}.")
                label_idx = label_names_list.index(value)
            fov_box = foreground_fov(labels[label_idx])
            if fov_box is None:
                return None
            box_vox = fov_box.astype(np.float32)

        elif source in ('p', 'point', 'points'):
            if points is None:
                raise ValueError(f"crop='{crop}' but no points were provided.")
            value_parts = value.split(':')
            if len(value_parts) == 1:
                batch_idx, point_ref = 0, value_parts[0]
            else:
                batch_idx, point_ref = int(value_parts[0]), value_parts[1]
            batch = points[batch_idx]
            if point_ref.lstrip('-').isdigit():
                point_vox = batch[int(point_ref)]
            else:
                batch_names = point_names[batch_idx] if point_names is not None else None
                if batch_names is None:
                    raise ValueError(f"crop='{crop}' uses a point name but no 'point_names' were provided.")
                if point_ref not in batch_names:
                    raise ValueError(f"Point name '{point_ref}' not found in point_names: {batch_names}.")
                point_vox = batch[batch_names.index(point_ref)]
            box_vox = np.stack([point_vox, point_vox])

        else:
            raise ValueError(f"Unknown crop prefix '{source}'. Expected 'l'/'label'/'labels' or 'p'/'point'/'points'.")

    elif isinstance(crop, np.ndarray) and crop.ndim == 2:
        # Box3D - crop directly, no margin applied.
        if affine is not None:
            box_vox = (crop - origin) / spacing
        else:
            box_vox = crop
        apply_margin = False

    else:
        # Point3D - create a box of crop_margin around the point.
        point = to_numpy(crop)
        if affine is not None:
            point_vox = (point - origin) / spacing
        else:
            point_vox = point
        box_vox = np.stack([point_vox, point_vox])

    if apply_margin:
        box_vox[0] -= margin_vox
        box_vox[1] += margin_vox

    # Clip to data bounds and convert to int.
    box_vox = np.stack([
        np.clip(np.floor(box_vox[0]).astype(np.int32), 0, size - 1),
        np.clip(np.ceil(box_vox[1]).astype(np.int32), 0, size - 1),
    ])
    return box_vox

def __resolve_crosshairs(
    crosshairs: Point3D | str | None,
    size: Size,
    affine: AffineMatrix | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    labels: BatchLabelImage | None = None,
    label_names: List[RegionID] | None = None,
    points: List[np.ndarray] | None = None,
    point_names: List[List[LandmarkID]] | None = None,
    ) -> Voxel | None:
    if crosshairs is None:
        return None
    return __resolve_point(crosshairs, size, affine=affine, centre_method=centre_method, label_names=label_names, labels=labels, point_names=point_names, points=points)

def __resolve_labels(
    labels: LabelImage | BatchLabelImage | None,
    dim: int,
    ) -> BatchLabelImage | None:
    if labels is None:
        return None
    # Normalise to batch form (B, ...).
    if labels.ndim == dim:
        labels = labels[np.newaxis]
    if labels.dtype != bool:
        if labels.min() < 0 or labels.max() > 1:
            logger.warn(f"Labels values are outside the range [0, 1]. Got min={labels.min():.3f}, max={labels.max():.3f}. Performing minmax norm.")
            labels_min, labels_max = labels.min(), labels.max()
            if labels_max > labels_min:
                labels = (labels - labels_min) / (labels_max - labels_min)
    return labels

def __resolve_point(
    idx: int | float | str | Point | None,
    size: Size,
    affine: AffineMatrix | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    labels: BatchLabelImage | None = None,
    label_names: List[RegionID] | None = None,
    points: List[np.ndarray] | None = None,
    point_names: List[List[str]] | None = None,
    ) -> Pixel | Voxel:
    if idx is None:
        idx = 'f:0.5'
    size = to_numpy(size)

    # Point.
    if isinstance(idx, (tuple, list, np.ndarray)) and not isinstance(idx, bool):
        idx = np.asarray(idx, dtype=float).flatten()
        if len(idx) != len(size):
            raise ValueError(f"Expected a {len(size)}-element point but got {len(idx)} elements.")
        if affine is not None:
            idx = to_image_coords(idx, affine).astype(float)
        return np.clip(np.round(idx), 0, size - 1).astype(int)

    if not isinstance(idx, str):
        raise ValueError(f"Invalid idx: {idx}. Expected Point3D, str, or None.")

    source, value = idx.split(':', 1)

    # Proportion of field-of-view - applied equally to all axes.
    if source == 'f':
        p = float(value)
        return np.clip(np.round(p * (size - 1)), 0, size - 1).astype(int)

    # Label channels - by index (e.g. "labels:0") or name (e.g. "labels:Brainstem").
    if source in ('l', 'label', 'labels'):
        if labels is None:
            raise ValueError(f"idx='{idx}' but no labels were provided.")
        if value.lstrip('-').isdigit():
            label_idx = int(value)
        else:
            if label_names is None:
                raise ValueError(f"idx='{idx}' uses a label name but no 'label_names' were provided.")
            label_names_list = arg_to_list(label_names, str)
            if value not in label_names_list:
                raise ValueError(f"Label name '{value}' not found in label_names: {label_names_list}.")
            label_idx = label_names_list.index(value)
        if centre_method == 'com':
            point = centre_of_mass(labels[label_idx], affine=affine)
        elif centre_method == 'fov':
            point = foreground_fov_centre(labels[label_idx], affine=affine)
        else:
            raise ValueError(f"Unknown centre_method '{centre_method}'. Expected 'com' or 'fov'.")

    # Points.
    elif source in ('p', 'point', 'points'):
        if points is None:
            raise ValueError(f"idx='{idx}' but no points were provided.")
        value_parts = value.split(':')
        if len(value_parts) == 1:
            batch_idx, point_ref = 0, value_parts[0]
        else:
            batch_idx, point_ref = int(value_parts[0]), value_parts[1]
        batch = points[batch_idx]
        if point_ref.lstrip('-').isdigit():
            point = batch[int(point_ref)]
        else:
            batch_names = point_names[batch_idx] if point_names is not None else None
            if batch_names is None:
                raise ValueError(f"idx='{idx}' uses a point name but no 'point_names' were provided.")
            if point_ref not in batch_names:
                raise ValueError(f"Point name '{point_ref}' not found in point_names: {batch_names}.")
            point = batch[batch_names.index(point_ref)]

    else:
        raise ValueError(f"Unknown idx prefix '{source}'. Expected 'f', 'labels', or 'points'.")

    # Convert to image coords.
    if affine is not None:
        point = to_image_coords(point, affine)
    
    # Truncate to image bounds.
    point = np.clip(point, 0, size - 1)

    return point

def __resolve_points(
    points: Point | Points | BatchPoints | List[Points] | Landmark | Landmarks | None,
    affine: AffineMatrix | None = None,
    point_names: LandmarkID | List[LandmarkID] | None = None,
    # Returns a list of points (instead of a batchpoints) because this allows
    # different batch sizes.
    ) -> Tuple[List[Points] | None, List[List[LandmarkID]] | None]:
    if points is None:
        return None, None

    # Handle Landmark/Landmarks (pd.DataFrame/pd.Series).
    if isinstance(points, (pd.DataFrame, pd.Series)):
        if point_names is None:
            point_names = points['landmark-id'].astype(str).tolist()
        points = landmarks_to_points(points)

    # Normalise to List[Points].
    if isinstance(points, list):
        # List[Points] — convert each batch element, normalise (dim,) → (1, dim).
        points = [to_numpy(p) for p in points]
        points = [p[np.newaxis] if p.ndim == 1 else p for p in points]
    else:
        if not isinstance(points, np.ndarray):
            points = to_numpy(points)
        # Normalise to (B, N, dim): single point (dim,) → (1, 1, dim), unbatched (N, dim) → (1, N, dim).
        if points.ndim == 1:
            points = [points[np.newaxis]]
        elif points.ndim == 2:
            points = [points]
        else:
            points = [p for p in points]

    # Resolve point names — applied uniformly across all batches.
    n_points = points[0].shape[0] if points else 0
    if point_names is None:
        per_point_names = [str(i) for i in range(n_points)]
    else:
        per_point_names = arg_to_list(point_names, str)
        assert len(per_point_names) == n_points, f"Expected point_names of length {n_points} but got {len(per_point_names)}."

    # Convert from world to image coords.
    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        points = [(p - origin) / spacing for p in points]

    # Skip empty batches.
    batches = []
    batch_names_list = []
    for b in points:
        if b.shape[0] == 0:
            continue
        batches.append(b)
        batch_names_list.append(list(per_point_names))

    if not batches:
        logger.warn("Points array is empty. No points will be plotted.")
        return None, None

    return batches, batch_names_list

def __resolve_window(
    window: Window | None,
    affine: AffineMatrix | None = None,
    data: Image | None = None,
    labels: BatchLabelImage | None = None,
    label_margin: float = 100,
    vmin: float | None = None,
    vmax: float | None = None,
    ) -> tuple[float | None, float | None]:
    if window is not None:
        assert vmin is None, "vmin must be None if window is specified."
        assert vmax is None, "vmax must be None if window is specified."
    if window is None:
        return vmin, vmax
    if isinstance(window, str):
        if window in WINDOW_PRESETS:
            width, level = WINDOW_PRESETS[window]
        else:
            source, value = window.split(':', 1)
            if source in ('l', 'label', 'labels'):
                if labels is None:
                    raise ValueError(f"window='{window}' but no labels were provided.")
                if value.lstrip('-').isdigit():
                    label_idx = int(value)
                else:
                    raise ValueError(f"window='{window}' uses a label name but 'label_names' is not supported for window presets.")
                label = labels[label_idx]
                fov = foreground_fov(label, affine=affine)
                fov[0] -= label_margin
                fov[1] += label_margin
                fov_data = crop(data, fov)
                width = fov_data.max() - fov_data.min()
                level = (fov_data.max() + fov_data.min()) / 2
            else:
                raise ValueError(f"Unknown window preset '{window}'. Expected one of {list(WINDOW_PRESETS.keys())}, or 'labels:<idx>'.")
    else:
        width, level = window
    vmin = level - width / 2
    vmax = level + width / 2
    return vmin, vmax
