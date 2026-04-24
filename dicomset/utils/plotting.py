import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Literal

from ..typing import AffineMatrix, AffineMatrix2D, AffineMatrix3D, BatchBox, BatchBox3D, BatchLabelImage, BatchLabelImage2D, BatchLabelImage3D, BatchVoxelBox, Box, Box2D, Box3D, Image, Image2D, Image3D, LabelImage, LabelImage2D, LabelImage3D, Landmark3D, Landmarks3D, Number, Orientation, Pixel, PixelBox, Point, Point2D, Point3D, Points, Points2D, Points3D, RegionID, Size, View, Voxel, VoxelBox, Window
from .args import alias_kwargs, arg_to_list
from .conversion import to_numpy
from .geometry import affine_origin, affine_spacing, centre_of_mass, foreground_fov, foreground_fov_centre, to_image_coords
from .landmarks import landmarks_to_points
from .logging import logger

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

def _assert_orientation(
    orientation: Orientation,
    ) -> None:
    orientations = {'LAI', 'LAS', 'LPI', 'LPS', 'RAI', 'RAS', 'RPI', 'RPS'}
    if orientation not in orientations:
        raise ValueError(f"Invalid orientation '{orientation}'. Must be one of {orientations}.")

def _get_view_aspect(
    view: View,
    affine: AffineMatrix3D | None,
    ) -> float | None:
    if affine is None:
        return None
    spacing = affine_spacing(affine)
    axes = [i for i in range(3) if i != view]
    aspect = float(spacing[axes[1]] / spacing[axes[0]])
    return aspect

def _get_view_origin(
    view: View,
    orientation: Orientation = 'LPS',
    ) -> tuple[Literal['lower', 'upper'], Literal['lower', 'upper']]:
    _assert_orientation(orientation)
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

def _get_view_slice(
    view: View,
    data: np.ndarray,
    idx: int,
    ) -> np.ndarray:
    slicing: list[int | slice] = [slice(None)] * 3
    slicing[view] = idx
    return data[tuple(slicing)]

def _get_view_xy(
    view: View,
    values: tuple | np.ndarray,
    ) -> tuple:
    axes = [i for i in range(3) if i != view]
    return values[axes[0]], values[axes[1]]

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

def plot_slice(
    data: Image2D | None,
    affine: AffineMatrix2D | None = None,
    alpha: float = 0.3,
    ax: mpl.axes.Axes | None = None,
    cmap: str = 'gray',
    crop: Box2D | Point2D | str | None = None,
    crop_margin: int = 100,
    labels: LabelImage2D | BatchLabelImage2D | None = None,
    points: Point2D | Points2D | None = None,
    points_colour: str = 'yellow',
    return_axis: bool = False,
    show_hist: bool = False,
    show_point_idxs: bool = False,
    title: str | None = None,
    use_image_coords: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    window: Window | None = None,
    x_label: str | None = None,
    x_origin: Literal['lower', 'upper'] | None = 'lower',
    y_label: str | None = None,
    y_origin: Literal['lower', 'upper'] | None = 'upper',
    ) -> mpl.axes.Axes | None:
    if data is None:
        assert labels is not None, "Labels must be provided if data is None."
        data = np.zeros(labels.shape[-2:])

    # Resolve window to vmin/vmax.
    vmin, vmax = __resolve_window(window, vmin, vmax)

    # Normalise labels to batch form (B, X, Y).
    if labels is not None and labels.ndim == 2:
        labels = labels[np.newaxis]
    if labels is not None and labels.dtype != bool:
        if labels.min() < 0 or labels.max() > 1:
            logger.warn(f"Labels values are outside the range [0, 1]. Got min={labels.min():.3f}, max={labels.max():.3f}.")

    # Check for empty points array - could be filtered by the transform.
    if points is not None:
        if isinstance(points, (pd.DataFrame, pd.Series)):
            points = landmarks_to_points(points)
        elif not isinstance(points, np.ndarray):
            points = to_numpy(points)

        # Expand single points.
        if points.ndim == 1:
            points = points[np.newaxis, :]

        if points.shape[0] == 0:
            logger.warn("Points array is empty. No points will be plotted.")
            points = None
        else:
            assert points.shape[1] == 2, f"Expected points to have shape (N, 2) but got {points.shape}."

    # Resolve crop.
    crop_box = __resolve_crop(crop, crop_margin, data.shape, affine=affine, label_names=None, labels=labels, points=points)
    print(crop_box)

    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False

    # Aspect ratio from affine.
    aspect = None
    if affine is not None:
        spacing = affine_spacing(affine)
        aspect = float(spacing[1] / spacing[0])

    # Plot the image.
    if crop_box is not None:
        data = data[crop_box[0, 0]:crop_box[1, 0] + 1, crop_box[0, 1]:crop_box[1, 1] + 1]
    ax.imshow(data.T, aspect=aspect, cmap=cmap, origin=y_origin, vmax=vmax, vmin=vmin)

    # Plot labels.
    if labels is not None:
        palette = sns.color_palette('colorblind', len(labels))
        for i, l in enumerate(labels):
            if crop_box is not None:
                l = l[crop_box[0, 0]:crop_box[1, 0] + 1, crop_box[0, 1]:crop_box[1, 1] + 1]
            cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[i]))
            ax.imshow(l.T, alpha=alpha, cmap=cmap_label)
            ax.contour(l.T, colors=[palette[i]], levels=[.5], linestyles='solid')

    # Plot points.
    if points is not None:
        if affine is not None:
            spacing = affine_spacing(affine)
            origin = affine_origin(affine)
        if points_colour == 'gradient' and len(points) > 1:
            points_cmap = mpl.colors.LinearSegmentedColormap.from_list('warm_bright', ['#FFE600', '#FF8C00', '#FF3300', '#FF0066'])
            p_colours = [points_cmap(i / (len(points) - 1)) for i in range(len(points))]
        else:
            p_colours = [points_colour] * len(points)
        for pi, p in enumerate(points):
            p = (p - origin) / spacing if affine is not None else p
            if crop_box is not None:
                p -= crop_box[0]
            ax.scatter(p[0], p[1], c=[p_colours[pi]], marker='o', s=20, zorder=5)
            if show_point_idxs:
                ax.annotate(str(pi), (p[0], p[1]),
                    color=p_colours[pi], fontsize=8,
                    textcoords='offset points', xytext=(5, 5), zorder=5)

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
        s = affine_spacing(affine)
        o = affine_origin(affine)
        sx, sy = _get_view_xy(v, s)
        ox, oy = _get_view_xy(v, o)
        x_ticks = (x_ticks * sx + ox)
        y_ticks = (y_ticks * sy + oy)
    ax.set_xticklabels([f'{t:.1f}' for t in x_ticks])
    ax.set_yticklabels([f'{t:.1f}' for t in y_ticks])

    # Hide axis spines.
    for p in ['right', 'top', 'bottom', 'left']:
        ax.spines[p].set_visible(False)

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
)
def plot_volume(
    data: Image3D | None,
    affine: AffineMatrix3D | None = None,
    ax: mpl.axes.Axes | List[mpl.axes.Axes] | None = None,
    box: Box3D | BatchBox3D | RegionID | List[RegionID] | None = None,
    cmap: str = 'gray',
    crop: Box3D | Point3D | str | None = None,
    crop_margin: int = 100,
    dose: Image3D | None = None,
    dose_alpha_min: float = 0.3,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: float = 0.15,
    figsize: tuple[float, float] = (16, 6),
    idx: int | float | str | Point3D | None = None,
    labels: LabelImage3D | BatchLabelImage3D | None = None,
    label_names: RegionID | List[RegionID] | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    orientation: Orientation = 'LPS',
    label_alpha: float = 0.3,
    points: Point3D | Points3D | Landmark3D | Landmarks3D | None = None,
    points_colour: str = 'yellow',
    crosshairs: Point3D | str | None = None,
    crosshairs_colour: str = 'yellow',
    return_axis: bool = False,
    show_crosshairs_coords: bool = True,
    show_labels: bool = True,
    show_points: bool = True,
    show_point_idxs: bool = False,
    show_title: bool = True,
    use_image_coords: bool = False,
    view: int | list[int] | Literal['all'] = 'all',
    vmin: float | None = None,
    vmax: float | None = None,
    window: Window | None = None,
    ) -> np.ndarray | None:
    if data is None:
        assert labels is not None, "Labels must be provided if data is None."
        data = np.zeros(labels.shape[-3:])

    # Resolve window to vmin/vmax.
    vmin, vmax = __resolve_window(window, vmin, vmax)

    # Normalise labels to batch form (B, X, Y, Z).
    if labels is not None and labels.ndim == 3:
        labels = labels[np.newaxis]
    if labels is not None and labels.dtype != bool:
        if labels.min() < 0 or labels.max() > 1:
            logger.warn(f"Labels values are outside the range [0, 1]. Got min={labels.min():.3f}, max={labels.max():.3f}.")

    # Check for empty points array - could be filtered by the transform.
    if points is not None:
        if isinstance(points, (pd.DataFrame, pd.Series)):
            points = landmarks_to_points(points)
        elif not isinstance(points, np.ndarray):
            points = to_numpy(points)

        # Expand single points.
        if points.ndim == 1:
            points = points[np.newaxis, :]

        if points.shape[0] == 0:
            logger.warn("Points array is empty. No points will be plotted.")
            points = None
            if isinstance(idx, str) and idx.startswith('points:'):
                idx = None
        else:
            assert points.shape[1] == 3, f"Expected points to have shape (N, 3) but got {points.shape}."


    # Resolve views.
    views = list(range(3)) if view == 'all' else (view if isinstance(view, list) else [view])

    # Resolve idx and crosshairs to 3D voxel points.
    idx_vox = __resolve_point(idx, data.shape, affine=affine, centre_method=centre_method, label_names=label_names, labels=labels, points=points)

    # Resolve crop to a voxel bounding box.
    crop_box = __resolve_crop(crop, crop_margin, data.shape, affine=affine, label_names=label_names, labels=labels, points=points)

    # Resolve boxes for overlay.
    boxes = __resolve_boxes(box, data.shape, affine=affine, label_names=label_names, labels=labels)

    palette = sns.color_palette('colorblind', 20)

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
        x_axis, y_axis = _get_view_xy(v, list(range(3)))
        view_crop_box = crop_box[:, [x_axis, y_axis]] if crop_box is not None else None

        image = _get_view_slice(v, data, view_idx)
        if view_crop_box is not None:
            image = image[view_crop_box[0, 0]:view_crop_box[1, 0] + 1, view_crop_box[0, 1]:view_crop_box[1, 1] + 1]
        aspect = _get_view_aspect(v, affine)
        origin_x, origin_y = _get_view_origin(v, orientation=orientation)

        # The two non-view axes: first is displayed on x, second on y.
        col_ax.imshow(image.T, aspect=aspect, cmap=cmap, origin=origin_y, vmax=vmax, vmin=vmin)
        if origin_x == 'upper':
            col_ax.invert_xaxis()

        # Box overlays.
        if boxes is not None:
            # Get 2D boxes.
            x_axis, y_axis = _get_view_xy(v, list(range(3)))
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
            dose_slice = _get_view_slice(v, dose, view_idx)
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
            label_names_list = arg_to_list(label_names, str) if label_names is not None else None
            for j, lab in enumerate(labels):
                label_slice = _get_view_slice(v, lab, view_idx)
                if view_crop_box is not None:
                    label_slice = label_slice[view_crop_box[0, 0]:view_crop_box[1, 0] + 1, view_crop_box[0, 1]:view_crop_box[1, 1] + 1]
                cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[j]))
                col_ax.imshow(label_slice.T, alpha=label_alpha, aspect=aspect, cmap=cmap_label, origin=origin_y)
                col_ax.contour(label_slice.T, colors=[palette[j]], levels=[0.5], linestyles='solid')

            # Add legend on first view only.
            if label_names_list is not None and v == views[0]:
                handles = [mpl.patches.Patch(facecolor=palette[j], label=label_names_list[j]) for j in range(len(labels)) if j < len(label_names_list)]
                col_ax.legend(fontsize='small', framealpha=0.7, handles=handles, loc='upper right')

        # Point overlays.
        if show_points and points is not None:
            if affine is not None:
                spacing = affine_spacing(affine)
                origin = affine_origin(affine)
            if points_colour == 'gradient' and len(points) > 1:
                points_cmap = mpl.colors.LinearSegmentedColormap.from_list('warm_bright', ['#FFE600', '#FF8C00', '#FF3300', '#FF0066'])
                points_colours = [points_cmap(i / (len(points) - 1)) for i in range(len(points))]
            else:
                points_colours = ['yellow'] * len(points)
            for pi, p in enumerate(points):
                p = (p - origin) / spacing if affine is not None else p
                p_x, p_y = _get_view_xy(v, p)
                if view_crop_box is not None:
                    p_x -= view_crop_box[0, 0]
                    p_y -= view_crop_box[0, 1]
                col_ax.scatter(p_x, p_y, c=[points_colours[pi]], marker='o', s=20, zorder=5)
                if show_point_idxs:
                    col_ax.annotate(str(pi), (p_x, p_y),
                        color=points_colours[pi], fontsize=8,
                        textcoords='offset points', xytext=(5, 5), zorder=5)

        # Crosshairs.
        if crosshairs is not None:
            crosshairs_vox = __resolve_point(crosshairs, data.shape, affine=affine, centre_method=centre_method, label_names=label_names, labels=labels, points=points)
            ch_x, ch_y = _get_view_xy(v, crosshairs_vox)
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
                    ch_x_world, ch_y_world = _get_view_xy(v, crosshairs_vox * s + o)
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
            sx, sy = _get_view_xy(v, s)
            ox, oy = _get_view_xy(v, o)
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
            col_ax.set_title(title)

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
    ) -> BatchVoxelBox | None:
    if box is None:
        return None

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
    points: Points | None = None,
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
            point_world = points[int(value)]
            if affine is not None:
                point_vox = (point_world - origin) / spacing
            else:
                point_vox = point_world
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

def __resolve_point(
    idx: int | float | str | Point | None,
    size: Size,
    affine: AffineMatrix | None = None,
    centre_method: Literal['com', 'fov'] = 'com',
    labels: BatchLabelImage | None = None,
    label_names: List[RegionID] | None = None,
    points: Points | None = None,
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
        point = points[int(value)]

    else:
        raise ValueError(f"Unknown idx prefix '{source}'. Expected 'f', 'labels', or 'points'.")

    # Convert to image coords.
    if affine is not None:
        point = to_image_coords(point, affine)
    
    # Truncate to image bounds.
    point = np.clip(point, 0, size - 1)

    return point

def __resolve_window(
    window: Window | None,
    vmin: float | None,
    vmax: float | None,
    ) -> tuple[float | None, float | None]:
    if window is not None:
        assert vmin is None, "vmin must be None if window is specified."
        assert vmax is None, "vmax must be None if window is specified."
    if window is None:
        return vmin, vmax
    if isinstance(window, str):
        if window not in WINDOW_PRESETS:
            raise ValueError(f"Unknown window preset '{window}'. Expected one of {list(WINDOW_PRESETS.keys())}.")
        width, level = WINDOW_PRESETS[window]
    else:
        width, level = window
    vmin = level - width / 2
    vmax = level + width / 2
    return vmin, vmax
