from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .args import arg_to_list, bubble_args, resolve_filepath
    from .conversion import to_list, to_numpy, to_tensor
    from .debug import from_desc
    from .geometry import affine_origin, affine_spacing, centre_of_mass, combine_boxes, create_affine, foreground_fov, foreground_fov_centre, foreground_fov_width, fov, fov_centre, fov_width
    from .images import create_box_label
    from .io import load_csv, save_csv
    from .landmarks import landmarks_dim, landmarks_to_points, points_to_landmarks
    from .logging import logger
    from .metrics import centroid_error, dice, distances, tre, volume
    from .plotting import plot_hist, plot_slice, plot_volume
    from .python import filter_lists
    from .regions import region_to_list
    from .transforms import minmax, resample, spatial_transpose

__all__ = [
    'arg_to_list', 'bubble_args', 'resolve_filepath',
    'to_list', 'to_numpy', 'to_tensor',
    'from_desc',
    'affine_origin', 'affine_spacing', 'centre_of_mass', 'combine_boxes', 'create_affine', 'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width', 'fov', 'fov_centre', 'fov_width',
    'create_box_label',
    'load_csv', 'save_csv',
    'landmarks_dim', 'landmarks_to_points', 'points_to_landmarks',
    'logger',
    'centroid_error', 'distances', 'dice', 'tre', 'volume',
    'append_row',
    'plot_hist', 'plot_slice', 'plot_volume',
    'filter_lists',
    'region_to_list',
    'minmax', 'resample', 'spatial_transpose',
]

ARGS_IMPORTS = ['arg_to_list', 'bubble_args', 'resolve_filepath']
CONVERSION_IMPORTS = ['to_list', 'to_numpy', 'to_tensor']
DEBUG_IMPORTS = ['from_desc']
GEOMETRY_IMPORTS = [
    'affine_origin', 'affine_spacing', 'centre_of_mass', 'combine_boxes', 'create_affine',
    'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width', 'fov', 'fov_centre', 'fov_width'
]
IMAGES_IMPORTS = ['create_box_label']
IO_IMPORTS = ['load_csv', 'save_csv']
LANDMARKS_IMPORTS = ['landmarks_dim', 'landmarks_to_points', 'points_to_landmarks']
LOGGING_IMPORTS = ['logger']
METRICS_IMPORTS = ['centroid_error', 'distances', 'dice', 'tre', 'volume']
PANDAS_IMPORTS = ['append_row']
PLOTTING_IMPORTS = ['plot_hist', 'plot_slice', 'plot_volume']
PYTHON_IMPORTS = ['filter_lists']
REGIONS_IMPORTS = ['region_to_list']
TRANSFORMS_IMPORTS = ['minmax', 'resample', 'spatial_transpose']

def __getattr__(name):
    if name in ARGS_IMPORTS:
        from . import args
        return getattr(args, name)
    if name in CONVERSION_IMPORTS:
        from . import conversion
        return getattr(conversion, name)
    if name in DEBUG_IMPORTS:
        from . import debug
        return getattr(debug, name)
    if name in GEOMETRY_IMPORTS:
        from . import geometry
        return getattr(geometry, name)
    if name in IMAGES_IMPORTS:
        from . import images
        return getattr(images, name)
    if name in IO_IMPORTS:
        from . import io
        return getattr(io, name)
    if name in LANDMARKS_IMPORTS:
        from . import landmarks
        return getattr(landmarks, name)
    if name in LOGGING_IMPORTS:
        from . import logging
        return getattr(logging, name) 
    if name in METRICS_IMPORTS:
        from . import metrics
        return getattr(metrics, name)
    if name in PANDAS_IMPORTS:
        from . import pandas
        return getattr(pandas, name)
    if name in PLOTTING_IMPORTS:
        from . import plotting
        return getattr(plotting, name)
    if name in PYTHON_IMPORTS:
        from . import python
        return getattr(python, name)
    if name in REGIONS_IMPORTS:
        from . import regions
        return getattr(regions, name)
    if name in TRANSFORMS_IMPORTS:
        from . import transforms
        return getattr(transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
