from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .args import arg_to_list, bubble_args, resolve_filepath
    from .conversion import to_list, to_numpy, to_tensor
    from .geometry import affine_origin, affine_spacing, centre_of_mass, create_affine, foreground_fov, foreground_fov_centre, foreground_fov_width, fov, fov_centre, fov_width
    from .logging import logger
    from .metrics import centroid_error, dice, distances
    from .plotting import plot_hist, plot_slice, plot_volume
    from .transforms import minmax, resample, spatial_transpose

__all__ = [
    'arg_to_list', 'bubble_args', 'resolve_filepath',
    'to_list', 'to_numpy', 'to_tensor',
    'affine_origin', 'affine_spacing', 'centre_of_mass', 'create_affine', 'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width', 'fov', 'fov_centre', 'fov_width',
    'logger',
    'centroid_error', 'distances', 'dice',
    'plot_hist', 'plot_slice', 'plot_volume',
    'minmax', 'resample', 'spatial_transpose',
]

ARGS_IMPORTS = ['arg_to_list', 'bubble_args', 'resolve_filepath']
CONVERSION_IMPORTS = ['to_list', 'to_numpy', 'to_tensor']
GEOMETRY_IMPORTS = [
    'affine_origin', 'affine_spacing', 'centre_of_mass', 'create_affine',
    'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width', 'fov', 'fov_centre', 'fov_width'
]
LOGGING_IMPORTS = ['logger']
METRICS_IMPORTS = ['centroid_error', 'distances', 'dice']
PLOTTING_IMPORTS = ['plot_hist', 'plot_slice', 'plot_volume']
TRANSFORMS_IMPORTS = ['minmax', 'resample', 'spatial_transpose']

def __getattr__(name):
    if name in ARGS_IMPORTS:
        from . import args
        return getattr(args, name)
    if name in CONVERSION_IMPORTS:
        from . import conversion
        return getattr(conversion, name)
    if name in GEOMETRY_IMPORTS:
        from . import geometry
        return getattr(geometry, name)
    if name in LOGGING_IMPORTS:
        from . import logging
        return getattr(logging, name) 
    if name in METRICS_IMPORTS:
        from . import metrics
        return getattr(metrics, name)
    if name in PLOTTING_IMPORTS:
        from . import plotting
        return getattr(plotting, name)
    if name in TRANSFORMS_IMPORTS:
        from . import transforms
        return getattr(transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
