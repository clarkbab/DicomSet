from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .geometry import affine_origin, affine_spacing, centre_of_mass, create_affine, foreground_fov, foreground_fov_centre, fov, fov_centre
    from .metrics import centroid_error, dice, distances
    from .plotting import plot_slice, plot_volume
    from .transforms import resample, spatial_transpose

__all__ = [
    'affine_origin', 'affine_spacing', 'centre_of_mass', 'create_affine',
    'foreground_fov', 'foreground_fov_centre', 'fov', 'fov_centre',
    'centroid_error', 'distances', 'dice',
    'plot_slice', 'plot_volume',
    'resample', 'spatial_transpose',
]

GEOMETRY_IMPORTS = [
    'affine_origin', 'affine_spacing', 'centre_of_mass', 'create_affine',
    'foreground_fov', 'foreground_fov_centre', 'fov', 'fov_centre',
]
METRICS_IMPORTS = ['centroid_error', 'distances', 'dice']
PLOTTING_IMPORTS = ['plot_slice', 'plot_volume']
TRANSFORMS_IMPORTS = ['resample', 'spatial_transpose']

def __getattr__(name):
    if name in GEOMETRY_IMPORTS:
        from . import geometry
        return getattr(geometry, name)
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
