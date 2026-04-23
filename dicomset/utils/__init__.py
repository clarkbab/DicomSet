from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

LAZY_IMPORTS = {
    'args': ['arg_to_list', 'bubble_args', 'resolve_filepath'],
    'conversion': ['to_list', 'to_numpy', 'to_tensor'],
    'debug': ['from_desc'],
    'geometry': [
        'affine_origin', 'affine_spacing', 'centre_of_mass', 'combine_boxes', 'create_affine',
        'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width', 'fov', 'fov_centre', 'fov_width'
    ],
    'images': ['create_box_label'],
    'io': ['load_csv', 'load_nifti', 'load_numpy', 'save_csv', 'save_nifti', 'save_numpy'],
    'landmarks': ['landmarks_dim', 'landmarks_to_points', 'points_to_landmarks'],
    'load_utils': ['list', 'load'],
    'logging': ['logger'],
    'metrics': ['centroid_error', 'distances', 'dice', 'tre', 'volume'],
    'pandas': ['append_row'],
    'plotting': ['plot_hist', 'plot_slice', 'plot_volume'],
    'python': ['ensure_loaded', 'filter_lists'],
    'regions': ['region_to_list'],
    'transforms': ['minmax', 'resample', 'spatial_transpose'],
}

__all__ = [attr for attrs in LAZY_IMPORTS.values() for attr in attrs]

if TYPE_CHECKING:
    for module, attrs in LAZY_IMPORTS.items():
        for attr in attrs:
            exec(f"from .{module} import {attr}")

def __getattr__(name):
    for module, attrs in LAZY_IMPORTS.items():
        if name in attrs:
            return getattr(importlib.import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"Module {__name__} has no attribute {name}")
