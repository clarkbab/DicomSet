from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

LAZY_IMPORTS = {
    'args': ['arg_default', 'arg_to_list', 'bubble_args', 'landmarks_to_list', 'regions_to_list', 'resolve_filepath'],
    'assertions': ['assert_orientation', 'assert_shapes_equal'],
    'conversion': ['to_list', 'to_numpy', 'to_tensor', 'to_tuple'],
    'debug': ['from_desc'],
    'dicom': [
        'from_ct_dicom', 'from_rtdose_dicom', 'from_rtplan_dicom', 'from_rtstruct_dicom',
        'list_rtstruct_landmarks', 'list_rtstruct_regions', 'load_dicom', 'save_dicom',
        'to_ct_dicom', 'to_rtdose_dicom', 'to_rtstruct_dicom',
    ],
    'geometry': [
        'affine_origin', 'affine_spacing', 'centre_of_mass', 'change_orientation', 'combine_boxes', 'create_affine',
        'foreground_fov', 'foreground_fov_centre', 'foreground_fov_width', 'fov', 'fov_centre', 'fov_width'
    ],
    'images': ['create_box_label'],
    'io': ['load_csv', 'load_json', 'load_nifti', 'load_nrrd', 'load_numpy', 'save_csv', 'save_json', 'save_nifti', 'save_numpy'],
    'landmarks': ['landmarks_dim', 'landmarks_to_points', 'points_to_landmarks', 'replace_points'],
    'load_utils': ['list_datasets', 'load_dataset'],
    'logging': ['logger'],
    'metrics': ['centroid_error', 'distances', 'dice', 'tre', 'volume'],
    'pandas': ['append_row'],
    'plotting': ['plot_hist', 'plot_slice', 'plot_volume'],
    'python': ['ensure_loaded', 'filter_lists', 'flatten_list', 'sort_lists', 'unzip'],
    'transforms': [
        'crop', 'crop_affine', 'crop_points', 'from_sitk_image', 'hist_eq', 'minmax', 'one_hot_encode',
        'resample', 'sample', 'standardise', 'to_sitk_image', 'transpose'
    ],
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
