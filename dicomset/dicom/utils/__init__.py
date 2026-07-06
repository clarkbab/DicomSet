from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

LAZY_IMPORTS = {
    'conversion': ['convert_to_nifti'],
    'load': ['dataset_exists', 'list_datasets', 'load_dataset'],
}

_DICOM_ATTRS = [
    'from_ct_dicom', 'from_rtdose_dicom', 'from_rtplan_dicom', 'from_rtstruct_dicom',
    'list_rtstruct_ids', 'list_rtstruct_landmarks', 'list_rtstruct_regions', 'to_ct_dicom', 'to_rtstruct_dicom',
    'load_dicom', 'save_dicom',
]

__all__ = [attr for attrs in LAZY_IMPORTS.values() for attr in attrs] + _DICOM_ATTRS

if TYPE_CHECKING:
    for module, attrs in LAZY_IMPORTS.items():
        for attr in attrs:
            exec(f"from .{module} import {attr}")
    from dicomset.utils.dicom import (
        from_ct_dicom, from_rtdose_dicom, from_rtplan_dicom, from_rtstruct_dicom,
        list_rtstruct_landmarks, list_rtstruct_regions, to_ct_dicom, to_rtstruct_dicom,
        load_dicom, save_dicom,
    )

def __getattr__(name):
    if name in _DICOM_ATTRS:
        return getattr(importlib.import_module('dicomset.utils.dicom'), name)
    for module, attrs in LAZY_IMPORTS.items():
        if name in attrs:
            return getattr(importlib.import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"Module {__name__} has no attribute {name}")
