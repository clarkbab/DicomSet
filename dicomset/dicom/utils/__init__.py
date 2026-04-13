from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dicom import from_ct_dicom, from_rtdose_dicom, from_rtplan_dicom
    from .load import dataset_exists, list_datasets, load_dataset

__all__ = [
    'dataset_exists',
    'from_ct_dicom', 'from_rtdose_dicom', 'from_rtplan_dicom',
    'list_datasets', 'load_dataset',
]

DICOM_IMPORTS = ['from_ct_dicom', 'from_rtdose_dicom', 'from_rtplan_dicom']
LOAD_IMPORTS = ['dataset_exists', 'list_datasets', 'load_dataset']

def __getattr__(name):
    if name in DICOM_IMPORTS:
        from . import dicom
        return getattr(dicom, name)
    if name in LOAD_IMPORTS:
        from . import load
        return getattr(load, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
