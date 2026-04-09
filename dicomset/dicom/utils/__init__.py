from __future__ import annotations

from typing import TYPE_CHECKING

from .dicom import from_ct_dicom, from_rtdose_dicom, from_rtplan_dicom
if TYPE_CHECKING:
    from .load import dataset_exists, list_datasets, load_dataset

__all__ = [
    'dataset_exists', 
    'from_ct_dicom', 'from_rtdose_dicom', 'from_rtplan_dicom', 
    'list_datasets', 'load_dataset',
]

def __getattr__(name):
    if name in ('dataset_exists', 'load_dataset', 'list_datasets'):
        from .load import dataset_exists, load_dataset, list_datasets
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
