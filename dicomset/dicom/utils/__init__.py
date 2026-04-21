from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversion import convert_to_nifti
    from .dicom import from_ct_dicom, from_rtdose_dicom, from_rtplan_dicom, from_rtstruct_dicom, list_rtstruct_regions, to_ct_dicom, to_rtstruct_dicom
    from .io import load_dicom, save_dicom
    from .load import dataset_exists, list_datasets, load_dataset

__all__ = [
    'convert_to_nifti',
    'from_ct_dicom', 'from_rtdose_dicom', 'from_rtplan_dicom', 'from_rtstruct_dicom', 'list_rtstruct_regions', 'to_ct_dicom', 'to_rtstruct_dicom',
    'load_dicom', 'save_dicom',
    'dataset_exists', 'list_datasets', 'load_dataset',
]

CONVERSION_IMPORTS = ['convert_to_nifti']
DICOM_IMPORTS = ['from_ct_dicom', 'from_rtdose_dicom', 'from_rtplan_dicom', 'from_rtstruct_dicom', 'list_rtstruct_regions', 'to_ct_dicom', 'to_rtstruct_dicom']
IO_IMPORTS = ['load_dicom', 'save_dicom']
LOAD_IMPORTS = ['dataset_exists', 'list_datasets', 'load_dataset']

def __getattr__(name):
    if name in CONVERSION_IMPORTS:
        from . import conversion
        return getattr(conversion, name)
    if name in DICOM_IMPORTS:
        from . import dicom
        return getattr(dicom, name)
    if name in IO_IMPORTS:
        from . import io
        return getattr(io, name)
    if name in LOAD_IMPORTS:
        from . import load
        return getattr(load, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
