from typing import List

from ..dataset import Dataset
from ..dicom import DicomDataset
from ..dicom.utils.load import list_datasets as list_dicom_datasets
from ..nifti import NiftiDataset
from ..nifti.utils.load import list_datasets as list_nifti_datasets
from ..raw import RawDataset
from ..raw.utils.load import list_datasets as list_raw_datasets
from ..training import TrainingDataset
from ..training.utils.load import list_datasets as list_training_datasets
from ..typing import DatasetID, DatasetType

def load(
    name: str,
    type: DatasetType,
    **kwargs,
    ) -> Dataset:
    lower_type = type.lower()
    if lower_type == 'dicom':
        return DicomDataset(name, **kwargs)
    elif lower_type == 'nifti':
        return NiftiDataset(name, **kwargs)
    elif lower_type == 'raw':
        return RawDataset(name, **kwargs)
    elif lower_type == 'training':
        return TrainingDataset(name, **kwargs)
    else:
        raise ValueError(f"Dataset type '{type}' not found.")

def list(
    type: DatasetType,
    ) -> List[DatasetID]:
    lower_type = type.lower()
    if lower_type == 'dicom':
        return list_dicom_datasets()
    elif lower_type == 'nifti':
        return list_nifti_datasets()
    elif lower_type == 'raw':
        return list_raw_datasets()
    elif lower_type == 'training':
        return list_training_datasets()
    else:
        raise ValueError(f"Dataset type '{type}' not found.")
