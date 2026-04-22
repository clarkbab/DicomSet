from typing import List

from ..dataset import Dataset
from ..typing import DatasetID, DatasetType


def list(
    type: DatasetType,
    ) -> List[DatasetID]:
    lower_type = type.lower()
    if lower_type == 'dicom':
        from ..dicom.utils.load import list_datasets as list_dicom_datasets
        return list_dicom_datasets()
    elif lower_type == 'nifti':
        from ..nifti.utils.load import list_datasets as list_nifti_datasets
        return list_nifti_datasets()
    elif lower_type == 'raw':
        from ..raw.utils.load import list_datasets as list_raw_datasets
        return list_raw_datasets()
    elif lower_type == 'training':
        from ..training.utils.load import list_datasets as list_training_datasets
        return list_training_datasets()
    else:
        raise ValueError(f"Dataset type '{type}' not found.")

def load(
    name: str,
    type: DatasetType,
    **kwargs,
    ) -> Dataset:
    lower_type = type.lower()
    if lower_type == 'dicom':
        from ..dicom import DicomDataset
        return DicomDataset(name, **kwargs)
    elif lower_type == 'nifti':
        from ..nifti import NiftiDataset
        return NiftiDataset(name, **kwargs)
    elif lower_type == 'raw':
        from ..raw import RawDataset
        return RawDataset(name, **kwargs)
    elif lower_type == 'training':
        from ..training import TrainingDataset
        return TrainingDataset(name, **kwargs)
    else:
        raise ValueError(f"Dataset type '{type}' not found.")
