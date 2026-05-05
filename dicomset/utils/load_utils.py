from typing import List

from ..dataset import Dataset
from ..typing import DatasetID, DatasetType

def list_datasets(
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

def load_dataset(
    name: str,
    type: DatasetType | None = None,
    **kwargs,
    ) -> Dataset:
    if type is None:
        # Match by name only.
        types = ['raw', 'dicom', 'nifti', 'training']
        for t in types:
            datasets = list_datasets(t)
            if name in datasets:
                return load_dataset(name, t, **kwargs)

    lower_type = type.lower()
    if lower_type == 'raw':
        from ..raw import RawDataset
        return RawDataset(name, **kwargs)
    elif lower_type == 'dicom':
        from ..dicom import DicomDataset
        return DicomDataset(name, **kwargs)
    elif lower_type == 'nifti':
        from ..nifti import NiftiDataset
        return NiftiDataset(name, **kwargs)
    elif lower_type == 'training':
        from ..training import TrainingDataset
        return TrainingDataset(name, **kwargs)
    else:
        raise ValueError(f"Dataset type '{type}' not found.")
