import os
from typing import List

from ... import config
from ...typing import DatasetID
from ..dataset import DicomDataset

def dataset_exists(dataset_id: DatasetID) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'dicom', dataset_id)
    return os.path.exists(ds_path)

def list_datasets() -> List[DatasetID]:
    path = os.path.join(config.directories.datasets, 'dicom')
    return list(sorted(os.listdir(path))) if os.path.exists(path) else []

def load_dataset(dataset_id: DatasetID) -> DicomDataset:
    ds_path = os.path.join(config.directories.datasets, 'dicom', dataset_id)
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dicom dataset '{dataset_id}' not found at {ds_path}.")
    return DicomDataset(dataset_id)
