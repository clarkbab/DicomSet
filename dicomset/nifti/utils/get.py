import os
from typing import List

from ... import config
from ...typing import DatasetID
from ..dataset import NiftiDataset

def exists_dataset(dataset_id: DatasetID) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'nifti', dataset_id)
    return os.path.exists(ds_path)

def get_dataset(dataset_id: DatasetID) -> NiftiDataset:
    ds_path = os.path.join(config.directories.datasets, 'nifti', dataset_id)
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Nifti dataset '{id}' not found at {ds_path}.")
    return NiftiDataset(dataset_id)
    
def list_datasets() -> List[DatasetID]:
    path = os.path.join(config.directories.datasets, 'nifti')
    return list(sorted(os.listdir(path))) if os.path.exists(path) else []
