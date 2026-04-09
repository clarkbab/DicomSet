import os
from typing import List

from ... import config
from ...typing import DatasetID
from ..dataset import RawDataset

def dataset_exists(dataset_id: DatasetID) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'raw', dataset_id)
    return os.path.exists(ds_path)

def list_datasets() -> List[DatasetID]:
    path = os.path.join(config.directories.datasets, 'raw')
    return list(sorted(os.listdir(path))) if os.path.exists(path) else []

def load_dataset(dataset_id: DatasetID) -> RawDataset:
    ds_path = os.path.join(config.directories.datasets, 'raw', dataset_id)
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Raw dataset '{dataset_id}' not found at {ds_path}.")
    return RawDataset(dataset_id)
