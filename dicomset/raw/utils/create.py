import os
import shutil

from ... import config
from ...typing import DatasetID
from ..dataset import RawDataset

def create_dataset(
    dataset_id: DatasetID,
    recreate: bool = False,
    ) -> RawDataset:
    ds_path = os.path.join(config.directories.datasets, 'raw', dataset_id)
    if os.path.exists(ds_path):
        if recreate:
            shutil.rmtree(ds_path)
    os.makedirs(ds_path, exist_ok=True)
    return RawDataset(dataset_id)
