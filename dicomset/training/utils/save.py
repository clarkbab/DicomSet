import os
import shutil

from ... import config
from ...typing import DatasetID
from ..dataset import TrainingDataset

def create_dataset(
    dataset_id: DatasetID,
    recreate: bool = False,
    ) -> TrainingDataset:
    ds_path = os.path.join(config.directories.datasets, 'training', dataset_id)
    if os.path.exists(ds_path):
        if recreate:
            shutil.rmtree(ds_path)
    os.makedirs(ds_path, exist_ok=True)
    return TrainingDataset(dataset_id)
