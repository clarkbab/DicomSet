import os
import shutil

from ... import config
from ...typing import DatasetID
from ...utils.python import with_makeitso

def destroy(
    dataset_id: DatasetID,
    makeitso: bool = True,
    ) -> None:
    ds_path = os.path.join(config.directories.datasets, 'training', dataset_id)
    if os.path.exists(ds_path):
        with_makeitso(makeitso, lambda: shutil.rmtree(ds_path), f"Destroying training dataset '{dataset_id}' at {ds_path}.")
