import os
import shutil

from ... import config
from ...typing import DatasetID

def destroy(
    dataset_id: DatasetID,
    makeitso: bool = True,
    ) -> None:
    ds_path = os.path.join(config.directories.datasets, 'nifti', dataset_id)
    if os.path.exists(ds_path):
        with_makeitso(makeitso, lambda: shutil.rmtree(ds_path), f"Destroying nifti dataset '{id}' at {ds_path}.")
