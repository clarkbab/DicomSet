import os
import pydicom as dcm
from typing import List

from ...typing import DirPath, FilePath
from ...utils.args import resolve_filepath
from ...utils.io import is_dir, is_file

def load_dicom(
    filepath: FilePath,
    **kwargs,
    ) -> dcm.dataset.FileDataset:
    filepath = resolve_filepath(filepath)
    return dcm.dcmread(filepath, **kwargs)

def save_dicom(
    dicom: dcm.dataset.FileDataset | List[dcm.dataset.FileDataset],
    path: DirPath | FilePath | List[FilePath],
    overwrite: bool = True,
    ) -> None:
    # Save single dicom.
    if isinstance(dicom, dcm.dataset.FileDataset):
        assert is_file(path), f"Expected filepath for single DICOM, got directory path '{path}'."
        filepath = resolve_filepath(path)
        if os.path.exists(filepath) and not overwrite:
            raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        dcm.dcmwrite(filepath, dicom)
        return

    # Save multiple dicoms.
    else:
        assert isinstance(path, list) or is_dir(path), f"Expected directory path or list of filepaths for multiple DICOMs, got '{path}'."
        if is_dir(path):
            path = resolve_filepath(path) 
            filepaths = [os.path.join(path, f'{i:03}.dcm') for i in range(len(dicom))]            
        else:
            filepaths = [resolve_filepath(p) for p in path] 
        if len(filepaths) != len(dicom):
            raise ValueError(f"Number of DICOMs ({len(dicom)}) does not match number of filepaths ({len(filepaths)}).")
        for d, f in zip(dicom, filepaths):
            save_dicom(d, f, overwrite=overwrite)
