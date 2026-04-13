import ast
import json
import nibabel as nib
import nrrd
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from typing import Any, Dict, List, Tuple
import yaml

from ..typing import AffineMatrix3D, DirPath, FilePath, Image3D
from .args import arg_to_list, resolve_filepath
from .geometry import create_eye

def assert_writeable(filepath: FilePath | List[FilePath]) -> None:
    filepaths = arg_to_list(filepath, str)
    for f in filepaths:
        f = resolve_filepath(f)
        if os.path.exists(f):
            try:
                open(f, 'a')
            except (OSError, IOError):
                raise PermissionError(f"File '{f}' is open or read-only, cannot overwrite.")

def is_dir(
    path: DirPath | FilePath,
    ) -> bool:
    return not is_file(path)

def is_file(
    path: DirPath | FilePath,
    ) -> bool:
    path = resolve_filepath(path)
    _, ext = os.path.splitext(path)
    return bool(ext)

def load_csv(
    filepath: FilePath,
    exists_only: bool = False,
    filters: Dict[str, Any] = {},
    map_cols: Dict[str, str] = {},
    map_types: Dict[str, Any] = {},
    parse_cols: str | List[str] = [],
    **kwargs,
    ) -> pd.DataFrame | bool:
    filepath = resolve_filepath(filepath)
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"CSV at filepath '{filepath}' not found.")

    # Load CSV.
    map_types['patient-id'] = str
    map_types['study-id'] = str
    map_types['series-id'] = str
    df = pd.read_csv(filepath, dtype=map_types, **kwargs)

    # Map column names.
    df = df.rename(columns=map_cols)

    # Evaluate columns as literals.
    parse_cols = arg_to_list(parse_cols, str)
    for c in parse_cols:
        df[c] = df[c].apply(lambda s: ast.literal_eval(s))

    # Apply filters.
    for k, v in filters.items():
        df = df[df[k] == v]

    return df

def load_nifti(
    filepath: FilePath,
    ) -> Tuple[Image3D, AffineMatrix3D]:
    filepath = resolve_filepath(filepath)
    assert filepath.endswith('.nii') or filepath.endswith('.nii.gz'), "Filepath must end with .nii or .nii.gz"
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def load_nrrd(
    filepath: FilePath,
    ) -> Tuple[Image3D, AffineMatrix3D]:
    filepath = resolve_filepath(filepath)
    data, header = nrrd.read(filepath)
    affine = create_eye(3)
    affine[:3, :3] = header['space directions']
    affine[:3, 3] = header['space origin']
    affine[3, 3] = 1.0
    return data, affine

def load_numpy(
    filepath: FilePath,
    keys: str | List[str] = 'data',
    ) -> np.ndarray:
    assert filepath.endswith('.npy') or filepath.endswith('.npz'), "Filepath must end with .npy or .npz"
    data = np.load(filepath)
    if filepath.endswith('.npz'):
        keys = arg_to_list(keys, str)
        items = [data[k] for k in keys]
        items = items[0] if len(items) == 1 else items
    else:
        items = data
    return items

def load_yaml(filepath: FilePath) -> Any:
    filepath = resolve_filepath(filepath)
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def make_serialisable(
    data: Any,
    ) -> Any:
    if isinstance(data, dict):
        return {k: make_serialisable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_serialisable(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(make_serialisable(v) for v in data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
        return data.item()
    else:
        return data

def save_csv(
    data: pd.DataFrame,
    filepath: FilePath,
    index: bool = False,
    overwrite: bool = True,
    ) -> None:
    filepath = resolve_filepath(filepath)
    if os.path.exists(filepath) and not overwrite:
        raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath, index=index)

def save_json(
    data: Any,
    filepath: FilePath,
    overwrite: bool = True,
    ) -> None:
    filepath = resolve_filepath(filepath)
    if os.path.exists(filepath) and not overwrite:
        raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = make_serialisable(data)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def save_nifti(
    data: Image3D,
    affine: AffineMatrix3D,
    filepath: FilePath,
    ) -> None:
    filepath = resolve_filepath(filepath)
    assert filepath.endswith('.nii.gz') or filepath.endswith('.nii'), "Filepath must end with .nii or .nii.gz"
    if data.dtype == bool:
        data = data.astype(np.uint32)
    img = nib.nifti1.Nifti1Image(data, affine)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    nib.save(img, filepath)

def save_numpy(
    data: np.ndarray | List[np.ndarray] | Dict[str, np.ndarray],
    filepath: FilePath,
    keys: str | List[str] = 'data',
    ) -> None:
    filepath = resolve_filepath(filepath)
    assert filepath.endswith('.npy') or filepath.endswith('.npz'), "Filepath must end with .npy or .npz"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if filepath.endswith('.npz'):
        keys = arg_to_list(keys, str)
        if isinstance(data, dict):
            np.savez_compressed(filepath, **data)
        else:
            items = data if isinstance(data, list) else [data]
            np.savez_compressed(filepath, **{k: v for k, v in zip(keys, items)})
    else:
        np.save(filepath, data)

def save_transform(
    transform: sitk.Transform,
    filepath: FilePath,
    overwrite: bool = True,
    ) -> None:
    filepath = resolve_filepath(filepath)
    if os.path.exists(filepath) and not overwrite:
        raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sitk.WriteTransform(transform, filepath)

def save_yaml(
    data: Any,
    filepath: FilePath,
    ) -> None:
    filepath = resolve_filepath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f)
