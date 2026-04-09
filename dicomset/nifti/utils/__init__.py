from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .create import create_ct, create_dataset, create_index, create_region, create_registration_moved_image, create_registration_moved_landmarks, create_registration_moved_regions, create_registration_transform
    from .load import dataset_exists, load_ct, load_dataset, load_index, load_region, load_registration_moved_image, load_registration_moved_landmarks, load_registration_moved_regions, load_registration_transform
    from .rename import rename_patients

__all__ = [
    'create_ct', 'create_dataset', 'create_index', 'create_region',
    'create_registration_moved_image', 'create_registration_moved_landmarks',
    'create_registration_moved_regions', 'create_registration_transform',
    'dataset_exists', 'list_datasets', 'load_ct', 'load_dataset', 'load_index', 'load_region',
    'load_registration_moved_image', 'load_registration_moved_landmarks',
    'load_registration_moved_regions', 'load_registration_transform',
    'rename_patients',
]

CREATE_IMPORTS = [
    'create_ct', 'create_dataset', 'create_index', 'create_region',
    'create_registration_moved_image', 'create_registration_moved_landmarks',
    'create_registration_moved_regions', 'create_registration_transform',
]
LOAD_IMPORTS = [
    'dataset_exists', 'list_datasets', 'load_ct', 'load_dataset', 'load_index', 'load_region',
    'load_registration_moved_image', 'load_registration_moved_landmarks',
    'load_registration_moved_regions', 'load_registration_transform',
]
RENAME_IMPORTS = ['rename_patients']

def __getattr__(name):
    if name in CREATE_IMPORTS:
        from . import create
        return getattr(create, name)
    if name in LOAD_IMPORTS:
        from . import load
        return getattr(load, name)
    if name in RENAME_IMPORTS:
        from . import rename
        return getattr(rename, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
