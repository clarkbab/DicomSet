from __future__ import annotations


import importlib
from typing import TYPE_CHECKING

LAZY_IMPORTS = {
    'create': [
        'create_ct', 'create_dataset', 'create_index', 'create_region',
        'create_registered_image', 'create_registered_landmarks',
        'create_registered_regions', 'create_registration_transform',
    ],
    'load': [
        'dataset_exists', 'list_datasets', 'load_ct', 'load_dataset', 'load_index', 'load_region',
        'load_registered_image', 'load_registered_landmarks',
        'load_registered_regions', 'load_registration_transform',
    ],
    'rename': ['rename_patients'],
}

__all__ = [attr for attrs in LAZY_IMPORTS.values() for attr in attrs]

if TYPE_CHECKING:
    for module, attrs in LAZY_IMPORTS.items():
        for attr in attrs:
            exec(f"from .{module} import {attr}")

def __getattr__(name):
    for module, attrs in LAZY_IMPORTS.items():
        if name in attrs:
            return getattr(importlib.import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"Module {__name__} has no attribute {name}")
