from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

LAZY_IMPORTS = {
    'config': ['config_data'],
    'dicom': ['DicomDataset', 'DicomPatient', 'DicomSeries', 'DicomStudy'],
    'nifti': ['NiftiDataset', 'NiftiPatient', 'NiftiSeries', 'NiftiStudy'],
    # 'utils': ['load', 'list'],
    'utils': [('load_dataset', 'load'), ('list_datasets', 'list')]
}


# Support tuple-based aliasing in LAZY_IMPORTS
__all__ = []
for attrs in LAZY_IMPORTS.values():
    for attr in attrs:
        if isinstance(attr, tuple):
            __all__.append(attr[1])  # alias
        else:
            __all__.append(attr)

if TYPE_CHECKING:
    for module, attrs in LAZY_IMPORTS.items():
        for attr in attrs:
            exec(f"from .{module} import {attr}")


def __getattr__(name):
    for module, attrs in LAZY_IMPORTS.items():
        for attr in attrs:
            if isinstance(attr, tuple):
                real_name, alias = attr
                if name == alias:
                    mod = importlib.import_module(f"{__name__}.{module}")
                    return getattr(mod, real_name)
            else:
                if name == attr:
                    mod = importlib.import_module(f"{__name__}.{module}")
                    return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
