from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .resampling import resample
    from .transpose import spatial_transpose

__all__ = ['resample', 'spatial_transpose']

RESAMPLE_IMPORTS = ['resample']
TRANSPOSE_IMPORTS = ['spatial_transpose']

def __getattr__(name):
    if name in RESAMPLE_IMPORTS:
        from . import resampling
        return getattr(resampling, name)
    if name in TRANSPOSE_IMPORTS:
        from . import transpose
        return getattr(transpose, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
