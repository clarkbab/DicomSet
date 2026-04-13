from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .distance import centroid_error, distances
    from .overlap import dice

__all__ = ['centroid_error', 'distances', 'dice']

DISTANCE_IMPORTS = ['centroid_error', 'distances']
OVERLAP_IMPORTS = ['dice']

def __getattr__(name):
    if name in DISTANCE_IMPORTS:
        from . import distance
        return getattr(distance, name)
    if name in OVERLAP_IMPORTS:
        from . import overlap
        return getattr(overlap, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
