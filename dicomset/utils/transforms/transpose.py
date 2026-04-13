import numpy as np

from ...typing import BatchChannelImage, BatchImage, Image, SpatialDim

# Transposes spatial coordinates, whilst maintaining initial batch/channel dimensions.
def spatial_transpose(
    data: BatchChannelImage | BatchImage | Image,
    dim: SpatialDim = 3,
    ) -> BatchChannelImage | BatchImage | Image:
    from_axes = tuple(range(data.ndim - dim, data.ndim))     
    to_axes = tuple(reversed(from_axes)) 
    return np.moveaxis(data, from_axes, to_axes)
