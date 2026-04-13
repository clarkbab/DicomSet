import numpy as np
from typing import Callable

from ...typing import BatchImage, Image, SpatialDim
from ..logging import logger

def compute_spatial_transforms(
    spatial_transform_fn: Callable,
    data: BatchImage | Image,
    dim: SpatialDim | None = None, 
    **kwargs,
    ) -> BatchImage | Image:
    if data.ndim == 2:    # 2D image.
        return spatial_transform_fn(data, **kwargs) 
    elif data.ndim == 3:  # 2D batch or 3D image.
        # Could be 3D label or batch of 2D labels - assume 3D. 
        if dim is None or dim == 3: # 3D image.
            logger.warn(f"Transform function '{spatial_transform_fn.__name__}' received 3D array with no specified 'dim'. Assuming 3D labels. If these are batches of 2D labels, specify 'dim=2' to compute transform per image in batch.") 
            return spatial_transform_fn(data, **kwargs)
        elif dim == 2:                       # Batch of 2D images.
            results = []
            for d in data:
                results.append(spatial_transform_fn(d, **kwargs)) 
            return np.stack(results, axis=0)
    elif data.ndim == 4:  # Batch of 3D images.
        results = []
        for d in data:
            results.append(spatial_transform_fn(d, **kwargs))
        return np.stack(results, axis=0) 
    else:
        raise ValueError(f"Transform function '{spatial_transform_fn.__name__}' expects array of spatial dimension 2 or 3, with optional batch dimension. Got array of shape '{data.shape}' with inferred spatial dimension {data.ndim}. Specify 'dim' to override inference.")        
