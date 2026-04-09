from typing import Callable, List

from ...typing import Image, LabelImage, SpatialDim
from ...utils.logging import logger

# Allows us to compute metrics on image batches by applying metric to 
# the underlying spatial volumes.

# This doesn't allow for vectorised metric calcs - do metrics need to
# be more efficient?
def compute_spatial_metrics(
    spatial_metric_fn: Callable,
    # E.g. List[a, b] for dice or List[a] for centre of mass. 
    *data: List[Image] | List[LabelImage], 
    dim: SpatialDim | None = None, 
    **kwargs,
    ) -> float | List[float]:
    if data[0].ndim == 2:    # 2D image.
        return spatial_metric_fn(*data, **kwargs) 
    elif data[0].ndim == 3:  # 2D batch or 3D image.
        # Could be 3D label or batch of 2D labels - assume 3D. 
        if dim is None or dim == 3: # 3D image.
            logger.warn(f"Metric function '{spatial_metric_fn.__name__}' received 3D arrays with no specified 'dim'. Assuming 3D labels. If these are batches of 2D labels, specify 'dim=2' to compute metric per image in batch.") 
            return spatial_metric_fn(*data, **kwargs)
        elif dim == 2:                       # Batch of 2D images.
            # Split data into lists.
            interleaved_data = list(zip(*data)) 
            return [spatial_metric_fn(*d, **kwargs) for d in interleaved_data]
    elif data[0].ndim == 4:  # Batch of 3D images.
        # Split data into lists.
        interleaved_data = list(zip(*data)) 
        return [spatial_metric_fn(*d, **kwargs) for d in interleaved_data]
    else:
        raise ValueError(f"Metric function '{spatial_metric_fn.__name__}' expects arrays of spatial dimension 2 or 3, with optional batch dimension. Got arrays of shape '{data[0].shape}' with inferred spatial dimension {data[0].ndim}. Specify 'dim' to override inference.")        
