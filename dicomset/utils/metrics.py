import numpy as np
import SimpleITK as sitk
from surface_distance import compute_average_surface_distance, compute_robust_hausdorff, compute_surface_dice_at_tolerance, compute_surface_distances
from typing import Callable, Dict, List

from ..typing import AffineMatrix, BatchImage, BatchLabelImage, Image, LabelImage, Number, SpatialDim
from .args import arg_to_list, bubble_args
from .conversion import to_numpy
from .geometry import affine_spacing, centre_of_mass
from .logging import logger

# Allows us to compute metrics on image batches by applying metric to 
# the underlying spatial volumes.

# This doesn't allow for vectorised metric calcs - do metrics need to
# be more efficient?
def __spatial_centroid_error(
    a: LabelImage,
    b: LabelImage,
    affine: AffineMatrix | None = None,
    ) -> float:
    a = to_numpy(a, dtype=bool)
    b = to_numpy(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Metric 'centroid_error' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'centroid_error' can't be calculated on empty labels.")

    # Compute centroids - these are the same as centre-of-mass for label images.
    a_centroid = centre_of_mass(a, affine=affine)    
    b_centroid = centre_of_mass(b, affine=affine)

    # Compute error.
    error = np.linalg.norm(np.array(b_centroid) - np.array(a_centroid))

    return error

def __spatial_dice(
    a: LabelImage,
    b: LabelImage,
    ) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'dice' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")

    # 'SimpleITK' filter doesn't handle empty a/b.
    if a.sum() == 0 and b.sum() == 0:
        return 1.0

    # Convert types for SimpleITK.
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    a = sitk.GetImageFromArray(a)
    b = sitk.GetImageFromArray(b)
    filter = sitk.LabelOverlapMeasuresImageFilter()
    filter.Execute(a, b)
    dice = filter.GetDiceCoefficient()
    return dice

def __spatial_distances(
    a: LabelImage,
    b: LabelImage, 
    affine: AffineMatrix | None = None,    
    tols: Number | List[Number] | None = None, 
    ) -> Dict[str, float]:
    a = to_numpy(a, dtype=bool)
    b = to_numpy(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Metric 'distances' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'distances' can't be calculated on empty labels.")

    # Calculate surface distances.
    if affine is not None:
        spacing = affine_spacing(affine)
    else:
        spacing = (1,) * a.ndim
    surf_dists = compute_surface_distances(a, b, spacing) 

    # Compute metrics.
    metrics = {
        'hd': compute_robust_hausdorff(surf_dists, 100),
        'hd-95': compute_robust_hausdorff(surf_dists, 95),
        'msd': np.mean(compute_average_surface_distance(surf_dists)),
    }
    if tols is not None:
        tols = arg_to_list(tols, (int, float))
        for t in tols:
            metrics[f'surface-dice-{t}'] = compute_surface_dice_at_tolerance(surf_dists, t)

    return metrics

def __spatial_ncc(
    a: np.ndarray,
    b: np.ndarray,
    ) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'ncc' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if (a.dtype != np.float32 and a.dtype != np.float64) or (b.dtype != np.float32 and b.dtype != np.float64):
        raise ValueError(f"Metric 'ncc' expects float32/64 arrays. Got '{a.dtype}' and '{b.dtype}'.")

    # Calculate normalised cross-correlation.
    norm_a = (a - np.mean(a)) / np.std(a)
    norm_b = (b - np.mean(b)) / np.std(b)
    result = (1 / a.size) * np.sum(norm_a * norm_b)

    return result

@bubble_args(__spatial_centroid_error)
def centroid_error(
    a: LabelImage | BatchLabelImage,
    b: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> float | List[float]:
    return compute_channel_or_spatial_metrics(__spatial_centroid_error, a, b, dim=dim, **kwargs)

def compute_channel_or_spatial_metrics(
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

@bubble_args(__spatial_dice)
def dice(
    a: LabelImage | BatchLabelImage,
    b: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    ) -> float | List[float]:
    return compute_channel_or_spatial_metrics(__spatial_dice, a, b, dim=dim)

@bubble_args(__spatial_distances)
def distances(
    a: LabelImage | BatchLabelImage,
    b: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> Dict[str, float] | List[Dict[str, float]]:
    return compute_channel_or_spatial_metrics(__spatial_distances, a, b, dim=dim, **kwargs)

@bubble_args(__spatial_ncc)
def ncc(
    a: Image | BatchImage,
    b: Image | BatchImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> float | List[float]:
    return compute_channel_or_spatial_metrics(__spatial_ncc, a, b, dim=dim, **kwargs)
