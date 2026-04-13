import numpy as np
from surface_distance import compute_average_surface_distance, compute_robust_hausdorff, compute_surface_dice_at_tolerance, compute_surface_distances
from typing import Dict, List

from ...typing import AffineMatrix, BatchLabelImage, LabelImage, Number, SpatialDim
from ...utils.args import arg_to_list
from ...utils.conversion import to_numpy
from ...utils.geometry import affine_spacing, centre_of_mass
from ..python import bubble_args
from .shared import compute_spatial_metrics

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

# These are grouped because they all require surface distances calc
# which is fairly expensive.
@bubble_args(__spatial_centroid_error)
def centroid_error(
    a: LabelImage | BatchLabelImage,
    b: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> float | List[float]:
    return compute_spatial_metrics(__spatial_centroid_error, a, b, dim=dim, **kwargs)

@bubble_args(__spatial_distances)
def distances(
    a: LabelImage | BatchLabelImage,
    b: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> Dict[str, float] | List[Dict[str, float]]:
    return compute_spatial_metrics(__spatial_distances, a, b, dim=dim, **kwargs)
