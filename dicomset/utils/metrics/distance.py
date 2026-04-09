import numpy as np
from surface_distance import compute_average_surface_distance, compute_robust_hausdorff, compute_surface_dice_at_tolerance, compute_surface_distances
from typing import Dict, List

from ...typing import AffineMatrix, BatchLabelImage, LabelImage, Number, SpatialDim
from ...utils.args import arg_to_list
from ...utils.geometry import affine_spacing
from ..python import delegates_to
from .shared import compute_spatial_metrics

def __spatial_distances(
    a: LabelImage,
    b: LabelImage, 
    affine: AffineMatrix | None = None,    
    tols: Number | List[Number] | None = None, 
    ) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'distances' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(f"Metric 'distances' can't be calculated on labels.")

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

@delegates_to(__spatial_distances)
def distances(
    a: LabelImage | BatchLabelImage,
    b: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    ) -> Dict[str, float] | List[Dict[str, float]]:
    return compute_spatial_metrics(__spatial_distances, a, b, dim=dim)
