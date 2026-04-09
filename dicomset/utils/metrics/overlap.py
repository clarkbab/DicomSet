import numpy as np
import SimpleITK as sitk
from typing import List

from ...typing import BatchLabelImage, LabelImage, SpatialDim
from ...utils.python import delegates_to
from .shared import compute_spatial_metrics

def __spatial_dice(
    a: LabelImage,
    b: LabelImage,
    ) -> float | List[float]:
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

@delegates_to(__spatial_dice)
def dice(
    a: LabelImage | BatchLabelImage,
    b: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    ) -> float | List[float]:
    return compute_spatial_metrics(__spatial_dice, a, b, dim=dim)
