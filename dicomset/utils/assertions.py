import numpy as np
import torch
from typing import List

from ..typing import Orientation, Orientation2D, Orientation3D, SpatialDim

def assert_orientation(
    orientation: Orientation,
    dim: SpatialDim,
    ) -> None:
    if dim == 2:
        __assert_orientation_2d(orientation)
    elif dim == 3:
        __assert_orientation_3d(orientation)

def __assert_orientation_2d(
    orientation: Orientation2D,
    ) -> None:
    orientations = {'LI', 'LS', 'RI', 'RS'}
    if orientation not in orientations:
        raise ValueError(f"Invalid orientation '{orientation}' for dim=2. Must be one of {orientations}.")

def __assert_orientation_3d(
    orientation: Orientation3D,
    ) -> None:
    orientations = {'LAI', 'LAS', 'LPI', 'LPS', 'RAI', 'RAS', 'RPI', 'RPS'}
    if orientation not in orientations:
        raise ValueError(f"Invalid orientation '{orientation}' for dim=3. Must be one of {orientations}.")
    
def assert_shapes_equal(
    *args: List[np.ndarray | torch.Tensor],
    ) -> None:
    shapes = [arg.shape for arg in args]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All arrays must have the same shape. Got shapes {shapes}.")
