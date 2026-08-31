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
    if len(orientation) != 2:
        raise ValueError(f"Invalid orientation '{orientation}' for dim=2. Must be a 2-character string.")
    n_lr = sum(c in 'LR' for c in orientation)
    n_ap = sum(c in 'AP' for c in orientation)
    n_is = sum(c in 'IS' for c in orientation)
    if n_lr + n_ap + n_is != 2 or any(n == 2 for n in (n_lr, n_ap, n_is)):
        raise ValueError(f"Invalid orientation '{orientation}' for dim=2. Must have one char from each of two different pairs of {{L,R}}, {{A,P}}, {{I,S}}.")

def __assert_orientation_3d(
    orientation: Orientation3D,
    ) -> None:
    if len(orientation) != 3:
        raise ValueError(f"Invalid orientation '{orientation}' for dim=3. Must be a 3-character string.")
    n_lr = sum(c in 'LR' for c in orientation)
    n_ap = sum(c in 'AP' for c in orientation)
    n_is = sum(c in 'IS' for c in orientation)
    if n_lr != 1 or n_ap != 1 or n_is != 1:
        raise ValueError(f"Invalid orientation '{orientation}' for dim=3. Must have exactly one char from each of {{L,R}}, {{A,P}}, {{I,S}}. Got: LR={n_lr}, AP={n_ap}, IS={n_is}.")
    
def assert_shapes_equal(
    *args: List[np.ndarray | torch.Tensor],
    ) -> None:
    shapes = [arg.shape for arg in args]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All arrays must have the same shape. Got shapes {shapes}.")
