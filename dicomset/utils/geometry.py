import numpy as np
import pandas as pd
import scipy
from typing import List

from ..typing import AffineMatrix, Box, Image, LabelImage, Landmarks, Pixel, Point, Points, Size, Spacing, SpatialDim, Voxel
from .landmarks import landmarks_to_points, points_to_landmarks

def affine_origin(
    affine: AffineMatrix,
    ) -> Point:
    # Get origin.
    dim = affine.shape[0] - 1
    if dim == 2:
        origin = (affine[0, 2], affine[1, 2])
    else:
        origin = (affine[0, 3], affine[1, 3], affine[2, 3])

    return origin

def affine_spacing(
    affine: AffineMatrix,
    ) -> Spacing:
    # Get spacing.
    dim = affine.shape[0] - 1
    if dim == 2:
        spacing = (affine[0, 0], affine[1, 1])
    else:
        spacing = (affine[0, 0], affine[1, 1], affine[2, 2])

    return spacing

def assert_box_width(
    box: Box,
    ) -> None:
    dim = box.shape[1]
    for i in range(dim):
        width = box[1, i] - box[0, i]
        if width <= 0:
            raise ValueError(f"Box width must be positive, got '{box}'.")

def centre_of_mass(
    data: Image,
    affine: AffineMatrix | None = None,
    ) -> Point | Pixel | Voxel:
    if data.sum() == 0:
        return None 

    # Compute the centre of mass.
    com = scipy.ndimage.center_of_mass(data)
    if affine is not None:
        com = to_world_coords(com, affine)

    return com

def combine_boxes(
    *boxes: List[Box],
    ) -> Box:
    min = np.stack([box[0] for box in boxes]).min(axis=0)
    max = np.stack([box[1] for box in boxes]).max(axis=0)
    return np.stack([min, max])

def create_affine(
    spacing: Spacing,
    origin: Point,
    ) -> AffineMatrix:
    dim = len(spacing)
    affine = create_eye(dim)
    if dim == 2:
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[0, 2] = origin[0]
        affine[1, 2] = origin[1]
    else:
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        affine[0, 3] = origin[0]
        affine[1, 3] = origin[1]
        affine[2, 3] = origin[2]
    return affine

def create_eye(
    dim: SpatialDim,
    ) -> np.ndarray:
    return np.eye(dim + 1)

def foreground_fov(
    data: LabelImage,
    affine: AffineMatrix | None = None,
    ) -> Box | None:
    if data.sum() == 0:
        return None

    # Get fov of foreground objects.
    non_zero = np.argwhere(data != 0)
    fov_vox = np.stack([
        non_zero.min(axis=0),
        non_zero.max(axis=0),
    ])
    if affine is None:
        return fov_vox

    # Get fov in mm.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    fov_mm = fov_vox * spacing + origin

    return fov_mm

def foreground_fov_centre(
    data: LabelImage,
    affine: AffineMatrix | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel | None:
    fov_d = foreground_fov(data, affine=affine, **kwargs)
    if fov_d is None:
        return None
    fov_c = fov_d.sum(axis=0) / 2
    if affine is None:
        fov_c = np.round(fov_c).astype(np.int32)
        
    return fov_c

def foreground_fov_width(
    data: LabelImage,
    **kwargs,
    ) -> Size | None:
    # Get foreground fov.
    fov_fg = foreground_fov(data, **kwargs)
    if fov_fg is None:
        return None
    min, max = fov_fg
    fov_w = max - min

    return fov_w

def fov(
    size: Size,
    affine: AffineMatrix | None = None,
    ) -> Box:
    # Get fov in voxels.
    n_dims = len(size)
    fov_vox = np.stack([
        np.zeros(n_dims, dtype=np.int32),
        np.array(size, dtype=np.int32),
    ])
    if affine is None:
        return fov_vox

    # Get fov in mm.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    fov_mm = fov_vox * spacing + origin

    return fov_mm

def fov_centre(
    size: Size,
    affine: AffineMatrix | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel:
    # Get FOV.
    fov_d = fov(size, affine=affine, **kwargs)

    # Get FOV centre.
    fov_c = fov_d.sum(axis=0) / 2
    if affine is None:
        fov_c = np.round(fov_c).astype(np.int32)

    return fov_c

def fov_width(
    size: Size,
    affine: AffineMatrix | None = None,
    **kwargs,
    ) -> Size:
    fov_d = fov(size, affine=affine, **kwargs)
    
    # Get width.
    min, max = fov_d
    fov_w = max - min

    return fov_w

def to_image_coords(
    point: Point | Points | Landmarks,
    affine: AffineMatrix,
    ) -> Pixel | Voxel:
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    if isinstance(point, pd.DataFrame):
        landmarks = point
        points = landmarks_to_points(landmarks)
        points = np.round((np.array(points) - origin) / spacing).astype(np.int32)
        points = points_to_landmarks(points, landmarks['landmark-id'])
    else:
        points = np.round((np.array(point) - origin) / spacing).astype(np.int32)
    return points

def to_world_coords(
    point: Point | Point | Landmarks,
    affine: AffineMatrix,
    ) -> Point:
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    if isinstance(point, pd.DataFrame):
        landmarks = point
        points = landmarks_to_points(landmarks)
        points = (points * spacing + origin).astype(np.float32)
        points = points_to_landmarks(points, landmarks['landmark-id'])
    else:
        points = (np.array(point) * spacing + origin).astype(np.float32)
    return points
