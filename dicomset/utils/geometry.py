import numpy as np
import pandas as pd
import scipy
import torch
from typing import Callable, List

from ..typing import AffineMatrix, BatchChannelImage, BatchImage, BatchLabelImage, Box, Image, LabelImage, Landmarks, Orientation, Pixel, Point, Points, Size, Spacing, SpatialDim, Voxel
from .assertions import assert_orientation
from .conversion import to_numpy, to_tensor, to_tuple
from .landmarks import landmarks_to_points, points_to_landmarks
from .logging import logger

def affine_origin(
    affine: AffineMatrix,
    ) -> Point:
    affine, return_type = to_tensor(affine, return_type=True)

    # Get origin.
    dim = affine.shape[0] - 1
    if dim == 2:
        origin = to_tensor([affine[0, 2], affine[1, 2]], device=affine.device)
    else:
        origin = to_tensor([affine[0, 3], affine[1, 3], affine[2, 3]], device=affine.device)

    if return_type is np.ndarray:
         origin = to_numpy(origin)

    return origin

def affine_spacing(
    affine: AffineMatrix,
    ) -> Spacing:
    affine, return_type = to_tensor(affine, return_type=True)

    # Get spacing.
    dim = affine.shape[0] - 1
    if dim == 2:
        spacing = to_tensor([affine[0, 0], affine[1, 1]], device=affine.device)
    else:
        spacing = to_tensor([affine[0, 0], affine[1, 1], affine[2, 2]], device=affine.device)

    if return_type is np.ndarray:
         spacing = to_numpy(spacing)

    return spacing

def assert_box_width(
    box: Box,
    ) -> None:
    dim = box.shape[1]
    for i in range(dim):
        width = box[1, i] - box[0, i]
        if width <= 0:
            raise ValueError(f"Box width must be positive, got '{box}'.")

def __spatial_centre_of_mass(
    data: Image | LabelImage,
    affine: AffineMatrix | None = None,
    ) -> Point | Pixel | Voxel | None:
    if data.sum() == 0:
        return None

    # Compute the centre of mass.
    com = to_tuple(scipy.ndimage.center_of_mass(data))
    if affine is not None:
        com = to_world_coords(com, affine)

    return com

def compute_channel_or_spatial_geometry(
    geometry_fn: Callable,
    data: Image | BatchImage | BatchChannelImage,
    *args,
    combine_channels: bool = False,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> Box | Point | Pixel | Size | Voxel | List[Box | Point | Pixel | Size | Voxel | None] | None:
    if data.ndim == 2:    # 2D image.
        return geometry_fn(data, *args, **kwargs)
    elif data.ndim == 3:  # 2D batch or 3D image.
        if dim is None or dim == 3:  # 3D image.
            if dim is None:
                logger.warn(f"Geometry function '{geometry_fn.__name__}' received 3D array with no specified 'dim'. Assuming 3D image. If these are batches of 2D images, specify 'dim=2' to compute per image in batch.")
            return geometry_fn(data, *args, **kwargs)
        elif dim == 2:    # Batch of 2D images.
            if combine_channels:
                return geometry_fn(data, *args, **kwargs)
            else:
                return [geometry_fn(d, *args, **kwargs) for d in data]
    elif data.ndim == 4:  # 2D batch/channel or 3D batch.
        if dim is None or dim == 3:  # 3D batch.
            if dim is None:
                logger.warn(f"Geometry function '{geometry_fn.__name__}' received 4D array with no specified 'dim'. Assuming batch of 3D images. If these are batch/channels of 2D images, specify 'dim=2' to compute per image in batch.")
            if combine_channels:
                return geometry_fn(data, *args, **kwargs)
            else:
                return [geometry_fn(d, *args, **kwargs) for d in data]
        elif dim == 2:    # 2D batch/channel.
            results = []
            for b in data:
                if combine_channels:
                    results.append(geometry_fn(b, *args, **kwargs))
                else:
                    results.append([geometry_fn(c, *args, **kwargs) for c in b])
            return results
    elif data.ndim == 5:  # 3D batch/channel.
        results = []
        for b in data:
            if combine_channels:
                results.append(geometry_fn(b, *args, **kwargs))
                continue
            results.append([geometry_fn(c, *args, **kwargs) for c in b])
        return results
    else:
        raise ValueError(f"Geometry function '{geometry_fn.__name__}' expects array of spatial dimension 2 or 3, with optional batch dimension. Got array of shape '{data.shape}' with inferred spatial dimension {data.ndim}. Specify 'dim' to override inference.")

def centre_of_mass(
    data: Image | LabelImage | BatchImage | BatchLabelImage,
    affine: AffineMatrix | None = None,
    dim: SpatialDim | None = None,
    ) -> Point | Pixel | Voxel | List[Point | Pixel | Voxel | None] | None:
    return compute_channel_or_spatial_geometry(__spatial_centre_of_mass, data, affine=affine, dim=dim)

def change_orientation(
    affine: AffineMatrix,
    old_orientation: Orientation,
    new_orientation: Orientation,
    ) -> AffineMatrix:
    dim = affine.shape[0] - 1
    assert_orientation(old_orientation, dim)
    assert_orientation(new_orientation, dim)
    flip_axes = [o.lower() != n.lower() for o, n in zip(old_orientation, new_orientation)]
    affine = affine.copy()
    for a, flip in enumerate(flip_axes):
        if flip:
            affine[a, :] *= -1
    return affine

def combine_boxes(
    *boxes: List[Box],
    ) -> Box:
    min = np.stack([box[0] for box in boxes]).min(axis=0)
    max = np.stack([box[1] for box in boxes]).max(axis=0)
    return np.stack([min, max])

def create_affine(
    spacing: Spacing | None = None,
    origin: Point | None = None,
    dim: SpatialDim | None = None,
    ) -> AffineMatrix:
    # Resolve dim.
    if dim is None:
        if spacing is not None:
            dim = len(spacing)
        elif origin is not None:
            dim = len(origin)
        else:
            raise ValueError("Must provide 'dim' if 'spacing' and 'origin' are not provided.")
    if spacing is None:
        spacing = np.ones(dim)
    if origin is None:
        origin = np.zeros(dim)
    affine = np.eye(dim + 1)
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

def foreground_fov(
    data: LabelImage,
    affine: AffineMatrix | None = None,
    ) -> Box | None:
    if data.sum() == 0:
        return None

    data, return_type = to_tensor(data, return_type=True)
    affine = to_tensor(affine, device=data.device) if affine is not None else None

    # Get fov of foreground objects.
    non_zero = torch.argwhere(data != 0)
    fov = torch.stack([
        non_zero.min(dim=0).values,
        non_zero.max(dim=0).values,
    ])

    # Get fov in mm.
    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        fov = fov * spacing + origin

    if return_type is np.ndarray:
        fov = to_numpy(fov)

    return fov

def foreground_fov_centre(
    data: LabelImage,
    affine: AffineMatrix | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel | None:
    data, return_type = to_tensor(data, return_type=True)
    affine = to_tensor(affine, device=data.device) if affine is not None else None

    fov_d = foreground_fov(data, affine=affine, **kwargs)
    if fov_d is None:
        return None
    fov_c = fov_d.sum(axis=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)

    if return_type is np.ndarray:
        fov_c = to_numpy(fov_c)
        
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
    fov_w = max - min + 1

    return fov_w

def fov(
    size: Size,
    affine: AffineMatrix | None = None,
    ) -> Box:
    size, return_type = to_tensor(size, return_type=True)
    affine = to_tensor(affine, device=size.device) if affine is not None else None

    # Get fov in voxels.
    n_dims = len(size)
    fov_vox = torch.stack([
        torch.zeros(n_dims, dtype=torch.int32),
        size - 1,
    ], dim=0)
    if affine is None:
        return fov_vox

    # Get fov in mm.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    fov_mm = fov_vox * spacing + origin

    if return_type is np.ndarray:
         fov_mm = to_numpy(fov_mm)

    return fov_mm

def fov_centre(
    size: Size,
    affine: AffineMatrix | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel:
    size, return_type = to_tensor(size, return_type=True)
    affine = to_tensor(affine, device=size.device) if affine is not None else None

    # Get FOV.
    fov_d = fov(size, affine=affine, **kwargs)

    # Get FOV centre.
    fov_c = fov_d.sum(axis=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)

    if return_type is np.ndarray:
        fov_c = to_numpy(fov_c)

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


