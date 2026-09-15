import numpy as np
import pandas as pd
import scipy
import torch
from typing import Callable, List, Tuple

from ..typing import AffineMatrix, BatchChannelImage, BatchImage, BatchLabelImage, Box, Image, LabelImage, Landmark, Landmarks, Orientation, Pixel, Point, Points, Size, Spacing, SpatialDim, Voxel
from .args import alias_kwargs, arg_to_list
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
    dim = affine.shape[0] - 1
    spacing = torch.linalg.norm(affine[:dim, :dim], dim=0)

    if return_type is np.ndarray:
         spacing = to_numpy(spacing)

    return spacing

def affine_subset(
    affine: AffineMatrix,
    dims: List[SpatialDim] | SpatialDim,
    ) -> AffineMatrix:
    affine, return_type = to_tensor(affine, return_type=True)
    dims = arg_to_list(dims, int)

    # Keep rows/cols for the requested dims, plus the final homogeneous row/col.
    dim = affine.shape[0] - 1
    idxs = dims + [dim]
    subset = affine[np.ix_(idxs, idxs)]

    if return_type is np.ndarray:
        subset = to_numpy(subset)

    return subset

def assert_box_width(
    box: Box,
    ) -> None:
    dim = box.shape[1]
    for i in range(dim):
        width = box[1, i] - box[0, i]
        if width <= 0:
            raise ValueError(f"Box width must be positive, got '{box}'.")

@alias_kwargs(
    ('d', 'dim'),
)
def centre_of_mass(
    data: Image | LabelImage | BatchImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel | List[Point | Pixel | Voxel | None] | None:
    return compute_channel_or_spatial_geometry(__spatial_centre_of_mass, data, dim=dim, **kwargs)

def change_image_orientation(
    image: Image,
    old_orientation: Orientation,
    new_orientation: Orientation,
    affine: AffineMatrix | None = None,
    ) -> Tuple[Image, AffineMatrix]:
    dim = len(old_orientation)
    assert_orientation(old_orientation, dim)
    assert_orientation(new_orientation, dim)
    if affine is None:
        affine = create_affine(dim=dim)
    affine = affine.copy()

    # Permute axes by pairing LR, AP, and IS axes. 
    pair = lambda c: 0 if c in 'LR' else (1 if c in 'AP' else 2)
    old_pairs = [pair(c) for c in old_orientation]
    new_pairs = [pair(c) for c in new_orientation]
    perm = [old_pairs.index(p) for p in new_pairs]
    if perm != list(range(dim)):
        image = np.transpose(image, perm)
        affine[:dim, :dim] = affine[:dim, :dim][np.ix_(perm, perm)]     # Permute the rotation/scale part.
        affine[:dim, dim] = affine[perm, dim]                           # Permute the translation part.

    # Flip the data and update the affine translation to preserve the
    # position of the world origin.
    R = affine[:dim, :dim]
    o_i = np.linalg.inv(affine) @ np.append(np.zeros(dim), 1)
    o_i = o_i[:dim]
    o_i_new = o_i.copy()
    for i in range(dim):
        if old_orientation[perm[i]] != new_orientation[i]:
            n = image.shape[i]
            o_i_new[i] = n - o_i[i] - 1
            image = np.flip(image, axis=i).copy()
    affine[:dim, dim] = -R @ o_i_new

    return image, affine

def change_points_orientation(
    image_size: Size,
    points: Point | Points | Landmark | Landmarks,
    old_orientation: Orientation,
    new_orientation: Orientation,
    affine: AffineMatrix | None = None,
    ) -> Tuple[Point | Points | Landmark | Landmarks, AffineMatrix]:
    dim = len(old_orientation)
    assert_orientation(old_orientation, dim)
    assert_orientation(new_orientation, dim)
    image_size = to_numpy(image_size)
    points = to_numpy(points).copy()
    if affine is None:
        affine = create_affine(dim=dim)
    affine = affine.copy()
    is_single = points.ndim == 1
    if is_single:
        points = points[np.newaxis, :]

    # Permute axes by pairing LR, AP, and IS axes. 
    pair = lambda c: 0 if c in 'LR' else (1 if c in 'AP' else 2)
    old_pairs = [pair(c) for c in old_orientation]
    new_pairs = [pair(c) for c in new_orientation]
    perm = [old_pairs.index(p) for p in new_pairs]
    if perm != list(range(dim)):
        points = points[:, perm]
        affine[:dim, :dim] = affine[:dim, :dim][np.ix_(perm, perm)]     # Permute the rotation/scale part.
        affine[:dim, dim] = affine[perm, dim]                           # Permute the translation part.

    # Map points to old image coordinates.
    points = to_image_coords(points, affine)

    # Flip the data and update the affine translation to preserve the
    # position of the world origin.
    R = affine[:dim, :dim]
    o_i = np.linalg.inv(affine) @ np.append(np.zeros(dim), 1)
    o_i = o_i[:dim]
    o_i_new = o_i.copy()
    for i in range(dim):
        if old_orientation[perm[i]] != new_orientation[i]:
            n = image_size[i]
            o_i_new[i] = n - o_i[i] - 1
    affine[:dim, dim] = -R @ o_i_new

    # Map points to new image coords then new world coords.
    points = image_size - points - 1
    points = to_world_coords(points, affine)
    
    if is_single:
        points = points[0]

    return points, affine

def combine_boxes(
    *boxes: List[Box],
    ) -> Box:
    min = np.stack([box[0] for box in boxes]).min(axis=0)
    max = np.stack([box[1] for box in boxes]).max(axis=0)
    return np.stack([min, max])

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

@alias_kwargs(
    ('d', 'dim'),
)
def foreground_fov(
    data: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> Box | List[Box | None] | None:
    return compute_channel_or_spatial_geometry(__spatial_foreground_fov, data, dim=dim, **kwargs)

@alias_kwargs(
    ('d', 'dim'),
)
def foreground_fov_centre(
    data: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel | List[Point | Pixel | Voxel | None] | None:
    return compute_channel_or_spatial_geometry(__spatial_foreground_fov_centre, data, dim=dim, **kwargs)

@alias_kwargs(
    ('d', 'dim'),
)
def foreground_fov_width(
    data: LabelImage | BatchLabelImage,
    dim: SpatialDim | None = None,
    **kwargs,
    ) -> Size | List[Size | None] | None:
    return compute_channel_or_spatial_geometry(__spatial_foreground_fov_width, data, dim=dim, **kwargs)

def fov(
    size: Size,
    **kwargs,
    ) -> Box:
    return __spatial_fov(size, **kwargs)

def fov_centre(
    size: Size,
    **kwargs,
    ) -> Point | Pixel | Voxel:
    return __spatial_fov_centre(size, **kwargs)

# Transforms every corner of a voxel-space box through 'affine' and takes the axis-aligned
# bounds of the transformed corners - handles rotated affines correctly.
def fov_width(
    size: Size,
    **kwargs,
    ) -> Size:
    return __spatial_fov_width(size, **kwargs)

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_centre_of_mass(
    data: Image | LabelImage,
    affine: AffineMatrix | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel | None:
    if data.sum() == 0:
        return None

    # Compute the centre of mass.
    com = scipy.ndimage.center_of_mass(data)
    if affine is not None:
        com = to_world_coords(com, affine)

    return to_tuple(com) 

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_foreground_fov(
    data: LabelImage,
    affine: AffineMatrix | None = None,
    ) -> Box | None:
    if data.sum() == 0:
        return None

    data, return_type = to_tensor(data, return_type=True)
    affine = to_tensor(affine, device=data.device) if affine is not None else None

    # Get fov of foreground objects, in voxels.
    non_zero = torch.argwhere(data != 0)
    fov_vox = torch.stack([
        non_zero.min(dim=0).values,
        non_zero.max(dim=0).values,
    ]).type(torch.float32)

    # Get fov in mm - transforms all box corners to handle rotated affines.
    fov_d = __transform_box(fov_vox, affine) if affine is not None else fov_vox

    if return_type is np.ndarray:
        fov_d = to_numpy(fov_d)

    return fov_d

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_foreground_fov_centre(
    data: LabelImage,
    affine: AffineMatrix | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel | None:
    data, return_type = to_tensor(data, return_type=True)
    affine = to_tensor(affine, device=data.device) if affine is not None else None

    fov_d = foreground_fov(data, affine=affine, **kwargs)
    if fov_d is None:
        return None
    fov_d = to_tensor(fov_d, device=data.device)
    fov_c = fov_d.sum(axis=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)

    if return_type is np.ndarray:
        fov_c = to_numpy(fov_c)

    return fov_c

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_foreground_fov_width(
    data: LabelImage,
    **kwargs,
    ) -> Size | None:
    fov_fg = foreground_fov(data, **kwargs)
    if fov_fg is None:
        return None
    min, max = fov_fg
    fov_w = max - min + 1

    return fov_w

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_fov(
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
    ], dim=0).type(torch.float32)
    if affine is None:
        if return_type is np.ndarray:
            fov_vox = to_numpy(fov_vox)
        return fov_vox

    # Get fov in mm - transforms all box corners to handle rotated affines.
    fov_mm = __transform_box(fov_vox, affine)

    if return_type is np.ndarray:
         fov_mm = to_numpy(fov_mm)

    return fov_mm

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_fov_centre(
    size: Size,
    affine: AffineMatrix | None = None,
    **kwargs,
    ) -> Point | Pixel | Voxel:
    size, return_type = to_tensor(size, return_type=True)
    affine = to_tensor(affine, device=size.device) if affine is not None else None

    # Get FOV.
    fov_d = fov(size, affine=affine, **kwargs)
    fov_d = to_tensor(fov_d, device=size.device)

    # Get FOV centre.
    fov_c = fov_d.sum(axis=0) / 2
    if affine is None:
        fov_c = torch.round(fov_c).type(torch.int32)

    if return_type is np.ndarray:
        fov_c = to_numpy(fov_c)

    return fov_c

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_fov_width(
    size: Size,
    **kwargs,
    ) -> Size:
    fov_d = fov(size, **kwargs)

    # Get width.
    min, max = fov_d
    fov_w = max - min

    return fov_w

@alias_kwargs(
    ('a', 'affine'),
)
def to_image_coords(
    points: Point | Points | Landmark | Landmarks,
    affine: AffineMatrix,
    truncate: bool = True,
    ) -> Pixel | Voxel | Landmark | Landmarks:
    def __to_image_coords(points, affine):
        inv_affine = np.linalg.inv(affine)
        is_single = points.ndim == 1
        if is_single:
            points = points[None]
        dim = affine.shape[0] - 1
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        result = (inv_affine @ points_h.T).T[:, :dim]
        if truncate:
            result = np.round(result).astype(np.int32)
        return result[0] if is_single else result

    if isinstance(points, (pd.DataFrame, pd.Series)):
        landmark_ids = points['landmark-id'] if isinstance(points, pd.DataFrame) else points.index[0]
        points = landmarks_to_points(points)
        points = __to_image_coords(points, affine)
        points = points_to_landmarks(points, landmark_ids)
    else:
        points = to_numpy(points)
        points = __to_image_coords(points, affine)
    return points

# Note! This changes the underlying world coordinate system.
# This method changes the mapping from image -> world coordinates whilst
# preserving the position of the origin in world coordinates.
# E.g. if changing from L+ to R+, the origin will remain at the same place
# on the patient (left upper lobe for example), but world coordinates will
# increase to patient right instead of left.
@alias_kwargs(
    ('a', 'affine'),
)
def to_world_coords(
    points: Point | Points | Landmark | Landmarks,
    affine: AffineMatrix,
    ) -> Point:
    def __to_world_coords(points, affine):
        is_single = points.ndim == 1
        if is_single:
            points = points[None]
        dim = affine.shape[0] - 1
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        result = ((affine @ points_h.T).T[:, :dim]).astype(np.float32)
        return result[0] if is_single else result

    if isinstance(points, (pd.DataFrame, pd.Series)):
        landmark_ids = points['landmark-id'] if isinstance(points, pd.DataFrame) else points.index[0]
        points = landmarks_to_points(points)
        points = __to_world_coords(points, affine)
        points = points_to_landmarks(points, landmark_ids)
    else:
        points = to_numpy(points)
        points = __to_world_coords(points, affine)
    return points

def __transform_box(
    box: Box,
    affine: AffineMatrix,
    ) -> Box:
    dim = box.shape[1]
    corners = torch.cartesian_prod(*[box[:, i] for i in range(dim)])
    corners = to_tensor(to_world_coords(to_numpy(corners), to_numpy(affine)), device=box.device)
    return torch.stack([corners.min(dim=0).values, corners.max(dim=0).values])
