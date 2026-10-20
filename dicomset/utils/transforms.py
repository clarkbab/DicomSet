from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import skimage as ski
import torch
from typing import Callable, List, Literal, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    import SimpleITK as sitk

from ..typing import AffineMatrix, BatchChannelImage, BatchChannelLabelImage, BatchImage, BatchLabelImage, BatchLabelMap, Box, ChannelImage, Image, LabelImage, LabelMap, Landmark, Landmarks, Number, Point, Points, Size, Spacing, SpatialDim
from .args import alias_kwargs, arg_to_list, bubble_args
from .conversion import to_numpy, to_tensor
from .geometry import affine_direction, affine_origin, affine_spacing, assert_box_width, create_affine, fov, to_image_coords, to_world_coords
from .landmarks import landmarks_to_points, replace_points
from .logging import logger
if TYPE_CHECKING:
    from ..dicom import DicomSeries
    from ..nifti import NiftiImageSeries

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_centre_crop(
    data: Image,
    size: Size,
    affine: AffineMatrix | None = None,
    ) -> Image:
    # Convert size from world coords (mm) to voxels if affine is provided.
    if affine is not None:
        spacing = affine_spacing(affine)
        size_vox = np.round(np.array(size) / spacing).astype(int)
    else:
        size_vox = np.array(size)

    # Calculate the crop box.
    to_crop = np.array(data.shape) - size_vox
    crop_start = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    crop_end = crop_start + size_vox
    crop_box = np.stack([crop_start, crop_end])

    return __spatial_crop(data, crop_box)

def __box_slices(
    box: Box,
    shape: Size,
    affine: AffineMatrix | None = None,
    ) -> Tuple[slice, ...]:
    box = __resolve_box(box, shape, affine=affine)
    assert_box_width(box)

    # Convert box to voxel coordinates.
    if affine is not None:
        box = to_image_coords(box, affine=affine)
        # A flipped axis (negative affine direction) swaps the min/max corners.
        box = np.sort(box, axis=0)

    # Clip the box to the image bounds.
    size = np.array(shape)
    box_min = np.array(box[0]).clip(0)
    box_max = (size - np.array(box[1])).clip(0)
    return tuple(slice(int(mn), int(s - mx)) for mn, mx, s in zip(box_min, box_max, size))

# Can be applied to spatial or channel image.
@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_crop(
    data: Image,
    crop_box: Box,
    affine: AffineMatrix | None = None,
    ) -> Image:
    data, return_type = to_tensor(data, return_type=True)
    data = data[__box_slices(crop_box, data.shape, affine=affine)]

    if return_type is np.ndarray:
        data = to_numpy(data)
    return data

# Can be applied to spatial or channel image.
@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_crop_or_pad(
    data: Image,
    crop_box: Box,
    affine: AffineMatrix | None = None,
    fill: float | Literal['max', 'mean', 'median', 'min'] = 'min',
    ) -> Image:
    data, return_type = to_tensor(data, return_type=True)
    crop_box = __resolve_box(crop_box, data.shape, affine=affine)
    assert_box_width(crop_box)
    fill = __resolve_fill(data, fill)

    # Convert box to voxel coordinates.
    if affine is not None:
        crop_box = to_image_coords(crop_box, affine=affine)

    # Perform padding - clip if padding is less than zero.
    size = np.array(data.shape)
    pad_min = (-np.array(crop_box[0])).clip(0)
    pad_max = (np.array(crop_box[1]) - size).clip(0)
    padding = list(reversed(list(zip(pad_min, pad_max))))  # torch 'pad' operates from back to front.
    padding = tuple(torch.tensor(padding).flatten().tolist())
    data = torch.nn.functional.pad(data, padding, value=fill)

    # Perform cropping.
    size = np.array(data.shape)
    crop_min = np.array(crop_box[0]).clip(0)
    crop_max = (size - (np.array(crop_box[1]) + pad_min)).clip(0)
    slices = tuple(slice(int(mn), int(s - mx)) for mn, mx, s in zip(crop_min, crop_max, size))
    data = data[slices]

    if return_type is np.ndarray:
        data = to_numpy(data)
    return data

# Equalises using the histogram of the intensities inside 'box' only (world coords if 'affine' is
# provided, otherwise voxels), but the resulting mapping is applied to the whole image.
@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_hist_eq(
    data: Image,
    box: Box | None = None,
    affine: AffineMatrix | None = None,
    ) -> Image:
    data, return_type = to_numpy(data, return_type=True)
    mask = None
    if box is not None:
        mask = np.zeros(data.shape, dtype=bool)
        mask[__box_slices(box, data.shape, affine=affine)] = True
        if not mask.any():
            raise ValueError(f"Histogram equalisation 'box' doesn't overlap with the image.")
    data = ski.exposure.equalize_hist(data, mask=mask)
    if return_type is torch.Tensor:
        data = to_tensor(data)
    return data

@alias_kwargs(
    ('dmn', 'data_min'),
    ('dmx', 'data_max'),
    ('mn', 'min'),
    ('mx', 'max'),
)
def __spatial_minmax(
    data: Image,
    data_min: Number | Literal['min'] = 'min',
    data_max: Number | Literal['max'] = 'max',
    # Takes the data min/max and puts them at these values.
    min: Number = 0.0,
    max: Number = 1.0,
    ) -> Image:
    data, return_type = to_tensor(data, return_type=True)
    if data_min == 'min':
        data_min = float(data.min())
    if data_max == 'max':
        data_max = float(data.max())
    data_width = data_max - data_min
    if data_width == 0:
        if return_type is np.ndarray:
            return to_numpy(data)
        return data
    # Normalise to [0, 1].
    data = (data - data_min) / data_width
    # Scale to [min, max].
    data = data * (max - min) + min
    # Data min/max can result in values outside the [min, max] range - clip these.
    data = data.clamp(min, max)
    if return_type is np.ndarray:
        data = to_numpy(data)
    return data

@alias_kwargs(
    ('b', 'background'),
    ('nc', 'n_classes'),
)
def __spatial_one_hot_encode(
    data: LabelMap,
    background: bool = False,
    n_classes: int | None = None,
    ) -> ChannelLabelImage:
    data, return_type = to_tensor(data, dtype=torch.long, return_type=True)
    if n_classes is None:
        n_classes = int(data.max()) + 1
    label = torch.zeros((n_classes, *data.shape), dtype=torch.bool)
    label.scatter_(0, data.unsqueeze(0), True)
    if not background:
        label = label[1:]
    if return_type is np.ndarray:
        label = to_numpy(label)
    return label

# Pulls image data/affine from "data/affine" or "image" series.
# Output size/affine is pulled from "output_size/affine" or "output_image" series. 
@alias_kwargs(
    ('a', 'affine'),
    ('f', 'fill'),
)
def __spatial_pad(
    data: Image,
    box: Box,
    affine: AffineMatrix | None = None,
    fill: float | Literal['min'] = 'min',
    ) -> Image:
    assert_box_width(box)
    data, return_type = to_tensor(data, return_type=True)
    fill = __resolve_fill(data, fill)

    # Convert box to voxel coordinates.
    if affine is not None:
        min_mm, max_mm = box
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        min = tuple(np.round((np.array(min_mm) - origin) / spacing).astype(int))
        max = tuple(np.round((np.array(max_mm) - origin) / spacing).astype(int))
    else:
        min, max = box

    # Perform padding - clip if padding is less than zero.
    size = np.array(data.shape)
    pad_min = (-np.array(min)).clip(0)
    pad_max = (max - size).clip(0)
    padding = list(reversed(list(zip(pad_min, pad_max))))  # torch 'pad' operates from back to front.
    padding = tuple(torch.tensor(padding).flatten().tolist())
    data = torch.nn.functional.pad(data, padding, value=fill)

    if return_type is np.ndarray:
        data = to_numpy(data)
    return data

# Could pass 'data/affine' or 'image' which will pull both.
@alias_kwargs(
    ('d', 'data'),
    ('a', 'affine'),
    ('f', 'fill'),
    ('i', 'image'),
    ('oa', 'output_affine'),
    ('oi', 'output_image'),
    ('os', 'output_size'),
    ('rt', 'return_transform'),
    ('t', 'transform'),
)
def __spatial_resample(
    data: Image | None = None,
    affine: AffineMatrix | None = None,
    fill: Number | Literal['max', 'mean', 'median', 'min'] = 'min',
    image: DicomSeries | NiftiImageSeries | None = None,
    output_affine: AffineMatrix | None = None, 
    output_image: DicomSeries | NiftiImageSeries | None = None,
    output_size: Size | None = None,       # Unless specified, is set to maintain image FOV.
    return_transform: bool = False,
    transform: sitk.Transform | None = None,     # This transforms points not intensities. I.e. positive transform will move image in negative direction.
    ) -> Image | Tuple[Image, sitk.Transform]:
    # Get input data/affine.
    return_type = None
    if image is not None:
        assert data is None, "Data can't be provided when image is provided."
        assert affine is None, "Affine can't be provided when image is provided."
        data = image.data
        affine = image.affine
    else:
        assert data is not None, "Data must be provided when image is not provided."
        data, return_type = to_numpy(data, return_type=True)
        if affine is None:
            affine = create_affine(dim=data.ndim)

    # Get output size/affine.
    if output_image is not None:
        assert output_size is None, "Output size can't be provided when output image is provided."
        assert output_affine is None, "Output affine can't be provided when output image is provided."
        output_size = output_image.size
        output_affine = output_image.affine        
    else:
        # Don't set output size, this depends on output spacing,
        # i.e. if double the output spacing, we should halve the size
        # to maintain the field-of-view.
        # if output_size is None:
        #     output_size = data.shape    
        if output_affine is None:
            output_affine = affine
    output_spacing = affine_spacing(output_affine)
    output_origin = affine_origin(output_affine) 
    output_direction = affine_direction(output_affine)

    # Slow import so postponing until method call.
    import SimpleITK as sitk

    # Convert to sitk datatypes.
    if data.dtype == bool:
        data = data.astype(np.uint8)

    # Create 'sitk' image.
    img = to_sitk_image(data, affine=affine, dim=data.ndim)

    # Set resample filter params.
    filter = sitk.ResampleImageFilter()
    fill = __resolve_fill(data, fill)
    filter.SetDefaultPixelValue(fill)
    if data.dtype == bool:
        filter.SetInterpolator(sitk.sitkNearestNeighbor)
    filter.SetOutputSpacing(output_spacing)
    filter.SetOutputOrigin(output_origin)
    filter.SetOutputDirection(output_direction.flatten())
    if output_size is not None:
        filter.SetSize(output_size)
    else:
        # Choose output size that maintains the image field-of-view.
        size_factor = np.array(img.GetSpacing()) / filter.GetOutputSpacing()

        # Magic formula is: n_new = f * (n - 1) + 1
        # I think I worked this out by trial and error, but what's going on here is:
        # (n - 1) is the number of intervals between voxels (voxel fov), multiplied by the size
        # factor gives the "voxel fov" of the new image. Plus one to get number of voxels.
        # E.g. downsampling by a factor of 2, from 5 voxels: f = 0.5, 0.5 * (5 - 1) + 1 = 3.
        size = tuple(int(np.ceil(f * (s - 1) + 1)) for f, s in zip(size_factor, img.GetSize()))
        filter.SetSize(size)
    if transform is not None:
        filter.SetTransform(transform)

    # Perform resampling.
    img = filter.Execute(img)

    # Get output data.
    image, _ = from_sitk_image(img)

    # Set return types.
    if data.dtype == bool:
        image = image.astype(bool)

    if return_type is torch.Tensor:
        image = to_tensor(image)

    if return_transform:
        return image, filter.GetTransform()
    else:
        return image

# What does this do?
# Applies 'transform_fn' to each image in a batch, or to each
# channel in a channel image. If 'combine_channels' is True, then
# all channels are passed to the 'transform_fn' - this is useful
# when normalising across channels for example.
@alias_kwargs(
    ('a', 'affine'),
    ('f', 'fill'),
    ('sc', 'sample_col'),
    ('sz', 'sample_size'),
    ('sp', 'sample_spacing'),
    ('t', 'transform'),
)
def __spatial_sample(
    data: Image,
    points: Point | Points | Landmark | Landmarks,
    affine: AffineMatrix | None = None,
    fill: Number | Literal['min'] = 'min',
    sample_col: str = 'sample',
    sample_size: Size | None = None,    # Defaults to point sample - e.g. (0, 0, 0) for 3D.
    sample_spacing: Spacing | None = None,  # Defaults to 1.0 isotropic.
    transform: sitk.Transform | None = None,
    **kwargs,
    ) -> List[float] | Landmarks:
    data = to_numpy(data)
    was_landmarks = False
    if isinstance(points, (pd.DataFrame, pd.Series)):
        was_landmarks = True
        landmarks_tmp = points.copy()
        points = landmarks_to_points(landmarks_tmp)
    if points.ndim == 1:
        points = points[None, :]
    dim = points.shape[1]
    if sample_size is None:
        sample_size = (1,) * dim
    if sample_spacing is None:
        sample_spacing = (1.0,) * dim

    # Slow import so postponing until method call.
    import SimpleITK as sitk

    # Convert to sitk datatypes.
    is_boolean = data.dtype == bool
    if is_boolean:
        data = data.astype(np.uint8)
    if sample_spacing is not None:
        sample_spacing = tuple(float(s) for s in sample_spacing)

    # Create 'sitk' image and set parameters.
    img = to_sitk_image(data, affine=affine, dim=dim)

    # Create resample filter.
    filter = sitk.ResampleImageFilter()
    fill = __resolve_fill(data, fill)
    filter.SetDefaultPixelValue(fill)
    if is_boolean:
        filter.SetInterpolator(sitk.sitkNearestNeighbor)
    if sample_size == (1, 1, 1):     # Sample a single point.
        output_size = (1, 1, 1)
    else:
        output_size = tuple(int(np.ceil(f / s)) for f, s in zip(sample_size, sample_spacing))
    filter.SetSize(output_size)
    if transform is not None:
        filter.SetTransform(transform)

    # Perform resampling.
    # It's hard to make this more efficient, as the resample image filter only accepts
    # regular grids of sampling points.
    samples = []
    for p in points:
        origin = tuple(np.array(p) - (np.array(sample_size) / 2))
        filter.SetOutputOrigin(origin)
        rimg = filter.Execute(img)
        rimg, _ = from_sitk_image(rimg)
        s = rimg.mean()    # Take average of sample grid.
        # r = bool(r) if is_boolean else float(r)
        samples.append(s)

    # Convert to return format.
    if was_landmarks:
        result = landmarks_tmp
        result[sample_col] = samples
    else:
        result = samples

    return result

@alias_kwargs(
    ('m', 'mean'),
    ('s', 'std'),
)
def __standardise(
    data: Image,
    # Takes the data mean/std and puts them at these values.
    mean: Number = 0.0,
    std: Number = 1.0,
    ) -> Image:
    data, return_type = to_tensor(data, return_type=True)
    if mean == 'mean':
        mean = float(data.mean())
    if std == 'std':
        std = float(data.std())
    if std == 0:
        if return_type is np.ndarray:
            return to_numpy(data)
        return data
    # Standardise to mean 0 and std 1.
    data = (data - mean) / std
    if return_type is np.ndarray:
        data = to_numpy(data)
    return data

# Transposes spatial coordinates, whilst maintaining initial batch/channel dimensions.
@bubble_args(__spatial_centre_crop)
def centre_crop(
    data: BatchImage | Image,
    *args,
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_centre_crop, data, *args, **kwargs)

@alias_kwargs(
    ('a', 'affine'),
)
def centre_crop_points(
    points: Point | Points | Landmark | Landmarks,
    size: Size,
    image_size: Size,
    affine: AffineMatrix | None = None,
    ) -> Points | Landmarks:
    # Convert size from world coords (mm) to voxels if affine is provided.
    if affine is not None:
        spacing = affine_spacing(affine)
        size_vox = np.round(np.array(size) / spacing).astype(int)
    else:
        size_vox = np.array(size)

    # Compute crop box in voxels (same logic as __spatial_centre_crop).
    to_crop = np.array(image_size) - size_vox
    crop_start = np.sign(to_crop) * np.ceil(np.abs(to_crop / 2)).astype(int)
    crop_end = crop_start + size_vox
    crop_box = np.stack([crop_start, crop_end])

    return crop_points(points, crop_box, affine=affine)

@alias_kwargs(
    ('cc', 'combine_channels'),
    ('d', 'dim'),
)
def compute_channel_or_spatial_transforms(
    transform_fn: Callable,
    data: Image | BatchImage | BatchChannelImage,
    *args,
    combine_channels: bool = False,
    dim: SpatialDim | None = None, 
    **kwargs,
    ) -> Image | BatchImage | BatchChannelImage:
    if data.ndim == 2:    # 2D image.
        return transform_fn(data, *args, **kwargs) 
    elif data.ndim == 3:  # 2D batch or 3D image.
        if dim is None:
            logger.warn(f"Transform function '{transform_fn.__name__}' received 3D array with no specified 'dim'. Assuming 3D labels. If these are batches of 2D labels, specify 'dim=2' to compute transform per image in batch.") 
        if dim is None or dim == 3: # Single 3D image.
            return transform_fn(data, *args, **kwargs)
        elif dim == 2:                       # Batch of 2D images.
            # Assume that a dim=2, 3D array is (C, X, Y).
            if combine_channels:
                return transform_fn(data, *args, **kwargs)
            else:
                results = []
                for d in data:
                    results.append(transform_fn(d, *args, **kwargs))
                return stack(results)
    elif data.ndim == 4:  # 2D batch/channel or 3D batch.
        if dim is None or dim == 3:
            # Assume that a dim=3, 4D array is (C, X, Y, Z).
            if dim is None:
                logger.warn(f"Transform function '{transform_fn.__name__}' received 4D array with no specified 'dim'. Assuming batch of 3D labels. If these are batch/channels of 2D labels, specify 'dim=2' to compute transform per image in batch.")
            if combine_channels:
                return transform_fn(data, *args, **kwargs)
            else:
                results = []
                for d in data:
                    results.append(transform_fn(d, *args, **kwargs))
                return stack(results)
        elif dim == 2:
            results = []
            for b in data:
                if combine_channels:
                    # Pass all channels to the transform function.
                    results.append(transform_fn(b, *args, **kwargs))
                else:
                    # Transform each channel separately.
                    channel_results = []
                    for c in b:
                        channel_results.append(transform_fn(c, *args, **kwargs))
                    results.append(stack(channel_results))
            return stack(results)
    elif data.ndim == 5:  # 3D batch/channel.
        results = []
        for b in data:
            # Pass all channels to the transform function.
            if combine_channels:
                results.append(transform_fn(b, *args, **kwargs))
                continue
            # Transform each channel separately.
            channel_results = []
            for c in b:
                channel_results.append(transform_fn(c, *args, **kwargs))
            results.append(stack(channel_results))
        return stack(results)
    else:
        raise ValueError(f"Transform function '{transform_fn.__name__}' expects array of spatial dimension 2 or 3, with optional batch dimension. Got array of shape '{data.shape}' with inferred spatial dimension {data.ndim}. Specify 'dim' to override inference.")        

@bubble_args(__spatial_crop)
def crop(
    data: BatchImage | Image,
    *args,
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_crop, data, *args, **kwargs)

# To/from sitk image need to be here for circular import reasons (spatial transpose).
@alias_kwargs(
    ('a', 'affine'),
)
def crop_affine(
    affine: AffineMatrix,
    crop_box: Box,
    ) -> AffineMatrix:
    spacing = affine_spacing(affine)
    return create_affine(spacing, crop_box[0])

# You might expect this to change point values so that they remain in the same
# position on an image after cropping.
# This is not the case if an affine is passed, as point locations won't change 
# in world coordinates and plotting code should just accept the new affine after
# cropping to correctly place points.
# With no affine, point values should change to reflect the new image coordinates.
@bubble_args(__spatial_crop_or_pad)
def crop_or_pad(
    data: BatchImage | Image,
    *args,
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_crop_or_pad, data, *args, **kwargs)

@alias_kwargs(
    ('a', 'affine'),
)
def crop_points(
    points: Point | Points | Landmark | Landmarks,
    crop: Box,
    affine: AffineMatrix | None = None,
    ) -> Points | Landmarks:
    was_landmarks = False
    if isinstance(points, (pd.DataFrame, pd.Series)):
        was_landmarks = True
        landmarks_tmp = points.copy()
        points = landmarks_to_points(landmarks_tmp)
    if points.ndim == 1:
        points = points[None, :]

    # Determine which points fall within the crop FOV.
    if affine is not None:
        # Points are in world coords; convert voxel crop box to world coords for comparison.
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        crop_world = np.array(crop) * spacing + origin
        to_keep = np.all((points >= crop_world[0]) & (points < crop_world[1]), axis=1)
    else:
        to_keep = np.all((points >= crop[0]) & (points < crop[1]), axis=1)

    # Shift points to new image coordinates if no affine (world coords are unchanged by cropping).
    if affine is None:
        points = points - crop[0]

    # Convert back to landmarks if necessary.
    if was_landmarks:
        points = replace_points(landmarks_tmp, points)

    # Remove points outside the crop.
    points = points[to_keep]

    return points

def from_sitk_image(
    img: sitk.Image,
    ) -> Tuple[Image, AffineMatrix]:
    # Slow import so postponing until method call.
    import SimpleITK as sitk
    data = sitk.GetArrayFromImage(img)
    # SimpleITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    dim = len(spacing)
    direction = np.array(img.GetDirection()).reshape(dim, dim)
    affine = create_affine(spacing, origin, direction=direction)
    return data, affine

@bubble_args(__spatial_hist_eq)
def hist_eq(
    data: Image | BatchImage,
    **kwargs,
    ) -> Image | BatchImage:
    return compute_channel_or_spatial_transforms(__spatial_hist_eq, data, **kwargs)

# What if we want to pass image series? This is probably bad
# design as we're then loading data and cropping in the same function.
# Better to keep loading and processing separate.
# We do this for spatial_resample though and it is convenient there due
# to all the input/output params...
@bubble_args(__spatial_minmax)
def minmax(
    data: Image | BatchImage,
    **kwargs,
    ) -> Image | BatchImage:
    return compute_channel_or_spatial_transforms(__spatial_minmax, data, **kwargs)

@bubble_args(__spatial_one_hot_encode)
def one_hot_encode(
    image: LabelMap | BatchLabelMap,
    **kwargs,
    ) -> ChannelLabelImage | BatchChannelLabelImage:
    return compute_channel_or_spatial_transforms(__spatial_one_hot_encode, image, **kwargs)

@bubble_args(__spatial_pad)
def pad(
    image: Image,
    *args, 
    **kwargs,
    ) -> Image:
    return compute_channel_or_spatial_transforms(__spatial_pad, image, *args, **kwargs)

@alias_kwargs(
    ('d', 'data'),
)
@bubble_args(__spatial_resample)
def resample(
    data: BatchImage | Image | None = None, 
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_resample, data, **kwargs) if data is not None else __spatial_resample(**kwargs)

@alias_kwargs(
    ('a', 'affine'),
)
def __resolve_box(
    box: Box,
    size: Size,
    affine: AffineMatrix | None = None,
    ) -> Box:
    size_fov = fov(size, affine=affine)
    box = np.where(np.isnan(box), size_fov, box)
    return box

@bubble_args(__spatial_sample)
def sample(
    data: BatchImage | Image,
    *args,
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_sample, data, *args, **kwargs)

def __spatial_transpose(
    data: Image,
    ) -> Image:
    data, return_type = to_tensor(data, return_type=True)
    data = data.T
    if return_type is np.ndarray:
        data = to_numpy(data)
    return data

def stack(
    results: List[np.ndarray, torch.Tensor],
    ) -> np.ndarray | torch.Tensor:
    if results and isinstance(results[0], torch.Tensor):
        return torch.stack(results, dim=0)
    return np.stack(results, axis=0)

# Handles 2/3D batch/channel/spatial images and passes them to the
# spatial or channel transform. Channel is needed because some transforms
# merge data across channels - e.g. minmax normalisation.
@bubble_args(__standardise)
def standardise(
    data: Image | BatchImage,
    **kwargs,
    ) -> Image | BatchImage:
    return compute_channel_or_spatial_transforms(__standardise, data, **kwargs)

@alias_kwargs(
    ('a', 'affine'),
)
def __spatial_window(
    data: Image,
    affine: AffineMatrix | None = None,
    box: Box | None = None,
    range: Tuple[float | None, float | None] | str | None = None,
    ) -> Image:
    data, return_type = to_tensor(data, return_type=True)
    if box is None and range is None:
        raise ValueError("Window: 'box' or 'range' must be provided.")
    if box is not None and range is not None:
        raise ValueError("Window: only 'box' or 'range' must be provided, not both.")

    # Get min/max intensities.
    if range is not None:
        if isinstance(range, str):
            # E.g. "p:99" keeps the central 99% of intensities.
            if not range.startswith('p:'):
                raise ValueError(f"Unknown window range '{range}'. Expected a (min, max) tuple or a percentile string like 'p:99'.")
            percentile = float(range[2:])
            if not 0 <= percentile <= 100:
                raise ValueError(f"Window percentile must be in [0, 100]. Got '{percentile}'.")
            tail = (100 - percentile) / 2
            vmin, vmax = np.percentile(to_numpy(data), [tail, 100 - tail])
            # Keep integer images integer-valued.
            if not data.is_floating_point():
                vmin, vmax = int(np.floor(vmin)), int(np.ceil(vmax))
        else:
            vmin, vmax = range
    else:
        region = crop(data, box, affine=affine)
        vmin, vmax = region.min(), region.max()

    # Apply the intensity window.
    data = torch.clamp(data, min=vmin, max=vmax)

    if return_type is np.ndarray:
        data = to_numpy(data)
    return data

# Intensity transform.
@bubble_args(__spatial_window)
def window(
    data: Image | BatchImage,
    **kwargs,
    ) -> Image | BatchImage:
    return compute_channel_or_spatial_transforms(__spatial_window, data, **kwargs)

@alias_kwargs(
    ('a', 'affine'),
    ('d', 'dim'),
)
def to_sitk_image(
    data: ChannelImage | Image,
    affine: AffineMatrix | None = None,
    dim: SpatialDim = 3,
    ) -> sitk.Image:
    # Slow import so postponing until method call.
    import SimpleITK as sitk
    data = to_numpy(data)
    # Multi-channel sitk images must be stored as vector images.
    is_vector = True if data.ndim == dim + 1 else False

    # Convert to SimpleITK data types.
    if data.dtype == bool:
        data = data.astype(np.uint8)
    
    # SimpleITK **sometimes** flips the data coordinates (x, y, z) -> (z, y, x) when converting from numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    # Preprocessing, such as np.transpose and np.moveaxis can change the numpy array indexing style
    # from the default C-style to Fortran-style. SimpleITK will flip coordinates for C-style but not F-style.
    # data = __spatial_transpose(data)
    data = transpose(data, dim=dim)
    # We can use 'copy' to reset the indexing to C-style and ensure that SimpleITK flips coordinates. If we
    # don't do this, code called before 'to_sitk' could affect the behaviour of 'GetImageFromArray', which
    # was very confusing for me.
    data = data.copy()
    if is_vector:
        # Sitk expects vector dimension to be last.
        data = np.moveaxis(data, 0, -1)
    img = sitk.GetImageFromArray(data, isVector=is_vector)
    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        direction = affine_direction(affine)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction.flatten())

    return img

@alias_kwargs(
    ('a', 'affine'),
    ('d', 'dim'),
)
def to_transform(
    dvf: ChannelImage,
    affine: AffineMatrix | None = None,
    dim: SpatialDim = 3,
    ) -> sitk.Transform:
    import SimpleITK as sitk
    dvf = dvf.astype(np.float64)
    if dvf.shape[0] != dim:
        raise ValueError(f"Expected DVF with {dim} channels (first axis), got {dvf.shape[0]}")
    dvf_img = to_sitk_image(dvf, affine=affine, dim=dim)
    return sitk.DisplacementFieldTransform(dvf_img)

def transform_points(
    points: Points | Landmarks,
    transform: sitk.Transform,
    ) -> Points | Landmarks:
    is_landmarks = True if isinstance(points, pd.DataFrame) else False
    if is_landmarks:
        lms = points.copy()
        points = points[list(range(3))].to_numpy()
    assert points.shape[1] in (2, 3)
    points = points.astype(np.float64)  # sitk expects double.
    
    # Apply transform to points.
    points_t = []
    for p in points:
        p_t = transform.TransformPoint(p)
        points_t.append(p_t)
    points_t = np.vstack(points_t)

    if is_landmarks:
        lms[list(range(3))] = points_t
        output = lms
    else:
        output = points_t

    return output

def transpose(
    data: BatchChannelImage | BatchImage | Image,
    *args,
    **kwargs,
    ) -> BatchChannelImage | BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_transpose, data, *args, **kwargs)

def __resolve_fill(
    data: np.ndarray,
    fill: Number | Literal['max', 'mean', 'median', 'min'] = 'min',
    ) -> Number:
    if isinstance(fill, (int, float)):
        return fill
    if fill == 'mean':
        return float(np.mean(data))
    if fill == 'median':
        return float(np.median(data))
    if fill == 'min':
        return float(np.min(data))
    raise ValueError(f"Unsupported fill value: {fill}")

def rotate_affine(
    affine: AffineMatrix,
    theta: float | Tuple[float],
    ):
    dim = affine.shape[0] - 1
    R = __rotation_matrix(theta, dim)

    new_affine = affine.copy()
    new_affine[:dim, :dim] = R @ affine[:dim, :dim]
    new_affine[:dim, dim]  = R @ affine[:dim, dim]

    return new_affine

# Rotates points about the world origin - same convention as 'rotate_affine'.
def rotate_points(
    points: Point | Points | Landmark | Landmarks,
    theta: float | Tuple[float],
    ) -> Points | Landmarks:
    was_landmarks = False
    if isinstance(points, (pd.DataFrame, pd.Series)):
        was_landmarks = True
        landmarks_tmp = points.copy()
        points = landmarks_to_points(landmarks_tmp)
    else:
        points = to_numpy(points)
    is_single = points.ndim == 1
    if is_single:
        points = points[np.newaxis, :]

    dim = points.shape[1]
    R = __rotation_matrix(theta, dim)
    points = points @ R.T

    if is_single:
        points = points[0]
    if was_landmarks:
        points = replace_points(landmarks_tmp, points)

    return points

def __rotation_matrix(
    theta: float | Tuple[float],
    dim: SpatialDim,
    ) -> np.ndarray:
    if dim == 2:
        assert isinstance(theta, (float, int)), "For 2D rotation, theta must be a single float or int."
        theta = np.radians(theta)
        R = np.array([
            [np.cos(theta),  np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
    elif dim == 3:
        thetas = arg_to_list(theta, (int, float),  broadcast=3)
        assert len(thetas) == 3, "For 3D rotation, theta must have three elements."
        tx, ty, tz = np.radians(thetas)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tx), np.sin(tx)],
            [0, -np.sin(tx), np.cos(tx)]
        ])
        Ry = np.array([
            [np.cos(ty), 0, -np.sin(ty)],
            [0, 1, 0],
            [np.sin(ty), 0, np.cos(ty)]
        ])
        Rz = np.array([
            [np.cos(tz), np.sin(tz), 0],
            [-np.sin(tz), np.cos(tz), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
    else:
        raise ValueError(f"Rotation not supported for dimension {dim}.")

    return R

# Rotates 'image' by 'theta' about the world origin (same convention as 'rotate_affine'/
# 'rotate_points'), then resamples onto an axis-aligned grid - extended to cover the full
# rotated field-of-view - so callers (e.g. slicing/plotting code) can assume an unrotated
# affine. 'affine' describes the image's location in world coords (defaults to identity).
# Call separately for each of image/dose/labels, passing 'output_affine'/'output_size' from
# the first call to keep them co-registered.
@alias_kwargs(
    ('a', 'affine'),
    ('f', 'fill'),
    ('oa', 'output_affine'),
    ('os', 'output_size'),
)
def rotate_image(
    image: Image | LabelImage | BatchImage | BatchLabelImage,
    theta: float | Tuple[float],
    affine: AffineMatrix | None = None,
    fill: Number | Literal['max', 'mean', 'median', 'min'] = 'min',
    output_affine: AffineMatrix | None = None,
    output_size: Size | None = None,
    ) -> Tuple[Image | LabelImage | BatchImage | BatchLabelImage, AffineMatrix]:
    dim = affine.shape[0] - 1 if affine is not None else image.ndim
    if affine is None:
        affine = create_affine(dim=dim)
    affine = rotate_affine(affine, theta)

    if output_affine is None or output_size is None:
        size = np.array(image.shape[-dim:])
        corners_vox = np.array(list(itertools.product(*[(0, s - 1) for s in size])), dtype=np.float32)
        corners_world = to_world_coords(corners_vox, affine)
        bbox_min = corners_world.min(axis=0)
        bbox_max = corners_world.max(axis=0)

        # Project the unit cube along each axis to determine the spacing required to
        # resample the rotated image without data loss.
        out_spacing = np.abs(affine[:dim, :dim]).sum(axis=1)
        if output_size is None:
            output_size = tuple(int(s) for s in np.ceil((bbox_max - bbox_min) / out_spacing).astype(int) + 1)
        if output_affine is None:
            output_affine = create_affine(dim=dim, origin=bbox_min, spacing=out_spacing)

    image = resample(data=image, affine=affine, dim=dim, fill=fill, output_affine=output_affine, output_size=output_size)
    return image, output_affine

# Union (logical OR) of multiple boolean labels of the same shape.
def combine_labels(
    labels: List[BatchLabelImage | LabelImage],
    ) -> BatchLabelImage | LabelImage:
    assert len(labels) > 0, "At least one label must be provided."
    _, return_type = to_tensor(labels[0], return_type=True)
    labels_t = [to_tensor(l, dtype=torch.bool) for l in labels]
    combined = torch.stack(labels_t, dim=0).any(dim=0)
    if return_type is np.ndarray:
        combined = to_numpy(combined)
    return combined
