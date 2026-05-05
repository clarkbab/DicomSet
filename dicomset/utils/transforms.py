from __future__ import annotations

import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage as ski
from typing import Callable, Literal, Tuple, TYPE_CHECKING

from ..typing import AffineMatrix, BatchChannelImage, BatchImage, ChannelImage, Image, Number, Size, SpatialDim
from .args import bubble_args
from .geometry import affine_origin, affine_spacing, assert_box_width, create_affine, fov, to_image_coords
from .landmarks import landmarks_to_points, points_to_landmarks, replace_points
from .logging import logger
if TYPE_CHECKING:
    from ..dicom import DicomSeries
    from ..nifti import NiftiImageSeries

# Handles 2/3D batch/channel/spatial images and passes them to the
# spatial or channel transform. Channel is needed because some transforms
# merge data across channels - e.g. minmax normalisation.
def __minmax(
    data: Image,
    data_min: Number | Literal['min'] = 'min',
    data_max: Number | Literal['max'] = 'max',
    # Takes the data min/max and puts them at these values.
    min: Number = 0.0,
    max: Number = 1.0,
    ) -> Image:
    if data_min == 'min':
        data_min = float(data.min())
    if data_max == 'max':
        data_max = float(data.max())
    data_width = data_max - data_min
    if data_width == 0:
        return data
    # Normalise to [0, 1].
    data = (data - data_min) / data_width
    # Scale to [min, max].
    data = data * (max - min) + min

    # Data min/max can result in values outside the [min, max] range - clip these.
    data = np.clip(data, min, max)

    return data

# Can be applied to spatial or channel image.
def __spatial_resample(
    data: Image | None = None,
    affine: AffineMatrix | None = None,
    fill: Number | Literal['min'] = 'min',
    image: DicomSeries | NiftiImageSeries | None = None,
    output_affine: AffineMatrix | None = None, 
    output_image: DicomSeries | NiftiImageSeries | None = None,
    output_size: Size | None = None,       # Unless specified, is set to maintain image FOV.
    return_transform: bool = False,
    transform: sitk.Transform | None = None,     # This transforms points not intensities. I.e. positive transform will move image in negative direction.
    ) -> Image | Tuple[Image, sitk.Transform]:
    # Get input data/affine.
    if image is not None:
        assert data is None, "Data can't be provided when image is provided."
        assert affine is None, "Affine can't be provided when image is provided."
        data = image.data
        affine = image.affine
    else:
        assert data is not None, "Either 'data' or 'image' must be provided."
        assert affine is not None, "Either 'affine' or 'image' must be provided."
    spacing = affine_spacing(affine)
    origin = affine_origin(affine) 

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

    # Convert to sitk datatypes.
    if data.dtype == bool:
        data = data.astype(np.uint8) 

    # Create 'sitk' image.
    img = to_sitk_image(data, affine=affine, dim=data.ndim)

    # Set resample filter params.
    filter = sitk.ResampleImageFilter()
    if isinstance(fill, str) and fill == 'min':
        fill = float(data.min())
    filter.SetDefaultPixelValue(fill)
    if data.dtype == bool:
        filter.SetInterpolator(sitk.sitkNearestNeighbor)
    filter.SetOutputSpacing(output_spacing)
    filter.SetOutputOrigin(output_origin)
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

    if return_transform:
        return image, filter.GetTransform()
    else:
        return image

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
        if dim is None or dim == 3: # 3D image.
            logger.warn(f"Transform function '{transform_fn.__name__}' received 3D array with no specified 'dim'. Assuming 3D labels. If these are batches of 2D labels, specify 'dim=2' to compute transform per image in batch.") 
            return transform_fn(data, *args, **kwargs)
        elif dim == 2:                       # Batch of 2D images.
            # Assume that a dim=2, 3D array is (C, X, Y).
            if combine_channels:
                return transform_fn(data, *args, **kwargs)
            else:
                results = []
                for d in data:
                    results.append(transform_fn(d, *args, **kwargs))
                return np.stack(results, axis=0)
    elif data.ndim == 4:  # 2D batch/channel or 3D batch.
        if dim is None or dim == 3:
            # Assume that a dim=3, 4D array is (C, X, Y, Z).
            logger.warn(f"Transform function '{transform_fn.__name__}' received 4D array with no specified 'dim'. Assuming batch of 3D labels. If these are batch/channels of 2D labels, specify 'dim=2' to compute transform per image in batch.")    
            if combine_channels:
                return transform_fn(data, *args, **kwargs)
            else:
                results = []
                for d in data:
                    results.append(transform_fn(d, *args, **kwargs))
                return np.stack(results, axis=0)
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
                    results.append(np.stack(channel_results, axis=0))
            return np.stack(results, axis=0) 
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
            results.append(np.stack(channel_results, axis=0))
        return np.stack(results, axis=0) 
    else:
        raise ValueError(f"Transform function '{transform_fn.__name__}' expects array of spatial dimension 2 or 3, with optional batch dimension. Got array of shape '{data.shape}' with inferred spatial dimension {data.ndim}. Specify 'dim' to override inference.")        

# Pulls image data/affine from "data/affine" or "image" series.
# Output size/affine is pulled from "output_size/affine" or "output_image" series. 
def from_sitk_image(
    img: sitk.Image,
    ) -> Tuple[Image, AffineMatrix]:
    data = sitk.GetArrayFromImage(img)
    # SimpleITK always flips the data coordinates (x, y, z) -> (z, y, x) when converting to numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    data = data.transpose()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    affine = create_affine(spacing, origin)
    return data, affine

@bubble_args(__minmax)
def minmax(
    data: Image | BatchImage,
    **kwargs,
    ) -> Image | BatchImage:
    return compute_channel_or_spatial_transforms(__minmax, data, **kwargs)

# Transposes spatial coordinates, whilst maintaining initial batch/channel dimensions.
@bubble_args(__spatial_resample)
def resample(
    data: BatchImage | Image | None = None, 
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_resample, data, **kwargs) if data is not None else __spatial_resample(**kwargs)

# To/from sitk image need to be here for circular import reasons (spatial transpose).
def spatial_transpose(
    data: BatchChannelImage | BatchImage | Image,
    dim: SpatialDim = 3,
    ) -> BatchChannelImage | BatchImage | Image:
    from_axes = tuple(range(data.ndim - dim, data.ndim))     
    to_axes = tuple(reversed(from_axes)) 
    return np.moveaxis(data, from_axes, to_axes)

def to_sitk_image(
    data: ChannelImage | Image,
    affine: AffineMatrix | None = None,
    dim: SpatialDim = 3,
    ) -> sitk.Image:
    # Multi-channel sitk images must be stored as vector images.
    is_vector = True if data.ndim == dim + 1 else False

    # Convert to SimpleITK data types.
    if data.dtype == bool:
        data = data.astype(np.uint8)
    
    # SimpleITK **sometimes** flips the data coordinates (x, y, z) -> (z, y, x) when converting from numpy.
    # See C- (row-major) vs. Fortran- (column-major) style indexing.
    # Preprocessing, such as np.transpose and np.moveaxis can change the numpy array indexing style
    # from the default C-style to Fortran-style. SimpleITK will flip coordinates for C-style but not F-style.
    data = spatial_transpose(data, dim=dim)
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
        img.SetSpacing(spacing)
        img.SetOrigin(origin)

    return img

def __spatial_crop(
    data: Image,
    crop_box: Box,
    affine: AffineMatrix | None = None,
    ) -> ImageArray:
    crop_box = __resolve_box(crop_box, data.shape, affine=affine)
    assert_box_width(crop_box)

    # Convert box to voxel coordinates.
    if affine is not None:
        crop_box = to_image_coords(crop_box, affine=affine)

    # Perform cropping.
    size = np.array(data.shape)
    crop_min = np.array(crop_box[0]).clip(0)
    crop_max = (size - np.array(crop_box[1])).clip(0)
    slices = tuple(slice(min, s - max) for min, max, s in zip(crop_min, crop_max, size))
    data = data[slices]

    return data

# Handle 'np.nan' values in box.
def __resolve_box(
    box: Box,
    size: Size,
    affine: AffineMatrix | None = None,
    ) -> Box:
    size_fov = fov(size, affine=affine)
    box = np.where(np.isnan(box), size_fov, box)
    return box

# What if we want to pass image series? This is probably bad
# design as we're then loading data and cropping in the same function.
# Better to keep loading and processing separate.
# We do this for spatial_resample though and it is convenient there due
# to all the input/output params...
@bubble_args(__spatial_crop)
def crop(
    data: BatchImage | Image,
    *args,
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_crop, data, *args, **kwargs)

# We're not actually cropping the affine, it's just what would the new affine
# before the image after a crop? Maybe crop should return this?
def crop_affine(
    affine: AffineMatrix,
    crop_box: Box,
    ) -> AffineMatrix:
    origin = affine_origin(affine)
    spacing = affine_spacing(affine)
    affine = create_affine(spacing, crop_box[0])
    return affine

# Cropping doesn't actually do anything to point locations, it just removes
# points that are outside the box.
def crop_points(
    points: Point | Points | Landmark | Landmarks,
    crop: Box,
    ) -> Points | Landmarks:
    was_landmarks = False
    if isinstance(points, (pd.DataFrame, pd.Series)):
        was_landmarks = True
        landmarks_tmp = points.copy()
        points = landmarks_to_points(landmarks_tmp)
    if points.ndim == 1:
        points = points[None, :]

    # Get decision variables.
    to_keep = np.stack((points >= crop[0], points < crop[1]), axis=1)
    to_keep = np.all(to_keep, axis=(1, 2))

    # Convert back to landmarks if necessary.
    if was_landmarks:
        points = replace_points(landmarks_tmp, points)

    # Remove points.
    points = points[to_keep]

    return points

# How is this different from "resample"?
# This samples all the points passed to the image, and you can specify the region that is sampled
# for each point by passing size/spacing - this creates a sampling grid centred on the point. 
# Either returns a list of samples, or adds them to the dataframe using 'sample_col'.
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
    if fill == 'min':
        fill = float(data.min())
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

@bubble_args(__spatial_sample)
def sample(
    data: BatchImage | Image,
    *args,
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_sample, data, *args, **kwargs)

def hist_eq(
    data: Image,
    ) -> Image:
    return ski.exposure.equalize_hist(data)
