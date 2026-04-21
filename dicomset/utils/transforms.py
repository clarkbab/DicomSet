from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from typing import Callable, Literal, Tuple, TYPE_CHECKING

from ..typing import AffineMatrix, BatchChannelImage, BatchImage, Image, Number, Size, SpatialDim
from .args import bubble_args
from .conversion import from_sitk_image, to_sitk_image
from .geometry import affine_origin, affine_spacing
from .logging import logger
if TYPE_CHECKING:
    from ..dicom import DicomSeries
    from ..nifti import NiftiImageSeries

# Handles 2/3D batch/channel/spatial images and passes them to the
# spatial or channel transform. Channel is needed because some transforms
# merge data across channels - e.g. minmax normalisation.
def __minmax(
    data: Image,
    ) -> Image:
    data_width = data.max() - data.min()
    if data_width == 0:
        return data
    data = (data - data.min()) / data_width
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
    combine_channels: bool = False,
    dim: SpatialDim | None = None, 
    **kwargs,
    ) -> Image | BatchImage | BatchChannelImage:
    if data.ndim == 2:    # 2D image.
        return transform_fn(data, **kwargs) 
    elif data.ndim == 3:  # 2D batch or 3D image.
        if dim is None or dim == 3: # 3D image.
            logger.warn(f"Transform function '{transform_fn.__name__}' received 3D array with no specified 'dim'. Assuming 3D labels. If these are batches of 2D labels, specify 'dim=2' to compute transform per image in batch.") 
            return transform_fn(data, **kwargs)
        elif dim == 2:                       # Batch of 2D images.
            # Assume that a dim=2, 3D array is (C, X, Y).
            if combine_channels:
                return transform_fn(data, **kwargs)
            else:
                results = []
                for d in data:
                    results.append(transform_fn(d, **kwargs)) 
                return np.stack(results, axis=0)
    elif data.ndim == 4:  # 2D batch/channel or 3D batch.
        if dim is None or dim == 3:
            # Assume that a dim=3, 4D array is (C, X, Y, Z).
            logger.warn(f"Transform function '{transform_fn.__name__}' received 4D array with no specified 'dim'. Assuming batch of 3D labels. If these are batch/channels of 2D labels, specify 'dim=2' to compute transform per image in batch.")    
            if combine_channels:
                return transform_fn(data, **kwargs)
            else:
                results = []
                for d in data:
                    results.append(transform_fn(d, **kwargs))
                return np.stack(results, axis=0) 
        elif dim == 2:
            results = []
            for b in data:
                if combine_channels:
                    # Pass all channels to the transform function.
                    results.append(transform_fn(b, **kwargs))
                else:
                    # Transform each channel separately.
                    channel_results = []
                    for c in b:
                        channel_results.append(transform_fn(c, **kwargs))
                    results.append(np.stack(channel_results, axis=0))
            return np.stack(results, axis=0) 
    elif data.ndim == 5:  # 3D batch/channel.
        results = []
        for b in data:
            # Pass all channels to the transform function.
            if combine_channels:
                results.append(transform_fn(b, **kwargs))
                continue
            # Transform each channel separately.
            channel_results = []
            for c in b:
                channel_results.append(transform_fn(c, **kwargs))
            results.append(np.stack(channel_results, axis=0))
        return np.stack(results, axis=0) 
    else:
        raise ValueError(f"Transform function '{transform_fn.__name__}' expects array of spatial dimension 2 or 3, with optional batch dimension. Got array of shape '{data.shape}' with inferred spatial dimension {data.ndim}. Specify 'dim' to override inference.")        

# Pulls image data/affine from "data/affine" or "image" series.
# Output size/affine is pulled from "output_size/affine" or "output_image" series. 
@bubble_args(__minmax)
def minmax(
    data: Image | BatchImage,
    **kwargs,
    ) -> Image | BatchImage:
    return compute_channel_or_spatial_transforms(__minmax, data, **kwargs)

@bubble_args(__spatial_resample)
def resample(
    data: BatchImage | Image | None = None, 
    **kwargs,
    ) -> BatchImage | Image:
    return compute_channel_or_spatial_transforms(__spatial_resample, data, **kwargs) if data is not None else __spatial_resample(**kwargs)

# Transposes spatial coordinates, whilst maintaining initial batch/channel dimensions.
def spatial_transpose(
    data: BatchChannelImage | BatchImage | Image,
    dim: SpatialDim = 3,
    ) -> BatchChannelImage | BatchImage | Image:
    from_axes = tuple(range(data.ndim - dim, data.ndim))     
    to_axes = tuple(reversed(from_axes)) 
    return np.moveaxis(data, from_axes, to_axes)
