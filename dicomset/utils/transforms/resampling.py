from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from typing import Literal, Tuple, TYPE_CHECKING

from ...typing import AffineMatrix, BatchImage, Image, Number, Size
from ...utils.conversion import from_sitk_image, to_sitk_image
from ...utils.geometry import affine_origin, affine_spacing
from ...utils.python import bubble_args
from .shared import compute_spatial_transforms
if TYPE_CHECKING:
    from ...dicom import DicomSeries
    from ...nifti import NiftiImageSeries

# Pulls image data/affine from "data/affine" or "image" series.
# Output size/affine is pulled from "output_size/affine" or "output_image" series. 
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

@bubble_args(__spatial_resample)
def resample(
    data: BatchImage | Image | None = None, 
    **kwargs,
    ) -> BatchImage | Image:
    return compute_spatial_transforms(__spatial_resample, data, **kwargs) if data is not None else __spatial_resample(**kwargs)
