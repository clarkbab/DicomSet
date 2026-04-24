from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch
from typing import List, Tuple

from ..typing import AffineMatrix, ChannelImage, Image, Number, SpatialDim
from .args import bubble_args
from .geometry import affine_origin, affine_spacing, create_affine

def to_numpy(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray | torch.Tensor | torch.Size,
    broadcast: int | None = None,
    dtype: np.dtype | None = None,
    return_type: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray | type] | None:
    if data is None:
        if return_type:
            return None, None
        return None

    # Record input type.
    if return_type:
        input_type = type(data)

    # Convert data to array.
    if isinstance(data, (bool, float, int, str)):
        data = np.array([data])
    elif isinstance(data, (list, tuple)):
        data = np.array(data)
    elif isinstance(data, torch.Size):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Set data type.
    if dtype is not None:
        data = data.astype(dtype)

    # Broadcast if required.
    if broadcast is not None and len(data) == 1:
        data = np.repeat(data, broadcast)

    if return_type:
        return data, input_type
    else:
        return data

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

@bubble_args(to_numpy)
def to_list(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray,
    **kwargs,
    ) -> List[bool | Number | str] | None:
    if data is None:
        return None 
    return to_numpy(data, **kwargs).tolist()

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

def to_tensor(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray | torch.Tensor | torch.Size,
    broadcast: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    return_type: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor | None, type] | None:
    # Record input type.
    if return_type:
        input_type = type(data)

    # Convert to tensor.
    if isinstance(data, (bool, float, int, str)):
        device = torch.device('cpu') if device is None else device  
        data = torch.tensor([data], device=device, dtype=dtype)
    elif isinstance(data, (list, tuple, np.ndarray, torch.Size)):
        device = torch.device('cpu') if device is None else device  
        data = torch.tensor(data, device=device, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        device = data.device if device is None else device
        dtype = data.dtype if dtype is None else dtype
        data = data.to(device=device, dtype=dtype)

    # Broadcast if required.
    if broadcast is not None and len(data) == 1:
        data = data.repeat(broadcast)

    if return_type:
        return data, input_type
    else:
        return data

# SimpleITK handles 2-4D images.
@bubble_args(to_numpy)
def to_tuple(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray,
    decimals: int | None = None,
    **kwargs,
    ) -> Tuple[bool | Number | str, ...] | None:
    if data is None:
        return None 
    # Convert to tuple.
    data = tuple(to_numpy(data, **kwargs).tolist())

    # Round elements if required.
    if decimals is not None:
        data = tuple(round(x, decimals) if isinstance(x, float) else x for x in data)

    return data
