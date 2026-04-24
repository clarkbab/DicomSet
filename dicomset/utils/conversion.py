from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple

from ..typing import Number
from .args import bubble_args

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

@bubble_args(to_numpy)
def to_list(
    data: bool | Number | str | List[bool | Number | str] | np.ndarray,
    **kwargs,
    ) -> List[bool | Number | str] | None:
    if data is None:
        return None 
    return to_numpy(data, **kwargs).tolist()

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
