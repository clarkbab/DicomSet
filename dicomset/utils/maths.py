import numpy as np
from typing import List

from ..typing import Number
from .conversion import to_list, to_numpy

def round(
    x: Number | List[Number] | np.ndarray,
    # Can round either by number of decimal places, or by tolerance.
    dp: int | None = None,
    tol: Number | None = None,
    ) -> Number | List[Number] | np.ndarray:
    x, return_type = to_numpy(x, return_type=True)
    assert (dp is not None) ^ (tol is not None), "Specify either dp or tol, not both."
    if dp is not None:
        x = np.round(x, dp)
    else:
        x = tol * np.round(x / tol)
    if return_type is int or return_type is float:
        return return_type(x[0])
    elif return_type is list:
        return to_list(x)
    else:
        return x