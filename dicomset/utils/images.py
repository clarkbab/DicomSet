import numpy as np

from ..typing import AffineMatrix, Box, Image, Size
from .geometry import affine_origin, affine_spacing, to_image_coords

def create_box_label(
    size: Size,
    box: Box,
    affine: AffineMatrix | None = None,
    ) -> Image:
    # Convert box to image coordinates.
    if affine is not None:
        spacing = affine_spacing(affine)
        origin = affine_origin(affine)
        box = to_image_coords(box, affine)
    box = box.astype(int)

    # Create box label.
    image = np.zeros(size, dtype=bool)
    slices = tuple(slice(box[0, i], box[1, i]) for i in range(len(size)))
    image[slices] = True

    return image
