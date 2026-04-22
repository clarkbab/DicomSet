from ....typing import AffineMatrix3D, Box3D, Image3D, Point3D, Size3D, Spacing3D
from ....utils.geometry import affine_origin, affine_spacing, fov
from ....utils.python import ensure_loaded, get_private_attr
from ..series import NiftiSeries

# Abstract class.
class NiftiImageSeries(NiftiSeries):
    def __init__(
        self,
        *args,
        **kwargs,
        ) -> None:
        super().__init__(*args, **kwargs)

    @property
    @ensure_loaded('__affine', '__load_data')
    def affine(self) -> AffineMatrix3D:
        return get_private_attr(self, '__affine')

    @property
    @ensure_loaded('__data', '__load_data')
    def data(self) -> Image3D:
        return get_private_attr(self, '__data')

    @ensure_loaded('__data', '__load_data')
    def fov(
        self,
        **kwargs,
        ) -> Box3D:
        return fov(get_private_attr(self, '__data'), get_private_attr(self, '__affine'), **kwargs)

    @property
    @ensure_loaded('__affine', '__load_data')
    def origin(self) -> Point3D:
        return affine_origin(get_private_attr(self, '__affine'))

    @property
    @ensure_loaded('__data', '__load_data')
    def size(self) -> Size3D:
        return get_private_attr(self, '__data').shape

    @property
    @ensure_loaded('__affine', '__load_data')
    def spacing(self) -> Spacing3D:
        return affine_spacing(get_private_attr(self, '__affine'))
