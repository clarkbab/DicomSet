import os

from .typing import DirPath, Orientation, SpatialDim
from .utils.assertions import assert_orientation

DEFAULT_DIM = 3
DEFAULT_ORIENTATION_2D = 'LS'
DEFAULT_ORIENTATION_3D = 'LPS'

class Directories:
    @property
    def data(self):
        if data is None:
            raise ValueError("Data directory is not set. Please set it using the 'set_data' function or the 'DS_DATA' environment variable.")
        return data

    @property
    def datasets(self):
        filepath = os.path.join(self.data, 'datasets')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def files(self):
        filepath = os.path.join(self.data, 'files')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def models(self):
        filepath = os.path.join(self.data, 'models')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def runs(self):
        filepath = os.path.join(self.data, 'runs')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

def init_data() -> DirPath | None:
    d = os.environ.get('DS_DATA')
    if d is not None:
        if not os.path.exists(d):
            raise ValueError(f"Data directory '{d}' does not exist.")
        return d
    return d

def init_dim() -> SpatialDim:
    dim = os.environ.get('DS_DIM')
    if dim is not None:
        try:
            dim = int(dim)
        except ValueError:
            raise ValueError(f"DS_DIM environment variable must be an integer (2 or 3), got '{dim}'.")
        if dim not in (2, 3):
            raise ValueError(f"DS_DIM environment variable must be 2 or 3, got {dim}.")
        return dim
    return DEFAULT_DIM

def init_orientation() -> tuple[Orientation, Orientation]:
    o2d = os.environ.get('DS_ORIENTATION_2D')
    if o2d is not None:
        assert_orientation(o2d, 2)
    else:
        o2d = DEFAULT_ORIENTATION_2D
    o3d = os.environ.get('DS_ORIENTATION_3D')
    if o3d is not None:
        assert_orientation(o3d, 3)
    else:
        o3d = DEFAULT_ORIENTATION_3D
    return o2d, o3d

def get_dim() -> SpatialDim:
    return dim

def get_orientation(
    dim: SpatialDim,
    ) -> Orientation:
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}.")
    if dim == 2:
        return orientation_2d
    elif dim == 3:
        return orientation_3d

def set_data(d: DirPath) -> None:
    if not os.path.exists(d):
        raise ValueError(f"Data directory '{d}' does not exist.")
    global data
    data = d

def set_dim(
    d: SpatialDim,
    ) -> None:
    if d not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {d}.")
    global dim
    dim = d

def set_orientation(
    o: Orientation,
    dim: SpatialDim,
    ) -> None:
    assert_orientation(o, dim)
    global orientation_2d, orientation_3d
    if dim == 2:
        orientation_2d = o
    elif dim == 3:
        orientation_3d = o

dirs = Directories()
data = init_data()
dim = init_dim()
orientation_2d, orientation_3d = init_orientation()
