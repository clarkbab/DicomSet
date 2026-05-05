from jaxtyping import Bool, Float, Int
import numpy as np
import pandas as pd
import pydicom as dcm
from typing import Literal, Tuple

# First-order types (composed of basic types).
# Splitting by 'order' allows for easier managing of type dependencies.
AffineMatrix2D = Float[np.ndarray, "3 3"]
AffineMatrix3D = Float[np.ndarray, "4 4"]
BatchImage2D = Float[np.ndarray, "B X Y"]
BatchImage3D = Float[np.ndarray, "B X Y Z"]
BatchChannelImage2D = Float[np.ndarray, "B C X Y"]
BatchChannelImage3D = Float[np.ndarray, "B C X Y Z"]
BatchChannelLabelImage2D = Bool[np.ndarray, "B C X Y"]
BatchChannelLabelImage3D = Bool[np.ndarray, "B C X Y Z"]
Box2D = Float[np.ndarray, "2 2"]
Box3D = Float[np.ndarray, "2 3"]
BatchBox2D = Float[np.ndarray, "B 2 2"]
BatchBox3D = Float[np.ndarray, "B 2 3"]
BatchLabelImage2D = Bool[np.ndarray, "B X Y"]
BatchLabelImage3D = Bool[np.ndarray, "B X Y Z"]
BatchPixelBox = Int[np.ndarray, "B 2 2"]
BatchVoxelBox = Int[np.ndarray, "B 2 3"]
ChannelImage2D = Float[np.ndarray, "C X Y"]
ChannelImage3D = Float[np.ndarray, "C X Y Z"]
ChannelLabelImage2D = Bool[np.ndarray, "C X Y"]
ChannelLabelImage3D = Bool[np.ndarray, "C X Y Z"]
CtDicom = dcm.dataset.FileDataset
DatasetID = str
DatasetType = Literal['dicom', 'nifti', 'raw', 'training']
DicomModality = Literal['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
DirPath = str
DiskLandmarkID = str
DiskRegionID = str  # Actual names of regions in rtstruct file or nifti file name.
SpatialDim = Literal[2, 3]
FilePath = str
GroupID = str       # E.g. lung, hn
Image2D = Float[np.ndarray, "X Y"]
Image3D = Float[np.ndarray, "X Y Z"]
Indices2D = Int[np.ndarray, "N 2"]
Indices3D = Int[np.ndarray, "N 3"]
LabelImage2D = Bool[np.ndarray, "X Y"]
LabelImage3D = Bool[np.ndarray, "X Y Z"]
Landmark2D = pd.Series
Landmark3D = pd.Series
Landmarks2D = pd.DataFrame
Landmarks3D = pd.DataFrame
LandmarkID = str
ModelID = str
NiftiModality = Literal['ct', 'dose', 'landmarks', 'mr', 'plan', 'regions']
Number = int | float
Orientation = Literal['LAI', 'LAS', 'LPI', 'LPS', 'RAI', 'RAS', 'RPI', 'RPS']
PatientID = str
Point2D = Tuple[float, float] | Float[np.ndarray, "2"]
Point3D = Tuple[float, float, float] | Float[np.ndarray, "3"]
Points2D = Float[np.ndarray, "N 2"]
Points3D = Float[np.ndarray, "N 3"]
Pixel = Tuple[int, int] | Int[np.ndarray, "2"]
PixelBox = Int[np.ndarray, "2 2"]
Pixels = Int[np.ndarray, "N 2"]
RegExp = str
RegionID = str
RegionList = str
RtStructDicom = dcm.dataset.FileDataset
SampleID = int
SeriesID = str
Size2D = Tuple[int, int] | Int[np.ndarray, "2"]
Size3D = Tuple[int, int, int] | Int[np.ndarray, "3"]
Spacing2D = Tuple[float, float] | Float[np.ndarray, "2"]
Spacing3D = Tuple[float, float, float] | Float[np.ndarray, "3"]
SplitID = int
StudyID = str
View = Literal[0, 1, 2]
Voxel = Tuple[int, int, int] | Int[np.ndarray, "3"]
VoxelBox = Int[np.ndarray, "2 3"]
Voxels = Int[np.ndarray, "N 3"]
Window = Tuple[float, float]

# Second-order types (composed of first-order types).
AffineMatrix = AffineMatrix2D | AffineMatrix3D
BatchBox = BatchBox2D | BatchBox3D
BatchChannelImage = BatchChannelImage2D | BatchChannelImage3D
BatchChannelLabelImage = BatchChannelLabelImage2D | BatchChannelLabelImage3D
BatchImage = BatchImage2D | BatchImage3D
BatchLabelImage = BatchLabelImage2D | BatchLabelImage3D
Box = Box2D | Box3D
ChannelImage = ChannelImage2D | ChannelImage3D
ID = DatasetID | PatientID | SeriesID | StudyID
Image = Image2D | Image3D
Indices = Indices2D | Indices3D
LabelImage = LabelImage2D | LabelImage3D
Landmark = Landmark2D | Landmark3D
Landmarks = Landmarks2D | Landmarks3D
Point = Point2D | Point3D
Points = Points2D | Points3D
Size = Size2D | Size3D
Spacing = Spacing2D | Spacing3D
