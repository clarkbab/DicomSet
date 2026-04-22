from __future__ import annotations

import os
import pandas as pd
from typing import Any, Dict, List, TYPE_CHECKING

from ... import config
from ...typing import AffineMatrix3D, Box3D, CtDicom, Image3D, Point3D, SeriesID, Size3D, Spacing3D
from ...utils.geometry import affine_origin, affine_spacing, fov
from ...utils.python import ensure_loaded, get_private_attr
from ..utils.dicom import from_ct_dicom
from ..utils.io import load_dicom
from .series import DicomSeries
if TYPE_CHECKING:
    from ..dataset import DicomDataset
    from ..patient import DicomPatient
    from ..study import DicomStudy

class DicomCtSeries(DicomSeries):
    def __init__(
        self,
        dataset: DicomDataset,
        patient: DicomPatient,
        study: DicomStudy,
        id: SeriesID,
        index: pd.DataFrame,
        index_policy: Dict[str, Any],
        ) -> None:
        super().__init__('ct', dataset, patient, study, id, index=index, index_policy=index_policy)
        dspath = os.path.join(config.directories.datasets, 'dicom', self._dataset.id, 'data', 'patients')
        relpaths = list(index['filepath'])
        abspaths = [os.path.join(dspath, p) for p in relpaths]
        self.__filepaths = abspaths

    @property
    @ensure_loaded('__data', '__load_data')
    def affine(self) -> AffineMatrix3D:
        return self.__affine

    @property
    @ensure_loaded('__data', '__load_data')
    def data(self) -> Image3D:
        return self.__data
    
    @property
    @ensure_loaded('__data', '__load_data')
    def dicoms(self) -> List[CtDicom]:
        return self.__dicoms

    @property
    def filepath(self) -> str:
        return self.__filepaths[0]

    @property
    def filepaths(self) -> List[str]:
        return self.__filepaths

    @ensure_loaded('__data', '__load_data')
    def fov(
        self,
        **kwargs,
        ) -> Box3D:
        return fov(self.__data.shape, affine=self.__affine, **kwargs)

    def __load_data(self) -> None:
        # Load dicoms.
        # Sort CTs by z position, smallest first.
        dicoms = [load_dicom(f, force=False) for f in self.__filepaths]
        dicoms = list(sorted(dicoms, key=lambda c: c.ImagePositionPatient[2]))
        self.__dicoms = dicoms

        # Consistency is checked during indexing.
        # TODO: Change 'check_consistency' to be more granular and set based on the index policy.
        self.__data, self.__affine = from_ct_dicom(dicoms)

    @property
    @ensure_loaded('__data', '__load_data')
    def origin(self) -> Point3D:
        return affine_origin(self.__affine)

    @property
    @ensure_loaded('__data', '__load_data')
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded('__data', '__load_data')
    def spacing(self) -> Spacing3D:
        return affine_spacing(self.__affine)

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepaths']
for p in props:
    setattr(DicomCtSeries, p, property(lambda self, p=p: get_private_attr(self, f'__{p}')))
