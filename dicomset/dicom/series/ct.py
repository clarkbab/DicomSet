from __future__ import annotations

import os
import pandas as pd
import pydicom as dcm
from typing import Any, Callable, Dict, List, TYPE_CHECKING

from ... import config
from ...typing import AffineMatrix3D, Box3D, Image3D, Point3D, SeriesID, Size3D, Spacing3D
from ...utils.geometry import affine_origin, affine_spacing, fov
from ...utils.python import has_private_attr
from ..utils.dicom import from_ct_dicom
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

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    def affine(self) -> AffineMatrix3D:
        return self.__affine

    @property
    @ensure_loaded
    def data(self) -> Image3D:
        return self.__data
    
    # Could return 'CTFile' objects - this would align with other series, but would create a lot of objects in memory.
    @property
    def dicoms(self) -> List[dcm.dataset.FileDataset]:
        # Sort CTs by z position, smallest first.
        ct_dicoms = [dcm.dcmread(f, force=False) for f in self.__filepaths]
        ct_dicoms = list(sorted(ct_dicoms, key=lambda c: c.ImagePositionPatient[2]))
        return ct_dicoms

    @property
    def filepath(self) -> str:
        return self.__filepaths[0]

    @property
    def filepaths(self) -> List[str]:
        return self.__filepaths

    @ensure_loaded
    def fov(
        self,
        **kwargs,
        ) -> Box3D:
        return fov(self.__data.shape, affine=self.__affine, **kwargs)

    def __load_data(self) -> None:
        # Consistency is checked during indexing.
        # TODO: Change 'check_consistency' to be more granular and set based on the index policy.
        self.__data, self.__affine = from_ct_dicom(self.dicoms)

    @property
    @ensure_loaded
    def origin(self) -> Point3D:
        return affine_origin(self.__affine)

    @property
    @ensure_loaded
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> Spacing3D:
        return affine_spacing(self.__affine)

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepaths']
for p in props:
    setattr(DicomCtSeries, p, property(lambda self, p=p: getattr(self, f'_{DicomCtSeries.__name__}__{p}')))
