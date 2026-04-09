from __future__ import annotations

import os
import pandas as pd
import pydicom as dcm
from typing import Any, Callable, Dict, TYPE_CHECKING

from ... import config
from ...typing import Box3D, Image3D, Point3D, SeriesID, Size3D, Spacing3D
from ...utils.geometry import fov
from ...utils.python import has_private_attr
from ..utils.dicom import from_rtdose_dicom
from .series import DicomSeries
if TYPE_CHECKING:
    from ..dataset import DicomDataset
    from ..patient import DicomPatient
    from ..study import DicomStudy

DICOM_RTDOSE_REF_RTPLAN_KEY = 'RefRTPLANSOPInstanceUID'

class DicomRtDoseSeries(DicomSeries):
    def __init__(
        self,
        dataset: DicomDataset,
        patient: DicomPatient,
        study: DicomStudy,
        id: SeriesID,
        index: pd.Series,
        index_policy: Dict[str, Any],
        ) -> None:
        super().__init__('rtdose', dataset, patient, study, id, index=index, index_policy=index_policy)
        dspath = os.path.join(config.directories.datasets, 'dicom', self._dataset.id, 'data', 'patients')
        self.__filepath = os.path.join(dspath, index['filepath'])

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> Image3D:
        return self.__data

    @property
    def dicom(self) -> dcm.dataset.FileDataset:
        return dcm.dcmread(self.__filepath)

    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Box3D:
        return fov(self.__data, origin=self.__origin, spacing=self.__spacing, **kwargs)

    def __load_data(self) -> None:
        self.__data, self.__spacing, self.__origin = from_rtdose_dicom(self.dicom)

    @property
    @ensure_loaded
    def origin(self) -> Point3D:
        return self.__origin

    @property
    @ensure_loaded
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> Spacing3D:
        return self.__spacing

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepath', 'ref_rtplan']
for p in props:
    setattr(DicomRtDoseSeries, p, property(lambda self, p=p: getattr(self, f'_{DicomRtDoseSeries.__name__}__{p}')))
