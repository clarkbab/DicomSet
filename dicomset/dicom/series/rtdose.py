from __future__ import annotations

import os
import pandas as pd
from typing import Any, Dict, TYPE_CHECKING

from ... import config
from ...typing import AffineMatrix3D, Box3D, Image3D, Point3D, SeriesID, Size3D, Spacing3D
from ...utils.geometry import affine_origin, affine_spacing, fov
from ...utils.python import ensure_loaded, get_private_attr
from ..utils.dicom import from_rtdose_dicom
from ..utils.io import load_dicom
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
        dspath = os.path.join(config.directories.datasets, 'dicom', self.__dataset.id, 'data', 'patients')
        self.__filepath = os.path.join(dspath, index['filepath'])

    @property
    @ensure_loaded('__affine', '__load_data')
    def affine(self) -> AffineMatrix3D:
        return self.__affine

    @property
    @ensure_loaded('__data', '__load_data')
    def data(self) -> Image3D:
        return self.__data

    @property
    @ensure_loaded('__dicom', '__load_data')
    def dicom(self) -> RtDoseDicom:
        return self.__dicom

    @ensure_loaded('__data', '__load_data')
    def fov(
        self,
        **kwargs) -> Box3D:
        return fov(self.__data.shape, self.__affine, **kwargs)

    def __load_data(self) -> None:
        self.__dicom = load_dicom(self.__filepath)
        self.__data, self.__affine = from_rtdose_dicom(self.__dicom)

    @property
    @ensure_loaded('__affine', '__load_data')
    def origin(self) -> Point3D:
        return affine_origin(self.__affine)

    @property
    @ensure_loaded('__data', '__load_data')
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded('__affine', '__load_data')
    def spacing(self) -> Spacing3D:
        return affine_spacing(self.__affine)

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepath', 'ref_rtplan']
for p in props:
    setattr(DicomRtDoseSeries, p, property(lambda self, p=p: get_private_attr(self, f'__{p}')))
