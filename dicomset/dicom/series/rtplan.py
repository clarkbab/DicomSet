from __future__ import annotations

import os
import pandas as pd
import pydicom as dcm
from typing import Any, Dict, TYPE_CHECKING

from ... import config
from ...typing import SeriesID
from ..utils.io import load_dicom
from .series import DicomSeries
if TYPE_CHECKING:
    from ..dataset import DicomDataset
    from ..patient import DicomPatient
    from ..study import DicomStudy

DICOM_RTPLAN_REF_RTSTRUCT_KEY = 'RefRTSTRUCTSOPInstanceUID'

class DicomRtPlanSeries(DicomSeries):
    def __init__(
        self,
        dataset: DicomDataset,
        pat: DicomPatient,
        study: DicomStudy,
        id: SeriesID,
        index: pd.Series,
        index_policy: Dict[str, Any],
        ) -> None:
        super().__init__('rtplan', dataset, pat, study, id, index=index, index_policy=index_policy)
        dspath = os.path.join(config.directories.datasets, 'dicom', self._dataset.id, 'data', 'patients')
        self.__filepath = os.path.join(dspath, index['filepath'])

    @property
    def dicom(self) -> dcm.dataset.FileDataset:
        return load_dicom(self.__filepath)

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepath', 'ref_rtstruct']
for p in props:
    setattr(DicomRtPlanSeries, p, property(lambda self, p=p: getattr(self, f'_{DicomRtPlanSeries.__name__}__{p}')))


