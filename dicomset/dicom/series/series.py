from datetime import datetime as dt

from ...mixins import IndexMixin
from ...series import Series
from ...typing import DicomModality
from ...utils.python import get_private_attr, set_private_attr
from ..utils.dicom import DICOM_DATE_FORMAT, DICOM_TIME_FORMAT

# Abstract class.
class DicomSeries(IndexMixin, Series):
    def __init__(
        self,
        modality: DicomModality,
        *args,
        **kwargs,
        ) -> None:
        set_private_attr(self, '__modality', modality)
        super().__init__(*args, **kwargs)

    @property
    def date(self) -> str:
        date_str = self.index['study-date']
        time_str = self.index['study-time']
        return f'{date_str}:{time_str}'

    @property
    def datetime(self) -> dt:
        parsed_dt = dt.strptime(self.date, f'{DICOM_DATE_FORMAT}:{DICOM_TIME_FORMAT}')
        return parsed_dt

    @property
    def modality(self) -> DicomModality:
        return get_private_attr(self, '__modality')
 