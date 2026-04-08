
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...dicom.series import DicomSeries
from ...mixins import IndexMixin
from ...series import Series
from ...typing import NiftiModality

NIFTI_DICOM_MODALITY_MAP = dict(
    ct='ct',
    dose='rtdose',
    landmarks='rtstruct',
    mr='mr',
    plan='rtplan',
    regions='rtstruct',
)

class NiftiSeries(IndexMixin, Series):
    def __init__(
        self,
        modality: NiftiModality,
        *args,
        **kwargs,
        ) -> None:
        self._modality = modality
        self.__dicom_modality = NIFTI_DICOM_MODALITY_MAP[self._modality]
        super().__init__(*args, **kwargs)

    @property
    def date(self) -> str | None:
        # May implement in dicom -> nifti processing in future.
        return None

    @property
    def dicom(self) -> DicomSeries:
        raise ValueError("Subclasses of 'NiftiSeries' must implement 'dicom' method.")

    @property
    def modality(self) -> NiftiModality:
        return self._modality
