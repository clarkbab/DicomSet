
from __future__ import annotations

from typing import TYPE_CHECKING

from ...mixins import IndexMixin
from ...series import Series
from ...typing import NiftiModality
from ...utils.python import get_private_attr, set_private_attr
if TYPE_CHECKING:
    from ...dicom.series import DicomSeries

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
        set_private_attr(self, '__modality', modality)
        self.__dicom_modality = NIFTI_DICOM_MODALITY_MAP[get_private_attr(self, '__modality')]
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
        return get_private_attr(self, '__modality')
