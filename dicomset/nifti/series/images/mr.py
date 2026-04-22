from __future__ import annotations

import os
import pandas as pd
from typing import TYPE_CHECKING

from .... import config
from ....dicom import DicomDataset
from ....typing import SeriesID
from ....utils.io import load_nifti, load_nrrd
from ....utils.python import get_private_attr
from ..shared import IMAGE_EXTENSIONS
from .image import NiftiImageSeries
if TYPE_CHECKING:
    from ....dicom import DicomMrSeries
    from ...dataset import NiftiDataset
    from ...patient import NiftiPatient
    from ...study import NiftiStudy

class NiftiMrSeries(NiftiImageSeries):
    def __init__(
        self,
        dataset: NiftiDataset,
        pat: NiftiPatient,
        study: NiftiStudy,
        id: SeriesID,
        index: pd.DataFrame | None = None
        ) -> None:
        super().__init__('mr', dataset, pat, study, id, index=index)
        basepath = os.path.join(config.directories.datasets, 'nifti', self._dataset.id, 'data', 'patients', self._pat.id, self._study.id, self._modality, self._id)
        filepath = None
        for e in IMAGE_EXTENSIONS:
            fpath = f"{basepath}{e}"
            if os.path.exists(fpath):
                filepath = fpath
        if filepath is None:
            raise ValueError(f"No nifti mr series found for study '{self._study.id}'. Filepath: {basepath}, with extensions {IMAGE_EXTENSIONS}.")
        self.__filepath = filepath

    @property
    def dicom(self) -> DicomMrSeries:
        if self.__index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self.__index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self._dataset.id) & (index['patient-id'] == self._pat.id) & (index['study-id'] == self._study.id) & (index['series-id'] == self._id) & (index['modality'] == 'mr')].drop_duplicates()
        assert len(index) == 1, f"Expected 1 index entry for DICOM MR series '{self._id}', but found {len(index)}. Index: {index}"
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).mr_series(row['dicom-series-id'])

    def __load_data(self) -> None:
        if self.__filepath.endswith('.nii') or self.__filepath.endswith('.nii.gz'):
            self.__data, self.__affine = load_nifti(self.__filepath)
        elif self.__filepath.endswith('.nrrd'):
            self.__data, self.__affine = load_nrrd(self.__filepath)
        else:
            raise ValueError(f'Unsupported file format: {self.__filepath}')

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepath']
for p in props:
    setattr(NiftiMrSeries, p, property(lambda self, p=p: get_private_attr(self, f'__{p}')))
