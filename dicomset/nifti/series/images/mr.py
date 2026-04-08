from __future__ import annotations

import os
import pandas as pd
from typing import Callable, TYPE_CHECKING

from .... import config
from ....dicom import DicomDataset
if TYPE_CHECKING:
    from ....dicom import DicomMrSeries
from ....typing import Image3D, Point3D, SeriesID, Size3D, Spacing3D
from ....utils.io import load_nifti, load_nrrd
from ....utils.python import has_private_attr
if TYPE_CHECKING:
    from ...dataset import NiftiDataset
    from ...patient import NiftiPatient
    from ...study import NiftiStudy
from .image import NiftiImageSeries

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
        extensions = ['.nii', '.nii.gz', '.nrrd']
        basepath = os.path.join(config.directories.datasets, 'nifti', self._dataset.id, 'data', 'patients', self._pat.id, self._study.id, self._modality, self._id)
        filepath = None
        for e in extensions:
            fpath = f"{basepath}{e}"
            if os.path.exists(fpath):
                filepath = fpath
        if filepath is None:
            raise ValueError(f"No nifti mr series found for study '{self._study.id}'. Filepath: {basepath}, with extensions {extensions}.")
        self.__filepath = filepath

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                if self.__filepath.endswith('.nii') or self.__filepath.endswith('.nii.gz'):
                    self.__data, self.__affine = load_nifti(self.__filepath)
                elif self.__filepath.endswith('.nrrd'):
                    self.__data, self.__affine = load_nrrd(self.__filepath)
                else:
                    raise ValueError(f'Unsupported file format: {self.__filepath}')
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> Image3D:
        return self.__data

    @property
    def dicom(self) -> DicomMrSeries:
        if self._index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self._index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self._dataset.id) & (index['patient-id'] == self._pat.id) & (index['study-id'] == self._study.id) & (index['series-id'] == self._id) & (index['modality'] == 'mr')].drop_duplicates()
        assert len(index) == 1
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).mr_series(row['dicom-series-id'])

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
props = ['filepath']
for p in props:
    setattr(NiftiMrSeries, p, property(lambda self, p=p: getattr(self, f'_{NiftiMrSeries.__name__}__{p}')))
