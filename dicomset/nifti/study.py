from __future__ import annotations

import os
import pandas as pd
from typing import Dict, List, Literal, TYPE_CHECKING

from .. import config
from ..dicom.dataset import DicomDataset
from ..mixins import IndexMixin
from ..region_map import RegionMap
from ..study import Study
from ..typing import NiftiModality, SeriesID, StudyID
from ..utils.args import arg_to_list, resolve_id
from ..utils.logging import logger
from .series import NiftiCtSeries, NiftiDoseSeries, NiftiLandmarksSeries, NiftiMrSeries, NiftiRegionsSeries, NiftiSeries
from .series.shared import IMAGE_EXTENSIONS
if TYPE_CHECKING:
    from ..dicom.study import DicomStudy
    from .dataset import NiftiDataset
    from .patient import NiftiPatient
    from .series import NiftiImageSeries

class NiftiStudy(IndexMixin, Study):
    def __init__(
        self,
        dataset: NiftiDataset,
        patient: NiftiPatient,
        id: StudyID,
        ct_from: NiftiStudy | None = None,
        index: pd.DataFrame | None = None,
        region_map: RegionMap | None = None,
        ) -> None:
        super().__init__(dataset, patient, id, ct_from=ct_from, index=index, region_map=region_map)
        self.__path = os.path.join(config.directories.datasets, 'nifti', self.__dataset.id, 'data', 'patients', self.__patient.id, self.__id)
        if not os.path.exists(self.__path):
            raise ValueError(f"No nifti study '{self.__id}' found at path: {self.__path}")

    def default_series(
        self,
        modality: NiftiModality,
        ) -> NiftiSeries | None:
        serieses = self.list_series(modality)
        if len(serieses) > 1:
            logger.warn(f"More than one '{modality}' series found for '{self}', defaulting to latest.")
        return self.series(serieses[-1], modality) if len(serieses) > 0 else None

    @property
    def dicom(self) -> DicomStudy:
        if self.__index is None:
            raise ValueError(f"Missing 'index.csv' for dataset '{self.__dataset.id}', cannot find corresponding dicom study.")
        index = self.__index[['dataset', 'patient-id', 'study-id', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id']]
        index = index[(index['dataset'] == self.__dataset.id) & (index['patient-id'] == self.__patient.id) & (index['study-id'] == self.__id)].drop_duplicates()
        assert len(index) == 1, f"Expected 1 index entry for DICOM study '{self.__id}', but found {len(index)}. Index: {index}"
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id'])

    def has_series(
        self,
        id: SeriesID,
        modality: NiftiModality,
        ) -> bool:
        return id in self.list_series(modality)

    def list_series(
        self,
        modality: NiftiModality,
        series_id: SeriesID | List[SeriesID] | Literal['all'] = 'all',
        ) -> List[SeriesID]:
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        if modality == 'ct':
            if self.__ct_from is None:
                dirpath = os.path.join(self.__path, 'ct')
                series_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
                series_ids = [i.replace(e, '') for i in series_ids for e in IMAGE_EXTENSIONS if i.endswith(e)]
            else:
                series_ids = self.__ct_from.list_series(modality)
        elif modality == 'dose':
            dirpath = os.path.join(self.__path, 'dose')
            series_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            series_ids = [i.replace(e, '') for i in series_ids for e in IMAGE_EXTENSIONS if i.endswith(e)]
        elif modality == 'landmarks':
            dirpath = os.path.join(self.__path, 'landmarks')
            series_ids = list(sorted(f.replace('.csv', '') for f in os.listdir(dirpath))) if os.path.exists(dirpath) else []
        elif modality == 'mr':
            dirpath = os.path.join(self.__path, 'mr')
            series_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            series_ids = [i.replace(e, '') for i in series_ids for e in IMAGE_EXTENSIONS if i.endswith(e)]
        elif modality == 'regions':
            dirpath = os.path.join(self.__path, 'regions')
            series_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
        else:
            raise ValueError(f"Unknown modality '{modality}'.")

        # Filter by series ID.
        if series_id != 'all':
            series_ids = arg_to_list(series_id, str)
            series_ids = [s for s in series_ids if s in series_ids]

        return series_ids

    @property
    def origin(self) -> Dict[str, str]:
        if self.__index is None:
            raise ValueError(f"No 'index.csv' provided for dataset '{self.__dataset}'.")
        info = self.__index.iloc[0].to_dict()
        info = {k: info[k] for k in ['dicom-dataset', 'dicom-patient-id', 'dicom-study-id']}
        return info

    def series(
        self,
        id: SeriesID | int,
        modality: NiftiModality,
        ) -> NiftiImageSeries | NiftiLandmarksSeries | NiftiRegionsSeries:
        if modality == 'ct':
            id = resolve_id(id, lambda: self.list_series('ct'))
            if self.__ct_from is None:
                index = self.__index[(self.__index['dataset'] == self.__dataset.id) & (self.__index['patient-id'] == self.__patient.id) & (self.__index['study-id'] == self.__id) & (self.__index['series-id'] == id) & (self.__index['modality'] == 'ct')].copy() if self.__index is not None else None
                return NiftiCtSeries(self.__dataset, self.__patient, self, id, index=index)
            else:
                return self.__ct_from.series(id, modality)
        elif modality == 'dose':
            id = resolve_id(id, lambda: self.list_series('dose'))
            # Could multiple series have the same series-id? Yeah definitely.
            index = self.__index[(self.__index['dataset'] == self.__dataset.id) & (self.__index['patient-id'] == self.__patient.id) & (self.__index['study-id'] == self.__id) & (self.__index['series-id'] == id) & (self.__index['modality'] == 'dose')].copy() if self.__index is not None else None
            return NiftiDoseSeries(self.__dataset, self.__patient, self, id, index=index)
        elif modality == 'landmarks':
            id = resolve_id(id, lambda: self.list_series('landmarks'))
            index = self.__index[(self.__index['dataset'] == self.__dataset.id) & (self.__index['patient-id'] == self.__patient.id) & (self.__index['study-id'] == self.__id) & (self.__index['series-id'] == id) & (self.__index['modality'] == 'landmarks')].copy() if self.__index is not None else None
            ref_ct = self.default_series('ct')
            ref_dose = self.default_series('dose')
            return NiftiLandmarksSeries(self.__dataset, self.__patient, self, id, index=index, ref_ct=ref_ct, ref_dose=ref_dose)
        elif modality == 'mr':
            id = resolve_id(id, lambda: self.list_series('mr'))
            index = self.__index[(self.__index['dataset'] == self.__dataset.id) & (self.__index['patient-id'] == self.__patient.id) & (self.__index['study-id'] == self.__id) & (self.__index['series-id'] == id) & (self.__index['modality'] == 'mr')].copy() if self.__index is not None else None
            return NiftiMrSeries(self.__dataset, self.__patient, self, id, index=index)
        elif modality == 'regions':
            id = resolve_id(id, lambda: self.list_series('regions'))
            index = self.__index[(self.__index['dataset'] == self.__dataset.id) & (self.__index['patient-id'] == self.__patient.id) & (self.__index['study-id'] == self.__id) & (self.__index['series-id'] == id) & (self.__index['modality'] == 'regions')].copy() if self.__index is not None else None
            return NiftiRegionsSeries(self.__dataset, self.__patient, self, id, index=index, region_map=self.__region_map)
        else:
            raise ValueError(f"Unknown NiftiSeries modality '{modality}'.")

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add 'list_{mod}_series' methods.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'list_{m}_series', lambda self, m=m, **kwargs: self.list_series(m, **kwargs))

# Add '{mod}_series' methods.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'{m}_series', lambda self, series, m=m, **kwargs: self.series(series, m, **kwargs))
    
# Add 'has_{mod}' properties.
# Note that 'has_landmarks' refers to the landmarks series, whereas 'has_landmark' is used for
# a single landmark ID. Same for regions. Could be confusing.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'has_{m}', property(lambda self, m=m: self.default_series(m) is not None))

# Add 'default_{mod}' properties.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'default_{m}', property(lambda self, m=m: self.default_series(m)))
    
# Add image filepath shortcuts from 'default_series(mod)'
mods = ['ct', 'mr', 'dose']
for m in mods:
    setattr(NiftiStudy, f'{m}_filepath', property(lambda self, m=m: getattr(self.default_series(m), 'filepath') if self.default_series(m) is not None else None))
setattr(NiftiStudy, 'region_filepaths', lambda self, region: self.default_series('regions').filepaths(region) if self.default_series('regions') is not None else None)

# Add image property shortcuts from 'default_series(mod)'.
mods = ['ct', 'mr', 'dose']
props = ['affine', 'data', 'fov', 'origin', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiStudy, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_series(m), p) if self.default_series(m) is not None else None))

# Add landmark/region method shortcuts from 'default_series(mod)'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(NiftiStudy, f'has_{m[:-1]}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'has_{m[:-1]}')(*args, **kwargs) if self.default_series(m) is not None else False)
    setattr(NiftiStudy, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'list_{m}')(*args, **kwargs) if self.default_series(m) is not None else [])
    setattr(NiftiStudy, f'{m}_data', lambda self, *args, m=m, **kwargs: self.default_series(m).data(*args, **kwargs) if self.default_series(m) is not None else None)
