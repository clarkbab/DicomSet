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
if TYPE_CHECKING:
    from ..dicom.study import DicomStudy
    from .dataset import NiftiDataset
    from .patient import NiftiPatient
    from .series import NiftiImageSeries

class NiftiStudy(IndexMixin, Study):
    def __init__(
        self,
        dataset: NiftiDataset,
        pat: NiftiPatient,
        id: StudyID,
        ct_from: NiftiStudy | None = None,
        index: pd.DataFrame | None = None,
        region_map: RegionMap | None = None,
        ) -> None:
        super().__init__(dataset, pat, id, ct_from=ct_from, index=index, region_map=region_map)
        self.__path = os.path.join(config.directories.datasets, 'nifti', self._dataset.id, 'data', 'patients', self._pat.id, self._id)
        if not os.path.exists(self.__path):
            raise ValueError(f"No nifti study '{self._id}' found at path: {self.__path}")

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
        if self._index is None:
            raise ValueError(f"Missing 'index.csv' for dataset '{self._dataset.id}', cannot find corresponding dicom study.")
        index = self._index[['dataset', 'patient-id', 'study-id', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id']]
        index = index[(index['dataset'] == self._dataset.id) & (index['patient-id'] == self._pat.id) & (index['study-id'] == self._id)].drop_duplicates()
        assert len(index) == 1, f"Expected 1 index entry for DICOM study '{self._id}', but found {len(index)}. Index: {index}"
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
            if self._ct_from is None:
                dirpath = os.path.join(self.__path, 'ct')
                series_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
                series_ids = [i.replace(e, '') for i in series_ids for e in image_extensions if i.endswith(e)]
            else:
                series_ids = self._ct_from.list_series(modality)
        elif modality == 'dose':
            dirpath = os.path.join(self.__path, 'dose')
            series_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            series_ids = [i.replace(e, '') for i in series_ids for e in image_extensions if i.endswith(e)]
        elif modality == 'landmarks':
            dirpath = os.path.join(self.__path, 'landmarks')
            series_ids = list(sorted(f.replace('.csv', '') for f in os.listdir(dirpath))) if os.path.exists(dirpath) else []
        elif modality == 'mr':
            dirpath = os.path.join(self.__path, 'mr')
            series_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            series_ids = [i.replace(e, '') for i in series_ids for e in image_extensions if i.endswith(e)]
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
        if self._index is None:
            raise ValueError(f"No 'index.csv' provided for dataset '{self._dataset}'.")
        info = self._index.iloc[0].to_dict()
        info = {k: info[k] for k in ['dicom-dataset', 'dicom-patient-id', 'dicom-study-id']}
        return info

    def series(
        self,
        id: SeriesID,
        modality: NiftiModality,
        ) -> NiftiImageSeries | NiftiLandmarksSeries | NiftiRegionsSeries:
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        if modality == 'ct':
            id = resolve_id(id, lambda: self.list_series('ct'))
            if self._ct_from is None:
                index = self._index[(self._index['dataset'] == self._dataset.id) & (self._index['patient-id'] == self._pat.id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'ct')].copy() if self._index is not None else None
                return NiftiCtSeries(self._dataset, self._pat, self, id, index=index)
            else:
                return self._ct_from.series(id, modality)
        elif modality == 'dose':
            id = resolve_id(id, lambda: self.list_series('dose'))
            # Could multiple series have the same series-id? Yeah definitely.
            index = self._index[(self._index['dataset'] == self._dataset.id) & (self._index['patient-id'] == self._pat.id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'dose')].copy() if self._index is not None else None
            return NiftiDoseSeries(self._dataset, self._pat, self, id, index=index)
        elif modality == 'landmarks':
            id = resolve_id(id, lambda: self.list_series('landmarks'))
            index = self._index[(self._index['dataset'] == self._dataset.id) & (self._index['patient-id'] == self._pat.id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'landmarks')].copy() if self._index is not None else None
            ref_ct = self.default_series('ct')
            ref_dose = self.default_series('dose')
            return NiftiLandmarksSeries(self._dataset, self._pat, self, id, index=index, ref_ct=ref_ct, ref_dose=ref_dose)
        elif modality == 'mr':
            id = resolve_id(id, lambda: self.list_series('mr'))
            index = self._index[(self._index['dataset'] == self._dataset.id) & (self._index['patient-id'] == self._pat.id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'mr')].copy() if self._index is not None else None
            return NiftiMrSeries(self._dataset, self._pat, self, id, index=index)
        elif modality == 'regions':
            id = resolve_id(id, lambda: self.list_series('regions'))
            index = self._index[(self._index['dataset'] == self._dataset.id) & (self._index['patient-id'] == self._pat.id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'regions')].copy() if self._index is not None else None
            return NiftiRegionsSeries(self._dataset, self._pat, self, id, index=index, region_map=self._region_map)
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
