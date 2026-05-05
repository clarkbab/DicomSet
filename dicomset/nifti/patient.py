from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import List, Literal, TYPE_CHECKING

from .. import config
from ..dicom.dataset import DicomDataset
from ..mixins import IndexMixin
from ..patient import Patient
from ..region_map import RegionMap
from ..typing import PatientID, StudyID
from ..utils.args import arg_to_list, resolve_id
from .study import NiftiStudy
if TYPE_CHECKING:
    from ..dicom.patient import DicomPatient
    from .dataset import NiftiDataset

class NiftiPatient(IndexMixin, Patient):
    def __init__(
        self,
        dataset: NiftiDataset,
        id: PatientID,
        ct_from: NiftiPatient | None = None,
        index: pd.DataFrame | None = None,
        excluded_labels: List[str] | None = None,
        region_map: RegionMap | None = None,
        ) -> None:
        super().__init__(dataset, id, ct_from=ct_from, index=index, region_map=region_map)
        self.__path = os.path.join(config.directories.datasets, 'nifti', self.__dataset.id, 'data', 'patients', self.__id)
        if not os.path.exists(self.__path):
            raise ValueError(f"No nifti patient '{self.__id}' found at path: {self.__path}")

    @property
    def default_study(self) -> NiftiStudy | None:
        studys = self.list_studies()
        return self.study(studys[-1]) if len(studys) > 0 else None

    @property
    def dicom(self) -> DicomPatient:
        if self.__index is None:
            raise ValueError(f"Missing 'index.csv' for dataset '{self.__dataset.id}', cannot find corresponding dicom patient.")
        index = self.__index[['dataset', 'patient-id', 'dicom-dataset', 'dicom-patient-id']]
        index = index[(index['dataset'] == self.__dataset.id) & (index['patient-id'] == self.__id)].drop_duplicates()
        assert len(index) == 1, f"Expected 1 index entry for DICOM patient '{self.__id}', but found {len(index)}. Index: {index}"
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id'])

    def has_study(
        self,
        study_id: StudyID | List[StudyID],
        any: bool = False,
        **kwargs,
        ) -> bool:
        real_ids = self.list_studies(study_id=study_id, **kwargs)
        req_ids = arg_to_list(study_id, StudyID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    def list_studies(
        self,
        study_id: StudyID | List[StudyID] | Literal['all'] = 'all',
        ) -> List[StudyID]:
        # Might have to deal with sorting at some point for 'default_study'.
        # Right now sorting is just alphabetical, which is fine if we're using anonymous IDs,
        # as they're sorted during DICOM -> NIFTI conversion.
        ids = list(sorted(os.listdir(self.__path)))
        if study_id != 'all':
            study_ids = arg_to_list(study_id, str)
            all_ids = ids.copy()
            ids = []
            for i, id in enumerate(all_ids):
                # Check if any of the passed 'studys' references this ID.
                for j, sid in enumerate(study_ids):
                    if sid.startswith('i:'):
                        # Check if idx refer
                        idx = int(sid.split(':')[1])
                        if i == idx or (idx < 0 and i == len(all_ids) + idx):   # Allow negative indexing.
                            ids.append(id)
                            break
                    elif id == sid:
                        ids.append(id)
                        break

        return ids

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

    def study(
        self,
        id: StudyID | int,
        **kwargs,
        ) -> NiftiStudy:
        id = resolve_id(id, self.list_studies)
        index = self.__index[self.__index['study-id'] == id].copy() if self.__index is not None else None

        # Get 'ct_from' study.
        if self.__ct_from is not None and self.__ct_from.has_study(id):
            ct_from = self.__ct_from.study(id)
        else:
            ct_from = None

        return NiftiStudy(self.__dataset, self, id, ct_from=ct_from, index=index, region_map=self.__region_map, **kwargs)

# Add properties/methods from 'default_study'.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiPatient, f'default_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))
    setattr(NiftiPatient, f'has_{m}', property(lambda self, m=m: getattr(self.default_study, f'has_{m}') if self.default_study is not None else None))
    setattr(NiftiPatient, f'{m}_series', lambda self, *args, m=m: self.default_study.series(*args, m) if self.default_study is not None else None)
    setattr(NiftiPatient, f'list_{m}_series', lambda self, *args, m=m: self.default_study.list_series(*args, m) if self.default_study is not None else None)
setattr(NiftiPatient, 'list_series', lambda self, *args, **kwargs: self.default_study.list_series(*args, **kwargs) if self.default_study is not None else None)

# Add image properties from 'default_study'.
mods = ['ct', 'dose', 'mr']
props = ['affine', 'data', 'fov', 'origin', 'size', 'spacing']
for m in mods:
    setattr(NiftiPatient, f'{m}_filepath', property(lambda self, m=m: getattr(self.default_study, f'{m}_filepath') if self.default_study is not None else None))
    for p in props:
        setattr(NiftiPatient, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_study, f'{m}_{p}') if self.default_study is not None else None))
setattr(NiftiPatient, 'region_filepaths', lambda self, region: self.default_study.region_filepaths(region) if self.default_study is not None else None)

# Add landmarks/regions methods from 'default_study'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(NiftiPatient, f'has_{m[:-1]}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'has_{m[:-1]}')(*args, **kwargs) if self.default_study is not None else False)
    setattr(NiftiPatient, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'list_{m}')(*args, **kwargs) if self.default_study is not None else [])
    setattr(NiftiPatient, f'{m}_data', lambda self, *args, m=m, **kwargs: getattr(self.default_study, f'{m}_data')(*args, **kwargs) if self.default_study is not None else None)
