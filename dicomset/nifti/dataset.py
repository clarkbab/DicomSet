import numpy as np
import os
import pandas as pd
import re
from typing import List, Literal

from .. import config
from ..dataset import CT_FROM_REGEXP, Dataset
from ..dicom import DicomDataset
from ..mixins import IndexMixin
from ..region_map import RegionMap
from ..typing import DatasetID, GroupID, PatientID, RegionID
from ..utils import logging
from ..utils.args import alias_kwargs, arg_to_list, resolve_id
from ..utils.io import load_csv
from ..utils.python import ensure_loaded
from .patient import NiftiPatient

class NiftiDataset(IndexMixin, Dataset):
    def __init__(
        self,
        id: DatasetID,
        ) -> None:
        self.__path = os.path.join(config.directories.datasets, 'nifti', str(id))
        if not os.path.exists(self.__path):
            raise ValueError(f"No nifti dataset '{id}' found at path: {self.__path}")
        ct_from = None
        for f in os.listdir(self.__path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from = match.group(1)
        ct_from = NiftiDataset(ct_from) if ct_from is not None else None
        super().__init__(id, ct_from=ct_from)

    @property
    def dicom(self) -> DicomDataset:
        if self.__index is None:
            raise ValueError(f"Missing 'index.csv' for dataset '{self._dataset.id}', cannot find corresponding dicom dataset.")
        ds = self.__index['dicom-dataset'].unique().tolist()
        assert len(ds) == 1
        return DicomDataset(ds[0])

    def has_patient(
        self,
        patient_id: PatientID | List[PatientID] | Literal['all'] = 'all',
        all: bool = False,
        ) -> bool:
        all_pats = self.list_patients()
        subset_pats = self.list_patients(patient_id=patient_id)
        n_overlap = len(np.intersect1d(all_pats, subset_pats))
        return n_overlap == len(all_pats) if all else n_overlap > 0

    # 'list_landmarks' can accept 'landmarks' keyword to filter - saves code elsewhere.
    def list_landmarks(self, *args, **kwargs) -> List[str]:
        # Load each patient.
        landmarks = []
        patient_ids = self.list_patients()
        for pat_id in patient_ids:
            pat_landmarks = self.patient(pat_id).list_landmarks(*args, **kwargs)
            landmarks += pat_landmarks
        landmarks = list(sorted(np.unique(landmarks)))
        return landmarks

    @alias_kwargs(
        ('g', 'group_id'),
        ('p', 'patient_id'),
        ('r', 'region_id'),
    )
    @ensure_loaded('__index', '__load_data')
    def list_patients(
        self,
        exclude: PatientID | List[PatientID] | Literal['all'] | None = None,
        group_id: GroupID | List[GroupID] | Literal['all'] = 'all',
        patient_id: PatientID | List[PatientID] | Literal['all'] = 'all',
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        ) -> List[PatientID]:
        # Load patients from filenames.
        dirpath = os.path.join(self.__path, 'data', 'patients')
        all_ids = list(sorted(os.listdir(dirpath)))

        # Filter by region ID.
        ids = all_ids.copy()
        if region_id != 'all':
            all_regions = self.list_regions()

            # # Check that 'regions' are valid.
            # for r in regions:
            #     if not r in all_regions:
            #         logging.warning(f"Filtering by region '{r}' but it doesn't exist in dataset '{self}'.")

            def filter_fn(p: PatientID) -> bool:
                pat_regions = self.patient(p).list_regions(region_id=region_id)
                if len(pat_regions) > 0:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        # Filter by group ID.
        if group_id != 'all':
            if self.__groups is None:
                raise ValueError(f"File 'groups.csv' not found for dicom dataset '{self._id}'.")
            all_groups = self.list_groups()
            group_ids = arg_to_list(group_id, str, literals={ 'all': all_groups })
            for g in group_ids:
                if g not in all_groups:
                    raise ValueError(f"Group {g} not found.")

            def filter_fn(p: PatientID) -> bool:
                pat_groups = self.__groups[self.__groups['patient-id'] == p]
                if len(pat_groups) == 0:
                    return False
                elif len(pat_groups) > 1:
                    raise ValueError(f"Patient {p} is a member of more than one group.")
                pat_group = pat_groups.iloc[0]['group-id']
                if pat_group in group_ids:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        # Filter by 'exclude'.
        if exclude is not None:
            exclude = arg_to_list(exclude, PatientID)
            ids = [p for p in ids if p not in exclude]

        # Filter by patient ID.
        if patient_id != 'all':
            patient_ids = arg_to_list(patient_id, PatientID)

            # # Check that 'patient_ids' are valid.
            # for p in patient_ids:
            #     if not p in all_ids:
            #         logging.warning(f"Filtering by patient ID '{p}' but it doesn't exist in dataset '{self}'.")

            filt_ids = []
            for i, id in enumerate(ids):
                # Check if any of the passed 'patient_ids' references this ID.
                for j, pid in enumerate(patient_ids):
                    if pid.startswith('i:'):
                        if '-' in pid and not 'i:-' in pid:   # Make sure negative indexing doesn't match - probably better with a regexp.
                            # Format 'i:4-8'.
                            min_idx, max_idx = pid.split(':')[1].split('-')
                            min_idx, max_idx = int(min_idx), int(max_idx)
                            if i >= min_idx and i < max_idx:
                                filt_ids.append(id)
                                break
                        else:
                            # Format: 'i:4' or 'i:-4'.
                            idx = int(pid.split(':')[1])
                            if i == idx or (idx < 0 and i == len(ids) + idx):   # Allow negative indexing.
                                filt_ids.append(id)
                                break
                    elif id == pid:
                        filt_ids.append(id)
                        break
            ids = filt_ids
                        
            # if isinstance(patient_ids, str):
            #     # Check for special group format.
            #     regexp = r'^group:(\d+):(\d+)$'
            #     match = re.match(regexp, patient_ids)
            #     if match:
            #         group = int(match.group(1))
            #         num_groups = int(match.group(2))
            #         group_size = int(np.ceil(len(patient_ids) / num_groups))
            #         patient_ids = patient_ids[group * group_size:(group + 1) * group_size]
            #     else:
            #         patient_ids = arg_to_list(patient_ids, PatientID)
            # else:
            #     patient_ids = arg_to_list(patient_ids, PatientID)

        return ids

    def list_regions(
        self,
        patient_id: PatientID | List[PatientID] | Literal['all'] = 'all', 
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        ) -> List[RegionID]:
        # Load all patients.
        patient_ids = self.list_patients(patient_id=patient_id)

        # Trawl the depths for region IDs.
        ids = []
        for p in patient_ids:
            pat = self.patient(p)
            study_ids = pat.list_studies()
            for s in study_ids:
                study = pat.study(s)
                series_ids = study.list_regions_series(region_id=region_id)
                for s in series_ids:
                    series = study.regions_series(s)
                    ids += series.list_regions(region_id=region_id)
        ids = list(str(i) for i in np.unique(ids))

        return ids

    def __load_data(self) -> None:
        # Load index.
        filepath = os.path.join(self.__path, 'index.csv')
        if os.path.exists(filepath):
            map_types = { 'dicom-patient-id': str, 'patient-id': str, 'study-id': str, 'series-id': str }
            self.__index = load_csv(filepath, map_types=map_types)
        else:
            self.__index = None

        # Load groups.
        filepath = os.path.join(self.__path, 'groups.csv')
        self.__groups = load_csv(filepath) if os.path.exists(filepath) else None

        # Load region map.
        self.__region_map = RegionMap.load(self.__path)

    # Copied from 'mymi/reports/dataset/nift.py' to avoid circular dependency.
    def __load_patient_regions_report(
        self,
        exists_only: bool = False,
        ) -> pd.DataFrame | bool:
        filepath = os.path.join(self.__path, 'reports', 'region-count.csv')
        if os.path.exists(filepath):
            if exists_only:
                return True
            else:
                return load_csv(filepath)
        else:
            if exists_only:
                return False
            else:
                raise ValueError(f"Patient regions report doesn't exist for dataset '{self}'.")

    @property
    def n_patients(self) -> int:
        return len(self.list_patients())

    @ensure_loaded('__index', '__load_data')
    def patient(
        self,
        id: PatientID | None = None,
        group_id: GroupID | List[GroupID] | Literal['all'] = 'all',
        n: int | None = None,
        **kwargs) -> NiftiPatient:
        id = resolve_id(id, lambda: self.list_patients(group_id=group_id))
        if n is not None:
            if id is not None:
                raise ValueError("Cannot specify both 'id' and 'n'.")
            id = self.list_patients()[n]

        # Filter indexes to include only rows relevant to the new patient.
        index = self.__index[self.__index['patient-id'] == str(id)].copy() if self.__index is not None else None
        self.__excluded_labels = None
        exc_df = self.__excluded_labels[self.__excluded_labels['patient-id'] == str(id)].copy() if self.__excluded_labels is not None else None

        # Get 'ct_from' patient.
        if self._ct_from is not None and self._ct_from.has_patient(id):
            ct_from = self._ct_from.patient(id)
        else:
            ct_from = None

        return NiftiPatient(self, id, ct_from=ct_from, excluded_labels=exc_df, index=index, region_map=self.__region_map, **kwargs)

    @property
    @ensure_loaded('__region_map', '__load_data')
    def region_map(self) -> RegionMap | None:
        return self.__region_map

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
