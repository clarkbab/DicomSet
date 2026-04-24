import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm
from typing import Callable, List, Literal

from .. import config
from ..dataset import CT_FROM_REGEXP, Dataset
from ..mixins import IndexWithErrorsMixin
from ..region_map import RegionMap
from ..typing import DatasetID, GroupID, PatientID, RegionID
from ..utils.args import alias_kwargs, arg_to_list, resolve_id
from ..utils.io import load_csv, load_yaml
from ..utils.logging import logger
from ..utils.python import ensure_loaded
from ..utils.regions import region_to_list
from .index import DICOM_ERROR_INDEX_COLS, DICOM_INDEX_COLS, build_index as build_index_base, index_exists
from .patient import DicomPatient

class DicomDataset(Dataset, IndexWithErrorsMixin):
    def __init__(
        self,
        id: DatasetID,
        ) -> None:
        self.__path = os.path.join(config.directories.datasets, 'dicom', str(id))
        if not os.path.exists(self.__path):
            raise ValueError(f"No dicom dataset '{id}' found at path: {self.__path}")
        ct_from = None
        for f in os.listdir(self.__path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from = match.group(1)
        ct_from = DicomDataset(ct_from) if ct_from is not None else None
        super().__init__(id, ct_from=ct_from)

    def build_index(
        self,
        **kwargs,
        ) -> None:
        build_index_base(self.__id, **kwargs)

    def has_patient(
        self,
        patient: PatientID | List[PatientID],
        any: bool = False,
        **kwargs,
        ) -> bool:
        real_ids = self.list_patients(patient_id=patient, **kwargs)
        req_ids = arg_to_list(patient, str)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    @alias_kwargs(
        ('g', 'group_id'),
        ('p', 'patient_id'),
        ('r', 'region_id'),
    )
    @ensure_loaded(
        ('__index', '__load_index'),
        ('__groups', '__load_groups'),
    )
    def list_patients(
        self,
        group_id: GroupID | List[GroupID] | Literal['all'] = 'all',
        patient_id: PatientID | List[PatientID] | Literal['all'] = 'all', 
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        show_progress: bool = False,
        sort: Callable[DicomPatient, int] | None = None,    # Not yet implemented.
        use_mapping: bool = True,
        use_regions_report: bool = True,
        ) -> List[PatientID]:
        # Load the region report for fast region filtering.
        if region_id != 'all' and use_regions_report:
            # Can't use 'load_patient_regions_report' due to circularity.
            filename = 'regions.csv' if use_mapping else 'unmapped-regions.csv'
            filepath = os.path.join(self.__path, 'reports', filename)
            if os.path.exists(filepath):
                region_ids = region_to_list(region_id, region_map=self.region_map)
                df = pd.read_csv(filepath)
                df = df[df['region'].isin(region_ids)]
                ids = list(sorted(df['patient-id'].unique()))
            else:
                logger.warn(f"No patient regions report for dataset '{self}'. Would speed up queries filtered by 'region'.")
        else:
            # Load patient IDs from index.
            ids = list(sorted(self.__index['patient-id'].unique()))

            # Filter by region ID.
            if region_id != 'all':
                def filter_fn(p: PatientID) -> bool:
                    pat_regions = self.patient(p).list_regions(region_id=region_id, use_mapping=use_mapping)
                    return True if len(pat_regions) > 0 else False
                ids = list(filter(filter_fn, tqdm(ids, disable=not show_progress)))

        # Filter by patient ID.
        if patient_id != 'all':
            patient_ids = arg_to_list(patient_id, str)
            all_ids = ids.copy()
            ids = []
            for i, id in enumerate(all_ids):
                # Check if any of the passed 'patient_ids' references this ID.
                for j, pid in enumerate(patient_ids):
                    if pid.startswith('i:'):
                        # Check if idx refer
                        idx = int(pid.split(':')[1])
                        if i == idx or (idx < 0 and i == len(all_ids) + idx):   # Allow negative indexing.
                            ids.append(id)
                            break
                    elif id == pid:
                        ids.append(id)
                        break

        # Filter by group ID.
        if group_id != 'all':
            if self.__groups is None:
                raise ValueError(f"File 'groups.csv' not found for dicom dataset '{self.__id}'.")
            all_groups = self.list_groups()
            group_ids = arg_to_list(group_id, str, literals={ 'all': all_groups })
            for g in group_ids:
                if g not in all_groups:
                    raise ValueError(f"Group '{g}' not found.")

            def filter_fn(p: PatientID) -> bool:
                pat_groups = self.__groups[self.__groups['patient-id'] == p]
                if len(pat_groups) == 0:
                    return False
                elif len(pat_groups) > 1:
                    raise ValueError(f"Patient '{p}' is a member of more than one group.")
                pat_group = pat_groups.iloc[0]['group-id']
                if pat_group in group_ids:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        return ids

    def list_regions(
        self,
        group_id: GroupID | List[GroupID] | Literal['all'] = 'all',
        patient_id: PatientID | List[PatientID] | Literal['all'] = 'all',
        use_mapping: bool = True,
        use_regions_report: bool = True,
        ) -> List[RegionID]:
        # Load all patients.
        patient_ids = self.list_patients(group=group_id, patient_id=patient_id)

        if use_regions_report:
            # Can't use 'load_patient_regions_report' due to circularity.
            filename = 'regions.csv' if use_mapping else 'unmapped-regions.csv'
            filepath = os.path.join(self.__path, 'reports', filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df = df[df['patient-id'].isin(patient_ids)]
                ids = list(sorted(df['region'].unique()))
                return ids
            else:
                logger.warn(f"No regions report for dataset '{self}'. Would speed up 'list_regions' query.")

        # Get patient regions.
        # Trawl the depths for region IDs.
        ids = []
        for p in tqdm(patient_ids):
            pat = self.patient(p)
            study_ids = pat.list_studies()
            for s in study_ids:
                study = pat.study(s)
                series_ids = study.list_rtstruct_series()
                for s in series_ids:
                    series = study.rtstruct_series(s)
                    ids += series.list_regions(use_mapping=use_mapping)
        ids = list(np.unique(ids))

        return ids

    def __load_groups(self) -> None:
        # Load groups.
        filepath = os.path.join(self.__path, 'groups.csv')
        self.__groups = load_csv(filepath) if os.path.exists(filepath) else None

    def __load_index(self) -> None:
        # Trigger index build if necessary.
        if not index_exists(self.__id):
            build_index_base(self.__id) 

        # Load index.
        filepath = os.path.join(self.__path, 'index.csv')
        try:
            self.__index = load_csv(filepath, eval_cols='mod-spec', map_types=DICOM_INDEX_COLS)
        except pd.errors.EmptyDataError as e:
            logger.info(f"Index empty for dataset '{self}'. Error: {e}")
            self.__index = pd.DataFrame(columns=DICOM_INDEX_COLS.keys())

        # Load error index.
        try:
            filepath = os.path.join(self.__path, 'index-errors.csv')
            self.__index_errors = load_csv(filepath, eval_cols='mod-spec', map_types=DICOM_ERROR_INDEX_COLS)
        except pd.errors.EmptyDataError as e:
            logger.info(f"Error index empty for dataset '{self}'.")
            self.__index_errors = pd.DataFrame(columns=DICOM_ERROR_INDEX_COLS.keys())

        # Load index policy.
        filepath = os.path.join(self.__path, 'index-policy.yaml')
        self.__index_policy = load_yaml(filepath)

    def __load_region_map(self) -> None:
        self.__region_map = RegionMap.load(self.__path)

    @ensure_loaded(
        ('__index', '__load_index'),
        ('__region_map', '__load_region_map'),
    )
    def patient(
        self,
        id: PatientID,
        group_id: GroupID | List[GroupID] | Literal['all'] = 'all',
        **kwargs,
        ) -> DicomPatient:
        id = resolve_id(id, lambda: self.list_patients(group_id=group_id))
        if not self.has_patient(id):
            raise ValueError(f"Patient '{id}' not found in dataset '{self}'.")
        index = self.__index[self.__index['patient-id'] == str(id)]
        index_errors = self.__index_errors[self.__index_errors['patient-id'] == str(id)]
        ct_from = self.__ct_from.patient(id) if self.__ct_from is not None and self.__ct_from.has_patient(id) else None
        return DicomPatient(self, id, index, self.__index_policy, index_errors, config=self.__config, ct_from=ct_from, region_map=self.__region_map, **kwargs)

    @property
    @ensure_loaded('__region_map', '__load_region_map')
    def region_map(self) -> RegionMap | None:
        return self.__region_map

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
