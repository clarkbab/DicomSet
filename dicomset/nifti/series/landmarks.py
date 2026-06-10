from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import List, Literal, TYPE_CHECKING

from ... import config
from ...dataset import Dataset
from ...dicom import DicomDataset
from ...patient import Patient
from ...struct_map import StructMap
from ...study import Study
from ...typing import LandmarkID, Landmarks3D, SeriesID
from ...utils.args import alias_kwargs, arg_to_list, landmarks_to_list
from ...utils.geometry import to_image_coords
from ...utils.io import load_csv
from ...utils.landmarks import landmarks_to_points
from ...utils.python import ensure_loaded, get_private_attr
from ...utils.transforms import sample
from .series import NiftiSeries
if TYPE_CHECKING:
    from ...dicom import DicomRtStructSeries
    from .images import NiftiCtSeries, NiftiDoseSeries

class NiftiLandmarksSeries(NiftiSeries):
    def __init__(
        self,
        dataset: Dataset,
        patient: Patient,
        study: Study,
        id: SeriesID,
        index: pd.DataFrame | None = None,
        ref_ct: NiftiCtSeries | None = None,
        ref_dose: NiftiDoseSeries | None = None,
        struct_map: StructMap | None = None,
        ) -> None:
        super().__init__('landmarks', dataset, patient, study, id, index=index)
        self.__filepath = os.path.join(config.dirs.datasets, 'nifti', self.__dataset.id, 'data', 'patients', self.__patient.id, self.__study.id, self.__modality, f'{self.__id}.csv')
        if not os.path.exists(self.__filepath):
            raise ValueError(f"No NiftiLandmarksSeries '{self.__id}' found for study '{self.__study.id}'. Filepath: {self.__filepath}")
        self.__ref_ct = ref_ct
        self.__ref_dose = ref_dose
        self.__struct_map = struct_map

    @alias_kwargs(
        (('l', 'landmark', 'landmarks', 'landmark_id'), 'landmark_ids'),
    )
    @ensure_loaded('__data', '__load_data')
    def data(
        self,
        landmark_ids: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
        n: int | None = None,
        points_only: bool = False,
        sample_ct: bool = False,
        sample_dose: bool = False,
        use_world_coords: bool = True,
        **kwargs,
        ) -> Landmarks3D:

        # Load landmarks.
        landmarks_data = self.__data.copy()
        landmarks_data = landmarks_data.rename(columns={ '0': 0, '1': 1, '2': 2 })
        if not use_world_coords:
            if self.__ref_ct is None:
                raise ValueError(f"Cannot convert landmarks to image coordinates without 'ref_ct'.")
            landmarks_data = to_image_coords(landmarks_data, self.__ref_ct.affine)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        landmarks_data = landmarks_data.sort_values('landmark-id')

        # Filter by landmark ID.
        if landmark_ids != 'all':
            landmark_ids = self.list_landmarks(landmark_ids=landmark_ids)
            landmarks_data = landmarks_data[landmarks_data['landmark-id'].isin(landmark_ids)]

        # Filter by number of rows.
        if n is not None:
            landmarks_data = landmarks_data.iloc[:n]

        # Add sampled CT intensities.
        if sample_ct:
            if self.__ref_ct is None:
                raise ValueError(f"Cannot sample CT intensities without 'ref_ct'.")
            ct_values = sample(self.__ref_ct.data, landmarks_to_points(landmarks_data), affine=self.__ref_ct.affine, **kwargs)
            landmarks_data['ct-series-id'] = self.__ref_ct.id
            landmarks_data['ct'] = ct_values

        # Add sampled dose intensities.
        if sample_dose:
            if self.__ref_dose is None:
                raise ValueError(f"Cannot sample dose intensities without 'ref_dose'.")
            dose_values = sample(self.__ref_dose.data, landmarks_to_points(landmarks_data), affine=self.__ref_dose.affine, **kwargs)
            landmarks_data['dose-series-id'] = self.__ref_dose.id
            landmarks_data['dose'] = dose_values

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if 'patient-id' not in landmarks_data.columns:
            landmarks_data.insert(0, 'patient-id', self.__patient.id)
        if 'study-id' not in landmarks_data.columns:
            landmarks_data.insert(1, 'study-id', self.__study.id)
        if 'series-id' not in landmarks_data.columns:
            landmarks_data.insert(2, 'series-id', self.__id)

        if points_only:
            landmarks_data = landmarks_to_points(landmarks_data)

        return landmarks_data

    @property
    def dicom(self) -> DicomRtStructSeries:
        if self.__index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self.__index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self.__dataset.id) & (index['patient-id'] == self.__patient.id) & (index['study-id'] == self.__study.id) & (index['series-id'] == self.__id) & (index['modality'] == 'landmarks')].drop_duplicates()
        assert len(index) == 1, f"Expected 1 index entry for DICOM landmarks series '{self.__id}', but found {len(index)}. Index: {index}"
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).rtstruct_series(row['dicom-series-id'])

    @alias_kwargs(
        (('l', 'landmark', 'landmarks', 'landmark_id'), 'landmark_ids'),
    )
    def has_landmark(
        self,
        landmark_ids: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
        any: bool = False,
        **kwargs,
        ) -> bool:
        all_ids = self.list_landmarks(**kwargs)
        landmark_ids = arg_to_list(landmark_ids, LandmarkID, literals={ 'all': all_ids })
        n_overlap = len(np.intersect1d(landmark_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(landmark_ids)

    def __load_data(self) -> None:
        self.__data = load_csv(self.__filepath)

    @alias_kwargs(
        (('l', 'landmark', 'landmarks', 'landmark_id'), 'landmark_ids'),
    )
    @ensure_loaded('__data', '__load_data')
    def list_landmarks(
        self,
        landmark_ids: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
        landmark_regexp: str | None = None,
        use_mapping: bool = True,
        ) -> List[LandmarkID]:
        if self.__struct_map is None:
            use_mapping = False

        # Get landmarks regexps.
        if landmark_regexp is None and self.__struct_map is not None:
            landmark_regexp = self.__struct_map.landmark_regexps

        # Load landmark IDs.
        true_disk_landmarks = list(sorted(self.__data['landmark-id']))

        # Map disk landmarks back to API landmarks.
        if landmark_ids == 'all':
            if use_mapping:
                # Map back to the API landmark names.
                api_landmarks = [self.__struct_map.map_disk_to_api(r) for r in true_disk_landmarks]
                api_landmarks = [l for ls in api_landmarks for l in (ls if isinstance(ls, list) else [ls])]
            else:
                api_landmarks = true_disk_landmarks
        else:
            landmark_ids = landmarks_to_list(landmark_ids, disk_landmark_ids=true_disk_landmarks, literals={ 'all': self.list_landmarks }, struct_map=self.__struct_map)
            api_landmarks = []
            for r in landmark_ids:
                # Only keep landmarks that map to one or more disk landmarks.
                if use_mapping:
                    disk_landmarks = self.__struct_map.map_api_to_disk(r, disk_ids=true_disk_landmarks)
                    if len(np.intersect1d(disk_landmarks, true_disk_landmarks)) > 0:
                        api_landmarks.append(r)
                else:
                    if r in true_disk_landmarks:
                        api_landmarks.append(r)

        return list(sorted(set(api_landmarks)))

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
    
# Add properties.
props = ['filepath']
for p in props:
    setattr(NiftiLandmarksSeries, p, property(lambda self, p=p: get_private_attr(self, f'__{p}')))
