from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import List, Literal, TYPE_CHECKING

from ... import config
from ...dataset import Dataset
from ...dicom import DicomDataset
from ...patient import Patient
from ...study import Study
from ...training import sample
from ...typing import LandmarkID, Landmarks3D, SeriesID
from ...utils.args import arg_to_list
from ...utils.python import get_private_attr
from ...utils.geometry import to_image_coords
from ...utils.io import load_csv
from ...utils.landmarks import landmarks_to_points
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
        ) -> None:
        super().__init__('landmarks', dataset, patient, study, id, index=index)
        self.__filepath = os.path.join(config.directories.datasets, 'nifti', self._dataset.id, 'data', 'patients', self._pat.id, self._study.id, self._modality, f'{self._id}.csv')
        if not os.path.exists(self.__filepath):
            raise ValueError(f"No NiftiLandmarksSeries '{self._id}' found for study '{self._study.id}'. Filepath: {self.__filepath}")
        self.__ref_ct = ref_ct
        self.__ref_dose = ref_dose

    def data(
        self,
        points_only: bool = False,
        landmark_id: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
        n: int | None = None,
        sample_ct: bool = False,
        sample_dose: bool = False,
        use_world_coords: bool = True,
        **kwargs,
        ) -> Landmarks3D:

        # Load landmarks.
        landmarks_data = load_csv(self.__filepath)
        landmarks_data = landmarks_data.rename(columns={ '0': 0, '1': 1, '2': 2 })
        if not use_world_coords:
            if self.__ref_ct is None:
                raise ValueError(f"Cannot convert landmarks to image coordinates without 'ref_ct'.")
            landmarks_data = to_image_coords(landmarks_data, self.__ref_ct.affine)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        landmarks_data = landmarks_data.sort_values('landmark-id')

        # Filter by landmark ID.
        if landmark_id != 'all':
            landmark_ids = self.list_landmarks(landmark_id=landmark_id)
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
            landmarks_data.insert(0, 'patient-id', self._pat.id)
        if 'study-id' not in landmarks_data.columns:
            landmarks_data.insert(1, 'study-id', self._study.id)
        if 'series-id' not in landmarks_data.columns:
            landmarks_data.insert(2, 'series-id', self._id)

        if points_only:
            return landmarks_data[range(3)].to_numpy().astype(np.float32)
        else:
            return landmarks_data

    @property
    def dicom(self) -> DicomRtStructSeries:
        if self.__index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self.__index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self._dataset.id) & (index['patient-id'] == self._pat.id) & (index['study-id'] == self._study.id) & (index['series-id'] == self._id) & (index['modality'] == 'landmarks')].drop_duplicates()
        assert len(index) == 1, f"Expected 1 index entry for DICOM landmarks series '{self._id}', but found {len(index)}. Index: {index}"
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).rtstruct_series(row['dicom-series-id'])

    def has_landmark(
        self,
        landmark_id: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
        any: bool = False,
        **kwargs,
        ) -> bool:
        all_ids = self.list_landmarks(**kwargs)
        landmark_ids = arg_to_list(landmark_id, LandmarkID, literals={ 'all': all_ids })
        n_overlap = len(np.intersect1d(landmark_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(landmark_ids)

    def list_landmarks(
        self,
        landmark_id: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
        ) -> List[LandmarkID]:
        # Load landmark IDs.
        landmarks_data = load_csv(self.__filepath)
        ids = list(sorted(landmarks_data['landmark-id']))

        if landmark_id == 'all':
            return ids

        if isinstance(landmark_id, float) and landmark_id > 0 and landmark_id < 1:
            # Take non-random subset of landmarks.
            ids = p_landmarks(ids, landmark_id)
        else:
            # Filter based on passed landmarks.
            landmark_ids = arg_to_list(landmark_id, LandmarkID)
            ids = [i for i in ids if i in landmark_ids]

        return ids

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
    
# Add properties.
props = ['filepath']
for p in props:
    setattr(NiftiLandmarksSeries, p, property(lambda self, p=p: get_private_attr(self, f'__{p}')))
