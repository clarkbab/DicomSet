from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    import SimpleITK as sitk

from ... import config
from ...typing import AffineMatrix3D, BatchLabelImage3D, DatasetID, Image3D, LabelImage3D, Landmarks3D, ModelID, NiftiModality, PatientID, RegionID, SeriesID, StudyID
from ...utils.args import arg_to_list
from ...utils.io import load_csv, load_nifti, load_transform
from ..dataset import NiftiDataset

def dataset_exists(dataset_id: DatasetID) -> bool:
    ds_path = os.path.join(config.dirs.datasets, 'nifti', dataset_id)
    return os.path.exists(ds_path)

def list_datasets() -> List[DatasetID]:
    path = os.path.join(config.dirs.datasets, 'nifti')
    return list(sorted(os.listdir(path))) if os.path.exists(path) else []
    
def load_ct(
    dataset: DatasetID,
    patient_id: PatientID,
    study_id: StudyID,
    series_id: SeriesID,
    ) -> Tuple[Image3D, AffineMatrix3D]:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'patients', patient_id, study_id, 'ct', f'{series_id}.nii.gz')
    return load_nifti(filepath)

def load_dataset(dataset_id: DatasetID) -> NiftiDataset:
    ds_path = os.path.join(config.dirs.datasets, 'nifti', dataset_id)
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Nifti dataset '{id}' not found at {ds_path}.")
    return NiftiDataset(dataset_id)

def load_index(
    dataset: DatasetID,
    ) -> pd.DataFrame:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'index.csv')
    return load_csv(filepath)

def load_region(
    dataset: DatasetID,
    patient_id: PatientID,
    study_id: StudyID,
    series_id: SeriesID,
    region_id: RegionID,
    ) -> Tuple[LabelImage3D, AffineMatrix3D]:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'patients', patient_id, study_id, 'regions', series_id, f'{region_id}.nii.gz')
    return load_nifti(filepath)

def load_registered_image(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    modality: NiftiModality,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> Tuple[Image3D, AffineMatrix3D]:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, modality, f'{model}.nii.gz')
    return load_nifti(filepath)

def load_registered_landmarks(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    landmark_ids: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> Landmarks3D:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'landmarks', f'{model}.csv')
    df = load_csv(filepath, map_cols=dict((str(i), i) for i in range(3)))
    if landmark_ids != 'all':
        landmark_ids = arg_to_list(landmark_ids, str)
        df = df[df['landmark-id'].isin(landmark_ids)]
    return df

def load_registered_regions(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    region_ids: RegionID | List[RegionID],
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> Tuple[RegionID | List[RegionID] | None, LabelImage3D | BatchLabelImage3D | None, AffineMatrix3D | None]:
    region_ids = arg_to_list(region_ids, str)
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    loaded_region_ids = []
    datas = []
    affine = None
    for r in region_ids:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'regions', r, f'{model}.nii.gz')
        if not os.path.exists(filepath):
            continue
        loaded_region_ids.append(r)
        d, a = load_nifti(filepath)
        datas.append(d)
        affine = a
    if len(datas) == 0:
        loaded_region_ids = None
        data = None
    else:
        loaded_region_ids = loaded_region_ids if len(loaded_region_ids) > 1 else loaded_region_ids[0]
        data = np.stack(datas) if len(datas) > 1 else datas[0]
    return loaded_region_ids, data, affine

def load_registration_transform(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> sitk.Transform:
    import SimpleITK as sitk    # Slow import.
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'transform', f'{model}.hdf5')
    return load_transform(filepath)

def load_registration(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    landmark_ids: LandmarkID | List[LandmarkID] | Literal['all'] | None = None,
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    region_ids: RegionID | List[RegionID] | Literal['all'] | None = None,
    ) -> Tuple[sitk.Transform | None, Tuple[CtImage | None, AffineMatrix3D | None], DoseImage | None, Landmarks | None, Tuple[RegionID | List[RegionID] | None, RegionLabel | RegionsLabel | None]]:
    # Load components.
    transform = load_registration_transform(dataset, fixed_patient_id, model, fixed_series_id=fixed_series_id, fixed_study_id=fixed_study_id, moving_patient_id=moving_patient_id, moving_series_id=moving_series_id, moving_study_id=moving_study_id)
    ct, affine = load_registered_image(dataset, fixed_patient_id, model, 'ct', fixed_series_id=fixed_series_id, fixed_study_id=fixed_study_id, moving_patient_id=moving_patient_id, moving_series_id=moving_series_id, moving_study_id=moving_study_id)
    dose, _ = load_registered_image(dataset, fixed_patient_id, model, 'dose', fixed_series_id=fixed_series_id, fixed_study_id=fixed_study_id, moving_patient_id=moving_patient_id, moving_series_id=moving_series_id, moving_study_id=moving_study_id)
    if landmark_ids is not None:
        landmarks_data = load_registered_landmarks(dataset, fixed_patient_id, model, landmark_ids=landmark_ids, fixed_series_id=fixed_series_id, fixed_study_id=fixed_study_id, moving_patient_id=moving_patient_id, moving_series_id=moving_series_id, moving_study_id=moving_study_id)
    else:
        landmarks_data = None
    if region_ids is not None:
        loaded_region_ids, regions_data, _ = load_registered_regions(dataset, fixed_patient_id, model, region_ids=region_ids, fixed_series_id=fixed_series_id, fixed_study_id=fixed_study_id, moving_patient_id=moving_patient_id, moving_series_id=moving_series_id, moving_study_id=moving_study_id)
    else:
        loaded_region_ids, regions_data = None, None
    return transform, (ct, affine), dose, landmarks_data, (loaded_region_ids, regions_data)
