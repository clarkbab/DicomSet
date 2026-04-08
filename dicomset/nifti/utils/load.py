import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from typing import List, Tuple

from ... import config
from ...typing import AffineMatrix3D, BatchLabelImage3D, DatasetID, Image3D, LabelImage3D, Landmarks3D, ModelID, NiftiModality, PatientID, RegionID, SeriesID, StudyID
from ...utils.args import arg_to_list
from ...utils.io import load_csv, load_nifti
from ..dataset import NiftiDataset

def exists_dataset(dataset_id: DatasetID) -> bool:
    ds_path = os.path.join(config.directories.datasets, 'nifti', dataset_id)
    return os.path.exists(ds_path)

def get_dataset(dataset_id: DatasetID) -> NiftiDataset:
    ds_path = os.path.join(config.directories.datasets, 'nifti', dataset_id)
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Nifti dataset '{id}' not found at {ds_path}.")
    return NiftiDataset(dataset_id)
    
def list_datasets() -> List[DatasetID]:
    path = os.path.join(config.directories.datasets, 'nifti')
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

def load_registration_moved_image(
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

def load_registration_moved_landmarks(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> Landmarks3D:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'landmarks', f'{model}.csv')
    return load_csv(filepath)

def load_registration_moved_regions(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    region_id: RegionID | List[RegionID],
    model: ModelID,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> Tuple[LabelImage3D | BatchLabelImage3D, AffineMatrix3D]:
    region_ids = arg_to_list(region_id, str)
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    data_list = []
    affine = None
    for r in region_ids:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'regions', r, f'{model}.nii.gz')
        d, a = load_nifti(filepath)
        data_list.append(d)
        affine = a
    data = np.stack(data_list) if len(data_list) > 1 else data_list[0]
    return data, affine

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
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'transform', f'{model}.hdf5')
    return sitk.ReadTransform(filepath)
