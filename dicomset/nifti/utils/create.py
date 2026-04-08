import os
import pandas as pd
import shutil
import SimpleITK as sitk

from ... import config
from ...typing import AffineMatrix3D, DatasetID, Image3D, LabelImage3D, Landmarks3D, ModelID, NiftiModality, PatientID, RegionID, SeriesID, StudyID
from ...utils.io import save_csv, save_nifti, save_transform
from ..dataset import NiftiDataset

def create_ct(
    dataset: DatasetID,
    patient_id: PatientID,
    study_id: StudyID,
    series_id: SeriesID,
    data: Image3D,
    affine: AffineMatrix3D,
    ) -> None:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'patients', patient_id, study_id, 'ct', f'{series_id}.nii.gz')
    save_nifti(data, affine, filepath)

def create_dataset(
    dataset_id: DatasetID,
    recreate: bool = False,
    ) -> NiftiDataset:
    ds_path = os.path.join(config.directories.datasets, 'nifti', dataset_id)
    if os.path.exists(ds_path):
        if recreate:
            shutil.rmtree(ds_path)
    os.makedirs(ds_path, exist_ok=True)
    return NiftiDataset(dataset_id)

def create_index(
    dataset: DatasetID,
    index: pd.DataFrame,
    ) -> None:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'index.csv')
    save_csv(index, filepath)

def create_region(
    dataset: DatasetID,
    patient_id: PatientID,
    study_id: StudyID,
    series_id: SeriesID,
    region_id: RegionID,
    data: LabelImage3D,
    affine: AffineMatrix3D,
    ) -> None:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'patients', patient_id, study_id, 'regions', series_id, f'{region_id}.nii.gz')
    save_nifti(data, affine, filepath)

def create_registration_moved_image(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    data: Image3D,
    affine: AffineMatrix3D,
    modality: NiftiModality,
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_study_id: StudyID = 'study_0',
    ) -> None:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, moving_patient_id, moving_study_id, modality, f'{model}.nii.gz')
    save_nifti(data, affine, filepath)

def create_registration_moved_landmarks(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    data: Landmarks3D,
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_study_id: StudyID = 'study_0',
    ) -> None:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, moving_patient_id, moving_study_id, 'landmarks', f'{model}.csv')
    save_csv(data, filepath)

def create_registration_moved_region(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    region_id: RegionID,
    model: ModelID,
    data: LabelImage3D,
    affine: AffineMatrix3D,
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_study_id: StudyID = 'study_0',
    ) -> None:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, moving_patient_id, moving_study_id, 'regions', region_id, f'{model}.nii.gz')
    save_nifti(data, affine, filepath)

def create_registration_transform(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    transform: sitk.Transform,
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_study_id: StudyID = 'study_0',
    ) -> None:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, moving_patient_id, moving_study_id, 'transform', f'{model}.hdf5')
    save_transform(transform, filepath)
