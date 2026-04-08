import os
import pandas as pd
import shutil
import SimpleITK as sitk
from typing import List

from ... import config
from ...typing import AffineMatrix3D, BatchLabelImage3D, DatasetID, Image3D, LabelImage3D, Landmarks3D, ModelID, NiftiModality, PatientID, RegionID, SeriesID, StudyID
from ...utils.args import arg_to_list
from ...utils.io import save_csv, save_nifti, save_transform
from ..dataset import NiftiDataset

def save_ct(
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

def save_index(
    dataset: DatasetID,
    index: pd.DataFrame,
    ) -> None:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'index.csv')
    save_csv(index, filepath)

def save_region(
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

def save_registration_moved_image(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    data: Image3D,
    affine: AffineMatrix3D,
    modality: NiftiModality,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> None:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, modality, f'{model}.nii.gz')
    save_nifti(data, affine, filepath)

def save_registration_moved_landmarks(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    data: Landmarks3D,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> None:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'landmarks', f'{model}.csv')
    save_csv(data, filepath)

def save_registration_moved_regions(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    data: LabelImage3D | BatchLabelImage3D,
    affine: AffineMatrix3D,
    region_id: RegionID | List[RegionID], 
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> None:
    region_ids = arg_to_list(region_id, str)
    if len(data.shape) == 3:
        data = data[None, ...]  # Add batch dimension if not present
    assert len(region_ids) == data.shape[0], "Number of region IDs must match the number of regions in the data."
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    for r, d in zip(region_ids, data): 
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'regions', r, f'{model}.nii.gz')
        save_nifti(d, affine, filepath)

def save_registration_transform(
    dataset: DatasetID,
    fixed_patient_id: PatientID,
    model: ModelID,
    transform: sitk.Transform,
    fixed_series_id: SeriesID = 'series_0',
    fixed_study_id: StudyID = 'study_1',
    moving_patient_id: PatientID | None = None,
    moving_series_id: SeriesID = 'series_0',
    moving_study_id: StudyID = 'study_0',
    ) -> None:
    set = NiftiDataset(dataset)
    moving_patient_id = fixed_patient_id if moving_patient_id is None else moving_patient_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_patient_id, fixed_study_id, fixed_series_id, moving_patient_id, moving_study_id, moving_series_id, 'transform', f'{model}.hdf5')
    save_transform(transform, filepath)
