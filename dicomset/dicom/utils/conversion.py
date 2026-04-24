from __future__ import annotations

import os
import pandas as pd
import re
import shutil
from time import time
from tqdm import tqdm
from typing import Callable, List, Literal, TYPE_CHECKING

from ...dataset import CT_FROM_REGEXP
from ...region_map import RegionMap
from ...typing import GroupID, LandmarkID, PatientID, RegionID
from ...utils.args import arg_to_list
from ...utils.io import save_csv, save_nifti
from ...utils.logging import logger
from ...utils.pandas import append_row
from ..dataset import DicomDataset
from ..patient import DicomPatient
from ..series.mr import DicomMrSeries
from ..series.rtdose import DicomRtDoseSeries
from ..series.rtstruct import DicomRtStructSeries
if TYPE_CHECKING:
    from ..study import DicomStudy

# By default conversion doesn't remove old data but it can be removed.
# recreate_dataset: removes everything.
# recreate_patients: removes the 'data/patients' folder.
# recreate_ct: removes all 'data/patients/<patient_id>/studies/<study_id>/series/ct' folders.
# etc...

# How do we start the conversion from an intermediate patient without
# breaking the patient IDs?
# - Add a "start_from_patient_id" argument.
# - Or add recreate_patient_id argument that could 
#
# Now "all_patient_ids" refers to everthing that should exist in the destination
# dataset, whereas "patient_ids" are the patients that are currently being processed.
# We still need to create index entries for all patients, as we've made the choice
# to recreate the index upon each conversion so that it doesn't become stale.
# Plus updating it is a nightmare - see "DicomDataset.build_index".

# Note that rerunning "convert_to_nifti" is flimsy as it depends heavily
# on the ordering of everything (patients, studies, series) for the anonymisation.
# For example if we have patients [1, 2, 4, 5], run conversion, and then
# add patient 3 and run conversion again - patient_2 will now rever to dicom 
# patient 3 instead of 4. Currently, when dicom ordering is changed we really
# need recreate_dataset/patients=True.
# 
# This can be remedied somewhat by the sort methods, e.g. "study_sort" that
# could place a new study last using custom sorting, rather than the default
# sorting which is by datetime.

def convert_to_nifti(
    dataset: str,
    anonymise_ct: bool = True,
    anonymise_dose: bool = True,
    anonymise_landmarks: bool = True,
    anonymise_mr: bool = True,
    anonymise_patients: bool = True,
    anonymise_regions: bool = True,
    anonymise_studies: bool = True,
    # These flags handle which series are converted. Maybe there's a bunch of rtdose
    # files in the source dicom dataset but we don't need these for our current project.
    convert_ct: bool = True,
    convert_dose: bool = True,
    convert_landmarks: bool = True,
    convert_mr: bool = True,
    convert_regions: bool = True,
    dest_dataset: str | None = None,
    filter_pats_by_landmarks: bool = False,
    filter_pats_by_regions: bool = False,
    group_id: GroupID | List[GroupID] | Literal['all'] = 'all',
    landmark_id: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
    patient_id: PatientID | List[PatientID] | Literal['all'] = 'all',
    # These flags handle the case where you want CT (for example) to be present in the
    # output dataset, but you don't want to convert it again.
    reconvert_ct: bool = True,
    reconvert_dose: bool = True,
    reconvert_landmarks: bool = True,
    reconvert_mr: bool = True,
    reconvert_regions: bool = True,
    # These flags remove data before conversion. Maybe there was a bunch of
    # dose series that got out of sync and we only want to remove them.
    recreate_dataset: bool = False,
    recreate_patients: bool = False,
    recreate_ct: bool = False,
    recreate_dose: bool = False,
    recreate_landmarks: bool = False,
    recreate_patient_id: PatientID | List[PatientID] | str | Literal['all'] = 'all',
    recreate_regions: bool = False,
    region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
    sort_cts: Callable[DicomCTSeries, int] | None = None,
    sort_doses: Callable[DicomRtDoseSeries, int] | None = None,
    sort_rtstructs: Callable[DicomRtStructSeries, int] | None = None, # Landmarks/regions are currently tied by rtstruct series ID.
    sort_mrs: Callable[DicomMrSeries, int] | None = None,
    sort_patients: Callable[DicomPatient, int] | None = None,
    sort_studies: Callable[DicomStudy, int] | None = None,
    ) -> None:
    start = time()

    # Create NIFTI dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    nifti_set = create_nifti_dataset(dest_dataset, recreate=recreate_dataset)

    # Load all patients.
    dicom_set = DicomDataset(dataset)
    okwargs = dict(group_id=group_id, patient_id=patient_id)
    if filter_pats_by_landmarks and landmark_id is not None: 
        okwargs['landmark_id'] = landmark_id
    if filter_pats_by_regions and region_id is not None:
        okwargs['region_id'] = region_id
    all_dicom_patient_ids = dicom_set.list_patients(sort=sort_patients, **okwargs)
    logger.info(f"Loaded patients: {all_dicom_patient_ids}")

    # Determine IDs.
    # There are all the IDs, but only a subset of these may be recreated.
    if anonymise_patients:
        all_patient_ids = [f'pat_{i}' for i in range(len(all_dicom_patient_ids))]
    else:
        all_patient_ids = all_dicom_patient_ids

    # If processing only a subset of patients..
    if recreate_patient_id != 'all':
        recreate_patient_ids = arg_to_list(recreate_patient_id, str)
        patient_ids = []
        for p in recreate_patient_ids:
            # Handle slice notation, e.g. "PAT5:" or ":PAT5".
            if p.startswith(':'):
                end_id = p[1:]
                patient_ids += all_patient_ids[:all_patient_ids.index(end_id)]
            elif p.endswith(':'):
                start_id = p[:-1]
                patient_ids += all_patient_ids[all_patient_ids.index(start_id):]
            elif ':' in p:
                start_id, end_id = p.split(':')
                patient_ids += all_patient_ids[all_patient_ids.index(start_id):all_patient_ids.index(end_id)]
            else:
                patient_ids.append(p)
        patient_ids = list(sorted(set(patient_ids)))
        dicom_patient_ids = [dp for p, dp in zip(all_patient_ids, all_dicom_patient_ids) if p in patient_ids]
    else:
        patient_ids = all_patient_ids
        dicom_patient_ids = all_dicom_patient_ids
    logger.info(f"Processing patients: {dicom_patient_ids}")

    # Remove existing data.
    # This is useful when we only want to remove, for example, the dose series.
    if not recreate_dataset:
        for p in patient_ids:
            dirpath = os.path.join(nifti_set.path, 'data', 'patients', p)
            if os.path.exists(dirpath):
                # Are recreate_patients/studies the same thing?
                # Yes - plus recreate_patients is just a shortcut for recreate_ct/dose/landmarks/etc.
                if recreate_patients:
                    logger.info(f"Removing existing patient directory: {dirpath}")
                    shutil.rmtree(dirpath)
                else:
                    # Recreate specified modalities.
                    mods = ['ct', 'dose', 'landmarks', 'regions']
                    recreate_mods = [recreate_ct, recreate_dose, recreate_landmarks, recreate_regions]
                    mods = [m for r, m in zip(recreate_mods, mods) if r]
                    study_ids = os.listdir(dirpath)
                    for m in mods:
                        for s in study_ids:
                            series_dirpath = os.path.join(dirpath, s, m)
                            if os.path.exists(series_dirpath):
                                logger.info(f"Removing existing series directory: {series_dirpath}")
                                shutil.rmtree(series_dirpath)

    # Remove markers.
    files = os.listdir(nifti_set.path)
    for f in files:
        if f.startswith('__DICOM_CONVERSION_TIME_MINS_'):
            os.remove(os.path.join(nifti_set.path, f))

    # Check if index is open and therefore can't be overwritten.
    filepath = os.path.join(nifti_set.path, 'index.csv')
    if os.path.exists(filepath):
        try:
            open(filepath, 'a')
        except PermissionError:
            logger.error(f"Index file '{filepath}' is currently open and cannot be overwritten. Please close it before running conversion.")
            return

    # Check '__ct_from_' for DICOM dataset.
    ct_from = None
    for f in os.listdir(dicom_set.path):
        match = re.match(CT_FROM_REGEXP, f)
        if match:
            ct_from = match.group(1)

    # Add '__ct_from_' tag to NIFTI dataset.
    if ct_from is not None:
        filepath = os.path.join(nifti_set.path, f'__CT_FROM_{ct_from}__')
        open(filepath, 'w').close()

    # Copy region map.
    region_map = RegionMap.load(dicom_set.path)
    if region_map is not None:
        filepath = region_map.filepath
        destpath = os.path.join(nifti_set.path, os.path.basename(filepath))
        shutil.copy(filepath, destpath)

    # Copy 'groups.csv' file.
    filepath = os.path.join(dicom_set.path, 'groups.csv')
    if os.path.exists(filepath):
        destpath = os.path.join(nifti_set.path, 'groups.csv')
        shutil.copy(filepath, destpath)

    # Recreate the index every time. It's not that expensive and
    # leads to fewer issues.
    cols = {
        'dataset': str,
        'patient-id': str,
        'study-id': str,
        'series-id': str,
        'modality': str,
        'dicom-dataset': str,
        'dicom-patient-id': str,
        'dicom-study-id': str,
        'dicom-series-id': str,
        'dicom-modality': str,
    }
    filepath = os.path.join(nifti_set.path, 'index.csv')
    index = pd.DataFrame(columns=cols.keys())

    # Write patient data.
    # We iterate over all patients, in order to create the correct index, but
    # in reality only a subset may be processed if this is a second pass (using
    # recreate_patient_id).
    for p, dp in tqdm(zip(all_patient_ids, all_dicom_patient_ids)):
        patient = dicom_set.patient(dp)
        logger.info(f"Inspecting: {patient}")

        # Get study IDs.
        dicom_study_ids = patient.list_studies(sort=sort_studies)
        if anonymise_studies:
            study_ids = [f'study_{i}' for i in range(len(dicom_study_ids))]
        else:
            study_ids = dicom_study_ids

        # Convert studies. 
        for s, ds in zip(study_ids, dicom_study_ids):
            study = patient.study(ds)
            logger.info(f"Inspecting: {study}")

            if convert_ct and ct_from is None:
                # Get series IDs.
                dicom_series_ids = study.list_ct_series(sort=sort_cts)
                if anonymise_ct:
                    series_ids = [f'series_{i}' for i in range(len(dicom_series_ids))] 
                else:
                    series_ids = dicom_series_ids

                # Convert CT series.
                for sr, dsr in zip(series_ids, dicom_series_ids):
                    series = study.ct_series(dsr)
                    logger.info(f"Inspecting: {series}")

                    filepath = os.path.join(nifti_set.path, 'data', 'patients', p, s, 'ct', f'{sr}.nii.gz')
                    create_index_entry = False

                    # Don't write the data if that patient is in the recreate subset.
                    if p in patient_ids:
                        # Create CT series.
                        # Doesn't overwrite data, if we want to replace some existing data, need to use
                        # a 'recreate' tag, which will remove existing patient data.
                        if not os.path.exists(filepath):
                            logger.info(f"Writing: {series} -> {filepath}")
                            save_nifti(series.data, series.affine, filepath)
                        create_index_entry = True
                    elif os.path.exists(filepath):
                        create_index_entry = True

                    # Add index entry.
                    # We destroy the index on each run, so even if the data already existed, this entry
                    # should be added to the index. We could probably create the index in a separate step
                    # before writing...
                    if create_index_entry:
                        data = {
                            'dataset': dest_dataset,
                            'patient-id': p,
                            'study-id': s,
                            'series-id': sr,
                            'modality': 'ct',
                            'dicom-dataset': dataset,
                            'dicom-patient-id': dp,
                            'dicom-study-id': ds,
                            'dicom-series-id': dsr,
                            'dicom-modality': 'ct',
                        }
                        index = append_row(index, data)

            # Convert MR series.
            if convert_mr:
                dicom_series_ids = study.list_mr_series(sort=sort_mrs)
                if anonymise_mr:
                    series_ids = [f'series_{i}' for i in range(len(dicom_series_ids))]
                else:
                    series_ids = dicom_series_ids

                for sr, dsr in zip(series_ids, dicom_series_ids):
                    series = study.mr_series(dsr)
                    logger.info(f"Inspecting: {series}")
                    
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', p, s, 'mr', f'{sr}.nii.gz')
                    create_index_entry = False

                    if p in patient_ids:
                        if not os.path.exists(filepath):
                            logger.info(f"Writing: {series} -> {filepath}")
                            save_nifti(series.data, series.affine, filepath)
                        create_index_entry = True
                    elif os.path.exists(filepath):
                        create_index_entry = True

                    # Add index entry.
                    if create_index_entry:
                        data = {
                            'dataset': dataset,
                            'patient-id': p,
                            'study-id': s,
                            'series-id': sr,
                            'modality': 'mr',
                            'dicom-dataset': dataset,
                            'dicom-patient-id': dp,
                            'dicom-study-id': ds,
                            'dicom-series-id': dsr,
                            'dicom-modality': 'mr',
                        }
                        index = append_row(index, data)

            # Convert RTSTRUCT series.
            if convert_landmarks or convert_regions:
                dicom_series_ids = study.list_rtstruct_series(sort=sort_rtstructs)

                # Create landmarks.
                if convert_landmarks and landmark_id is not None:
                    if anonymise_landmarks:
                        series_ids = [f'series_{i}' for i in range(len(dicom_series_ids))]
                    else:
                        series_ids = dicom_series_ids

                    for sr, dsr in zip(series_ids, dicom_series_ids):
                        series = study.rtstruct_series(dsr)
                        logger.info(f"Inspecting: {series}")

                        filepath = os.path.join(nifti_set.path, 'data', 'patients', p, s, 'landmarks', f'{sr}.csv')
                        create_index_entry = False

                        if p in patient_ids:
                            _, landmarks_data = series.landmarks_data(add_ids=False, landmark_id=landmark_id)
                            if landmarks_data is not None:
                                if not os.path.exists(filepath):
                                    logger.info(f"Writing: {series} -> {filepath}")
                                    save_csv(landmarks_data, filepath)
                                create_index_entry = True
                        elif os.path.exists(filepath):
                            create_index_entry = True

                        # Add index entry.
                        if create_index_entry:
                            data = {
                                'dataset': dataset,
                                'patient-id': p,
                                'study-id': s,
                                'series-id': sr,
                                'modality': 'landmarks',
                                'dicom-dataset': dataset,
                                'dicom-patient-id': dp,
                                'dicom-study-id': ds,
                                'dicom-series-id': dsr,
                                'dicom-modality': 'rtstruct',
                            }
                            index = append_row(index, data)

                # Create regions.
                if convert_regions and region_id is not None:
                    if anonymise_regions:
                        series_ids = [f'series_{i}' for i in range(len(dicom_series_ids))]
                    else:
                        series_ids = dicom_series_ids

                    for sr, dsr in zip(series_ids, dicom_series_ids):
                        series = study.rtstruct_series(dsr)
                        logger.info(f"Inspecting: {series}")
                        region_ids = series.list_regions(region_id=region_id)

                        create_index_entry = False
                        for r in region_ids:
                            filepath = os.path.join(nifti_set.path, 'data', 'patients', p, s, 'regions', sr, f'{r}.nii.gz')
                            if p in patient_ids:
                                # Do we need to load this?
                                # We might be requesting a region that this patient doesn't have,
                                # so we need to load the rtstruct to check this.
                                # On reconversion, we're missing CT only. So we'd like to skip
                                # the data loading, but we need to know whether to create and index
                                # entry for this series. If the series exists, then is it fair to say
                                # there is at least one region present? Not really - we could be restricting
                                # the number of regions heavily, e.g. region_id=GTV and we don't want to create
                                # index entries for series without the GTV - and hence no actual data in the
                                # nifti dataset.
                                # Realistically, this conversion will only be run once or maybe a few times at
                                # the beginning of a project, so it doesn't have to be super efficient.
                                _, region_data = series.regions_data(region_id=r)
                                region_data = region_data[0]    # Region data is a batch.
                                if region_data is not None:
                                    if not os.path.exists(filepath):
                                        logger.info(f"Writing: {series} -> {filepath}")
                                        save_nifti(region_data, series.ref_ct.affine, filepath)
                                    create_index_entry = True
                            elif os.path.exists(filepath):
                                create_index_entry = True

                        if create_index_entry:
                            # Add index entry.
                            data = {
                                'dataset': dataset,
                                'patient-id': p,
                                'study-id': s,
                                'series-id': sr,
                                'modality': 'regions',
                                'dicom-dataset': dataset,
                                'dicom-patient-id': dp,
                                'dicom-study-id': ds,
                                'dicom-series-id': dsr,
                                'dicom-modality': 'rtstruct',
                            }
                            index = append_row(index, data)

            # Convert RTDOSE series.
            if convert_dose:
                dicom_series_ids = study.list_rtdose_series(sort=sort_doses)
                if anonymise_dose:
                    series_ids = [f'series_{i}' for i in range(len(dicom_series_ids))]
                else:
                    series_ids = dicom_series_ids

                for sr, dsr in zip(series_ids, dicom_series_ids):
                    series = study.rtdose_series(dsr)
                    logger.info(f"Inspecting: {series}")
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', p, s, 'dose', f'{sr}.nii.gz')
                    create_index_entry = False
                    if p in patient_ids:
                        if not os.path.exists(filepath):
                            logger.info(f"Writing: {series} -> {filepath}")
                            save_nifti(series.data, series.affine, filepath)
                        create_index_entry = True
                    elif os.path.exists(filepath):
                        create_index_entry = True

                    if create_index_entry:    
                        # Add index entry.
                        data = {
                            'dataset': dataset,
                            'patient-id': p,
                            'study-id': s,
                            'series-id': sr,
                            'modality': 'dose',
                            'dicom-dataset': dataset,
                            'dicom-patient-id': dp,
                            'dicom-study-id': ds,
                            'dicom-series-id': dsr,
                            'dicom-modality': 'rtdose',
                        }
                        index = append_row(index, data)

    # Save index.
    if len(index) > 0:
        index = index.astype(cols)
        index = index.sort_values(['patient-id', 'study-id', 'series-id'])   # Required if adding patients to existing converted dataset.
    filepath = os.path.join(nifti_set.path, 'index.csv')
    save_csv(index, filepath)
