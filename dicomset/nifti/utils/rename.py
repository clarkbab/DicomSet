import os
from typing import Callable, List, Literal

from ...typing import DatasetID, PatientID
from ...utils.io import load_csv, save_csv
from ...utils.logging import logger
from ..dataset import NiftiDataset

def rename_patients(
    dataset: DatasetID,
    rename_fn: Callable,
    dicom_rename_fn: Callable | None = None,
    makeitso: bool = False,
    patient_id: PatientID | List[PatientID] | Literal['all'] = 'all',
    rename_evaluations: bool = True,
    rename_folders: bool = True,
    rename_indexes: bool = True,
    rename_predictions: bool = True,
    rename_reports: bool = True,
    ) -> None:
    # Rename evaluations.
    set = NiftiDataset(dataset)
    patient_ids = set.list_patients(patient_ids=patient_id)
    if rename_evaluations:
        # Rename registration evaluations.
        dirpath = os.path.join(set.path, 'data', 'evaluations', 'registration')
        if os.path.exists(dirpath):
            models = os.listdir(dirpath)
            for m in models:
                modalities = ['landmarks', 'regions']
                for mod in modalities:
                    filepath = os.path.join(dirpath, m, f'{mod}.csv')
                    if os.path.exists(filepath):
                        lm_df = load_csv(filepath)
                        lm_df['patient-id'] = lm_df['patient-id'].apply(rename_fn)
                        lm_df['moving-patient-id'] = lm_df['moving-patient-id'].apply(rename_fn)
                        if makeitso:
                            save_csv(lm_df, filepath, index=False)
                        else:
                            logger.info(f"Rename patient IDs for registration evaluation {filepath}.")

    # Rename folders.
    if rename_folders:
        for p in patient_ids:
            new_pat_id = rename_fn(p)
            srcpath = os.path.join(set.path, 'data', 'patients', p)
            destpath = os.path.join(set.path, 'data', 'patients', new_pat_id)
            if makeitso:
                os.rename(srcpath, destpath)
            else:
                logger.info(f"Rename patient folder from {srcpath} to {destpath}.")

    # Rename index and error index.
    if rename_indexes:
        index = set.index
        for i, r in index.iterrows():
            old_pat_id = r['patient-id']
            new_pat_id = rename_fn(old_pat_id)
            if dicom_rename_fn is not None:
                old_dicom_pat_id = r['dicom-patient-id']
                new_dicom_pat_id = dicom_rename_fn(old_dicom_pat_id)
            if makeitso:
                index.at[i, 'patient-id'] = new_pat_id
                index.at[i, 'nifti-patient-name'] = new_pat_id
                if dicom_rename_fn is not None:
                    index.at[i, 'dicom-patient-id'] = new_dicom_pat_id
                    index.at[i, 'dicom-patient-name'] = new_dicom_pat_id
            else:
                logger.info(f"Rename patient ID from {old_pat_id} to {new_pat_id} in index.")
                if dicom_rename_fn is not None:
                    logger.info(f"Rename DICOM patient ID from {old_dicom_pat_id} to {new_dicom_pat_id} in index.")
                else:
                    index.at[i, 'patient-id'] = new_pat_id
                    index.at[i, 'nifti-patient-name'] = new_pat_id
                    if dicom_rename_fn is not None:
                        index.at[i, 'dicom-patient-id'] = new_dicom_pat_id
                        index.at[i, 'dicom-patient-name'] = new_dicom_pat_id
        if makeitso:
            filepath = os.path.join(set.path, 'index.csv')
            save_csv(index, filepath, index=True)

    # Rename predictions.
    if rename_predictions:
        # # Rename registration predictions.
        # dirpath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients')
        # if os.path.exists(dirpath):
        #     old_patient_ids = os.listdir(dirpath)
        #     for o in old_patient_ids:
        #         # Rename moving patients.
        #         pat_dirpath = os.path.join(dirpath, o)
        #         studys = os.listdir(pat_dirpath)
        #         for s in studys:
        #             study_dirpath = os.path.join(pat_dirpath, s)
        #             old_moving_patient_ids = os.listdir(study_dirpath)
        #             for oo in old_moving_patient_ids:
        #                 new_moving_pat_id = rename_fn(oo)
        #                 if makeitso:
        #                     srcpath = os.path.join(study_dirpath, oo)
        #                     destpath = os.path.join(study_dirpath, new_moving_pat_id)
        #                     os.rename(srcpath, destpath)
        #                 else:
        #                     logger.info(f"Rename moving patient ID from {oo} to {new_moving_pat_id} in registration predictions.")

        #         # Rename fixed patient.
        #         new_pat_id = rename_fn(o)
        #         if makeitso:
        #             destpath = os.path.join(dirpath, new_pat_id)
        #             os.rename(pat_dirpath, destpath)
        #         else:
        #             logger.info(f"Rename fixed patient ID from {o} to {new_pat_id} in registration predictions.") 

        # Rename timing files.
        dirpath = os.path.join(set.path, 'data', 'predictions', 'registration', 'timing')
        if os.path.exists(dirpath):
            files = os.listdir(dirpath)
            for f in files:
                filepath = os.path.join(dirpath, f)
                if os.path.exists(filepath):
                    t_df = load_csv(filepath)
                    t_df['patient-id'] = t_df['patient-id'].apply(rename_fn)
                    if makeitso:
                        save_csv(t_df, filepath)
                    else:
                        logger.info(f"Rename patient IDs in registration predictions timing file {filepath}.")

    # Rename reports.
    if rename_reports:
        dirpath = os.path.join(set.path, 'data', 'reports')
        if os.path.exists(dirpath):
            files = os.listdir(dirpath)
            for f in files:
                if f.endswith('.csv'):
                    filepath = os.path.join(dirpath, f)
                    df = load_csv(filepath)
                    for i, r in df.iterrows():
                        old_pat_id = r['patient-id']
                        new_pat_id = rename_fn(old_pat_id)
                        if makeitso:
                            df.at[i, 'patient-id'] = new_pat_id
                            df.at[i, 'patient-name'] = new_pat_id
                        else:
                            logger.info(f"Rename patient ID from {old_pat_id} to {new_pat_id} in report {f}.")
                    if makeitso:
                        save_csv(df, filepath)
