import os
import pandas as pd
from tqdm import tqdm

from ...typing import DatasetID
from ...utils.geometry import diameter, volume
from ...utils.io import save_csv
from ...utils.pandas import append_row
from ..dataset import NiftiDataset

def create_regions_report(
    dataset: DatasetID,
    ) -> pd.DataFrame:
    # List patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()

    # Create dataframe.
    cols = {
        'patient-id': str,
        'study-id': str,
        'series-id': str,
        'region': str,
        'metric': str,
        'value': float,
    }
    df = pd.DataFrame(columns=cols.keys())

    # Add rows.
    for p in tqdm(pat_ids):
        pat = set.patient(p)
        study_ids = pat.list_studies()
        for s in study_ids:
            study = pat.study(s)
            series_ids = study.list_regions_series()
            for ss in series_ids:
                # Load data.
                series = study.regions_series(ss)
                regions, regions_data = series.data()
                ref_affine = series.ref_ct.affine

                for r, rd in zip(regions, regions_data):
                    # Handle empty labels.
                    if rd.sum() == 0:
                        continue

                    # Calculate metrics.
                    metrics = [
                        'diameter-cm',
                        'diameter-mm',
                        'volume-cm3',
                        'volume-mm3',
                    ]
                    diam_mm = diameter(rd, affine=ref_affine)
                    vol_mm3 = volume(rd, affine=ref_affine)
                    values = [
                        diam_mm / 10,
                        diam_mm,
                        vol_mm3 / 1000,
                        vol_mm3,
                    ]

                    for m, v in zip(metrics, values):
                        data = {
                            'patient-id': p,
                            'study-id': s,
                            'series-id': ss,
                            'region': r,
                            'metric': m,
                            'value': v,
                        }
                        df = append_row(df, data)

    # Save report.
    f = os.path.join(set.path, 'reports', 'regions.csv')
    save_csv(df, f)
