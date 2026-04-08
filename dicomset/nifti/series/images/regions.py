from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import List, Literal, Tuple, TYPE_CHECKING

from .... import config
from ....dicom import DicomDataset
if TYPE_CHECKING:
    from ....dicom import DicomRtStructSeries
from ....regions_map import RegionsMap
from ....typing import BatchLabelImage3D, FilePath, RegionID, SeriesID
from ....utils.args import alias_kwargs, arg_to_list
from ....utils.io import load_nifti, load_nrrd
if TYPE_CHECKING:
    from ...dataset import NiftiDataset
    from ...patient import NiftiPatient
    from ...study import NiftiStudy
from .image import NiftiImageSeries

class NiftiRegionsSeries(NiftiImageSeries):
    def __init__(
        self,
        dataset: NiftiDataset,
        patient: NiftiPatient,
        study: NiftiStudy,
        id: SeriesID,
        index: pd.DataFrame | None = None,
        regions_map: RegionsMap | None = None,
        ) -> None:
        super().__init__('regions', dataset, patient, study, id, index=index)
        extensions = ['.nii', '.nii.gz', '.nrrd']
        dirpath = os.path.join(config.directories.datasets, 'nifti', self._dataset.id, 'data', 'patients', self._pat.id, self._study.id, self._modality, self._id)
        if not os.path.exists(dirpath):
            raise ValueError(f"No regions series '{self._id}' found for study '{self._study.id}'. Dirpath: {dirpath}")
        self.__path = dirpath
        self.__regions_map = regions_map

    @alias_kwargs([
        ('r', 'region_id'),
    ])
    def data(
        self,
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        return_regions: bool = False,
        use_mapping: bool = True,
        ) -> BatchLabelImage3D | Tuple[BatchLabelImage3D, List[RegionID]]:
        region_ids = self.list_regions(region_id=region_id, use_mapping=use_mapping)
        if self.__regions_map is None:
            use_mapping = False

        # Add regions data.
        regions_data = None    # We don't know the shape yet.
        for i, r in enumerate(region_ids):
            # Get disk regions.
            if use_mapping:
                disk_regions = self.__regions_map.map_region(r)
            else:
                disk_regions = [r]

            # Load and sum multiple regions.
            reg_data = []
            for reg in disk_regions:
                matched = False
                extensions = ['.nii', '.nii.gz', '.nrrd']
                for e in extensions:
                    filepath = os.path.join(self.__path, f'{reg}{e}')
                    if os.path.exists(filepath):
                        matched = True
                        if e in ('.nii', '.nii.gz'):
                            d, _ = load_nifti(filepath)
                        else:
                            d, _ = load_nrrd(filepath)
                        reg_data.append(d)
                        break
            if len(reg_data) == 0:
                continue
            reg_data = np.sum(reg_data, axis=0).clip(0, 1).astype(bool)

            # Add to main tensor.
            if regions_data is None:
                regions_data = np.zeros((len(region_ids), *reg_data.shape), dtype=bool)
            regions_data[i] = reg_data

        if return_regions:
            return regions_data, region_ids
        else:
            return regions_data

    @property
    def dicom(self) -> DicomRtStructSeries:
        if self._index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self._index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self._dataset.id) & (index['patient-id'] == self._pat.id) & (index['study-id'] == self._study.id) & (index['series-id'] == self._id) & (index['modality'] == 'regions')].drop_duplicates()
        assert len(index) == 1, f"Expected one row in index for series '{self.id}', but found {len(index)}. Index: {index}"
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).rtstruct_series(row['dicom-series-id'])

    def filepaths(
        self,
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        regions_ignore_missing: bool = True,
        ) -> List[FilePath]:
        region_ids = arg_to_list(region_id, str, literals={ 'all': self.list_regions })
        if not regions_ignore_missing and not self.has_region(region_ids):
            raise ValueError(f'Regions {region_ids} not found in series {self.id}.')
        region_ids = [r for r in region_ids if self.has_region(r)]  # Filter out missing regions.
        # Region mapping is many-to-one, so we could get multiple files on disk for the same mapped region.
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        disk_ids = self.__regions_map.inv_map_region(region_ids, disk_regions=self.list_regions(use_mapping=False)) if self.__regions_map is not None else region_ids
        disk_ids = arg_to_list(disk_ids, str)
        # Check all possible file extensions.
        filepaths = [os.path.join(self.__path, f'{i}{e}') for i in disk_ids for e in image_extensions if os.path.exists(os.path.join(self.__path, f'{i}{e}'))]
        return filepaths

    def has_region(
        self,
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        any: bool = False,
        **kwargs,
        ) -> bool:
        all_ids = self.list_regions(**kwargs)
        region_ids = arg_to_list(region_id, str, literals={ 'all': all_ids })
        n_overlap = len(np.intersect1d(region_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(region_ids)

    @alias_kwargs([
        ('r', 'region_id'),
        ('um', 'use_mapping'),
    ])
    def list_regions(
        self,
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        use_mapping: bool = True,
        ) -> List[RegionID]:
        if self.__regions_map is None:
            use_mapping = False

        true_disk_regions = self.__load_disk_regions()
        
        # Map disk regions back to API regions.
        if region_id == 'all':
            if use_mapping:
                # Map back to the API region names.
                api_regions = [self.__regions_map.unmap_region(r) for r in true_disk_regions]
                api_regions = [r for rs in api_regions for r in (rs if isinstance(rs, list) else [rs])]
            else:
                api_regions = true_disk_regions
        else:
            region_ids = arg_to_list(region_id, str)
            api_regions = []
            for r in region_ids:
                # Only keep regions that map to a one or more disk regions.
                if use_mapping:
                    disk_regions = self.__regions_map.map_region(r)
                    if len(np.intersect1d(disk_regions, true_disk_regions)) > 0:
                        api_regions.append(r)
                else:
                    if r in true_disk_regions:
                        api_regions.append(r)

        return list(sorted(set(api_regions)))

    def __load_disk_regions(self) -> List[RegionID]:
        extensions = ['.nii', '.nii.gz', '.nrrd']
        files = os.listdir(self.__path)
        disk_regions = [f.replace(e, '') for f in files for e in extensions if f.endswith(e)]
        return list(sorted(set(disk_regions)))

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
