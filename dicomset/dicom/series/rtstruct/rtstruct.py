from __future__ import annotations

import numpy as np
import os
import pandas as pd
import pydicom as dcm
import re
from typing import Any, Dict, List, Literal, Tuple, TYPE_CHECKING

from .... import config as conf
from ....regions_map import RegionsMap
from ....typing import BatchLabelImage3D, FilePath, LandmarkID, Landmarks3D, Points3D, RegionID, SeriesID, Voxels
from ....utils.args import alias_kwargs, arg_to_list
from ....utils.conversion import to_numpy
from ....utils.geometry import to_image_coords
from ....utils.python import filter_lists
from ....utils.regions import regions_to_list
from ...utils.dicom import from_rtstruct_dicom
from ...utils.io import load_dicom
from ..series import DicomSeries
from .rtstruct_converter import RtStructConverter
if TYPE_CHECKING:
    from ...dataset import DicomDataset
    from ...patient import DicomPatient
    from ...study import DicomStudy
    from ..ct import DicomCtSeries

DEFAULT_LANDMARK_REGEXP = r'^Marker \d+$'
DICOM_RTSTRUCT_REF_CT_KEY = 'RefCTSeriesInstanceUID'

class DicomRtStructSeries(DicomSeries):
    def __init__(
        self,
        dataset: DicomDataset,
        patient: DicomPatient,
        study: DicomStudy,
        id: SeriesID,
        ref_ct: DicomCtSeries,
        index: pd.Series,
        index_policy: Dict[str, Any],
        config: Dict[str, Any] | None = None,
        regions_map: RegionsMap | None = None,
        ) -> None:
        super().__init__('rtstruct', dataset, patient, study, id, config=config)
        self.__filepath = os.path.join(conf.directories.datasets, 'dicom', dataset.id, 'data', 'patients', index['filepath'])
        self.__modality = 'rtstruct'
        self.__ref_ct = ref_ct
        self.__regions_map = regions_map

    @property
    def dicom(self) -> dcm.dataset.FileDataset:
        return load_dicom(self.__filepath)

    @property
    def filepath(self) -> FilePath:
        return self.__filepath

    def has_landmark(
        self,
        landmark_id: LandmarkID | List[LandmarkID],
        any: bool = False,
        **kwargs,
        ) -> bool:
        all_ids = self.list_landmarks(**kwargs)
        landmark_ids = arg_to_list(landmark_id, str, literals={ 'all': all_ids })
        n_overlap = len(np.intersect1d(landmark_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(landmark_ids)

    def has_region(
        self,
        region_id: RegionID | List[RegionID],
        any: bool = False,
        **kwargs,
        ) -> bool:
        all_ids = self.list_regions(**kwargs)
        region_ids = arg_to_list(region_id, str)
        n_overlap = len(np.intersect1d(region_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(region_ids)

    @property
    def landmark_regexp(self) -> str:
        if self._config is not None and 'landmarks' in self._config and 'regexp' in self._config['landmarks']:
            return self._config['landmarks']['regexp']
        else:
            return DEFAULT_LANDMARK_REGEXP

    def landmarks_data(
        self,
        points_only: bool = False,
        landmark_id: LandmarkID | List[LandmarkID] = 'all',
        landmark_regexp: str | None = None,
        n: int | None = None,
        show_ids: bool = True,
        use_world_coords: bool = True,
        **kwargs,
        ) -> Landmarks3D | Points3D | Voxels:
        # Load landmarks.
        landmark_ids = self.list_landmarks(landmark_id=landmark_id, landmark_regexp=landmark_regexp, **kwargs)
        rtstruct_dicom = self.dicom
        lms = []
        for l in landmark_ids:
            lm = RtStructConverter.get_roi_landmark(rtstruct_dicom, l)
            lms.append(lm)
        if len(lms) == 0:
            return None
        
        # Convert to DataFrame.
        lms = np.vstack(lms)
        landmarks_data = pd.DataFrame(lms, index=landmark_ids).reset_index()
        landmarks_data = landmarks_data.rename(columns={ 'index': 'landmark-id' })
        if not use_world_coords:
            landmarks_data = to_image_coords(landmarks_data, self.ref_ct.affine)

        # Filter by number of rows.
        if n is not None:
            landmarks_data = landmarks_data.iloc[:n]

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if show_ids:
            if 'patient-id' not in landmarks_data.columns:
                landmarks_data.insert(0, 'patient-id', self._pat_id)
            if 'study-id' not in landmarks_data.columns:
                landmarks_data.insert(1, 'study-id', self._study_id)
            if 'series-id' not in landmarks_data.columns:
                landmarks_data.insert(2, 'series-id', self._id)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        sort_cols = []
        if 'patient-id' in landmarks_data.columns:
            sort_cols += ['patient-id']
        if 'study-id' in landmarks_data.columns:
            sort_cols += ['study-id']
        if 'series-id' in landmarks_data.columns:
            sort_cols += ['series-id']
        sort_cols += ['landmark-id']
        landmarks_data = landmarks_data.sort_values(sort_cols)

        if points_only:
            return to_numpy(landmarks_data[list(range(3))])
        else:
            return landmarks_data

    def list_landmarks(
        self,
        landmark_id: LandmarkID | List[LandmarkID] = 'all',
        landmark_regexp: str | None = None,
        ) -> List[LandmarkID]:
        if landmark_regexp is None:
            landmark_regexp = self.landmark_regexp
        ids = self.list_regions(filter_landmarks=False)
        # Both landmarks/regions are stored in rtstruct, but we only want objects like 'Marker 1' for example.
        ids = [l for l in ids if re.match(landmark_regexp, l)]
        if landmark_id != 'all':
            landmark_ids = regions_to_list(landmark_id)
            ids = [i for i in ids if i in landmark_ids]
        return ids

    # 1. Should return only regions when landmark_regexp is None, load regexp from config or default.
    # 2. Should return landmarks also when use_landmark_regexp is False.

    @alias_kwargs([
        ('r', 'region_id'),
    ])
    def list_regions(
        self,
        filter_landmarks: bool = True,
        landmark_regexp: str | None = None,
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        return_numbers: bool = False,
        return_unmapped: bool = False,
        use_mapping: bool = True,
        ) -> List[RegionID] | Tuple[List[RegionID], List[RegionID]] | Tuple[List[RegionID], List[int]] | Tuple[List[RegionID], List[RegionID], List[int]] | List[int]:
        if filter_landmarks and landmark_regexp is None:
            landmark_regexp = self.landmark_regexp
        if self.__regions_map is None:
            use_mapping = False

        # Get disk regions.
        rtstruct_dicom = self.dicom
        true_disk_regions = RtStructConverter.get_roi_names(rtstruct_dicom)
        if return_numbers:
            nums = RtStructConverter.get_roi_numbers(rtstruct_dicom)

        # Filter regions on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        if return_numbers:
            true_disk_regions, nums = filter_lists([true_disk_regions, nums], lambda i: RtStructConverter.has_roi_data(rtstruct_dicom, i[0]))
        else:
            true_disk_regions = list(filter(lambda i: RtStructConverter.has_roi_data(rtstruct_dicom, i), true_disk_regions))
        
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

        return list(sorted(api_regions))
        
    @property
    def ref_ct(self) -> DicomCtSeries:
        return self.__ref_ct

    @alias_kwargs([
        ('r', 'region_id'),
    ])
    def regions_data(
        self,
        region_id: RegionID | List[RegionID] | Literal['all'] = 'all',
        return_regions: bool = False,
        use_mapping: bool = True,
        **kwargs,
        ) -> BatchLabelImage3D:
        if self.__regions_map is None:
            use_mapping = False

        # Get patient regions. If 'use_mapping=True', return unmapped region names too - we'll
        # need these to load regions from RTSTRUCT dicom.
        if use_mapping:
            region_ids, disk_ids = self.list_regions(region_id=region_id, return_unmapped=True, use_mapping=use_mapping, **kwargs)
        else:
            region_ids = self.list_regions(region_id=region_id, use_mapping=False)

        # Load data from dicom.
        rtstruct_dicom = self.dicom
        regions_data = None    # We don't know the shape yet.
        if use_mapping:
            # Regions could be grouped, e.g. Chestwall_L/R -> Chestwall.
            for i, d in enumerate(disk_ids):
                labels = from_rtstruct_dicom(rtstruct_dicom, self.ref_ct.size, self.ref_ct.affine, region_id=d)
                regions_data[i] = np.maximum(labels, axis=0)
        else:
            # Load and store region using unmapped name.
            regions_data = from_rtstruct_dicom(rtstruct_dicom, self.ref_ct.size, self.ref_ct.affine, region_id=region_ids)

        if return_regions:
            return regions_data, region_ids 
        else: 
            return regions_data

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
