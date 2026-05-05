from __future__ import annotations

import numpy as np
import os
import pandas as pd
from typing import Any, Dict, List, Literal, Tuple, TYPE_CHECKING

from ... import config as conf
from ...region_map import RegionMap
from ...typing import BatchLabelImage3D, FilePath, LandmarkID, Landmarks3D, RegExp, RegionID, RegionList, RtStructDicom, SeriesID
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.landmarks import landmarks_to_points
from ...utils.python import ensure_loaded
from ...utils.regions import region_to_list
from ..utils.dicom import from_rtstruct_dicom, list_rtstruct_landmarks, list_rtstruct_regions
from ..utils.io import load_dicom
from .series import DicomSeries
if TYPE_CHECKING:
    from ..dataset import DicomDataset
    from ..patient import DicomPatient
    from ..study import DicomStudy
    from .ct import DicomCtSeries

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
        region_map: RegionMap | None = None,
        ) -> None:
        super().__init__('rtstruct', dataset, patient, study, id, config=config)
        self.__filepath = os.path.join(conf.directories.datasets, 'dicom', dataset.id, 'data', 'patients', index['filepath'])
        self.__ref_ct = ref_ct
        self.__region_map = region_map

    @property
    @ensure_loaded('__dicom', '__load_data')
    def dicom(self) -> RtStructDicom:
        return self.__dicom

    @property
    def filepath(self) -> FilePath:
        return self.__filepath

    @ensure_loaded('__data', '__load_data')
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

    @ensure_loaded('__data', '__load_data')
    def has_region(
        self,
        region_id: RegionID | List[RegionID],
        any: bool = False,
        **kwargs,
        ) -> bool:
        all_ids = self.list_regions(**kwargs)
        disk_regions = list_rtstruct_regions(self.dicom, landmark_regexp=landmark_regexp)
        region_ids = region_to_list(region_id, disk_regions=disk_regions, region_map=self.__region_map, **kwargs)
        n_overlap = len(np.intersect1d(region_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(region_ids)

    @property
    def landmark_regexp(self) -> str:
        if self.__config is not None and 'landmarks' in self.__config and 'regexp' in self.__config['landmarks']:
            return self.__config['landmarks']['regexp']
        else:
            return DEFAULT_LANDMARK_REGEXP

    @ensure_loaded('__data', '__load_data')
    def landmarks_data(
        self,
        add_ids: bool = True,
        landmark_id: LandmarkID | List[LandmarkID] | Literal['all'] = 'all',
        landmark_regexp: RegExp | List[RegExp] | None = None,
        points_only: bool = False,
        use_world_coords: bool = True,
        **kwargs,
        ) -> Landmarks3D:
        # Get landmarks regexps.
        if landmark_regexp is None and self.__region_map is not None:
            landmark_regexp = self.__region_map.landmark_regexps

        # Load landmarks.
        landmark_ids = self.list_landmarks(landmark_id=landmark_id, landmark_regexp=landmark_regexp, **kwargs)
        _, landmarks_data = from_rtstruct_dicom(self.dicom, self.ref_ct.size, self.ref_ct.affine, landmark_id=landmark_ids, landmark_regexp=landmark_regexp, region_id=None)
        if landmarks_data is None:
            return None

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if add_ids:
            if 'patient-id' not in landmarks_data.columns:
                landmarks_data.insert(0, 'patient-id', self.__patient.id)
            if 'study-id' not in landmarks_data.columns:
                landmarks_data.insert(1, 'study-id', self.__study.id)
            if 'series-id' not in landmarks_data.columns:
                landmarks_data.insert(2, 'series-id', self.__id)

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

        # Extract points.
        if points_only:
            landmarks_data = landmarks_to_points(landmarks_data)

        return landmarks_data

    @alias_kwargs(
        ('l', 'landmark_id'),
        ('lr', 'landmark_regexp'),
        ('um', 'use_mapping'),
    )
    @ensure_loaded('__data', '__load_data')
    def list_landmarks(
        self,
        landmark_id: LandmarkID | List[LandmarkID] = 'all',
        landmark_regexp: str | None = None,
        use_mapping: bool = True,
        ) -> List[LandmarkID]:
        if self.__region_map is None:
            use_mapping = False

        # Get landmarks regexps.
        if landmark_regexp is None and self.__region_map is not None:
            landmark_regexp = self.__region_map.landmark_regexps

        # Get disk landmarks.
        # Don't pass 'landmark_id' here as "list_rtstruct_landmarks" doesn't know about region mapping.
        true_disk_landmarks = list_rtstruct_landmarks(self.dicom, landmark_regexp=landmark_regexp)

        # Map disk landmarks back to API landmarks.
        if landmark_id == 'all':
            if use_mapping:
                # Map back to the API landmark names.
                api_landmarks = [self.__region_map.map_disk_to_regions(r) for r in true_disk_landmarks]
                api_landmarks = [r for rs in api_landmarks for r in (rs if isinstance(rs, list) else [rs])]
            else:
                api_landmarks = true_disk_landmarks
        else:
            landmark_ids = region_to_list(landmark_id, disk_regions=true_disk_landmarks, region_map=self.__region_map)
            api_landmarks = []
            for r in landmark_ids:
                # Only keep landmarks that map to one or more disk landmarks.
                if use_mapping:
                    disk_landmarks = self.__region_map.map_regions_to_disk(r, disk_regions=true_disk_landmarks)
                    if len(np.intersect1d(disk_landmarks, true_disk_landmarks)) > 0:
                        api_landmarks.append(r)
                else:
                    if r in true_disk_landmarks:
                        api_landmarks.append(r)

        return list(sorted(set(api_landmarks)))

    @alias_kwargs(
        ('lr', 'landmark_regexp'),
        ('r', 'region_id'),
        ('um', 'use_mapping'),
    )
    @ensure_loaded('__data', '__load_data')
    def list_regions(
        self,
        filter_landmarks: bool = True,
        landmark_regexp: RegExp | List[RegExp] | None = None,
        region_id: RegionID | List[RegionID] | RegionList | Literal['all'] = 'all',
        use_mapping: bool = True,
        ) -> List[RegionID]:
        if self.__region_map is None:
            use_mapping = False

        # Get landmarks regexps.
        if landmark_regexp is None and self.__region_map is not None:
            landmark_regexp = self.__region_map.landmark_regexps

        # Get disk regions.
        # Don't pass 'region_id' here as "list_rtstruct_regions" doesn't know about region mapping.
        true_disk_regions = list_rtstruct_regions(self.dicom, landmark_regexp=landmark_regexp)

        # Map disk regions back to API regions.
        if region_id == 'all':
            if use_mapping:
                # Map back to the API region names.
                api_regions = [self.__region_map.map_disk_to_regions(r) for r in true_disk_regions]
                api_regions = [r for rs in api_regions for r in (rs if isinstance(rs, list) else [rs])]
            else:
                api_regions = true_disk_regions
        else:
            region_ids = region_to_list(region_id, disk_regions=true_disk_regions, region_map=self.__region_map)
            api_regions = []
            for r in region_ids:
                # Only keep regions that map to a one or more disk regions.
                if use_mapping:
                    disk_regions = self.__region_map.map_regions_to_disk(r, disk_regions=true_disk_regions)
                    if len(np.intersect1d(disk_regions, true_disk_regions)) > 0:
                        api_regions.append(r)
                else:
                    if r in true_disk_regions:
                        api_regions.append(r)

        return list(sorted(set(api_regions)))

    # 1. Should return only regions when landmark_regexp is None, load regexp from config or default.
    # 2. Should return landmarks also when use_landmark_regexp is False.

    def __load_data(self) -> None:
        self.__dicom = load_dicom(self.__filepath)
        
    @property
    def ref_ct(self) -> DicomCtSeries:
        return self.__ref_ct

    @alias_kwargs(
        ('r', 'region_id'),
        ('rr', 'return_regions'),
        ('um', 'use_mapping'),
    )
    @ensure_loaded('__data', '__load_data')
    def regions_data(
        self,
        region_id: RegionID | List[RegionID] | RegionList | Literal['all'] = 'all',
        landmark_regexp: RegExp | List[RegExp] | None = None,
        return_regions: bool = True,
        use_mapping: bool = True,
        **kwargs,
        ) -> Tuple[List[RegionID], BatchLabelImage3D]:
        if self.__region_map is None:
            use_mapping = False

        # Get landmarks regexps.
        if landmark_regexp is None and self.__region_map is not None:
            landmark_regexp = self.__region_map.landmark_regexps

        # Get required regions.
        region_ids = self.list_regions(landmark_regexp=landmark_regexp, region_id=region_id, use_mapping=use_mapping, **kwargs)

        # Get disk regions.
        # These are need for matching regexps from the region map.
        true_disk_regions = list_rtstruct_regions(self.dicom, landmark_regexp=landmark_regexp)

        # Add regions data.
        # This is pretty inefficient as we load up the rtstruct dicom for each API region individually.
        # We should probably batch this operation.
        # This should be easy - just concatenate the list of resolved disk regions (I don't think the
        # from_rtstruct_dicom code sorts or uniques the list - this is important because different API
        # regions could map to the same disk region). An even better way would be to form a map from API
        # to disk regions (we already have) and just load the unique disk regions and then map the results
        # back for the region aggregation.

        # Map to disk regions.
        mapped_regions = {}
        for r in region_ids:
            if use_mapping and self.__region_map is not None:
                mapped_regions[r] = self.__region_map.map_regions_to_disk(r, disk_regions=true_disk_regions)
            else:
                mapped_regions[r] = [r]

        # Load up a unique set of disk regions.
        disk_regions = [ri for r in mapped_regions.values() for ri in r]
        unique_disk_regions = list(sorted(np.unique(disk_regions).tolist()))

        # We can assume all passed regions are loaded, as they come from 'list_regions' which is based on regions
        # present in the rtstruct dicom.
        _, labels = from_rtstruct_dicom(self.dicom, self.ref_ct.size, self.ref_ct.affine, landmark_id=None, region_id=unique_disk_regions)

        # Aggregate results for each API region.
        regions_data = None    # We don't know the shape yet.
        for i, r in enumerate(region_ids):
            disk_regions = mapped_regions[r]

            # Load and sum multiple regions.
            indices = [unique_disk_regions.index(dr) for dr in disk_regions]
            rlabels = labels[indices]
            reg_data = np.sum(rlabels, axis=0).clip(0, 1).astype(bool)

            # Add to main tensor.
            if regions_data is None:
                regions_data = np.zeros((len(region_ids), *reg_data.shape), dtype=bool)
            regions_data[i] = reg_data

        if return_regions:
            return region_ids, regions_data
        else:
            return regions_data

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
