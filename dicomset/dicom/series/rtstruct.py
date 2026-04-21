from __future__ import annotations

import numpy as np
import os
import pandas as pd
import re
from typing import Any, Callable, Dict, List, Literal, TYPE_CHECKING

from ... import config as conf
from ...region_map import RegionMap
from ...typing import BatchLabelImage3D, FilePath, LandmarkID, Landmarks3D, Points3D, RegionID, RegionList, RtStructDicom, SeriesID, Voxels
from ...utils.args import alias_kwargs, arg_to_list
from ...utils.conversion import to_numpy
from ...utils.landmarks import landmarks_to_points
from ...utils.python import has_private_attr
from ...utils.regions import region_to_list
from ..utils.dicom import from_rtstruct_dicom, list_rtstruct_regions
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
        self.__modality = 'rtstruct'
        self.__ref_ct = ref_ct
        self.__region_map = region_map

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__dicom'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def dicom(self) -> RtStructDicom:
        return self.__dicom

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
        region_ids = region_to_list(region_id, region_map=self.__region_map, **kwargs)
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
        add_ids: bool = True,
        points_only: bool = False,
        landmark_id: LandmarkID | List[LandmarkID] = 'all',
        landmark_regexp: str | None = None,
        use_world_coords: bool = True,
        **kwargs,
        ) -> Landmarks3D | Points3D | Voxels:
        # Load landmarks.
        landmark_ids = self.list_landmarks(landmark_id=landmark_id, landmark_regexp=landmark_regexp, **kwargs)
        _, landmarks_data = from_rtstruct_dicom(self.dicom, self.ref_ct.size, self.ref_ct.affine, landmark_id=landmark_ids, landmark_regexp=landmark_regexp, region_id=None)
        if landmarks_data is None:
            return None

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if add_ids:
            if 'patient-id' not in landmarks_data.columns:
                landmarks_data.insert(0, 'patient-id', self._pat.id)
            if 'study-id' not in landmarks_data.columns:
                landmarks_data.insert(1, 'study-id', self._study.id)
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
            return landmarks_to_points(landmarks_data)
        else:
            return landmarks_data

    def list_landmarks(
        self,
        landmark_id: LandmarkID | List[LandmarkID] = 'all',
        landmark_regexp: str | None = None,
        ) -> List[LandmarkID]:
        landmark_regexp = landmark_regexp or DEFAULT_LANDMARK_REGEXP
        ids = self.list_regions(filter_landmarks=False)
        # Both landmarks/regions are stored in rtstruct, but we only want objects like 'Marker 1' for example.
        ids = [l for l in ids if re.match(landmark_regexp, l)]
        if landmark_id != 'all':
            landmark_ids = region_to_list(landmark_id, region_map=self.__region_map)
            ids = [i for i in ids if i in landmark_ids]
        return ids

    @alias_kwargs(
        ('r', 'region_id'),
    )
    def list_regions(
        self,
        filter_landmarks: bool = True,
        landmark_regexp: str | None = None,
        region_id: RegionID | List[RegionID] | RegionList | Literal['all'] = 'all',
        use_mapping: bool = True,
        ) -> List[RegionID]:
        if filter_landmarks and landmark_regexp is None:
            landmark_regexp = self.landmark_regexp
        if self.__region_map is None:
            use_mapping = False

        # Get disk regions.
        true_disk_regions = list_rtstruct_regions(self.dicom)

        # Map disk regions back to API regions.
        if region_id == 'all':
            if use_mapping:
                # Map back to the API region names.
                api_regions = [self.__region_map.map_disk_to_regions(r) for r in true_disk_regions]
                api_regions = [r for rs in api_regions for r in (rs if isinstance(rs, list) else [rs])]
            else:
                api_regions = true_disk_regions
        else:
            region_ids = region_to_list(region_id, region_map=self.__region_map)
            api_regions = []
            for r in region_ids:
                # Only keep regions that map to a one or more disk regions.
                if use_mapping:
                    disk_regions = self.__region_map.map_region_to_disk(r)
                    if len(np.intersect1d(disk_regions, true_disk_regions)) > 0:
                        api_regions.append(r)
                else:
                    if r in true_disk_regions:
                        api_regions.append(r)

        # Filter landmarks. Currently at the region level, does this ever need to
        # be at disk level?
        if landmark_regexp is not None:
            api_regions = [r for r in api_regions if not re.match(landmark_regexp, r)]

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
    def regions_data(
        self,
        region_id: RegionID | List[RegionID] | RegionList | Literal['all'] = 'all',
        return_regions: bool = False,
        use_mapping: bool = True,
        **kwargs,
        ) -> BatchLabelImage3D:
        if self.__region_map is None:
            use_mapping = False

        # Get required regions.
        region_ids = self.list_regions(region_id=region_id, use_mapping=use_mapping, **kwargs)

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
                mapped_regions[r] = self.__region_map.map_region_to_disk(r)
            else:
                mapped_regions[r] = [r]

        # Load up a unique set of disk regions.
        unique_disk_regions = set()
        for r in mapped_regions.values():
            unique_disk_regions.update(r)
        unique_disk_regions = list(unique_disk_regions)
        labels, _ = from_rtstruct_dicom(self.dicom, self.ref_ct.size, self.ref_ct.affine, region_id=unique_disk_regions)

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
            return regions_data, region_ids
        else:
            return regions_data

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
