from __future__ import annotations

import numpy as np
import os
import re
from typing import Any, Dict, List

from .typing import DirPath, DiskRegionID, FilePath, RegExp, RegionID, RegionList
from .utils.args import arg_to_list
from .utils.conversion import to_list
from .utils.io import load_yaml
from .utils.python import ensure_loaded

RM_FILENAME_REGEXP = r"regions?[-_]map\.ya?ml"

# Example:
# All: [BrachialPlex, BrainStem, Cavity_Oral, Parotid, SpinalCord]
# BrachialPlex: [BrachialPlex_L, BrachialPlex_R]
# Nerves: [BrachialPlex, SpinalCord]
# Parotid: [Parotid_L, Parotid_R]
# BrainStem: Brainstem

# str => str mappings are true mappings, e.g. BrainStem: Brainstem replaces 
# the region ID on disk
# "Brainstem" with the API region "BrainStem". "Brainstem" will then not be 
# visible unless "use_mapping=False". str => List[str] mappings are "groupings"
# , e.g. "Parotid: [Parotid_L, Parotid_R]" and all names will be available 
# through the API. The List[str] should refer to disk regions unless these 
# mappings of these str items to real disk regions have been provided elsewhere
# in the region map.

class RegionMap:
    def __init__(
        self,
        # Maps regions to disk regions or lists of regions (or disk regions).
        # data: Dict[RegionID, DiskRegionID | RegionID | List[DiskRegionID | RegionID]],
        data: Dict[str, Any],
        filepath: FilePath,
        ) -> None:
        self.__data = data

    @property
    def filepath(self) -> FilePath:
        return self.__filepath

    @property
    def landmark_regexps(self) -> List[RegExp] | None:
        regexps = []
        for k, v in self.landmarks.items():
            for vi in v:
                if isinstance(vi, str) and vi.startswith('re:'):
                    regexps.append(vi[3:])
        return to_list(np.unique(regexps))

    @property
    @ensure_loaded('__landmarks', '__load_landmarks')
    def landmarks(self) -> Dict[str, Any]:
        return self.__landmarks

    @property
    @ensure_loaded('__regions', '__load_regions')
    def regions(self) -> Dict[RegionList, List[RegionID]]:
        return self.__regions

    @classmethod
    def load(
        cls,
        dirpath: DirPath,
        ) -> RegionMap | None:
        files = os.listdir(dirpath)
        rm_files = [f for f in files if re.match(RM_FILENAME_REGEXP, f)]
        if not rm_files:
            return None
        filepath = os.path.join(dirpath, rm_files[0])
        data = load_yaml(filepath)
        return cls(data, filepath)

    def __resolve_list(
        self,
        l: List[RegionID | LandmarkID | RegionList | RegExp],
        all_lists: Dict[str, List[RegionID | LandmarkID | RegionList | RegExp]],
    ) -> List[RegionID]:
        resolved = []
        for li in l:
            if li in all_lists:
                v = arg_to_list(all_lists[li], str)
                resolved.extend(self.__resolve_list(v, all_lists))
            else:
                resolved.append(li)
        return resolved

    def __load_landmarks(self) -> None:
        if 'landmarks' not in self.__data:
            self.__landmarks = None
            return

        landmarks = self.__data['landmarks']
        self.__landmarks = {}
        for k, v in landmarks.items():
            v = arg_to_list(v, str)
            self.__landmarks[k] = self.__resolve_list(v, landmarks)

    def __load_regions(self) -> None:
        if 'regions' not in self.__data:
            self.__regions = None
            return

        self.__regions = {}
        regions = self.__data['regions']
        for k, v in regions.items():
            v = arg_to_list(v, str)
            self.__regions[k] = self.__resolve_list(v, regions)

    def __load_mappings(self) -> None:
        if 'mappings' not in self.__data:
            self.__mappings = None
            return
        self.__mappings = self.__data['mappings']

    def map_disk_to_regions(
        self,
        region_id: DiskRegionID | List[DiskRegionID],
        ) -> List[RegionID]:
        region_ids = arg_to_list(region_id, str)

        api_regions = [] 
        for r in region_ids:
            # Check mappings.
            if self.mappings is not None:
                for k, v in self.mappings.items():
                    if v.startswith('re:'):
                        regex = re.compile(v[3:], flags=re.IGNORECASE)
                        if regex.match(r):
                            api_regions.append(k)
                    else:
                        disk_regs = arg_to_list(v, str)
                        if r in disk_regs:
                            # Add intermediate mappings - these are still API regions.
                            api_regions.append(k)

                            # Unmap these regions.
                            api_regs = self.map_disk_to_regions(k)
                            api_regions.extend(api_regs)

                    # Don't break, the same disk region could be included in multiple mappings.
                    continue

            # Disk regions are also API accessible.
            api_regions.append(r)

        return list(sorted(set(api_regions)))

    # Takes API regions and returns actual disk regions - these are the leaf nodes
    # of the regions map chains.
    def map_regions_to_disk(
        self,
        region_id: RegionID | List[RegionID],
        disk_regions: List[DiskRegionID] | None = None
        ) -> List[DiskRegionID]:
        region_ids = arg_to_list(region_id, str)

        disk_regions = [] 
        for r in region_ids:
            # Check mappings.
            matched = False
            if self.mappings is not None:
                for k, v in self.mappings.items():
                    if v.startswith('re:'):
                        assert disk_regions is not None, "Disk regions must be provided for regex mapping."
                        v = v[3:]
                        regex = re.compile(v, flags=re.IGNORECASE)
                        for d in disk_regions:
                            if regex.match(d):
                                disk_regions.append(k)
                    elif k == r:
                        matched = True
                        # "v" could be a list of regions.
                        # Map all these regions to disk as they could be intermediate
                        # (e.g. other API) region IDs.
                        disk_regs = arg_to_list(v, str)
                        disk_regs = [self.map_regions_to_disk(v) for v in disk_regs]
                        disk_regs = [vi for v in disk_regs for vi in (v if isinstance(v, list) else [v])]  # Flatten list of lists.
                        disk_regions.extend(disk_regs)
                        break

            # If not mapping, append as is.
            if not matched and r not in disk_regions:
                disk_regions.append(r)

        return list(sorted(set(disk_regions)))

    @property
    @ensure_loaded('__mappings', '__load_mappings')
    def mappings(self) -> Dict[str, Any]:
        return self.__mappings

    def region_list(
        self,
        name: RegionList,
        ) -> List[RegionID]:
        regions = self.regions
        if regions is None or not name in regions:
            raise ValueError(f"Region list '{name}' not found.")
        return regions[name]

    def __repr__(self) -> str:
        return str(self)

    # Takes disk regions and maps them to all possible API regions that they are
    # a part of, including themselves.
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(landmarks={self.landmarks}, mappings={self.mappings}, regions={self.regions})"
