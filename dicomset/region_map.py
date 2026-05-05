from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from .typing import DirPath, DiskRegionID, FilePath, RegExp, RegionID, RegionList
from .utils.args import arg_to_list
from .utils.io import load_yaml

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
        self.__landmarks_data = data['landmarks'] if 'landmarks' in data else None
        self.__lists_data = data['lists'] if 'lists' in data else None
        self.__mappings_data = data['mappings'] if 'mappings' in data else None
        self.__filepath = filepath

    @property
    def landmarks_data(self) -> Dict[str, Any]:
        return self.__landmarks_data

    @property
    def lists_data(self) -> Dict[str, Any]:
        return self.__lists_data

    @property
    def mappings_data(self) -> Dict[str, Any]:
        return self.__mappings_data

    @property
    def landmark_regexps(self) -> List[RegExp] | None:
        if self.__landmarks_data is None:
            return None
        regexps = []
        for k, v in self.__landmarks_data.items():
            if isinstance(v, str) and v.startswith('re:'):
                regexps.append(v[3:])
        return regexps

    @property
    def landmarks_data(self) -> Dict[str, Any]:
        return self.__landmarks_data

    @property
    def filepath(self) -> FilePath:
        return self.__filepath

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

    # Takes API regions and returns actual disk regions - these are the leaf nodes
    # of the regions map chains.
    def map_disk_to_regions(
        self,
        region_id: DiskRegionID | List[DiskRegionID],
        ) -> List[RegionID]:
        region_ids = arg_to_list(region_id, str)

        api_regions = [] 
        for r in region_ids:
            # Check mappings.
            if self.__mappings_data is not None:
                for k, v in self.__mappings_data.items():
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
            if self.__mappings_data is not None:
                for k, v in self.__mappings_data.items():
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

    def region_list(
        self,
        name: RegionList,
        ) -> List[RegionID]:
        if self.__lists_data is None or not name in self.__lists_data:
            raise ValueError(f"Region list '{name}' not found.")
        return self.__lists_data[name]

    def __repr__(self) -> str:
        return str(self)

    # Takes disk regions and maps them to all possible API regions that they are
    # a part of, including themselves.
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__data})"
