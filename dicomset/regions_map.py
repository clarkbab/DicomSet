from __future__ import annotations

import os
import re
from typing import Dict, List

from .typing import RegionID
from .utils.args import arg_to_list
from .utils.io import load_yaml

RM_FILE_REGEXP = r"regions?[-_]map\.ya?ml"

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

class RegionsMap:
    def __init__(
        self,
        data: Dict[RegionID, RegionID]) -> None:
        self.__data = data

    @property
    def data(self) -> Dict[RegionID, RegionID]:
        return self.__data

    @classmethod
    def load(
        cls,
        dirpath
        ) -> RegionsMap | None:
        files = os.listdir(dirpath)
        rm_files = [f for f in files if re.match(RM_FILE_REGEXP, f)]
        if not rm_files:
            return None
        data = load_yaml(os.path.join(dirpath, rm_files[0]))
        return cls(data)

    # Takes API regions and returns actual disk regions - these are the leaf nodes
    # of the regions map chains.
    def map_region(
        self,
        region_id: RegionID | List[RegionID],
        ) -> List[RegionID]:
        region_ids = arg_to_list(region_id, str)

        disk_regions = [] 
        for r in region_ids:
            matched = False

            # Check literal matches.
            literals = self.__data['literals'] if 'literals' in self.__data else self.__data if 'regexes' not in self.__data else None
            if literals is not None:
                for k, v in literals.items():
                    if k == r:
                        matched = True
                        disk_regs = arg_to_list(v, str)
                        # Map to disk regions - don't add intermediate mappings.
                        disk_regs = [self.map_region(v) for v in disk_regs]
                        disk_regs = [vi for v in disk_regs for vi in (v if isinstance(v, list) else [v])]  # Flatten list of lists.
                        disk_regions.extend(disk_regs)
                        break

            # # Check regex matches.
            # regexes = self.__data['regexes'] if 'regexes' in self.__data else None
            # if regexes is not None:
            #     for k, v in regexes.items():
            #         if re.match(k, region, flags=re.IGNORECASE):
            #             return v

            if not matched and r not in disk_regions:
                disk_regions.append(r)

        return list(sorted(set(disk_regions)))

    # Takes disk regions and maps them to all possible API regions that they are
    # a part of, including themselves.
    def unmap_region(
        self,
        region_id: RegionID | List[RegionID],
        ) -> List[RegionID]:
        region_ids = arg_to_list(region_id, str)

        api_regions = [] 
        for r in region_ids:
            # Check literal matches.
            literals = self.__data['literals'] if 'literals' in self.__data else self.__data if 'regexes' not in self.__data else None
            if literals is not None:
                for k, v in literals.items():
                    disk_regs = arg_to_list(v, str)
                    if r in disk_regs:
                        matched = True

                        # Add intermediate mappings - these are still API regions.
                        api_regions.append(k)

                        # Unmap these regions.
                        api_regs = self.unmap_region(k)
                        api_regions.extend(api_regs)

                        # Don't break, the same disk region could be included in multiple mappings.

            # Disk regions are also API accessible.
            api_regions.append(r)

        return list(sorted(set(api_regions)))
