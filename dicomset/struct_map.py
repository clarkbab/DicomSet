from __future__ import annotations

import numpy as np
import os
import re
from typing import Any, Dict, List

from .typing import DirPath, DiskLandmarkID, DiskRegionID, FilePath, LandmarkID, LandmarkList, RegExp, RegionID, RegionList
from .utils.args import arg_to_list
from .utils.conversion import to_list
from .utils.io import load_yaml
from .utils.python import ensure_loaded

SM_FILENAME_REGEXP = r"struct[-_]?map\.ya?ml"

class StructMap:
    def __init__(
        self,
        # Maps regions to disk regions or lists of regions (or disk regions).
        # data: Dict[RegionID, DiskRegionID | RegionID | List[DiskRegionID | RegionID]],
        data: Dict[str, Any],
        filepath: FilePath,
        ) -> None:
        self.__data = data
        self.__filepath = filepath

    def expand_list(
        self,
        id: LandmarkID | RegionID | LandmarkList | RegionList | List[LandmarkID | RegionID | LandmarkList | RegionList],
        disk_ids: List[DiskLandmarkID | DiskRegionID] | None = None,
        sort: bool = True,
        ) -> List[LandmarkID | RegionID]:
        ids = arg_to_list(id, str)
        expanded = []
        for i in ids:
            if self.lists is not None and i in self.lists:
                v = self.lists[i]
                # Resolve any regexps.
                for vi in v:
                    if vi.startswith('re:'):
                        assert disk_ids is not None, "Disk IDs must be provided for regexp mapping."
                        regex = re.compile(vi[3:], flags=re.IGNORECASE)
                        for d in disk_ids:
                            if regex.match(d):
                                expanded.append(d)
                    else:
                        expanded.append(vi)
            else:
                expanded.append(i)
        if sort:
            return list(sorted(set(expanded)))
        return list(dict.fromkeys(expanded))    # Remove duplicates without sorting.

    @property
    def filepath(self) -> FilePath:
        return self.__filepath

    @property
    def landmark_regexps(self) -> List[RegExp] | None:
        if self.landmarks is None:
            return None
        regexps = []
        for v in self.landmarks:
            if isinstance(v, str) and v.startswith('re:'):
                regexps.append(v[3:])
        return to_list(np.unique(regexps)) if regexps else None

    @property
    @ensure_loaded('__landmarks', '__load_landmarks')
    def landmarks(self) -> List[str] | None:
        return self.__landmarks

    @property
    @ensure_loaded('__lists', '__load_lists')
    def lists(self) -> Dict[LandmarkList | RegionList, List[LandmarkID | RegionID]]:
        return self.__lists

    @classmethod
    def load(
        cls,
        dirpath: DirPath,
        ) -> StructMap | None:
        files = os.listdir(dirpath)
        files = [f for f in files if re.match(SM_FILENAME_REGEXP, f, flags=re.IGNORECASE)]
        if not files:
            return None
        filepath = os.path.join(dirpath, files[0])
        data = load_yaml(filepath)
        return cls(data, filepath)

    def __load_landmarks(self) -> None:
        if 'landmarks' not in self.__data:
            self.__landmarks = None
            return

        landmarks = self.__data['landmarks']
        self.__landmarks = arg_to_list(landmarks, str)

    # This does some pre-compute to resolve lists of lists,
    # so that it doesn't need to be done each time we map.
    def __load_list(
        self,
        l: List[LandmarkID | RegionID | LandmarkList | RegionList | RegExp],
        ) -> List[LandmarkID | RegionID]:
        loaded = []
        for li in l:
            # Literal match with another list name.
            if li in self.__data['lists']:
                v = arg_to_list(self.__data['lists'][li], str)
                loaded.extend(self.__load_list(v))
            # Just copy the item as is.
            else:
                loaded.append(li)
        return loaded

    # This also does some pre-compute to resolve intermediate
    # mappings, e.g. [Brainstem, re:^Parotid(\s_)?L$] -> 
    # [BrainStem, Parotid_L] -> HN_OARs.
    def __load_lists(self) -> None:
        if 'lists' not in self.__data:
            self.__lists = None
            return

        self.__lists = {}
        for k, v in self.__data['lists'].items():
            v = arg_to_list(v, str)
            self.__lists[k] = self.__load_list(v)

    # Recursively resolves intermediate mappings so that recursion doesn't
    # need to be applied with each mapping call.
    def __load_mapping(
        self,
        l: List[LandmarkID | RegionID | RegExp],
        ) -> List[LandmarkID | RegionID]:
        loaded = []
        for li in l:
            # Literal match with another mapping name.
            if li in self.__data['mappings']:
                v = arg_to_list(self.__data['mappings'][li], str)
                loaded.extend(self.__load_mapping(v))
            # Just copy the item as is.
            else:
                loaded.append(li)
        return loaded

    def __load_mappings(self) -> None:
        if 'mappings' not in self.__data:
            self.__mappings = None
            return

        self.__mappings = {}
        for k, v in self.__data['mappings'].items():
            v = arg_to_list(v, str)
            self.__mappings[k] = self.__load_mapping(v)

    # Mappings have already been resolved to disk IDs during load.
    def map_api_to_disk(
        self,
        api_ids: LandmarkID | RegionID | List[LandmarkID | RegionID],
        disk_ids: List[DiskLandmarkID | DiskRegionID] | None = None,
        ) -> List[DiskRegionID]:
        api_ids = arg_to_list(api_ids, str)
        true_disk_ids = disk_ids

        disk_ids = [] 
        for i in api_ids:
            # Check mappings.
            if self.mappings is not None and i in self.mappings:
                values = self.mappings[i]
                for v in values: 
                    # Handle regexp - add all matching disk IDs.
                    if v.startswith('re:'):
                        assert true_disk_ids is not None, "Disk IDs must be provided for regexp mapping."
                        regex = re.compile(v[3:], flags=re.IGNORECASE)
                        for d in true_disk_ids:
                            if regex.match(d):
                                disk_ids.append(k)
                    else:
                        disk_ids.append(v)
            else:
                disk_ids.append(i)

        return list(sorted(set(disk_ids)))

    # This method takes a list of disk regions (and landmarks) and returns the full
    # list of API regions that use these disk regions. This could include recursive
    # definitions, e.g. if we have a mapping from Nerves -> Brainstem -> BrainStem.
    # Then, API regions for 'BrainStem' disk region should be: ['Brainstem', 'Nerves'].
    def map_disk_to_api(
        self,
        disk_ids: DiskLandmarkID | DiskRegionID | List[DiskLandmarkID | DiskRegionID],
        ) -> List[LandmarkID | RegionID]:
        disk_ids = arg_to_list(disk_ids, str)

        api_ids = [] 
        for i in disk_ids:
            # Check if this disk region maps to any API regions, otherwise, it is
            # an API region itself.
            if self.mappings is not None:
                for k, v in self.mappings.items():
                    # 'k' is the API region.
                    # 'v' is the list of disk regions (or potentially intermediate
                    # API regions if chaining).
                    for vi in v:
                        matched = False
                        if vi.startswith('re:'):
                            regex = re.compile(vi[3:], flags=re.IGNORECASE)
                            if regex.match(i):
                                matched = True
                        elif vi == i:
                            matched = True

                        if matched:
                            api_ids.append(k)

                            # Include API regions that comprise these
                            # API regions, and further up the chain.
                            api_regs = self.map_disk_to_api(k)
                            api_ids.extend(api_regs)

                    # Don't break, the disk region could be used in several mappings, i.e. it
                    # could map to multiple API regions.
                    continue

            # Disk regions are also API accessible.
            api_ids.append(i)

        return list(sorted(set(api_ids)))

    # Given any ID or list, returns a list of API IDs.
    # Doesn't do any mapping - mapping is only required if we're loading
    # from disk. Lists are all defined using API IDs.
    # Hold up though - we need to handle regexps here.
    # Specifically, a list may be defined in terms of regexps (which can't
    # be resolved at load time, they can only be resolved given an rtstruct/
    # regions series). This method needs to resolve these regexps, and so
    # will need a list of ids that are potential matches.
    @property
    @ensure_loaded('__mappings', '__load_mappings')
    def mappings(self) -> Dict[str, Any]:
        return self.__mappings

    def __repr__(self) -> str:
        return str(self)

    # Takes disk regions and maps them to all possible API regions that they are
    # a part of, including themselves.
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(landmarks={self.landmarks}, lists={self.lists}, mappings={self.mappings})"
