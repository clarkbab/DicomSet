import re
from typing import List, Literal

from ..region_map import RegionMap
from ..typing import DiskRegionID, RegionID, RegionList
from .args import arg_to_list

# Converts to a list and resolves region lists ("rl:<name>").
# TODO: Move this to RegionMap.
def region_to_list(
    region_id: RegionID | RegionList | List[RegionID | RegionList] | Literal['all'],
    disk_regions: List[DiskRegionID] | None = None,
    region_map: RegionMap | None = None,
    **kwargs,
    ) -> List[RegionID]:
    region_ids = arg_to_list(region_id, str, **kwargs)

    # Expand regions.
    regions = []
    for r in region_ids:
        if region_map is not None:
            # Look through region lists.
            for k, v in region_map.regions.items(): 
                if r == k:
                    regions += v
                    break

            # Look through landmark lists.
            for k, v in region_map.landmarks.items():
                if r == k:
                    for vi in v:
                        if vi.startswith('re:'):
                            assert disk_regions is not None, "Disk regions must be provided for regex mapping."
                            regex = re.compile(vi[3:], flags=re.IGNORECASE)
                            for d in disk_regions:
                                if regex.match(d):
                                    regions.append(d)
                        else:
                            regions.append(vi)
        else:
            regions.append(r)

    return list(sorted(set(regions)))
