from typing import List, Literal

from ..region_map import RegionMap
from ..typing import RegionID, RegionList
from .args import arg_to_list

# Converts to a list and resolves region lists ("rl:<name>").
def region_to_list(
    region_id: RegionID | RegionList | List[RegionID | RegionList] | Literal['all'],
    region_map: RegionMap | None = None,
    **kwargs,
    ) -> List[RegionID]:
    region_ids = arg_to_list(region_id, str, **kwargs)

    # Expand regions.
    regions = []
    for r in region_ids:
        if r.startswith('l:'): 
            # Load region list.
            list_name = r.split(':')[-1]
            if region_map is None:
                raise ValueError(f"Used region list '{list_name}' without a region map.")
            regions += region_map.region_list(list_name)
        else:
            regions.append(r)

    return list(sorted(set(regions)))
