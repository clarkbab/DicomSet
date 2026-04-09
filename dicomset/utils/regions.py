import os
from typing import List

from .. import config
from ..typing import RegionID
from .args import arg_to_list

# Behaves like 'arg_to_list', but also handles special 'rl:<region list>' format.
def regions_to_list(
    region: RegionID | List[RegionID],
    **kwargs,
    ) -> List[RegionID]:
    if region is None:
        return None

    regions = arg_to_list(region, str, **kwargs)

    # Expand 'rl:<region list>' to list of regions.
    expanded_regions = []
    for r in regions:
        if r.startswith('rl:'): 
            # Expand str to list of regions.
            rl_name = r.split(':')[-1]
            if hasattr(RegionList, rl_name):
                r = list(getattr(RegionList, rl_name))
            else:
                filepath = os.path.join(config.directories.config, 'region-lists', f'{rl_name}.csv')
                if not os.path.exists(filepath):
                    raise ValueError(f"Region list '{rl_name}' not found. Filepath: {filepath}")
                df = pd.read_csv(filepath, header=None)
                r = list(sorted(df[0]))
            expanded_regions += r
        else:
            expanded_regions.append(r)

    return expanded_regions
