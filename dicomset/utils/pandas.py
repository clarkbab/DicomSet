import numpy as np
import pandas as pd
from typing import Dict, List

def append_row(
    df: pd.DataFrame,
    data: Dict[str, int | float | str],
    index: int | str | List[int | str] | None = None,
    ) -> pd.DataFrame:
    # Create new index if necessary.
    if index is not None:
        # Create row index.
        if type(index) == list or type(index) == tuple:
            # Handle multi-indexes.
            index = pd.MultiIndex(codes=[[0] for i in index], levels=[[i] for i in index], names=df.index.names)
        else:
            index = pd.Index(data=[index], name=df.index.name)
    else:
        # Assign index to new row based on existing index.
        max_index = df.index.max()
        if np.isnan(max_index):
            idx = 0
        else:
            idx = max_index + 1
        index = pd.Index(data=[idx], name=df.index.name)

    # Create new dataframe.
    new_df = pd.DataFrame([data], columns=df.columns, index=index)

    # Just return new dataframe if existing is empty. 
    if len(df) == 0:
        df = new_df
    else:
        df = pd.concat((df, new_df), axis=0)
    
    return df

def concat_dataframes(
    *dfs: List[pd.DataFrame],
    ) -> pd.DataFrame:
    # Filter empty dataframes.
    filt_dfs = [df for df in dfs if len(df) > 0]

    # Perform concatenation.
    if len(filt_dfs) >= 2:
        df = pd.concat(filt_dfs, axis=0)
    elif len(filt_dfs) == 1:
        df = filt_dfs[0]
    else:
        df = dfs[0]

    return df
