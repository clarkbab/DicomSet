from __future__ import annotations

import os
import pandas as pd
from typing import List, TYPE_CHECKING

from ..typing import RegionID, SampleID, SplitID
from ..utils.regions import region_to_list
from .sample import TrainingSample
if TYPE_CHECKING:
    from .dataset import TrainingDataset

class HoldoutSplit:
    def __init__(
        self,
        dataset: TrainingDataset,
        id: SplitID,
        ) -> None:
        self.__dataset = dataset
        self._id = id
        self.__global_id = f"{self.__dataset}:{self._id}"
        self.__path = os.path.join(self.__dataset.path, 'data', str(self._id))
        if not os.path.exists(self.__path):
            raise ValueError(f"Training split '{self.__global_id}' does not exist.")
        self.__index = None

    @property
    def dataset(self) -> TrainingDataset:
        return self.__dataset

    @property
    def index(self) -> pd.DataFrame:
        if self.__index is None:
            ds_index = self.dataset.index
            self.__index = ds_index[ds_index['split'] == self._id].copy()
        return self.__index

    def list_samples(
        self,
        region_ids: RegionID | List[RegionID] | None = None,
        ) -> List[SampleID]:
        filter_regions = region_to_list(region_ids, literals={ 'all': self.dataset.regions })
        sample_ids = self.index['sample-id'].to_list()
        if filter_regions is None:
            return sample_ids

        # Return samples that have any of the passed regions.
        sample_ids = [s for s in sample_ids if self.sample(s).has_region(filter_regions, all=False)]
        return sample_ids

    @property
    def path(self) -> str:
        return self.__path

    def sample(
        self,
        sample_id: SampleID,
        ) -> TrainingSample:
        return TrainingSample(self, sample_id)

    def __str__(self) -> str:
        return self.__global_id
