import os
import pandas as pd
from typing import List, Literal

from .. import config
from ..dataset import Dataset
from ..typing import DatasetID, GroupID, LandmarkID, RegionID, SampleID, Spacing3D
from ..utils.args import arg_to_list
from .sample import TrainingSample
from .split import HoldoutSplit

class TrainingDataset(Dataset):
    def __init__(
        self,
        id: DatasetID,
        **kwargs,
        ) -> None:
        self.__path = os.path.join(config.directories.datasets, 'training', id)
        if not os.path.exists(self.__path):
            raise ValueError(f"No training dataset '{id}' found at path: {self.__path}")
        super().__init__(id)
    
    @property
    def label_types(self) -> List[str]:
        return self.params['label-types'] if 'label-types' in self.params else []

    @property
    def landmarks(self) -> List[LandmarkID]:
        return self.params['landmarks'] if 'landmarks' in self.params else []

    def list_groups(self) -> List[HoldoutSplit]:
        return list(self.index['group'].unique())
    
    def list_samples(
        self,
        group_id: GroupID | List[GroupID] | Literal['all'] = 'all',
        ) -> List[SampleID]:
        group_ids = arg_to_list(group_id, str, literals={ 'all': self.list_groups })
        samples = []
        for s in group_ids:
            group = self.group(s)
            samples += group.list_samples()
        samples = list(sorted(samples))
        return samples
    
    @property
    def n_input_channels(self) -> int:
        def_group = self.group(self.list_groups()[0])
        def_sample = def_group.sample(def_group.list_samples()[0])
        input = def_sample.input
        n_channels = input.shape[0]
        return n_channels

    @property
    def params(self) -> pd.DataFrame:
        if self.__params is None:
            filepath = os.path.join(self.path, 'params.csv')
            self.__params = pd.read_csv(filepath)
            self.__params = dict((p, v) for p, v in zip(self.__params['param'], self.__params['value']))
            eval_params = ['spacing', 'regions', 'landmarks', 'label-types']
            for e in eval_params:
                if e in self.__params:
                    self.__params[e] = eval(self.__params[e])
        return self.__params
    
    @property
    def regions(self) -> List[RegionID]:
        return self.params['regions'] if 'regions' in self.params else []

    def sample(
        self,
        id: SampleID,
        ) -> TrainingSample:
        group_ids = self.list_groups()
        for s in group_ids:
            split = self.split(s)
            samples = split.list_samples()
            if id in samples:
                return split.sample(id)
        raise ValueError(f"Unknown sample ID '{id}'.")
    
    @property
    def spacing(self) -> Spacing3D:
        return self.params['spacing']

    def split(
        self,
        name: str,
        ) -> HoldoutSplit:
        return HoldoutSplit(self, name)
