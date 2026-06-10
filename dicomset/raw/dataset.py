import os

from .. import config
from ..dataset import Dataset
from ..typing import DatasetID

class RawDataset(Dataset):
    def __init__(
        self,
        id: DatasetID,
        ) -> None:
        self.__dirpath = os.path.join(config.dirs.datasets, 'raw', str(id))
        if not os.path.exists(self.__dirpath):
            raise ValueError(f"No raw dataset '{id}' found at path: {self.__dirpath}")
        super().__init__(id)
    