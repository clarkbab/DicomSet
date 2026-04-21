import os

from .typing import DirPath

DATA_DIR = None
DATA_ENV_VAR = 'DS_DATA'

def config_data(dirpath: DirPath) -> None:
    global DATA_DIR
    DATA_DIR = dirpath

class Directories:
    @property
    def data(self):
        if DATA_DIR is not None:
            return DATA_DIR

        if DATA_ENV_VAR in os.environ:
            config_data(os.environ[DATA_ENV_VAR])
            return DATA_DIR
        else:
            raise ValueError(f"Environment variable '{DATA_ENV_VAR}=\"<directory>\"' must be set to work with data.")

    @property
    def datasets(self):
        filepath = os.path.join(self.data, 'datasets')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath

    @property
    def files(self):
        filepath = os.path.join(self.data, 'files')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        return filepath
    
    # @property
    # def tmp(self):
    #     filepath = os.path.join(self.root, 'tmp')
    #     if not os.path.exists(filepath):
    #         os.makedirs(filepath)
    #     return filepath

directories = Directories()
