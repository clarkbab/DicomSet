from __future__ import annotations

import os
import pandas as pd
from typing import Any, Callable, Dict, List

from .typing import DatasetID, DirPath, GroupID
from .utils.io import load_yaml

CT_FROM_REGEXP = r'^__CT_FROM_(.*)__$'

class Dataset:
    def __init__(
        self,
        id: DatasetID,
        ct_from: Dataset | None = None,
        ) -> None:
        self._id = str(id)
        self._ct_from = ct_from
        filepath = os.path.join(self._path, 'config.yaml')
        self._config = load_yaml(filepath) if os.path.exists(filepath) else {}

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_index'):
                self._load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def groups(self) -> pd.DataFrame:
        return self._groups

    @property
    def id(self) -> DatasetID:
        return self._id

    @ensure_loaded
    def list_groups(self) -> List[GroupID]:
        if self._groups is None:
            raise ValueError(f"File 'groups.csv' not found for dicom dataset '{self._id}'.")
        group_ids = list(sorted(self._groups['group-id'].unique()))
        return group_ids

    @property
    def path(self) -> DirPath:
        return self._path

    def print_notes(self) -> None:
        filepath = os.path.join(self._path, 'notes.txt')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                print(f.read())

    def __repr__(self) -> str:
        return str(self)

    def __str__(
        self,
        class_name: str,
        ) -> str:
        params = dict(
            id=self._id,
        )
        if self._ct_from is not None:
            params['ct_from'] = self._ct_from.id
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
