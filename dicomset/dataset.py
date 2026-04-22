from __future__ import annotations

import os
import pandas as pd
from typing import Any, Dict, List

from .typing import DatasetID, DirPath, GroupID
from .utils.io import load_yaml
from .utils.python import ensure_loaded, get_private_attr, set_private_attr, wrap_quotes

CT_FROM_REGEXP = r'^__CT_FROM_(.*)__$'

class Dataset:
    def __init__(
        self,
        id: DatasetID,
        ct_from: Dataset | None = None,
        ) -> None:
        set_private_attr(self, '__id', str(id))
        set_private_attr(self, '__ct_from', ct_from)
        filepath = os.path.join(get_private_attr(self, '__path'), 'config.yaml')
        set_private_attr(self, '__config', load_yaml(filepath) if os.path.exists(filepath) else {})

    @property
    def config(self) -> Dict[str, Any]:
        return get_private_attr(self, '__config')

    @property
    def groups(self) -> pd.DataFrame:
        return get_private_attr(self, '__groups', None)

    @property
    def id(self) -> DatasetID:
        return get_private_attr(self, '__id')

    @ensure_loaded('__groups', '__load_groups')
    def list_groups(self) -> List[GroupID]:
        groups = get_private_attr(self, '__groups', None)
        if groups is None:
            raise ValueError(f"File 'groups.csv' not found for dicom dataset '{get_private_attr(self, '__id')}'.")
        group_ids = list(sorted(groups['group-id'].unique()))
        return group_ids

    @property
    def path(self) -> DirPath:
        return get_private_attr(self, '__path')

    def print_notes(self) -> None:
        filepath = os.path.join(get_private_attr(self, '__path'), 'notes.txt')
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
            id=wrap_quotes(get_private_attr(self, '__id')),
        )
        ct_from = get_private_attr(self, '__ct_from')
        if ct_from is not None:
            params['ct_from'] = ct_from.id
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
