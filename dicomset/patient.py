from __future__ import annotations

from typing import Any, Dict

from .dataset import Dataset
from .region_map import RegionMap
from .typing import PatientID
from .utils.python import get_private_attr, set_private_attr, wrap_quotes

class Patient:
    def __init__(
        self,
        dataset: Dataset,
        id: PatientID,
        config: Dict[str, Any] | None = None,
        ct_from: Patient | None = None,
        region_map: RegionMap | None = None,
        ) -> None:
        set_private_attr(self, '__dataset', dataset)
        set_private_attr(self, '__config', config)
        set_private_attr(self, '__id', str(id))
        set_private_attr(self, '__ct_from', ct_from)
        set_private_attr(self, '__region_map', region_map)

    @property
    def ct_from(self) -> Patient | None:
        return get_private_attr(self, '__ct_from')

    @property
    def dataset(self) -> Dataset:
        return get_private_attr(self, '__dataset')

    @property
    def id(self) -> PatientID:
        return get_private_attr(self, '__id')

    @property
    def region_map(self) -> RegionMap | None:
        return get_private_attr(self, '__region_map')

    def __repr__(self) -> str:
        return str(self)

    def __str__(
        self,
        class_name: str,
        ) -> str:
        params = dict(
            dataset_id=wrap_quotes(get_private_attr(self, '__dataset').id),
            id=wrap_quotes(get_private_attr(self, '__id')),
        )
        ct_from = get_private_attr(self, '__ct_from')
        if ct_from is not None:
            params['ct_from'] = ct_from.id
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
