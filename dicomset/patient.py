from __future__ import annotations

from typing import Any, Dict

from .dataset import Dataset
from .region_map import RegionMap
from .typing import PatientID
from .utils.python import wrap_quotes

class Patient:
    def __init__(
        self,
        dataset: Dataset,
        id: PatientID,
        config: Dict[str, Any] | None = None,
        ct_from: Patient | None = None,
        region_map: RegionMap | None = None,
        ) -> None:
        self._dataset = dataset
        self._config = config
        self._id = str(id)
        self._ct_from = ct_from
        self._region_map = region_map

    @property
    def ct_from(self) -> Patient | None:
        return self._ct_from

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def id(self) -> PatientID:
        return self._id

    @property
    def region_map(self) -> RegionMap | None:
        return self._region_map

    def __repr__(self) -> str:
        return str(self)

    def __str__(
        self,
        class_name: str,
        ) -> str:
        params = dict(
            dataset_id=wrap_quotes(self._dataset.id),
            id=wrap_quotes(self._id),
        )
        if self._ct_from is not None:
            params['ct_from'] = self._ct_from.id
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
