from __future__ import annotations

from typing import Any, Dict

from .dataset import Dataset
from .regions_map import RegionsMap
from .typing import PatientID

class Patient:
    def __init__(
        self,
        dataset: Dataset,
        id: PatientID,
        config: Dict[str, Any] | None = None,
        ct_from: Patient | None = None,
        regions_map: RegionsMap | None = None,
        ) -> None:
        self._dataset = dataset
        self._config = config
        self._id = str(id)
        self._ct_from = ct_from
        self._regions_map = regions_map

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
    def regions_map(self) -> RegionsMap | None:
        return self._regions_map

    def __repr__(self) -> str:
        return str(self)

    def __str__(
        self,
        class_name: str,
        ) -> str:
        params = dict(
            dataset=self._dataset.id,
            id=self._id,
        )
        if self._ct_from is not None:
            params['ct_from'] = self._ct_from.id
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
