from __future__ import annotations

from typing import Any, Dict

from .dataset import Dataset
from .patient import Patient
from .regions_map import RegionsMap
from .typing import StudyID

class Study:
    def __init__(
        self,
        dataset: Dataset,
        pat: Patient,
        id: StudyID,
        config: Dict[str, Any] | None = None,
        ct_from: Study | None = None,
        regions_map: RegionsMap | None = None,
        ) -> None:
        self._dataset = dataset
        self._config = config
        self._pat = pat
        self._id = str(id)
        self._ct_from = ct_from
        self._regions_map = regions_map

    @property
    def ct_from(self) -> Study | None:
        return self._ct_from

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def id(self) -> StudyID:
        return self._id

    @property
    def pat(self) -> Patient:
        return self._pat

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
            pat=self._pat.id,
        )
        if self._ct_from is not None:
            params['ct_from'] = self._ct_from.id
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
