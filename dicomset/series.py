from __future__ import annotations

from typing import Any, Dict

from .dataset import Dataset
from .patient import Patient
from .study import Study
from .typing import SeriesID
from .utils.python import wrap_quotes

class Series:
    def __init__(
        self,
        dataset: Dataset,
        pat: Patient,
        study: Study,
        id: SeriesID,
        config: Dict[str, Any] | None = None,
        ) -> None:
        self._dataset = dataset
        self._config = config
        self._pat = pat
        self._study = study
        self._id = str(id)

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def id(self) -> SeriesID:
        return self._id

    @property
    def pat(self) -> Patient:
        return self._pat

    def __repr__(self) -> str:
        return str(self)

    def __str__(
        self,
        class_name: str,
        ) -> str:
        params = dict(
            dataset_id=wrap_quotes(self._dataset.id),
            id=wrap_quotes(self._id),
            patient_id=wrap_quotes(self._pat.id),
            study_id=wrap_quotes(self._study.id),
        )
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"

    @property
    def study(self) -> Study:
        return self._study
