from __future__ import annotations

from typing import Any, Dict

from .dataset import Dataset
from .patient import Patient
from .study import Study
from .typing import SeriesID
from .utils.python import get_private_attr, set_private_attr, wrap_quotes

class Series:
    def __init__(
        self,
        dataset: Dataset,
        patient: Patient,
        study: Study,
        id: SeriesID,
        config: Dict[str, Any] | None = None,
        ) -> None:
        set_private_attr(self, '__dataset', dataset)
        set_private_attr(self, '__config', config)
        set_private_attr(self, '__patient', patient)
        set_private_attr(self, '__study', study)
        set_private_attr(self, '__id', str(id))

    @property
    def dataset(self) -> Dataset:
        return get_private_attr(self, '__dataset')

    @property
    def id(self) -> SeriesID:
        return get_private_attr(self, '__id')

    @property
    def patient(self) -> Patient:
        return get_private_attr(self, '__patient')

    def __repr__(self) -> str:
        return str(self)

    def __str__(
        self,
        class_name: str,
        ) -> str:
        params = dict(
            dataset_id=wrap_quotes(get_private_attr(self, '__dataset').id),
            id=wrap_quotes(get_private_attr(self, '__id')),
            patient_id=wrap_quotes(get_private_attr(self, '__patient').id),
            study_id=wrap_quotes(get_private_attr(self, '__study').id),
        )
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"

    @property
    def study(self) -> Study:
        return get_private_attr(self, '__study')
