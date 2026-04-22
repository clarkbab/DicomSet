import pandas as pd
from typing import Any, Dict

from ..utils.python import ensure_loaded, get_private_attr, set_private_attr

class IndexMixin:
    def __init__(
        self,
        *args,
        **kwargs,
        ) -> None:
        if 'index' in kwargs:
            set_private_attr(self, '__index', kwargs.pop('index'))
        if 'index_policy' in kwargs:
            set_private_attr(self, '__index_policy', kwargs.pop('index_policy'))
        super().__init__(*args, **kwargs)

    @ensure_loaded('__index', '__load_index')
    def index(
        self,
        **filters,
        ) -> pd.DataFrame:
        index = get_private_attr(self, '__index').copy()
        for k, v in filters.items():
            index = index[index[k] == v]
        return index

    @property
    @ensure_loaded('__index_policy', '__load_index')
    def index_policy(self) -> Dict[str, Any]:
        return get_private_attr(self, '__index_policy')

class IndexWithErrorsMixin(IndexMixin):
    @ensure_loaded('__index_errors', '__load_index')
    def index_errors(
        self,
        **filters,
        ) -> pd.DataFrame:
        index_errors = get_private_attr(self, '__index_errors').copy()
        for k, v in filters.items():
            index_errors = index_errors[index_errors[k] == v]
        return index_errors
        