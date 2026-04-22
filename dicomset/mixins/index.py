import pandas as pd
from typing import Any, Dict

from ..utils.python import call_private_method, get_private_attr, set_private_attr

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

    def index(
        self,
        **filters,
        ) -> pd.DataFrame:
        if get_private_attr(self, '__index_policy') is None:
            call_private_method(self, '__load_index')
        index = get_private_attr(self, '__index').copy()
        for k, v in filters.items():
            index = index[index[k] == v]
        return index

    @property
    def index_policy(self) -> Dict[str, Any]:
        pass

class IndexWithErrorsMixin(IndexMixin):
    def index_errors(
        self,
        **filters,
        ) -> pd.DataFrame:
        # 'index_errors_fn' is not bound to the instance, so it's 'self' won't have access to '_index'.
        def index_errors_fn(_, **filters) -> pd.DataFrame:
            index_errors = get_private_attr(self, '__index_errors').copy()
            for k, v in filters.items():
                index_errors = index_errors[index_errors[k] == v]
            return index_errors
        index_errors_fn = self.__class__.ensure_loaded(index_errors_fn) if hasattr(self.__class__, 'ensure_loaded') else index_errors_fn
        return index_errors_fn(self, **filters)
        