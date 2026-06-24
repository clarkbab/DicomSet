from __future__ import annotations

import ast
from dicomset import config
from functools import wraps
import inspect
import numpy as np
import os
import textwrap
from typing import Any, Callable, Dict, List, Literal, Tuple, TYPE_CHECKING

from ..typing import DiskLandmarkID, DiskRegionID, FilePath, ID, LandmarkID, LandmarkList, RegionID, RegionList
from .python import isinstance_generic, version
if TYPE_CHECKING:
    from ..struct_map import StructMap

# Does 'arg' have a value?
def arg_default(
    arg: Any | List[Any] | None,
    return_arg: Any | None,  # Return this value if at least one arg is not None.
    default: Any,    # Return this value is all args are None.
    ) -> Any:
    if not isinstance(arg, list) and not isinstance(arg, tuple):
        args = [arg]
    else:
        args = arg
    all_none = True
    for a in args:
        if a is not None:
            all_none = False
            break
    if all_none:
        return default
    else:
        return return_arg if return_arg is not None else arg

class CallVisitor(ast.NodeVisitor):
    def __init__(
        self,
        inner_fn: Callable):
        self.__inner_fn = inner_fn
        self.__args = []
        self.__kwargs = []
    
    @property
    def args(self) -> List[str]:
        return self.__args
        
    @property
    def kwargs(self) -> List[str]:
        return self.__kwargs

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.__inner_fn.__name__:
            for a in node.args:
                if isinstance(a, ast.Starred):
                    self.__args.append('args')
                else:
                    self.__args.append(ast.unparse(a))
            for k in node.keywords:
                if k.arg is None:
                    self.__kwargs.append('kwargs')
                else:
                    self.__kwargs.append(k.arg)

def alias_kwargs(
    *aliases: Tuple[str | Tuple[str, ...], str],
    ) -> Callable:
    alias_map = {}
    for shortcuts, full_name in aliases:
        if isinstance(shortcuts, str):
            shortcuts = (shortcuts,)
        for shortcut in shortcuts:
            alias_map[shortcut] = full_name

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for shortcut, full_name in alias_map.items():
                if shortcut in kwargs:
                    kwargs[full_name] = kwargs.pop(shortcut)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def arg_to_list(
    arg: Any | None,
    types: Any | List[Any],     # Check if 'arg' matches any of these types.
    broadcast: int = 1,         # Expand a match to multiple elements, e.g. None -> [None, None, None].
    literals: Dict[Any, List[Any]] | None = None,   # Check if 'arg' matches any of these literal values.
    out_type: Any | None = None,    # Convert a match to a different output type.
    return_matched: bool = False,   # If arg matched 'types' or 'literals' then return True.
    ) -> List[Any] | Tuple[List[Any], bool]:
    # Convert types to list.
    if not isinstance(types, list) and not isinstance(types, tuple):
        types = [types]
    
    # Check literal matches.
    if literals is not None:
        for k, v in literals.items():
            if isinstance(arg, type(k)) and arg == k:
                arg = v

                # If arg is a function, run it now. This means the function
                # is not evaluated every time 'arg_to_list' is called, only when
                # the arg matches the appropriate literal (e.g. 'all').
                if isinstance(arg, Callable):
                    arg = arg()

                if return_matched:
                    return arg, True
                return arg

    # Check types.
    matched = False
    for t in types:
        if isinstance_generic(arg, t):
            matched = True
            arg = [arg] * broadcast
            break
        
    # Convert to output type.
    if matched and out_type is not None:
        arg = [out_type(a) for a in arg]

    if return_matched:
        return arg, matched
    return arg

def assert_2d(
    data: np.ndarray,
    message: str = "Data must be 2D.",
    ) -> None:
    assert data.ndim == 2, message

# Expands an arg to the required length based on 'dim'.
def assert_3d(
    data: np.ndarray,
    message: str = "Data must be 3D.",
    ) -> None:
    assert data.ndim == 3, message

def bubble_args(*inner_fns: Callable) -> Callable:
    if not version(gte='3.9'):
        # Ast unparse is not available in Python < 3.9.
        return lambda f: f

    def change_outer_fn_sig(outer_fn: Callable) -> Any:
        # Load params.
        outer_sig = inspect.signature(outer_fn)
        outer_params = dict(outer_sig.parameters)
        outer_params_args = dict((k, v) for k, v in outer_params.items() if v.default is inspect.Parameter.empty)
        outer_params_kwargs = dict((k, v) for k, v in outer_params.items() if v.default is not inspect.Parameter.empty)

        # Bubble some args from the inner function up to the outer function signature.
        bubbled_args = {}
        bubbled_kwargs = {}
        for f in inner_fns:
            inner_params = dict(inspect.signature(f).parameters)
            inner_args, inner_kwargs = get_inner_args(outer_fn, f)
            inner_params_args = dict((k, v) for k, v in inner_params.items() if v.default is inspect.Parameter.empty)
            inner_params_kwargs = dict((k, v) for k, v in inner_params.items() if v.default is not inspect.Parameter.empty)
            if 'args' in outer_params and 'args' in inner_args:
                for k, v in inner_params_args.items():
                    if k not in inner_args:  # I.e. not already passed by inner call.
                        bubbled_args[k] = v
            if 'kwargs' in outer_params and 'kwargs' in inner_kwargs:
                for k, v in inner_params_kwargs.items():
                    if k not in inner_kwargs:  # I.e. not already passed by inner call.
                        bubbled_kwargs[k] = v
                    
        # Create final signature.
        outer_params_args = dict((k, v) for k, v in outer_params_args.items() if k not in ['args', 'kwargs'])
        args = {}
        args = args | outer_params_args
        args = args | bubbled_args
        kwargs = {}
        kwargs = kwargs | outer_params_kwargs
        kwargs = kwargs | bubbled_kwargs
        # Sort alphabetically, but keyword-only params must be last.
        kw_only_kwargs = dict(sorted((k, v) for k, v in kwargs.items() if v.kind is inspect.Parameter.KEYWORD_ONLY))
        other_kwargs = dict(sorted((k, v) for k, v in kwargs.items() if v.kind is not inspect.Parameter.KEYWORD_ONLY))
        kwargs = other_kwargs | kw_only_kwargs
        params = args | kwargs
        
        outer_fn.__signature__ = outer_sig.replace(parameters=params.values())
        return outer_fn

    return change_outer_fn_sig

def get_inner_args(
    outer_fn: Callable,
    inner_fn: Callable,
    ) -> Tuple[List[str], List[str]]:
    source = textwrap.dedent(inspect.getsource(outer_fn))
    tree = ast.parse(source)
    visitor = CallVisitor(inner_fn)
    visitor.visit(tree)
    return visitor.args, visitor.kwargs

# Can't move this to StructMap because we don't want literal='all' behaviour in there.
@alias_kwargs(
    (('dl', 'disk_landmark', 'disk_landmarks', 'disk_landmark_id'), 'disk_landmark_ids'),
    (('l', 'landmark', 'landmarks', 'landmark_id'), 'landmark_ids'),
)
def landmarks_to_list(
    landmark_ids: LandmarkID | LandmarkList | List[LandmarkID | LandmarkList] | Literal['all'],
    disk_landmark_ids: DiskLandmarkID | List[DiskLandmarkID] | None = None,
    struct_map: StructMap | None = None,
    **kwargs,
    ) -> List[LandmarkID]:
    landmark_ids = arg_to_list(landmark_ids, str, **kwargs)
    if struct_map is not None:
        landmark_ids = struct_map.expand_list(landmark_ids, disk_ids=disk_landmark_ids)
    return list(sorted(set(landmark_ids)))

# Can't move this to StructMap because we don't want literal='all' behaviour in there.
@alias_kwargs(
    (('dr', 'disk_region', 'disk_regions', 'disk_region_id'), 'disk_region_ids'),
    (('r', 'region', 'regions', 'region_ids'), 'region_ids'),
)
def regions_to_list(
    region_ids: RegionID | RegionList | List[RegionID | RegionList] | Literal['all'],
    disk_region_ids: DiskRegionID | List[DiskRegionID] | None = None,
    struct_map: StructMap | None = None,
    **kwargs,
    ) -> List[RegionID]:
    region_ids = arg_to_list(region_ids, str, **kwargs)
    if struct_map is not None:
        region_ids = struct_map.expand_list(region_ids, disk_ids=disk_region_ids)
    return list(sorted(set(region_ids)))

def resolve_filepath(filepath: FilePath) -> FilePath:
    file_options = ['f', 'file', 'files']
    for f in file_options:
        if filepath.startswith(f"{f}:"):
            filepath = os.path.join(config.dirs.files, filepath[len(f) + 1:])
            break
    return filepath

def resolve_id(
    id: ID | int,
    all_ids: List[ID] | Callable[[], List[ID]],
    ) -> str:
    if isinstance(id, int) or id.startswith('i:'):
        if isinstance(id, int):
            idx = id
        else:
            idx = int(id.split(':')[1])
        ids = all_ids()
        if idx > len(ids) - 1:
            raise ValueError(f"Index ({idx}) was larger than list (len={len(ids)}).")
        id = ids[idx]

    return id
