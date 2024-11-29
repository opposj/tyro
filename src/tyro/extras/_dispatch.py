import collections
import dataclasses
import importlib
import itertools
import operator
import typing
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import copy
from pathlib import Path
from types import GenericAlias
from typing import Annotated, Any, Callable, Self, TypedDict, TypeVar, Union, cast

from camel_converter import to_pascal

from ..conf import ConsolidateSubcommandArgs, FlagConversionOff, StringToType
from ..conf import arg as tyro_arg
from ..conf import subcommand as tyro_subcommand


class _ConfigFactory:
    def __new__(cls, config_name: str):
        class _(type):  # noqa: N801
            disp_dict: dict[tuple[tuple[str, str], ...], Any] = {}

            @classmethod
            def register(cls, cls_, **kwargs):
                insert_key = tuple(iter(kwargs.items()))
                if insert_key in cls.disp_dict:
                    _err_msg = (
                        f"Ambiguous definition found for "
                        f"<{cls_.__module__ + "." + cls_.__qualname__}> and "
                        f"<{cls_.disp_dict[insert_key].__module__ + "." + cls_.disp_dict[insert_key].__qualname__}> "
                        f"with keys: {dict(insert_key)}."
                    )
                    raise ValueError(_err_msg)
                cls.disp_dict[insert_key] = cls_

            def __post_new__(
                cls: type[Self],
                name: str,
                bases: tuple[Any],
                namespace: dict[str, Any],
                **kwargs,
            ):
                new_cls = super().__new__(cls, name, bases, namespace)
                cls.register(new_cls, **kwargs)
                return new_cls

        _.__name__ = to_pascal(f"{config_name}_config_meta")

        class __(metaclass=_): ...

        __.__name__ = to_pascal(f"{config_name}_config")
        _.__new__ = _.__post_new__

        return __


AtomConfig = _ConfigFactory("atom")
DefaultConfig = _ConfigFactory("default")


class _DispInfo(TypedDict):
    disp_key: tuple[str, str]
    disp_cls: dict[tuple[tuple[str, str], ...], tuple[int, type[AtomConfig]]] | None
    disp_dflt: dict[tuple[tuple[str, str], ...], type[DefaultConfig]] | None
    accu_disp_keys: tuple[tuple[str, str], ...]
    next_disps: "list[_DispInfo] | None"


def _import_module(
    module_name: str,
    *,
    recursive: bool = True,
    include_namespace_packages: bool = False,
) -> None:
    module_imported = importlib.import_module(module_name)

    # Import a single module.
    if module_imported.__package__ != module_name:
        return

    if module_imported.__file__ is None and not include_namespace_packages:
        return

    module_path = Path(module_imported.__path__[0])

    # Find all .py modules under the current directory recursively.
    modules = module_path.glob("*.py")
    [
        importlib.import_module(f"{module_name}.{f_name[:-3]}")
        for f in modules
        if not (f_name := f.name).startswith("_") and f_name != "__init__.py"
    ]

    if recursive:
        # Find all python directory modules.
        modules = (
            module_path.glob("*/__init__.py")
            if not include_namespace_packages
            else module_path.glob("*/*.py")
        )
        [
            _import_module(
                f"{module_name}.{d_name}",
                recursive=recursive,
                include_namespace_packages=include_namespace_packages,
            )
            for d in modules
            if not (d_name := d.parent.name).startswith("_")
        ]


def tyro_dispatch[T](
    cfg_cls: type[T], search_modules: Sequence[str]
) -> tuple[T, tuple[tuple[str, str], ...]]:
    from .. import cli as tyro_cli

    # Import all configs from the specified modules.
    [_import_module(module) for module in search_modules]

    def _get_cli_keys_and_attr_names(
        _cls_key: tuple[tuple[str, str], ...], _cls: type[AtomConfig] | type[T]
    ) -> None:
        _cli_keys = tuple(tp.__name__ for tp in getattr(_cls, "__parameters__", ()))
        # TODO: Other things except dataclass might also work? (cannot be tuple)
        # TODO:  What if a dispatching/non-dispatching dataclass inherited from a dispatching dataclass?
        if _cli_keys and not dataclasses.is_dataclass(_cls):
            _err_msg = f"Expected a dataclass config to be dispatched, but got a {type(_cls)} instead."
            raise TypeError(_err_msg)

        # TakeNote: Use `get_type_hints` to fetch type info from both the class and its parent classes.
        _attr_names = {
            var.__name__: var_name
            for var_name, var in typing.get_type_hints(_cls).items()
            if isinstance(var, TypeVar) and var.__name__ in _cli_keys
        }

        def _remove_cli_keys(
            _cls: type[AtomConfig] | type[T],
            _cli_keys: tuple[str, ...],
            _excessive: Iterable[str],
            _attr_names: dict[str, str],
        ) -> tuple[str, ...]:
            # TODO: This function is not perfect. It might not work for all cases. It should be improved.
            warnings.warn(
                f"Unused type parameters found in the config "
                f'<{_cls.__module__ + "." + _cls.__qualname__}>: "{", ".join(_excessive)}". '
                f"The excessive parameters will be ignored.",
                RuntimeWarning,
                stacklevel=1,
            )
            _cls.__type_params__ = _cls.__parameters__ = tuple(
                _key for _key in _cls.__parameters__ if _key.__name__ not in _excessive
            )
            _unused_vars = [
                _attr_names.pop(_key) for _key in _excessive if _key in _attr_names
            ]
            # noinspection PyUnresolvedReferences,PyProtectedMember
            [
                getattr(_cls, dataclasses._FIELDS).pop(_var_name, None)
                for _var_name in _unused_vars
            ]
            return tuple(_key for _key in _cli_keys if _key not in _excessive)

        if _excessive_keys := (set(_cli_keys) - set(_attr_names.keys())):
            _cli_keys = _remove_cli_keys(_cls, _cli_keys, _excessive_keys, _attr_names)

        if _cli_keys:
            _all_disp_keys = set(
                map(lambda _x: _x[0][0], filter(bool, AtomConfig.disp_dict.keys()))
            )
            if _excessive_keys := (set(_cli_keys) - _all_disp_keys):
                _cli_keys = _remove_cli_keys(
                    _cls, _cli_keys, _excessive_keys, _attr_names
                )
            if len(_cli_keys) != len(_attr_names):
                raise AssertionError
            cli_keys[id(_cls)] = _cli_keys
            attr_names[id(_cls)] = _attr_names
            if _cls_key:
                spawn_keys[_cls_key[0]] = _cli_keys
            return

        if _cls is cfg_cls:
            # The case of a pure no-dispatching config.
            cli_keys[id(_cls)] = _cli_keys
            if _attr_names:
                raise AssertionError
            attr_names[id(_cls)] = _attr_names

    cli_keys: dict[int, tuple[str, ...]] = {}
    attr_names: dict[int, dict[str, str]] = {}
    spawn_keys: dict[tuple[str, str], tuple[str, ...]] = {}
    # noinspection PyTypeChecker
    [
        _get_cli_keys_and_attr_names(cls_key, cls)
        for cls_key, cls in itertools.chain(
            (((), cfg_cls),), iter(AtomConfig.disp_dict.items())
        )
    ]
    if len(cli_keys) != len(attr_names):
        raise AssertionError

    def _remove_unavail_cfgs(
        _atom_disp: dict[tuple[tuple[str, str], ...], AtomConfig],
        _dflt_disp: dict[tuple[tuple[str, str], ...], DefaultConfig],
        _cli_keys: dict[int, tuple[str, ...]],
    ):
        _cmb_cli_keys: tuple[str, ...] = tuple(
            itertools.chain.from_iterable(_cli_keys.values())
        )
        _atom_check_keys = set(_cmb_cli_keys)
        _dflt_check_keys = _atom_check_keys | set(
            itertools.chain.from_iterable(_atom_disp)
        )

        def _check_cfg(
            _cfg_dict: dict[tuple[tuple[str, str], ...], AtomConfig | DefaultConfig],
            _check_set: set[str],
        ):
            _keys_to_remove = []
            for _disp_key, _disp_cls in _cfg_dict.items():
                if not _disp_key:
                    if _check_set is _atom_check_keys:
                        warnings.warn(
                            f"Empty dispatching flag found in an `AtomConfig`: "
                            f'<{_disp_cls.__module__ + "." + _disp_cls.__qualname__}>. '
                            f"This config will be ignored.",
                            RuntimeWarning,
                            stacklevel=1,
                        )
                        _keys_to_remove.append(_disp_key)
                    continue

                _cli_key = _disp_key[0][0]
                if _cli_key not in _check_set:
                    warnings.warn(
                        f"Unknown dispatching flag found in the definition of "
                        f'<{_disp_cls.__module__ + "." + _disp_cls.__qualname__}>: "{_cli_key}". '
                        f"This config will be ignored.",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                    _keys_to_remove.append(_disp_key)
            [_cfg_dict.pop(_key) for _key in _keys_to_remove]

        _check_cfg(_atom_disp, _atom_check_keys)
        _check_cfg(_dflt_disp, _dflt_check_keys)

        # TODO: A more concise manner via `spawn_keys`?
        _excessive_keys = tuple(
            filter(
                lambda _key: len(
                    set(
                        _disp_key[0][0]
                        for _disp_key, _disp_cls in _atom_disp.items()
                        if _key in _cli_keys.get(id(_disp_cls), ())
                    )
                )
                > 1,
                (
                    _key
                    for _key, _count in collections.Counter(_cmb_cli_keys).items()
                    if _count > 1
                ),
            )
        )
        if _excessive_keys:
            _err_msg = f"Duplicate dispatching keys found:\n" + "\n".join(
                f"\t{_key}: {", ".join(
                    f"<{_cls.__module__ + '.' + _cls.__qualname__}>" 
                    for _cls in itertools.chain((cfg_cls,), AtomConfig.disp_dict.values()) 
                    if _key in _cli_keys.get(id(_cls), ()))}."
                for _key in _excessive_keys
            )
            raise ValueError(_err_msg)

    _remove_unavail_cfgs(AtomConfig.disp_dict, DefaultConfig.disp_dict, cli_keys)
    cls_map = defaultdict(dict)
    [
        cls_map[_disp_key[0][0]].update({_disp_key: _disp_cls})
        for _disp_key, _disp_cls in AtomConfig.disp_dict.items()
    ]

    def _build_disp_info(
        _cli_keys: dict[int, tuple[str, ...]],
        _spawn_keys: dict[tuple[str, str], tuple[str, ...]],
        _attr_names: dict[int, dict[str, str]],
        _cls_map: dict[str, dict[tuple[tuple[str, str], ...], type[AtomConfig]]],
        _atom_disp: dict[tuple[tuple[str, str], ...], type[AtomConfig]],
        _dflt_disp: dict[tuple[tuple[str, str], ...], type[DefaultConfig]],
    ):
        _unique_cls_keys = set(itertools.chain.from_iterable(_atom_disp))

        def _cls_key_iter(_key: str):
            return (_cls_key for _cls_key in _unique_cls_keys if _cls_key[0] == _key)

        def _check_cfg_instance(
            _cls: type[AtomConfig] | type[T],
            _var: T,
            _cls_stack: tuple[type[AtomConfig], ...],
        ):
            if None in _cls_stack or type(None) in _cls_stack:
                raise AssertionError

            return all(
                (
                    isinstance(
                        (_var_value := getattr(_var, _var_name, None)), _cls_stack
                    )
                    and _check_cfg_instance(type(_var_value), _var_value, _cls_stack)
                )
                or _var_value is None
                for _var_name in _attr_names.get(id(_cls), {}).values()
            )

        def _check_cls_stack(
            _dflt: type[DefaultConfig], _cls_stack: tuple[type[AtomConfig], ...]
        ):
            _all_dflt_var = list(
                filter(lambda _var: isinstance(_var[1], cfg_cls), vars(_dflt).items())
            )
            _bad_dflt_var = list(
                filter(
                    lambda _var: not _check_cfg_instance(cfg_cls, _var[1], _cls_stack),
                    _all_dflt_var,
                )
            )

            if len(_bad_dflt_var) >= len(_all_dflt_var):
                return False

            [delattr(_dflt, _var_name) for _var_name, _ in _bad_dflt_var]
            return True

        def _get_cached_disp_info(
            _disp_info: _DispInfo,
            _cur_key: tuple[str, str],
            _cached_key: tuple[tuple[str, str], ...],
            _disp_cls: dict[tuple[tuple[str, str], ...], tuple[int, type[AtomConfig]]]
            | None,
            _disp_dflt: dict[tuple[tuple[str, str], ...], type[DefaultConfig]] | None,
        ):
            is_cached = True
            if _disp_info_cache.get(_cached_key) is None:
                _next_disp_info = _DispInfo(
                    disp_key=_cur_key,
                    disp_cls=_disp_cls,
                    disp_dflt=_disp_dflt,
                    accu_disp_keys=_cached_key,
                    next_disps=[],
                )
                _disp_info_cache[_cached_key] = _next_disp_info
                _disp_info["next_disps"].append(_next_disp_info)
                is_cached = False
            return _disp_info_cache[_cached_key], is_cached

        def _maxes[_T, _G](
            _a: Iterable[_T], _key: Callable[[_T], _G] | None = None
        ) -> tuple[_G, list[_T]]:
            if _key is None:
                _key = lambda _x: _x
            _a = iter(_a)
            try:
                _a0 = next(_a)
                _m, _max_list = _key(_a0), [_a0]
            except StopIteration:
                raise ValueError("_maxes() iterable argument is empty")
            for _s in _a:
                _k = _key(_s)
                if _k > _m:
                    _m, _max_list = _k, [_s]
                elif _k == _m:
                    _max_list.append(_s)
            return _m, _max_list

        def _iter_build(
            _cli_keys: dict[int, tuple[str, ...]],
            _dsp_case: tuple[tuple[int, tuple[str, str]], ...],
            _disp_info: _DispInfo,
            _stack_level: int,
            _cls_stack: list[
                tuple[
                    tuple[
                        tuple[
                            str,
                            str,
                        ],
                        ...,
                    ],
                    tuple[int, type[AtomConfig]],
                ]
            ],
        ) -> _DispInfo:
            if _dsp_case in _disp_case_cache:
                return _disp_info
            elif _dsp_case:
                _pure_case: tuple[tuple[str, str], ...] = tuple(
                    map(operator.itemgetter(1), _dsp_case)
                )
                _hier_level, _next_disp_key = _dsp_case[_stack_level]
                _next_disp_cls = _maxes(
                    cast(
                        Iterable[
                            tuple[
                                tuple[tuple[str, str], ...],
                                type[AtomConfig],
                            ]
                        ],
                        filter(
                            lambda _x: set(_x[0]) <= set(_pure_case),
                            _cls_map[_next_disp_key[0]].items(),
                        ),
                    ),
                    lambda _x: len(set(_x[0]) & set(_pure_case)),
                )[1]
                if len(_next_disp_cls) > 1:
                    warnings.warn(
                        f"Ambiguous dispatching combinations found:\n"
                        f"\t{"\n\t".join(
                            map(
                                lambda _kv: str(_kv[0]) + "->" + _kv[1][1].__module__ + "." + _kv[1][1].__qualname__, 
                                _next_disp_cls
                            )
                        )}\n"
                        f"The chosen dispatching is {_next_disp_cls[0][0]}.\n"
                        f"It is recommended to manually provide a preference. For example:\n"
                        f"\tAtomConfig.register({_next_disp_cls[0][1].__name__}, "
                        f"{", ".join(f"{_k}={_v}" for _k, _v in set(
                            itertools.chain.from_iterable(
                                map(operator.itemgetter(0), _next_disp_cls)
                            )
                        ))}).",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                if id(_next_disp_cls[0][1]) not in _cli_keys:
                    _dsp_case = _dsp_case[: _stack_level + 1] + tuple(
                        itertools.dropwhile(
                            lambda _x: _x[0] > _hier_level,
                            _dsp_case[_stack_level + 1 :],
                        )
                    )
                _cls_stack.append(
                    (_next_disp_cls[0][0], (_hier_level, _next_disp_cls[0][1]))
                )
            else:
                _next_disp_key = ("", "")

            _pure_case = tuple(map(operator.itemgetter(1), _dsp_case))
            if _stack_level < len(_dsp_case) - 1:
                _next_disp_info = _get_cached_disp_info(
                    _disp_info,
                    _next_disp_key,
                    _pure_case[: _stack_level + 1],
                    None,
                    None,
                )[0]
                _next_disp_info = _iter_build(
                    _cli_keys,
                    _dsp_case,
                    _next_disp_info,
                    _stack_level + 1,
                    _cls_stack,
                )
            else:
                # TODO: Add warning for filtered default configs.
                _pure_stack = tuple(_key_cls[1][1] for _key_cls in _cls_stack)
                _cur_dflt = list(
                    filter(
                        lambda _dflt: set(_dflt[0]) <= set(_pure_case)
                        and _check_cls_stack(_dflt[1], _pure_stack),
                        _dflt_disp.items(),
                    )
                )
                _next_disp_info, _is_cached = _get_cached_disp_info(
                    _disp_info,
                    _next_disp_key,
                    _pure_case,
                    dict(_cls_stack),
                    dict(_cur_dflt),
                )
                if _is_cached:
                    raise AssertionError
                _disp_case_cache.append(_dsp_case)

            return _next_disp_info

        def _get_disp_cases(
            _gb_cli_keys: dict[tuple[str, str], tuple[str, ...]],
            _lc_cli_keys: list[tuple[int, str]],
            _disp_stack: list[tuple[int, str]],
        ):
            if not _lc_cli_keys:
                yield tuple(_disp_stack)
                return

            _h, _disp = _lc_cli_keys.pop(0)
            _disp_choices = _cls_key_iter(_disp)
            for _ch in _disp_choices:
                _cur_stack = copy(_disp_stack)
                _cur_stack.append((_h, _ch))
                _cur_lc_cli_keys = copy(_lc_cli_keys)
                if _ch in _gb_cli_keys:
                    yield from _get_disp_cases(
                        _gb_cli_keys,
                        list(
                            itertools.zip_longest(
                                (), _gb_cli_keys[_ch], fillvalue=_h + 1
                            )
                        )
                        + _cur_lc_cli_keys,
                        _cur_stack,
                    )
                else:
                    yield from _get_disp_cases(
                        _gb_cli_keys, _cur_lc_cli_keys, _cur_stack
                    )

        _disp_info_cache = {}
        _disp_case_cache = []
        _disp_info = _DispInfo(
            disp_key=("", ""),
            disp_cls=None,
            disp_dflt=None,
            accu_disp_keys=(),
            next_disps=[],
        )
        for _disp_case in _get_disp_cases(
            _spawn_keys,
            list(itertools.zip_longest((), _cli_keys[id(cfg_cls)], fillvalue=0)),
            [],
        ):
            _iter_build(_cli_keys, _disp_case, _disp_info, 0, [])

        return _disp_info

    disp_info = _build_disp_info(
        cli_keys,
        spawn_keys,
        attr_names,
        cls_map,
        AtomConfig.disp_dict,
        DefaultConfig.disp_dict,
    )

    def _get_constructor(_disp_info: _DispInfo):
        def _(_: _arg_type(_disp_info)):
            if _disp_info["disp_key"][0]:
                return (
                    (_, _disp_info["disp_key"])
                    if not isinstance(_, tuple)
                    else (*_, _disp_info["disp_key"])
                )
            else:
                return _

        return _

    def _sanitize_dflts(
        _dflts: dict[tuple[tuple[str, str], ...], type[DefaultConfig]],
        _cli_order: tuple[str, ...],
    ):
        _rt = {}
        for _dflt_name, _dflt in _dflts.items():
            _dflt_name = tuple(
                (_key, _disp)
                for _n in _cli_order
                for _key, _disp in _dflt_name
                if _n == _key
            )
            _dflt_name_prefix = "_".join(_n[1] for _n in _dflt_name)
            _cur_cfgs = {
                (
                    f"{_dflt_name_prefix}_{_var_name}"
                    if _dflt_name_prefix
                    else f"{_var_name}"
                ): _var
                for _var_name, _var in vars(_dflt).items()
                if isinstance(_var, cfg_cls)
            }
            if _dup := set(_cur_cfgs) & set(_rt):
                _err_msg = f"Duplicate configuration found for {_dup}."
                raise ValueError(_err_msg)
            _rt.update(_cur_cfgs)
        if not _rt:
            _rt = {"no_default": None}
        return _rt

    def _build_cfg_type(
        _disp_cls: list[tuple[int, type[AtomConfig] | GenericAlias]],
        _cls_stack: list[type[AtomConfig] | GenericAlias],
        _latest_h: int,
    ):
        if not _disp_cls:
            return _cls_stack

        _h, _cls = _disp_cls.pop(0)
        if _h < _latest_h:
            _disp_cls.insert(0, (_h, _cls))
            return _cls_stack
        if not _disp_cls:
            _cls_stack.append(_cls)
            return _cls_stack

        if _h == _disp_cls[0][0]:
            _cls_stack.append(_cls)
            return _build_cfg_type(_disp_cls, _cls_stack, _h)
        elif _h < _disp_cls[0][0]:
            _cls_stack.append(_cls[*_build_cfg_type(_disp_cls, [], _h)])
            return _build_cfg_type(_disp_cls, _cls_stack, _h)
        else:
            _cls_stack.append(_cls)
            return _cls_stack

    def _arg_type(_disp_info: _DispInfo):
        if _disp_info["disp_cls"] is None:
            if _disp_info["disp_dflt"] is not None:
                raise AssertionError

            if len(_disp_info["next_disps"]) == 1:
                _single_disp = _disp_info["next_disps"][0]
                warnings.warn(
                    f"Only one dispatching flag found for the case:\n"
                    f"\t{_single_disp['accu_disp_keys'][:-1]}->'{_single_disp['disp_key'][0]}'\n"
                    f"Its dispatching will be skipped and reduced to a vanilla configuration "
                    f"w.r.t. '{_single_disp['disp_key'][1]}'.",
                    RuntimeWarning,
                    stacklevel=1,
                )

            return Annotated[
                Union.__getitem__(
                    tuple(
                        Annotated[
                            _get_constructor(_next_disp),
                            tyro_subcommand(
                                name=_next_disp["disp_key"][1],
                                description=f"Configuration for {_next_disp["disp_key"]}.",
                            ),
                        ]
                        for _next_disp in _disp_info["next_disps"]
                    )
                ),
                tyro_arg(name=""),
            ]
        assert _disp_info["disp_dflt"] is not None
        return Annotated[
            Union.__getitem__(
                tuple(
                    Annotated[
                        cfg_cls[
                            *_build_cfg_type(
                                list(_disp_info["disp_cls"].values()), [], 0
                            )
                        ]
                        if _disp_info["disp_cls"]
                        else cfg_cls,
                        tyro_subcommand(
                            name=k,
                            default=v,
                            description=f"Default configuration for {k}.",
                        ),
                    ]
                    for k, v in _sanitize_dflts(
                        _disp_info["disp_dflt"],
                        tuple(
                            map(operator.itemgetter(0), _disp_info["accu_disp_keys"])
                        ),
                    ).items()
                )
            ),
            tyro_arg(name=""),
        ]

    _dummy_launch_fn = _get_constructor(disp_info)

    rt = tyro_cli(
        _dummy_launch_fn,
        description="The main entry point for the project.",
        use_underscores=True,
        config=[
            FlagConversionOff,
            ConsolidateSubcommandArgs,
            StringToType,
        ],
    )

    if not isinstance(rt, tuple):
        return rt, ()
    else:
        return rt[0], tuple(reversed(rt[1:]))
