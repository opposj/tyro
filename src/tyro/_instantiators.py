"""Helper for using type annotations to recursively generate instantiator functions,
which map sequences of strings to the annotated type.

Some examples of type annotations and the desired instantiators:
```
    int

        lambda strings: int(str[0])

    List[int]

        lambda strings: list(
            [int(x) for x in strings]
        )

    List[Color], where Color is an enum

        lambda strings: list(
            [Color[x] for x in strings]
        )

    Tuple[int, float]

        lambda strings: tuple(
            typ(x)
            for typ, x in zip(
                (int, float),
                strings,
            )
        )
```
"""

import builtins
import collections.abc
import dataclasses
import datetime
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import yaml
from typing_extensions import Annotated, Literal, get_args, get_origin

from . import _resolver

# There are cases where typing.Literal doesn't match typing_extensions.Literal:
# https://github.com/python/typing_extensions/pull/148
try:
    from typing import Literal as LiteralAlternate
except ImportError:
    LiteralAlternate = Literal  # type: ignore

from . import _strings
from ._typing import TypeForm
from .conf import _markers

_StandardInstantiator = Callable[[List[str]], Any]
_AppendNargsInstantiator = Callable[[List[List[str]]], Any]
# Special case: the only time that argparse doesn't give us a string is when the
# argument action is set to `store_true` or `store_false`. In this case, we get
# a bool directly, and the field action can be a no-op.
_FlagInstantiator = Callable[[bool], bool]

Instantiator = Union[_StandardInstantiator, _AppendNargsInstantiator, _FlagInstantiator]

NoneType = type(None)
T = TypeVar("T", bound=enum.Enum)


@dataclasses.dataclass
class InstantiatorMetadata:
    # Unlike in vanilla argparse, we never set nargs to None. To make things simpler, we
    # instead use nargs=1.
    nargs: Optional[Union[int, Literal["*"]]]
    # Unlike in vanilla argparse, our metavar is always a string. We handle
    # sequences, multiple arguments, etc, manually.
    metavar: str
    choices: Optional[Tuple[str, ...]]
    action: Optional[Literal["append"]]

    def check_choices(self, strings: List[str]) -> None:
        if self.choices is not None and any(s not in self.choices for s in strings):
            raise ValueError(f"invalid choice: {strings} (choose from {self.choices}))")


class UnsupportedTypeAnnotationError(Exception):
    """Exception raised when an unsupported type annotation is detected."""


_builtin_set: Set[Hashable] = set(
    filter(
        lambda x: isinstance(x, Hashable),  # type: ignore
        vars(builtins).values(),  # type: ignore
    )
)


def is_type_string_converter(typ: Union[Callable, TypeForm[Any]]) -> bool:
    """Check if type is a string converter, i.e., (arg: Union[str, Any]) -> T."""
    param_count = 0
    has_var_positional = False
    try:
        signature = inspect.signature(typ)
    except ValueError:
        # pybind objects might not have a parsable signature. We try to be tolerant in this case.
        # One day this should be fixed with `__text_signature__`.
        return True

    type_annotations = _resolver.get_type_hints_with_backported_syntax(typ)
    # Some checks we can do if the signature is available!
    for i, param in enumerate(signature.parameters.values()):
        annotation = type_annotations.get(param.name, param.annotation)
        annotation = _resolver.TypeParamResolver.concretize_type_params(annotation)
        if i == 0 and not (
            (get_origin(annotation) is Union and str in get_args(annotation))
            or annotation in (str, inspect.Parameter.empty)
        ):
            return False
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif param.default is inspect.Parameter.empty and param.kind is not (
            inspect.Parameter.VAR_KEYWORD
        ):
            param_count += 1

    # Raise an error if parameters look wrong.
    if not (param_count == 1 or (param_count == 0 and has_var_positional)):
        return False
    return True


def instantiator_from_type(
    typ: Union[TypeForm[Any], Callable],
    markers: Set[_markers.Marker],
) -> Tuple[Instantiator, InstantiatorMetadata]:
    """Recursive helper for parsing type annotations.

    Returns two things:
    - An instantiator function, for instantiating the type from a string or list of
      strings. The latter applies when argparse's `nargs` parameter is set.
    - A metadata structure, which specifies parameters for argparse.
    """

    # Handle Any.
    if typ is Any:
        raise UnsupportedTypeAnnotationError("`Any` is not a parsable type.")

    # Handle NoneType.
    if typ is NoneType:

        def instantiator(strings: List[str]) -> None:
            # Note that other inputs should be caught by `choices` before the
            # instantiator runs.
            assert strings == ["None"]

        return instantiator, InstantiatorMetadata(
            nargs=1,
            metavar="{None}",
            choices=("None",),
            action=None,
        )

    # Instantiate os.PathLike annotations using pathlib.Path.
    # Ideally this should be implemented in a more general way, eg using
    # https://github.com/brentyi/tyro/pull/30
    if typ is os.PathLike:
        typ = pathlib.Path

    # Unwrap NewType + set metavar based on NewType name.
    # `isinstance(x, NewType)` doesn't work because NewType isn't a class until
    # Python 3.10, so we instead do a duck typing-style check.
    metavar = getattr(typ, "__name__", "").upper()
    typ, maybe_newtype_name = _resolver.unwrap_newtype_and_aliases(typ)
    if maybe_newtype_name is not None:
        metavar = maybe_newtype_name.upper()

    typ = _resolver.unwrap_annotated_and_aliases(typ)

    # Address container types. If a matching container is found, this will recursively
    # call instantiator_from_type().
    container_out = _instantiator_from_container_type(cast(TypeForm[Any], typ), markers)
    if container_out is not None:
        return container_out

    # At this point: Annotated[] and containers like list[T] are all handled,
    # any types with a non-None origin aren't supported.
    if get_origin(typ) is not None:
        raise UnsupportedTypeAnnotationError(  # pragma: no cover
            f"Unsupported type {typ} with origin {get_origin(typ)}"
        )

    # Validate that typ is a `(arg: str) -> T` type converter, as expected by argparse.
    if typ in (type, Type):
        raise UnsupportedTypeAnnotationError(
            "We do not support populating `type` instances from the command-line"
        )
    elif typ in _builtin_set:
        pass
    elif not callable(typ):
        raise UnsupportedTypeAnnotationError(
            f"Expected {typ} to be an `(arg: str) -> T` type converter, but is not"
            " callable."
        )
    elif not is_type_string_converter(typ):
        raise UnsupportedTypeAnnotationError(
            f"Expected type to be an `(arg: str) -> T` type converter, but {typ} is not"
            " a valid type converter."
        )

    # Use ISO 8601 standard for dates/times.
    if typ in (datetime.datetime, datetime.date, datetime.time):

        def instantiate(args: List[str]) -> Any:
            (arg,) = args
            try:
                # Type ignore is unnecessary for pyright but needed for mypy.
                return typ.fromisoformat(arg)  # type: ignore
            except ValueError:
                raise ValueError(
                    f"`{typ.__name__}.fromisoformat('{arg}')` failed. Dates "
                    "should be specified in ISO-8601 format: "
                    "https://en.wikipedia.org/wiki/ISO_8601"
                )

        return (
            instantiate,
            InstantiatorMetadata(
                nargs=1,
                metavar={
                    # Actually we take any ISO 8601 string here. So these
                    # metavars are a bit incomplete.
                    # datetime.datetime: "YYYY-MM-DD[THH:MM:SS]",
                    # datetime.date: "YYYY-MM-DD",
                    # datetime.time: "HH:MM[:SS]",
                    #
                    # More complete.
                    datetime.datetime: "YYYY-MM-DD[THH:MM:[SS[…]]]",
                    datetime.date: "YYYY-MM-DD",
                    datetime.time: "HH:MM[:SS[…]]",
                    #
                    # Even more complete!
                    # datetime.datetime: "YYYY-MM-DD[THH:MM[:SS[.fff]][±HH:MM|Z]]",
                    # datetime.date: "YYYY-MM-DD",
                    # datetime.time: "HH:MM[:SS[.fff]][±HH:MM|Z]",
                    #
                    # Not very informative but (1) precise and (2) succinct.
                    # datetime.datetime: "ISO-DATETIME",
                    # datetime.date: "ISO-DATE",
                    # datetime.time: "ISO-TIME",
                }[cast(Any, typ)],  # cast is for mypy, pyright works fine
                choices=None,
                action=None,
            ),
        )

    # Special case `choices` for some types. Booleans accept {True,False},
    # enums accept based on members.
    choices: Union[None, Tuple[str, ...]] = None
    autochoice_instantiate: Union[None, Callable[[str], Any]] = None
    if typ is bool:
        choices = ("True", "False")
        autochoice_instantiate = lambda x: {"True": True, "False": False}[x]
    elif inspect.isclass(typ) and issubclass(typ, enum.Enum):
        if _markers.EnumChoicesFromValues in markers:
            value_from_str_choice = {str(member.value): member for member in typ}
            choices = tuple(value_from_str_choice.keys())
            autochoice_instantiate = lambda x: value_from_str_choice[x]
        else:
            choices = tuple(typ.__members__.keys())
            autochoice_instantiate = lambda x: typ[x]

    if choices is not None:
        metavar = "{" + ",".join(choices) + "}"

    def instantiator_base_case(strings: List[str]) -> Any:
        """Given a type and and a string from the command-line, reconstruct an object. Not
        intended to deal with containers.

        This is intended to replace all calls to `type(string)`, which can cause unexpected
        behavior. As an example, note that the following argparse code will always print
        `True`, because `bool("True") == bool("False") == bool("0") == True`.
        ```
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", type=bool)

        print(parser.parse_args().flag)
        ```
        """
        assert len(get_args(typ)) == 0, f"TypeForm {typ} cannot be instantiated."
        (string,) = strings
        if autochoice_instantiate is not None:
            return autochoice_instantiate(string)
        elif typ is bytes:
            return bytes(string, encoding="ascii")  # type: ignore
        else:
            # We assume this base type can be called on a string to convert
            # from a string, eg int("5"), float("5."), Path("/home"), etc.
            return typ(string)  # type: ignore

    return instantiator_base_case, InstantiatorMetadata(
        nargs=1,
        metavar=metavar,
        choices=choices,
        action=None,
    )


@overload
def _instantiator_from_type_inner(
    typ: TypeForm,
    allow_sequences: Literal["fixed_length"],
    markers: Set[_markers.Marker],
) -> Tuple[Instantiator, InstantiatorMetadata]: ...


@overload
def _instantiator_from_type_inner(
    typ: TypeForm,
    allow_sequences: Literal[False],
    markers: Set[_markers.Marker],
) -> Tuple[_StandardInstantiator, InstantiatorMetadata]: ...


@overload
def _instantiator_from_type_inner(
    typ: TypeForm,
    allow_sequences: Literal[True],
    markers: Set[_markers.Marker],
) -> Tuple[Instantiator, InstantiatorMetadata]: ...


def _instantiator_from_type_inner(
    typ: TypeForm,
    allow_sequences: Literal["fixed_length", True, False],
    markers: Set[_markers.Marker],
) -> Tuple[Instantiator, InstantiatorMetadata]:
    """Thin wrapper over instantiator_from_type, with some extra asserts for catching
    errors."""
    out = instantiator_from_type(typ, markers)
    if out[1].nargs == "*":
        # We currently only use allow_sequences=False for options in Literal types,
        # which are evaluated using `type()`. It should not be possible to hit this
        # condition from polling a runtime type.
        assert allow_sequences
        if allow_sequences == "fixed_length" and not isinstance(out[1].nargs, int):
            raise UnsupportedTypeAnnotationError(
                f"{typ} is a variable-length sequence, which is ambiguous when nested."
                " For nesting variable-length sequences (example: List[List[int]]),"
                " `tyro.conf.UseAppendAction` can help resolve ambiguities."
            )
    return out


def _instantiator_from_container_type(
    typ: TypeForm[Any],
    markers: Set[_markers.Marker],
) -> Optional[Tuple[Instantiator, InstantiatorMetadata]]:
    """Attempt to create an instantiator from a container type. Returns `None` if no
    container type is found."""

    # Default generic types to strings.
    if typ in (dict, Dict):
        typ = Dict[str, str]
    elif typ in (tuple, Tuple):
        typ = Tuple[str, ...]  # type: ignore
    elif typ in (list, List, collections.abc.Sequence, Sequence):
        typ = List[str]
    elif typ in (set, Set):
        typ = Set[str]

    type_origin = get_origin(typ)
    if type_origin in (None, Annotated):
        return None

    for make, matched_origins in {
        _instantiator_from_sequence: (
            collections.abc.Sequence,
            frozenset,
            list,
            set,
            deque,
        ),
        _instantiator_from_tuple: (tuple,),
        _instantiator_from_dict: (dict, collections.abc.Mapping),
        _instantiator_from_union: (Union,),
        _instantiator_from_literal: (Literal, LiteralAlternate),
    }.items():
        if type_origin in matched_origins:
            return make(typ, markers)
    return None


def _instantiator_from_tuple(
    typ: TypeForm,
    markers: Set[_markers.Marker],
) -> Tuple[Instantiator, InstantiatorMetadata]:
    types = get_args(typ)
    typeset = set(types)  # Note that sets are unordered.
    typeset_no_ellipsis = typeset - {Ellipsis}  # type: ignore

    if typeset_no_ellipsis != typeset:
        # Ellipsis: variable argument counts. When an ellipsis is used, tuples must
        # contain only one type.
        assert len(typeset_no_ellipsis) == 1
        return _instantiator_from_sequence(typ, markers)

    else:
        instantiators: List[_StandardInstantiator] = []
        metas: List[InstantiatorMetadata] = []
        nargs = 0
        for t in types:
            a, b = _instantiator_from_type_inner(
                t, allow_sequences="fixed_length", markers=markers
            )
            instantiators.append(a)  # type: ignore
            metas.append(b)
            assert isinstance(b.nargs, int)
            nargs += b.nargs

        if _markers.StringToType in markers:

            def fixed_length_tuple_instantiator(strings: List[str]) -> Any:
                try:
                    assert len(strings) == 1
                    loaded: list[Any] = yaml.safe_load(strings[0])
                    assert len(loaded) == len(metas)
                except (yaml.YAMLError, AssertionError):
                    raise ValueError(
                        f"Failed to parse {strings[0]} as a tuple of {len(metas)} elements."
                    )

                # Make tuple.
                outs = []
                for i, meta in enumerate(metas):
                    assert isinstance(meta.nargs, int) and meta.nargs == 1
                    meta.check_choices([str(loaded[i])])
                    outs.append(instantiators[i]([str(loaded[i])]))
                return tuple(outs)

            return fixed_length_tuple_instantiator, InstantiatorMetadata(
                nargs=1,
                metavar="[" + ",".join(m.metavar for m in metas) + "]",
                choices=None,
                action=None,
            )

        def fixed_length_tuple_instantiator(strings: List[str]) -> Any:
            assert len(strings) == nargs

            # Make tuple.
            out = []
            i = 0
            for make, meta in zip(instantiators, metas):
                assert isinstance(meta.nargs, int)
                meta.check_choices(strings[i : i + meta.nargs])
                out.append(make(strings[i : i + meta.nargs]))
                i += meta.nargs
            return tuple(out)

        return fixed_length_tuple_instantiator, InstantiatorMetadata(
            nargs=nargs,
            metavar=" ".join(m.metavar for m in metas),
            choices=None,
            action=None,
        )


def _join_union_metavars(metavars: Iterable[str]) -> str:
    """Metavar generation helper for unions. Could be revisited.

    Examples:
        None, INT => NONE|INT
        {0,1,2}, {3,4} => {0,1,2,3,4}
        {0,1,2}, {3,4}, STR => {0,1,2,3,4}|STR
        {None}, INT [INT ...] => {None}|{INT [INT ...]}
        STR, INT [INT ...] => STR|{INT [INT ...]}
        STR, INT INT => STR|{INT INT}

    The curly brackets are unfortunately overloaded but alternatives all interfere with
    argparse internals.
    """
    metavars = tuple(metavars)
    merged_metavars = [metavars[0]]
    for i in range(1, len(metavars)):
        prev = merged_metavars[-1]
        curr = metavars[i]
        if (
            prev.startswith("{")
            and prev.endswith("}")
            and curr.startswith("{")
            and curr.endswith("}")
        ):
            merged_metavars[-1] = prev[:-1] + "," + curr[1:]
        else:
            merged_metavars.append(curr)

    for i, m in enumerate(merged_metavars):
        if " " in m:
            merged_metavars[i] = "{" + m + "}"

    return "|".join(merged_metavars)


def _instantiator_from_union(
    typ: TypeForm,
    markers: Set[_markers.Marker],
) -> Tuple[Instantiator, InstantiatorMetadata]:
    options = list(get_args(typ))
    if NoneType in options:
        # Move `None` types to the beginning.
        # If we have `Optional[str]`, we want this to be parsed as
        # `Union[NoneType, str]`.
        options.remove(NoneType)
        options.insert(0, NoneType)

    # General unions, eg Union[int, bool]. We'll try to convert these from left to
    # right.
    instantiators = []
    metas = []
    choices: Tuple[str, ...] | None = ()
    nargs: Optional[Union[int, Literal["*"]]] = 1
    first = True
    for t in options:
        a, b = _instantiator_from_type_inner(t, allow_sequences=True, markers=markers)
        instantiators.append(a)
        metas.append(b)
        if b.choices is None:
            choices = None
        elif choices is not None:
            choices = choices + b.choices

        if t is not NoneType:
            # Enforce that `nargs` is the same for all child types, except for
            # NoneType.
            if first:
                nargs = b.nargs
                first = False
            elif nargs != b.nargs:
                # Just be as general as possible if we see inconsistencies.
                nargs = "*"

    metavar: str
    metavar = _join_union_metavars(map(lambda x: cast(str, x.metavar), metas))

    def union_instantiator(strings: List[str]) -> Any:
        metadata: InstantiatorMetadata
        errors = []
        for i, (instantiator, metadata) in enumerate(zip(instantiators, metas)):
            # Check choices.
            if metadata.choices is not None and any(
                x not in metadata.choices for x in strings
            ):
                errors.append(
                    f"{options[i]}: {strings} does not match choices {metadata.choices}"
                )
                continue

            # Try passing input into instantiator.
            if len(strings) == metadata.nargs or (metadata.nargs == "*"):
                try:
                    return instantiator(strings)  # type: ignore
                except ValueError as e:
                    # Failed, try next instantiator.
                    errors.append(f"{options[i]}: {e.args[0]}")
            else:
                errors.append(
                    f"{options[i]}: input length {len(strings)} did not match expected"
                    f" argument count {metadata.nargs}"
                )
        raise ValueError(
            f"no type in {options} could be instantiated from"
            f" {strings}.\n\nGot errors:  \n- " + "\n- ".join(errors)
        )

    return union_instantiator, InstantiatorMetadata(
        nargs=nargs,
        metavar=metavar,
        choices=None if choices is None else tuple(set(choices)),
        action=None,
    )


def _instantiator_from_dict(
    typ: TypeForm,
    markers: Set[_markers.Marker],
) -> Tuple[Instantiator, InstantiatorMetadata]:
    key_type, val_type = get_args(typ)
    key_instantiator, key_meta = _instantiator_from_type_inner(
        key_type, allow_sequences="fixed_length", markers=markers
    )

    if _markers.UseAppendAction in markers:
        val_instantiator, val_meta = _instantiator_from_type_inner(
            val_type,
            allow_sequences=True,
            markers=markers - {_markers.UseAppendAction},
        )
        pair_metavar = f"{key_meta.metavar} {val_meta.metavar}"
        key_nargs = cast(int, key_meta.nargs)  # Casts needed for mypy but not pyright!
        val_nargs = val_meta.nargs
        assert isinstance(key_nargs, int)

        def append_dict_instantiator(strings: List[List[str]]) -> Any:
            out = {}
            for s in strings:
                out[key_instantiator(s[:key_nargs])] = val_instantiator(s[key_nargs:])  # type: ignore
            return out

        return append_dict_instantiator, InstantiatorMetadata(
            nargs=key_nargs + val_nargs if isinstance(val_nargs, int) else "*",
            metavar=pair_metavar,
            choices=None,
            action="append",
        )
    else:
        val_instantiator, val_meta = _instantiator_from_type_inner(
            val_type, allow_sequences="fixed_length", markers=markers
        )
        pair_metavar = f"{key_meta.metavar} {val_meta.metavar}"
        key_nargs = cast(int, key_meta.nargs)  # Casts needed for mypy but not pyright!
        val_nargs = cast(int, val_meta.nargs)
        assert isinstance(key_nargs, int)
        assert isinstance(val_nargs, int)
        pair_nargs = key_nargs + val_nargs

        def dict_instantiator(strings: List[str]) -> Any:
            out = {}
            if len(strings) % pair_nargs != 0:
                raise ValueError("incomplete set of key value pairs!")

            index = 0
            for _ in range(len(strings) // pair_nargs):
                assert isinstance(key_nargs, int)
                assert isinstance(val_nargs, int)
                k = strings[index : index + key_nargs]
                index += key_nargs
                v = strings[index : index + val_nargs]
                index += val_nargs

                if key_meta.choices is not None and any(
                    kj not in key_meta.choices for kj in k
                ):
                    raise ValueError(
                        f"invalid choice: {k} (choose from {key_meta.choices}))"
                    )
                if val_meta.choices is not None and any(
                    vj not in val_meta.choices for vj in v
                ):
                    raise ValueError(
                        f"invalid choice: {v} (choose from {val_meta.choices}))"
                    )
                out[key_instantiator(k)] = val_instantiator(v)  # type: ignore
            return out

        return dict_instantiator, InstantiatorMetadata(
            nargs="*",
            metavar=_strings.multi_metavar_from_single(pair_metavar),
            choices=None,
            action=None,
        )


def _instantiator_from_sequence(
    typ: TypeForm,
    markers: Set[_markers.Marker],
) -> Tuple[Instantiator, InstantiatorMetadata]:
    """Instantiator for variable-length sequences: list, sets, Tuple[T, ...], etc."""
    container_type = get_origin(typ)
    assert container_type is not None
    if container_type is collections.abc.Sequence:
        container_type = list

    if container_type is tuple:
        (contained_type, ell) = get_args(typ)
        assert ell == Ellipsis
    else:
        (contained_type,) = get_args(typ)

    if _markers.UseAppendAction in markers:
        make, inner_meta = _instantiator_from_type_inner(
            contained_type,
            allow_sequences=True,
            markers=markers - {_markers.UseAppendAction},
        )

        def append_sequence_instantiator(strings: List[List[str]]) -> Any:
            assert strings is not None
            return container_type(cast(_StandardInstantiator, make)(s) for s in strings)

        return append_sequence_instantiator, InstantiatorMetadata(
            nargs=inner_meta.nargs,
            metavar=inner_meta.metavar,
            choices=inner_meta.choices,
            action="append",
        )
    else:
        make, inner_meta = _instantiator_from_type_inner(
            contained_type,
            allow_sequences="fixed_length",
            markers=markers,
        )

        if _markers.StringToType in markers:

            def multi_metavar_from_single(single: str) -> str:
                if len(_strings.strip_ansi_sequences(single)) >= 32:
                    # Shorten long metavars
                    return f"[{single},...]"
                else:
                    return f"[{single},{single},...]"

            def sequence_instantiator(strings: List[str]) -> Any:
                try:
                    assert len(strings) == 1
                    loaded: list[Any] = yaml.safe_load(strings[0])
                    # Validate nargs.
                    if (
                        isinstance(inner_meta.nargs, int)
                        and len(loaded) % inner_meta.nargs != 0
                    ):
                        raise ValueError(
                            f"input {loaded} is of length {len(loaded)}, which is not"
                            f" divisible by {inner_meta.nargs}."
                        )
                except yaml.YAMLError:
                    raise ValueError(
                        f"Failed to parse {strings[0]} as a container for {inner_meta.metavar}."
                    )

                # Make tuple.
                out = []
                step = inner_meta.nargs if isinstance(inner_meta.nargs, int) else 1
                assert step == 1
                for i in range(0, len(loaded), step):
                    sub_str = [str(loaded[i])]
                    inner_meta.check_choices(sub_str)
                    out.append(make(sub_str)) # type: ignore
                assert container_type is not None
                return container_type(out)

            return sequence_instantiator, InstantiatorMetadata(
                nargs=1,
                metavar=multi_metavar_from_single(inner_meta.metavar),
                choices=None,
                action=None,
            )

        def sequence_instantiator(strings: List[str]) -> Any:
            # Validate nargs.
            if (
                isinstance(inner_meta.nargs, int)
                and len(strings) % inner_meta.nargs != 0
            ):
                raise ValueError(
                    f"input {strings} is of length {len(strings)}, which is not"
                    f" divisible by {inner_meta.nargs}."
                )

            # Make tuple.
            out = []
            step = inner_meta.nargs if isinstance(inner_meta.nargs, int) else 1
            for i in range(0, len(strings), step):
                out.append(make(strings[i : i + inner_meta.nargs]))  # type: ignore
            assert container_type is not None
            return container_type(out)

        return sequence_instantiator, InstantiatorMetadata(
            nargs="*",
            metavar=_strings.multi_metavar_from_single(inner_meta.metavar),
            choices=inner_meta.choices,
            action=None,
        )


def _instantiator_from_literal(
    typ: TypeForm,
    markers: Set[_markers.Marker],
) -> Tuple[_StandardInstantiator, InstantiatorMetadata]:
    choices = get_args(typ)
    str_choices = tuple(
        (x.value if _markers.EnumChoicesFromValues in markers else x.name)
        if isinstance(x, enum.Enum)
        else str(x)
        for x in choices
    )
    return (
        # Note that if string is not in str_choices, it will be caught from setting
        # `choices` below.
        lambda strings: choices[str_choices.index(strings[0])],
        InstantiatorMetadata(
            nargs=1,
            metavar="{" + ",".join(str_choices) + "}",
            choices=str_choices,
            action=None,
        ),
    )
