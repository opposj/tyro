"""Core public API."""
import argparse
import dataclasses
import sys
import warnings
from typing import Callable, Optional, Sequence, Type, TypeVar, Union, cast, overload

from . import _argparse_formatter, _arguments, _calling, _fields, _parsers
from . import _shtab as shtab
from . import _strings, conf

OutT = TypeVar("OutT")


# Overload notes:
# 1. Type[T] is almost a subtype of Callable[..., T]; the difference is types like
#    Union[T1, T2] which fall under the former but not the latter.
# 2. We really shouldn't need an overload here. But as of 1.1.268, it seems like it's
#    needed for pyright to understand that Union types are OK to pass in directly.
#    Hopefully we can just switch to a Union[Type[...], Callable[...]] in the future.


@overload
def cli(
    f: Type[OutT],
    *,
    prog: Optional[str] = None,
    description: Optional[str] = None,
    args: Optional[Sequence[str]] = None,
    default: Optional[OutT] = None,
) -> OutT:
    ...


@overload
def cli(
    f: Callable[..., OutT],
    *,
    prog: Optional[str] = None,
    description: Optional[str] = None,
    args: Optional[Sequence[str]] = None,
    default: Optional[OutT] = None,
) -> OutT:
    ...


def cli(
    f: Union[Type[OutT], Callable[..., OutT]],
    *,
    prog: Optional[str] = None,
    description: Optional[str] = None,
    args: Optional[Sequence[str]] = None,
    default: Optional[OutT] = None,
    **deprecated_kwargs,
) -> OutT:
    """Call or instantiate `f`, with inputs populated from an automatically generated
    CLI interface.

    `f` should have type-annotated inputs, and can be a function or type. Note that if
    `f` is a type, `dcargs.cli()` returns an instance.

    The parser is generated by populating helptext from docstrings and types from
    annotations; a broad range of core type annotations are supported.
    - Types natively accepted by `argparse`: str, int, float, pathlib.Path, etc.
    - Default values for optional parameters.
    - Booleans, which are automatically converted to flags when provided a default
      value.
    - Enums (via `enum.Enum`).
    - Various annotations from the standard typing library. Some examples:
      - `typing.ClassVar[T]`.
      - `typing.Optional[T]`.
      - `typing.Literal[T]`.
      - `typing.Sequence[T]`.
      - `typing.List[T]`.
      - `typing.Dict[K, V]`.
      - `typing.Tuple`, such as `typing.Tuple[T1, T2, T3]` or
        `typing.Tuple[T, ...]`.
      - `typing.Set[T]`.
      - `typing.Final[T]` and `typing.Annotated[T]`.
      - `typing.Union[T1, T2]`.
      - Various nested combinations of the above: `Optional[Literal[T]]`,
        `Final[Optional[Sequence[T]]]`, etc.
    - Hierarchical structures via nested dataclasses, TypedDict, NamedTuple,
      classes.
      - Simple nesting.
      - Unions over nested structures (subparsers).
      - Optional unions over nested structures (optional subparsers).
    - Generics (including nested generics).

    Completion script generation for interactive shells is also provided. To print a
    script that can be used for tab completion, pass in `--dcargs-print-completion
    {bash/zsh/tcsh}`.

    Args:
        f: Function or type.
        prog: The name of the program printed in helptext. Mirrors argument from
            `argparse.ArgumentParser()`.
        description: Description text for the parser, displayed when the --help flag is
            passed in. If not specified, `f`'s docstring is used. Mirrors argument from
            `argparse.ArgumentParser()`.
        args: If set, parse arguments from a sequence of strings instead of the
            commandline. Mirrors argument from `argparse.ArgumentParser.parse_args()`.
        default: An instance of `T` to use for default values; only supported
            if `T` is a dataclass, TypedDict, or NamedTuple. Helpful for merging CLI
            arguments with values loaded from elsewhere. (for example, a config object
            loaded from a yaml file)

    Returns:
        The output of `f(...)`, or an instance `f`. If `f` is a class, the two are
        typically equivalent.
    """
    return cast(
        OutT,
        _cli_impl(
            f,
            prog=prog,
            description=description,
            args=args,
            default=default,
            return_parser=False,
            **deprecated_kwargs,
        ),
    )


def get_parser(
    f: Union[Type[OutT], Callable[..., OutT]],
    *,
    # Note that we have no `args` argument, since this is only used when
    # parser.parse_args() is called.
    prog: Optional[str] = None,
    description: Optional[str] = None,
    default: Optional[OutT] = None,
) -> argparse.ArgumentParser:
    """Get the `argparse.ArgumentParser` object generated under-the-hood by
    `dcargs.cli()`. Useful for tools like `sphinx-argparse`, `argcomplete`, etc.

    For tab completion, we recommend using `dcargs.cli()`'s built-in `--dcargs-print-completion`
    flag."""
    return cast(
        argparse.ArgumentParser,
        _cli_impl(
            f,
            prog=prog,
            description=description,
            args=None,
            default=default,
            return_parser=True,
        ),
    )


def _cli_impl(
    f: Union[Type[OutT], Callable[..., OutT]],
    *,
    prog: Optional[str] = None,
    description: Optional[str],
    args: Optional[Sequence[str]],
    default: Optional[OutT],
    return_parser: bool,
    **deprecated_kwargs,
) -> Union[OutT, argparse.ArgumentParser]:
    """Helper for stitching the `dcargs` pipeline together.

    Converts `f` into a
    """
    if "default_instance" in deprecated_kwargs:
        warnings.warn(
            "`default_instance=` is deprecated! use `default=` instead.", stacklevel=2
        )
        default = deprecated_kwargs["default_instance"]
    if deprecated_kwargs.get("avoid_subparsers", False):
        f = conf.AvoidSubcommands[f]  # type: ignore
        warnings.warn(
            "`avoid_subparsers=` is deprecated! use `dcargs.conf.AvoidSubparsers[]`"
            " instead.",
            stacklevel=2,
        )

    # Internally, we distinguish between two concepts:
    # - "default", which is used for individual arguments.
    # - "default_instance", which is used for _fields_ (which may be broken down into
    #   one or many arguments, depending on various factors).
    #
    # This could be revisited.
    default_instance_internal: Union[_fields.NonpropagatingMissingType, OutT] = (
        _fields.MISSING_NONPROP if default is None else default
    )

    # We wrap our type with a dummy dataclass if it can't be treated as a nested type.
    # For example: passing in f=int will result in a dataclass with a single field
    # typed as int.
    if not _fields.is_nested_type(cast(Type, f), default_instance_internal):
        dummy_field = cast(
            dataclasses.Field,
            dataclasses.field(
                default=default if default is not None else dataclasses.MISSING
            ),
        )
        f = dataclasses.make_dataclass(
            cls_name="",
            fields=[(_strings.dummy_field_name, cast(Type, f), dummy_field)],
        )
        dummy_wrapped = True
    else:
        dummy_wrapped = False

    # Map a callable to the relevant CLI arguments + subparsers.
    parser_definition = _parsers.ParserSpecification.from_callable(
        f,
        description=description,
        parent_classes=set(),  # Used for recursive calls.
        parent_type_from_typevar=None,  # Used for recursive calls.
        default_instance=default_instance_internal,  # Overrides for default values.
        prefix="",  # Used for recursive calls.
    )

    # Read and fix arguments. If the user passes in --field_name instead of
    # --field-name, correct for them.
    args = sys.argv[1:] if args is None else args

    def fix_arg(arg: str) -> str:
        if not arg.startswith("--"):
            return arg
        if "=" in arg:
            arg, _, val = arg.partition("=")
            return arg.replace("_", "-") + "=" + val
        else:
            return arg.replace("_", "-")

    args = list(map(fix_arg, args))

    # If we pass in the --dcargs-print-completion flag: turn formatting tags, and get
    # the shell we want to generate a completion script for (bash/zsh/tcsh).
    #
    # Note that shtab also offers an add_argument_to() functions that fulfills a similar
    # goal, but manual parsing of argv is convenient for turning off formatting.
    print_completion = len(args) >= 2 and args[0] == "--dcargs-print-completion"

    completion_shell = None
    if print_completion:
        completion_shell = args[1]
    if print_completion or return_parser:
        _arguments.USE_RICH = False

    # Generate parser!
    with _argparse_formatter.ansi_context():
        parser = argparse.ArgumentParser(
            prog=prog,
            formatter_class=_argparse_formatter.DcargsArgparseHelpFormatter,
        )
        parser_definition.apply(parser)

        if return_parser:
            return parser

        if print_completion:
            _arguments.USE_RICH = True
            assert completion_shell in ("bash", "zsh", "tcsh",), (
                "Shell should be one `bash`, `zsh`, or `tcsh`, but got"
                f" {completion_shell}"
            )
            print(
                shtab.complete(
                    parser=parser,
                    shell=completion_shell,
                    root_prefix=f"dcargs_{parser.prog}",
                )
            )
            raise SystemExit()

        value_from_prefixed_field_name = vars(parser.parse_args(args=args))

    if dummy_wrapped:
        value_from_prefixed_field_name = {
            k.replace(_strings.dummy_field_name, ""): v
            for k, v in value_from_prefixed_field_name.items()
        }

    try:
        # Attempt to call `f` using whatever was passed in.
        out, consumed_keywords = _calling.call_from_args(
            f,
            parser_definition,
            default_instance_internal,
            value_from_prefixed_field_name,
            field_name_prefix="",
        )
    except _calling.InstantiationError as e:
        # Emulate argparse's error behavior when invalid arguments are passed in.
        parser.print_usage()
        print()
        print(e.args[0])
        raise SystemExit()

    assert len(value_from_prefixed_field_name.keys() - consumed_keywords) == 0, (
        f"Parsed {value_from_prefixed_field_name.keys()}, but only consumed"
        f" {consumed_keywords}"
    )

    if dummy_wrapped:
        out = getattr(out, _strings.dummy_field_name)
    return out
