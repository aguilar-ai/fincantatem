from functools import wraps
from typing import Callable, Optional, ParamSpec, TypeVar, overload

from .core import core_loop
from .domain.values import PresetIdentifier


def load_ipython_extension(ipython: object) -> None:
    from .ipython_ext import load_ipython_extension as _load

    _load(ipython)


def unload_ipython_extension(ipython: object) -> None:
    from .ipython_ext import unload_ipython_extension as _unload

    _unload(ipython)


P = ParamSpec("P")
R = TypeVar("R")


@overload
def finite(
    fn: Callable[P, R],
    *,
    preset: PresetIdentifier = "openrouter",
    snippets: bool = True,
    chat: bool = False,
    cautious: bool = False,
) -> Callable[P, R]: ...


@overload
def finite(
    fn: None = None,
    *,
    preset: PresetIdentifier = "openrouter",
    snippets: bool = True,
    chat: bool = False,
    cautious: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def finite(
    fn: Optional[Callable[P, R]] = None,
    *,
    preset: PresetIdentifier = "openrouter",
    snippets: bool = True,
    chat: bool = False,
    cautious: bool = False,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        @wraps(target)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return target(*args, **kwargs)
            except Exception as e:
                core_loop(
                    e, snippets=snippets, preset=preset, chat=chat, cautious=cautious
                )
                raise e

        return wrapper

    if fn is None:
        return decorate

    return decorate(fn)
