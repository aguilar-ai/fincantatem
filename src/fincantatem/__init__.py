# pyright: strict

from typing import Callable, Optional, ParamSpec, TypeVar, overload
from functools import wraps
import traceback

from .domain.constants import SYSTEM_PROMPT
from .domain.workflows import build_exception_context, build_prompt
from .domain.values import PresetIdentifier
from .domain.aggs import Message
from .lib.ports import *
from .lib.repl import repl_loop
from .lib.theme import (
    TRACEBACK_STYLE,
    ANALYSIS_STYLE,
    ACTION_HINT_RICH,
    ACTION_HINT_PLAIN,
    SPELL_HEADER_RICH,
    SPELL_HEADER_PLAIN,
)


# IPython/Jupyter integration:
# Allows `%load_ext fincantatem` to enable automatic help on cell failures.
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
                fs = FileSystem(cautious=cautious)
                interface = (
                    RichTextInterface()
                    if RichTextInterface.is_available()
                    else PlainTextInterface()
                )
                context = build_exception_context(e, fs)
                inference = InferenceApi()
                env = DecoratorEnv()
                prompt = build_prompt(context, "default", snippets=snippets)
                response_chunks = inference.call_stream(
                    env.read_env(preset=preset),
                    [
                        Message(role="system", content=SYSTEM_PROMPT),
                        Message(role="user", content=prompt),
                    ],
                )
                header = (
                    SPELL_HEADER_RICH
                    if RichTextInterface.is_available()
                    else SPELL_HEADER_PLAIN
                )
                interface.display(header)
                interface.display(traceback.format_exc(), **TRACEBACK_STYLE)

                response = interface.display_stream(response_chunks, **ANALYSIS_STYLE)

                if chat:
                    hint = (
                        ACTION_HINT_RICH
                        if RichTextInterface.is_available()
                        else ACTION_HINT_PLAIN
                    )
                    interface.display(hint)
                    interface.display("")

                    chat_session = Chat(
                        interface, prompt, response, exception_context=context
                    )
                    repl_loop(
                        chat_session, inference, env.read_env(preset=preset), interface
                    )
                import sys
                sys.excepthook = lambda *args, **kwargs: None # type: ignore
                raise e

        return wrapper

    if fn is None:
        return decorate

    return decorate(fn)
