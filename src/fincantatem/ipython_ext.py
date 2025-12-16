# pyright: strict

from __future__ import annotations

from typing import Any, Callable, Optional

from .domain.workflows import build_exception_context, build_prompt
from .domain.values import PresetIdentifier
from .lib.ports import (
    DecoratorEnv,
    FileSystem,
    InferenceApi,
    PlainTextInterface,
    RichTextInterface,
)


def load_ipython_extension(ipython: Any) -> None:
    """
    IPython extension entrypoint.

    Usage in a notebook:
        %load_ext fincantatem
    """

    def _on_cell_error(result: Any) -> None:
        err: Optional[BaseException] = getattr(
            result, "error_in_exec", None
        ) or getattr(result, "error_before_exec", None)
        if err is None:
            return

        if isinstance(err, (KeyboardInterrupt, SystemExit)):
            return

        fs = FileSystem()
        interface = (
            RichTextInterface()
            if RichTextInterface.is_available()
            else PlainTextInterface()
        )
        context = build_exception_context(
            err if isinstance(err, Exception) else Exception(str(err)), fs
        )
        env = DecoratorEnv()
        inference = InferenceApi()

        preset: PresetIdentifier = "openrouter"
        prompt = build_prompt(context, "default", snippets=True)
        response = inference.call(env.read_env(preset=preset), prompt)
        interface.display(response)

    setattr(ipython, "_fincantatem_post_run_cell_hook", _on_cell_error)
    ipython.events.register("post_run_cell", _on_cell_error)


def unload_ipython_extension(ipython: Any) -> None:
    hook: Optional[Callable[[Any], None]] = getattr(
        ipython, "_fincantatem_post_run_cell_hook", None
    )
    if hook is None:
        return
    try:
        ipython.events.unregister("post_run_cell", hook)
    finally:
        try:
            delattr(ipython, "_fincantatem_post_run_cell_hook")
        except Exception:
            pass
