import sys
import traceback
from typing import Sequence

from .domain.aggs import Message, StringableContext
from .domain.constants import SYSTEM_PROMPT
from .domain.framework import Framework
from .domain.values import PresetIdentifier
from .domain.workflows import build_exception_context, build_prompt
from .lib.framework import JaxFramework
from .lib.ports import (
    Chat,
    DecoratorEnv,
    InferenceApi,
    PlainTextInterface,
    RichTextInterface,
)
from .lib.ports.fs import FileSystem
from .lib.repl import repl_loop
from .lib.theme import (
    ACTION_HINT_PLAIN,
    ACTION_HINT_RICH,
    ANALYSIS_STYLE,
    SPELL_HEADER_PLAIN,
    SPELL_HEADER_RICH,
    TRACEBACK_STYLE,
)
from .lib.utils import pipe


def core_loop(
    e: Exception,
    *,
    snippets: bool = True,
    cautious: bool = False,
    preset: PresetIdentifier = "openrouter",
    chat: bool = False,
) -> None:
    fs = FileSystem(cautious=cautious)
    interface = (
        RichTextInterface()
        if RichTextInterface.is_available()
        else PlainTextInterface()
    )
    frameworks: Sequence[Framework[StringableContext]] = pipe(
        [JaxFramework() if JaxFramework.detect() else None],
        lambda ls: filter(None, ls),
        list,
    )
    context = build_exception_context(e, fs, frameworks)
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
        SPELL_HEADER_RICH if RichTextInterface.is_available() else SPELL_HEADER_PLAIN
    )
    interface.display(header)
    interface.display(traceback.format_exc(), **TRACEBACK_STYLE)

    response = interface.display_stream(response_chunks, **ANALYSIS_STYLE)

    if chat:
        hint = (
            ACTION_HINT_RICH if RichTextInterface.is_available() else ACTION_HINT_PLAIN
        )
        interface.display(hint)
        interface.display("")

        chat_session = Chat(interface, prompt, response, exception_context=context)
        repl_loop(chat_session, inference, env.read_env(preset=preset), interface)

    sys.excepthook = lambda *args, **kwargs: None
