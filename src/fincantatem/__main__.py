import runpy
import traceback

from .lib.ports import *
from .lib.repl import repl_loop
from .lib.theme import (
    TRACEBACK_STYLE,
    ANALYSIS_STYLE,
    ACTION_HINT_RICH,
    ACTION_HINT_PLAIN,
)
from .domain.workflows import build_exception_context, build_prompt
from .domain.constants import SYSTEM_PROMPT
from .domain.aggs import Message

if __name__ == "__main__":
    cli = CLIEnv()
    invocation = cli.read_args()

    if invocation.filename is None:
        raise ValueError("Filename is required")

    try:
        runpy.run_path(invocation.filename, run_name="__main__")
    except Exception as e:
        fs = FileSystem(cautious=invocation.cautious)
        interface = (
            RichTextInterface()
            if RichTextInterface.is_available()
            else PlainTextInterface()
        )
        context = build_exception_context(e, fs)
        inference = InferenceApi()
        prompt = build_prompt(context, "default", snippets=invocation.snippets)
        response_chunks = inference.call_stream(
            cli.read_env(preset=invocation.preset),
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ],
        )
        interface.display(traceback.format_exc(), **TRACEBACK_STYLE)
        response = interface.display_stream(response_chunks, **ANALYSIS_STYLE)

        if invocation.chat:
            hint = (
                ACTION_HINT_RICH
                if RichTextInterface.is_available()
                else ACTION_HINT_PLAIN
            )
            interface.display(hint)
            interface.display("")
            chat_session = Chat(interface, prompt, response)
            repl_loop(
                chat_session,
                inference,
                cli.read_env(preset=invocation.preset),
                interface,
            )

        raise e
