from ..domain.aggs import InferenceSettings
from ..domain.ports import Chat, InferenceApi, Interface
from ..domain.values import Response
from .theme import CHAT_RESPONSE_STYLE


def repl_loop(
    chat: Chat,
    inference: InferenceApi,
    settings: InferenceSettings,
    interface: Interface,
) -> None:
    while chat.ask_user(interface) is not None:
        response_chunks = inference.call_stream(settings, chat.get_messages())
        full_response = interface.display_stream(response_chunks, **CHAT_RESPONSE_STYLE)
        chat.add_response(Response(full_response))
