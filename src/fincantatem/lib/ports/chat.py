import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib.util import find_spec
from typing import List, Optional

from ...domain.aggs import ExceptionContext, Message
from ...domain.constants import SYSTEM_PROMPT
from ...domain.ports import Chat as DomainChat
from ...domain.ports import Interface
from ...domain.values import Prompt, Response
from ...lib.constants import CHAT_EXPORT_JSON_VERSION
from ...lib.utils import pipe
from ..theme import (
    CHAT_HELP_STYLE,
    CHAT_HELP_TEXT,
    CHAT_HELP_TEXT_PLAIN,
    CHAT_INFO_STYLE,
    USER_PROMPT_STYLE,
)


class ChatCommand(Enum):
    """Commands available in chat mode."""

    HELP = "/help"
    SAVE = "/save"
    QUIT = "/quit"
    QUIT_SHORT = "/q"


def _is_command(text: str) -> bool:
    """Check if text is a chat command."""
    return text.strip().startswith("/")


def _parse_command(text: str) -> Optional[ChatCommand]:
    """Parse a command string into a ChatCommand enum."""
    text = text.strip().lower()
    for cmd in ChatCommand:
        if text == cmd.value:
            return cmd
    return None


@dataclass
class ChatExportData:
    @dataclass
    class Exception:
        type: str
        message: str

    timestamp: int = field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp())
    )
    version: str = field(default=CHAT_EXPORT_JSON_VERSION)
    python_version: Optional[str] = None

    exception: Optional[Exception] = None
    messages: List[Message[Prompt | Response]] = field(default_factory=list)


class Chat(DomainChat):
    def __init__(
        self,
        interface: Interface,
        initial_prompt: Prompt,
        analysis: Response,
        exception_context: Optional[ExceptionContext] = None,
    ):
        self.interface = interface
        self.exception_context: ExceptionContext | None = exception_context
        self.messages: List[Message[Prompt | Response]] = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=initial_prompt),
            Message(role="assistant", content=analysis),
        ]

    def _handle_help(self, interface: Interface) -> None:
        """Display help message."""
        if find_spec("rich.console.Console") is not None:
            interface.display(CHAT_HELP_TEXT, **CHAT_HELP_STYLE)
        else:
            interface.display(CHAT_HELP_TEXT_PLAIN)

    def _handle_save(self, interface: Interface) -> Optional[str]:
        """Save chat history as JSON and return the filename."""
        timestamp, filename = pipe(
            datetime.now(timezone.utc),
            lambda dt: (
                dt.isoformat(),
                f"fincantatem_chat_{dt.strftime('%Y-%m-%d_%H-%M')}.json",
            ),
        )

        export_data = ChatExportData()

        if self.exception_context:
            export_data.python_version = self.exception_context.python_version
            export_data.exception = ChatExportData.Exception(
                type=str(self.exception_context.exception_type_name),
                message=str(self.exception_context.exception_message),
            )

        export_data.messages = [
            Message(role=msg.role, content=msg.content)
            for msg in self.messages
            if msg.role != "system"
        ]

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(asdict(export_data), f, indent=2, ensure_ascii=False)

            interface.display(
                f"  ✧ Chat saved to: {filename}",
                **CHAT_INFO_STYLE,
            )
            return filename
        except Exception as e:
            interface.display(f"  ! Failed to save: {e}", **CHAT_INFO_STYLE)
            return None

    def ask_user(self, interface: Interface) -> Optional[Prompt]:
        prompt = interface.prompt("", **USER_PROMPT_STYLE)
        if prompt is None:
            return None

        # Handle commands
        if _is_command(prompt):
            command = _parse_command(prompt)

            if command == ChatCommand.QUIT or command == ChatCommand.QUIT_SHORT:
                return None

            if command == ChatCommand.HELP:
                self._handle_help(interface)
                return self.ask_user(interface)

            if command == ChatCommand.SAVE:
                self._handle_save(interface)
                return self.ask_user(interface)

            interface.display(
                f"  ! Unknown command: {prompt}. Type /help for available commands.",
                **CHAT_INFO_STYLE,
            )
            return self.ask_user(interface)

        self.messages.append(Message(role="user", content=Prompt(prompt)))
        return Prompt(prompt)

    def get_messages(self) -> List[Message[Prompt | Response]]:
        return self.messages

    def add_response(self, response: Response) -> None:
        self.messages.append(Message(role="assistant", content=response))
