from ...domain.values import Response, Prompt
from ...domain.aggs import Message, ExceptionContext
from ...domain.constants import SYSTEM_PROMPT
from ...domain.ports import Chat as DomainChat
from ...domain.ports import Interface
from ..theme import (
    USER_PROMPT_STYLE,
    CHAT_HELP_STYLE,
    CHAT_HELP_TEXT,
    CHAT_HELP_TEXT_PLAIN,
    CHAT_INFO_STYLE,
)
from typing import Optional, List, Any
from enum import Enum
import json
from datetime import datetime, timezone


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


class Chat(DomainChat):
    def __init__(
        self,
        interface: Interface,
        initial_prompt: Prompt,
        analysis: Response,
        exception_context: Optional[ExceptionContext] = None,
    ):
        self.interface = interface
        self.exception_context = exception_context
        self.messages: List[Message[Prompt | Response]] = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=initial_prompt),
            Message(role="assistant", content=analysis),
        ]

    # NOTE: LLM-written code
    def _handle_help(self, interface: Interface) -> None:
        """Display help message."""
        # Check if we have Rich available
        try:
            from rich.console import Console  # type: ignore

            interface.display(CHAT_HELP_TEXT, **CHAT_HELP_STYLE)
        except ImportError:
            interface.display(CHAT_HELP_TEXT_PLAIN)

    # NOTE: LLM-written code
    def _handle_save(self, interface: Interface) -> Optional[str]:
        """Save chat history as JSON and return the filename."""
        timestamp = datetime.now(timezone.utc).isoformat()
        safe_timestamp = timestamp.replace(":", "-").replace("+", "_")
        filename = f"fincantatem_chat_{safe_timestamp}.json"

        # Build the export structure
        export_data: dict[str, Any] = {
            "timestamp": timestamp,
            "version": "1.0",
        }

        # Add exception context if available
        if self.exception_context:
            export_data["exception"] = {
                "type": str(self.exception_context.exception_type_name),
                "message": str(self.exception_context.exception_message),
                "python_version": str(self.exception_context.python_version),
            }

        # Add messages (skip system prompt for cleaner export)
        export_data["messages"] = [
            {"role": str(msg.role), "content": str(msg.content)}
            for msg in self.messages
            if str(msg.role) != "system"
        ]

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

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
                # Recursively ask for next input
                return self.ask_user(interface)

            if command == ChatCommand.SAVE:
                self._handle_save(interface)
                # Recursively ask for next input
                return self.ask_user(interface)

            # Unknown command
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
