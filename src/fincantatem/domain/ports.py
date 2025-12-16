# pyright: strict

from typing import Protocol, Any, Optional, Iterator, List

from .values import (
    SourceCode,
    SourceCodePath,
    SourceCodeSnippet,
    LineNumberOffset,
    Prompt,
    Response,
)
from .aggs import InferenceSettings, Invocation, Message


class FileSystem(Protocol):
    def fetch_source_code_from_path(self, path: SourceCodePath) -> SourceCode: ...
    def fetch_source_code_snippet_from_path(
        self,
        path: SourceCodePath,
        line_number_offset: LineNumberOffset,
        *,
        before: int,
        after: int,
    ) -> SourceCodeSnippet: ...

    def fetch_source_code_from_frame(
        self, frame: Any
    ) -> tuple[SourceCode, LineNumberOffset]: ...
    def fetch_source_code_snippet_from_frame(self, frame: Any) -> SourceCodeSnippet: ...


class InferenceApi(Protocol):
    def call(self, settings: InferenceSettings, prompt: Prompt) -> Response: ...
    def call_stream(
        self, settings: InferenceSettings, messages: List[Message[Prompt | Response]]
    ) -> Iterator[str]: ...


class Interface(Protocol):
    def display(self, message: str, **kwargs: Any) -> None: ...
    def display_stream(self, chunks: Iterator[str], **kwargs: Any) -> str: ...
    def prompt(self, prompt: str, **kwargs: Any) -> Optional[str]: ...

    @staticmethod
    def is_available() -> bool: ...


class CLIEnv(Protocol):
    def read_args(self) -> Invocation: ...
    def read_env(self, preset: Optional[str] = None) -> InferenceSettings: ...


class DecoratorEnv(Protocol):
    def read_args(self, **kwargs: Any) -> Invocation: ...
    def read_env(self, preset: Optional[str] = None) -> InferenceSettings: ...


class Chat(Protocol):
    def ask_user(self, interface: Interface) -> Optional[Prompt]: ...
    def get_messages(self) -> List[Message[Prompt | Response]]: ...
    def add_response(self, response: Response) -> None: ...
