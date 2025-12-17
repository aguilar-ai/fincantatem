from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Protocol, Sequence, TypeVar

from .constants import INFERENCE_PRESETS
from .values import (
    ApiKey,
    ArrayDtype,
    ArrayShape,
    Cause,
    Context,
    DeviceName,
    ExceptionAttributes,
    ExceptionMessage,
    ExceptionTypeName,
    FnName,
    FrameworkName,
    InferenceApiIdentifier,
    InferenceApiUrl,
    LineNumberOffset,
    ModelId,
    PresetIdentifier,
    PythonVersion,
    Role,
    SourceCode,
    SourceCodePath,
    SourceCodeSnippet,
    Traceback,
    TransformationType,
)


class StringableContext(Protocol):
    def frame_context_string(self, index: int) -> Optional[str]: ...


S = TypeVar("S", covariant=True, bound=StringableContext)

# ------------------------------ Inference ------------------------------

T = TypeVar("T")


@dataclass
class Message(Generic[T]):
    role: Role
    content: T


@dataclass
class InferenceSettings:
    identifier: InferenceApiIdentifier
    url: InferenceApiUrl
    model: Optional[ModelId] = None
    api_key: Optional[ApiKey] = None

    @classmethod
    def preset(cls, identifier: PresetIdentifier) -> "InferenceSettings":
        match identifier:
            case "openrouter":
                url, model = INFERENCE_PRESETS["openrouter"]
                return cls(
                    identifier=InferenceApiIdentifier("openrouter"),
                    url=url,
                    model=model,
                )
            case "openai":
                url, model = INFERENCE_PRESETS["openai"]
                return cls(
                    identifier=InferenceApiIdentifier("openai"),
                    url=url,
                    model=model,
                )

    @classmethod
    def custom(
        cls,
        identifier: InferenceApiIdentifier,
        url: InferenceApiUrl,
        model: Optional[ModelId] = None,
        api_key: Optional[ApiKey] = None,
    ) -> "InferenceSettings":
        return cls(
            identifier=identifier,
            url=url,
            model=model,
            api_key=api_key,
        )


# ------------------------------ Exception ------------------------------


@dataclass(frozen=True)
class SourceCodeBundle:
    path: SourceCodePath
    code: SourceCode
    snippet: SourceCodeSnippet
    line_number_offset: LineNumberOffset
    code_start_line_number_offset: LineNumberOffset
    function_name: FnName
    local_vars: Optional[Dict[str, Any]]


@dataclass
class ExceptionContext:
    python_version: PythonVersion
    exception_type_name: ExceptionTypeName
    exception_message: ExceptionMessage
    exception_attributes: Optional[ExceptionAttributes]
    traceback: Traceback
    cause: Optional[Cause]
    context: Optional[Context]
    immediate_source_code_bundle: SourceCodeBundle
    source_code_bundles: List[SourceCodeBundle]
    framework_context: List[StringableContext]


# ------------------------------ CLI & Decorator ------------------------------


@dataclass
class Invocation:
    filename: Optional[str]
    preset: PresetIdentifier
    full_source: bool
    locals: bool
    chat: bool = False
    cautious: bool = False


# ------------------------------ Framework ------------------------------


@dataclass(frozen=True)
class ArrayMetadata:
    """Framework-agnostic array metadata"""

    name: str
    shape: ArrayShape
    dtype: ArrayDtype
    device: DeviceName


@dataclass(frozen=True)
class FrameContext:
    """Framework-agnostic per-frame context"""

    arrays: Sequence[ArrayMetadata]
    transformation_context: Optional[TransformationType]


@dataclass(frozen=True)
class FrameworkContext:
    """Base framework context"""

    name: FrameworkName
    version: str
    frame_contexts: Sequence[FrameContext]
