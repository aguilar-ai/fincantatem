# pyright: strict

from .values import *
from .constants import INFERENCE_PRESETS
from typing import Optional, List, Dict, Any, TypeVar
from dataclasses import dataclass

# ------------------------------ Inference ------------------------------

T = TypeVar("T")


@dataclass
class Message[T]:
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


# ------------------------------ CLI & Decorator ------------------------------


@dataclass
class Invocation:
    filename: Optional[str]
    preset: PresetIdentifier
    snippets: bool
    locals: bool
    chat: bool = False
    cautious: bool = False
