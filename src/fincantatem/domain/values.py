# pyright: strict

from typing import NewType, Literal, Dict, Any

# ------------------------------ Inference ------------------------------

# Identifier for an inference API
InferenceApiIdentifier = NewType("InferenceApiIdentifier", str)

# Identifies a model on an inference API
ModelId = NewType("ModelId", str)

# URL to an OpenAI-compatible inference API
InferenceApiUrl = NewType("InferenceApiUrl", str)

# API key for an inference API
ApiKey = NewType("ApiKey", str)

# Identifier for a preset
PresetIdentifier = Literal["openrouter", "openai"]

# Prompt Template identifier
PromptTemplateIdentifier = Literal["default"]

# Prompt Template
PromptTemplate = NewType("PromptTemplate", str)

# Prompt
Prompt = NewType("Prompt", str)

Response = NewType("Response", str)

Role = Literal["system", "user", "assistant"]

PromptVar = Literal[
    "python_version",
    "exception_type_name",
    "exception_message",
    "exception_attributes",
    "immediate",
    "traceback",
    "call_stack_section",
    "cause_section",
    "context_section",
]

# ------------------------------ Exception ------------------------------

PythonVersion = NewType("PythonVersion", str)
ExceptionTypeName = NewType("ExceptionTypeName", str)
ExceptionMessage = NewType("ExceptionMessage", str)
ExceptionAttributes = NewType("ExceptionAttributes", Dict[str, Any])

# Full traceback string
Traceback = NewType("Traceback", str)

# Cause of the exception
Cause = NewType("Cause", str)

# Context of the exception
Context = NewType("Context", str)

# Function name
FnName = NewType("FnName", str)

# Source code path
SourceCodePath = NewType("SourceCodePath", str)

# Source code
SourceCode = NewType("SourceCode", str)

# Source code snippet
SourceCodeSnippet = NewType("SourceCodeSnippet", str)

# Line number offset (1-based)
LineNumberOffset = NewType("LineNumberOffset", int)

# Secret
Secret = NewType("Secret", str)

# Redacted secret
RedactedSecret = NewType("RedactedSecret", str)
