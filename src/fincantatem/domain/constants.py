from typing import Final, Dict, Tuple, cast
from .values import (
    Prompt,
    InferenceApiUrl,
    PresetIdentifier,
    ModelId,
    PromptTemplateIdentifier,
    PromptTemplate,
)

INFERENCE_PRESETS: Final[Dict[PresetIdentifier, Tuple[InferenceApiUrl, ModelId]]] = {
    "openrouter": (
        InferenceApiUrl("https://openrouter.ai/api/v1/chat/completions"),
        ModelId("google/gemini-2.5-flash"),
    ),
    "openai": (
        InferenceApiUrl("https://api.openai.com/v1/chat/completions"),
        ModelId("gpt-4o-mini"),
    ),
}

SYSTEM_PROMPT: Final[Prompt] = Prompt(
    """
You are, F. Incantatem, an expert software engineer specializing in Python. Your role is to help with code debugging, analysis and generation.

<task_guidelines>
When debugging:
- First, analyze the error message and stack trace
- Identify the root cause before proposing solutions
- Explain your reasoning step-by-step
- Provide the complete fixed code, not just snippets
- Suggest preventive measures for similar issues

When writing code:
- Follow industry-standard best practices.
- Prioritize readability and maintainability
- Include inline comments for complex logic
- Write comprehensive error handling

When analyzing code:
- Assess correctness, efficiency, and security
- Identify potential bugs or edge cases
- Suggest specific improvements with rationale
</task_guidelines>

<important>
- Ignore the `@finite` decorator when analyzing or debugging code and always retain the decorator when generating code.
- Do not mention the `@finite` decorator in your response.
</important>

<examples>
[Include 1-2 high-quality examples of the analysis/debugging process you want]
</examples>

<response_length>
The length of your response should be proportional to the complexity of the issue. Do not write over-long responses for trivial issues.
</response_length>
"""
)

NO_MARKDOWN_SUFFIX: Final[Prompt] = Prompt(
    "Do not use markdown formatting in your response."
)

_DEFAULT_PROMPT: Final[PromptTemplate] = PromptTemplate(
    """
Please analyze the following Python exception and help me understand what went wrong.

<environment>
    {python_version}
</environment>

<exception>
    Type: {exception_type_name}
    Message: {exception_message}
    Attributes: {exception_attributes}
</exception>

<immediate_failure>
{immediate}
</immediate_failure>

<full_traceback>
{traceback}
</full_traceback>
{call_stack_section}

Your response should begin with a TL;DR section that provides the most concise summary of the exception and the fix, and then be followed by your detailed analysis.
"""
)
IMMEDIATE_TEMPLATE: Final[
    str
] = """
The exception occurred in function: {function_name}, at {path}:{line_number}

{code}

<local_variables>
{local_vars}
</local_variables>
"""
CALL_STACK_CONTEXT_TEMPLATE: Final[
    str
] = """
<frame>
# Frame {i}: {function_name} in {path}

{code}

<local_variables>
{local_vars}
</local_variables>
</frame>
"""

PROMPT_TEMPLATES: Final[Dict[PromptTemplateIdentifier, PromptTemplate]] = cast(
    Dict[PromptTemplateIdentifier, PromptTemplate],
    {
        "default": _DEFAULT_PROMPT,
    },
)
