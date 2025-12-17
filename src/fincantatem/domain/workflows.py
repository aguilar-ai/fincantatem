import json
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence

from .aggs import ExceptionContext, SourceCodeBundle, StringableContext
from .constants import CALL_STACK_CONTEXT_TEMPLATE, IMMEDIATE_TEMPLATE, PROMPT_TEMPLATES
from .errors import FileSystemAccessError
from .framework import Framework
from .ports import FileSystem
from .values import (
    Cause,
    Context,
    ExceptionAttributes,
    ExceptionMessage,
    ExceptionTypeName,
    FnName,
    LineNumberOffset,
    Prompt,
    PromptTemplateIdentifier,
    PromptVar,
    PythonVersion,
    SourceCodePath,
    Traceback,
)

SNIPPET_BEFORE: int = 3
SNIPPET_AFTER: int = 3


def fetch_source_code_bundle(
    e: Exception, fs: FileSystem, frameworks: Sequence[Framework[StringableContext]]
) -> List[SourceCodeBundle]:
    tb = e.__traceback__
    bundle: List[SourceCodeBundle] = []
    while tb:
        frame = tb.tb_frame
        lineno = tb.tb_lineno
        function_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        local_vars = frame.f_locals

        is_wrapper_frame = function_name == "wrapper" and filename.endswith(
            "__init__.py"
        )
        is_ipython_frame = "/IPython/core/interactiveshell.py" in filename
        if is_wrapper_frame or is_ipython_frame:
            tb = tb.tb_next
            continue

        try:
            source_code = fs.fetch_source_code_from_path(SourceCodePath(filename))
            snippet = fs.fetch_source_code_snippet_from_path(
                SourceCodePath(filename),
                LineNumberOffset(lineno),
                before=SNIPPET_BEFORE,
                after=SNIPPET_AFTER,
            )
            code_start = LineNumberOffset(1)
        except Exception as e:
            try:
                source_code, code_start = fs.fetch_source_code_from_frame(frame)
                snippet = fs.fetch_source_code_snippet_from_frame(frame)
            except Exception as inner:
                raise FileSystemAccessError(
                    f"Failed to fetch source code for frame {frame}"
                ) from inner

        bundle.append(
            SourceCodeBundle(
                path=SourceCodePath(filename),
                code=source_code,
                snippet=snippet,
                line_number_offset=LineNumberOffset(lineno),
                code_start_line_number_offset=code_start,
                function_name=FnName(function_name),
                local_vars=local_vars,
            )
        )

        tb = tb.tb_next

    return bundle


def build_exception_context(
    exception: Exception,
    file_system: FileSystem,
    frameworks: Sequence[Framework[StringableContext]],
) -> ExceptionContext:
    python_version = PythonVersion(sys.version.split(" ")[0])
    exception_type_name = ExceptionTypeName(type(exception).__name__)
    exception_message = ExceptionMessage(str(exception))
    traceback_str = Traceback(traceback.format_exc(limit=None))
    cause = (
        Cause(repr(exception.__cause__)) if exception.__cause__ is not None else None
    )
    context = (
        Context(repr(exception.__context__))
        if exception.__context__ is not None
        else None
    )

    exception_attributes = None
    if hasattr(exception, "__dict__"):
        exception_attributes = ExceptionAttributes(exception.__dict__)
    elif hasattr(exception, "__slots__"):
        exception_attributes = ExceptionAttributes(
            {
                slot: getattr(exception, slot)
                for slot in getattr(exception, "__slots__", [])
            }
        )
    if exception_attributes is not None and len(exception_attributes) == 0:
        exception_attributes = None

    source_code_bundles = fetch_source_code_bundle(exception, file_system, frameworks)

    return ExceptionContext(
        python_version=python_version,
        exception_type_name=exception_type_name,
        exception_message=exception_message,
        exception_attributes=exception_attributes,
        traceback=traceback_str,
        cause=cause,
        context=context,
        immediate_source_code_bundle=source_code_bundles[-1],
        source_code_bundles=source_code_bundles[:-1],
        framework_context=[
            ctx
            for framework in frameworks
            if (ctx := framework.extract_context(source_code_bundles)) is not None
        ],
    )


def build_prompt(
    context: ExceptionContext,
    template_identifier: PromptTemplateIdentifier,
    *,
    snippets: bool,
) -> Prompt:
    def _render_code_block(
        *,
        b: SourceCodeBundle,
        start_line_number_offset: LineNumberOffset,
        code: str,
    ) -> str:
        header = (
            f"{b.path}:{int(b.line_number_offset)} ({b.function_name})\n```python\n"
        )
        out = header
        for idx, line in enumerate(code.split("\n")):
            line_no = int(start_line_number_offset) + idx
            marker = ">>> " if line_no == int(b.line_number_offset) else "    "
            out += f"{marker}{line_no:4d}: {line}\n"
        out += "```\n"
        return out

    def _render_snippet(b: SourceCodeBundle) -> str:
        snippet_start = (
            b.code_start_line_number_offset
            if int(b.code_start_line_number_offset) != 1
            else LineNumberOffset(max(1, int(b.line_number_offset) - SNIPPET_BEFORE))
        )
        return _render_code_block(
            b=b, start_line_number_offset=snippet_start, code=str(b.snippet)
        )

    def _render_full_source(b: SourceCodeBundle) -> str:
        return _render_code_block(
            b=b,
            start_line_number_offset=b.code_start_line_number_offset,
            code=str(b.code),
        )

    def format_locals(local_vars: Optional[Dict[str, Any]]) -> str:
        """Format local variables for display."""
        if not local_vars:
            return "No local variables captured"

        lines: List[str] = []
        for name, value in local_vars.items():
            value_str = repr(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            lines.append(f"  {name} = {value_str}")

        return "\n".join(lines)

    # NOTE: This code is generated by Claude Opus 4
    def get_framework_info_for_frame(frame_index: int) -> str:
        """Get combined framework context string for a given frame index."""
        framework_info_parts: List[str] = []
        for fw_ctx in context.framework_context:
            frame_str = fw_ctx.frame_context_string(frame_index)
            if frame_str:
                framework_info_parts.append(frame_str)
        return "\n".join(framework_info_parts) if framework_info_parts else ""

    # NOTE: This code is generated by Claude Opus 4
    def render_call_stack_frame(display_index: int, b: SourceCodeBundle) -> str:
        # Convert reversed display index back to original frame index
        original_frame_index = len(context.source_code_bundles) - 1 - display_index
        return CALL_STACK_CONTEXT_TEMPLATE.format(
            i=display_index,
            function_name=b.function_name,
            path=b.path,
            code=_render_snippet(b) if snippets else _render_full_source(b),
            local_vars=format_locals(b.local_vars),
            framework_info=get_framework_info_for_frame(original_frame_index),
        )

    # The immediate frame is the last frame in the original source_code_bundles list
    # context.source_code_bundles excludes the last frame, so its length equals the immediate frame's index
    immediate_frame_index = len(context.source_code_bundles)

    template = PROMPT_TEMPLATES[template_identifier]
    values: Dict[PromptVar, str] = {
        "python_version": context.python_version,
        "exception_type_name": context.exception_type_name,
        "exception_message": context.exception_message,
        "exception_attributes": (
            json.dumps(context.exception_attributes, indent=2, sort_keys=True)
            if context.exception_attributes is not None
            else ""
        ),
        "immediate": IMMEDIATE_TEMPLATE.format(
            function_name=context.immediate_source_code_bundle.function_name,
            path=context.immediate_source_code_bundle.path,
            line_number=context.immediate_source_code_bundle.line_number_offset,
            code=(
                _render_snippet(context.immediate_source_code_bundle)
                if snippets
                else _render_full_source(context.immediate_source_code_bundle)
            ),
            local_vars=format_locals(context.immediate_source_code_bundle.local_vars),
            framework_info=get_framework_info_for_frame(immediate_frame_index),
        ),
        "traceback": context.traceback,
    }

    call_stack_frames = "\n".join(
        [
            render_call_stack_frame(i, b)
            for i, b in enumerate(reversed(context.source_code_bundles))
        ]
    )
    if call_stack_frames:
        values["call_stack_section"] = (
            f"\n<call_stack_context>\n{call_stack_frames}\n</call_stack_context>\n"
        )
    else:
        values["call_stack_section"] = ""

    try:
        return Prompt(str(template).format(**values))
    except Exception:
        raise
