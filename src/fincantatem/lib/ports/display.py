from typing import Iterator, Optional, Tuple, TypedDict, Literal, Unpack, List
import re

from rich.console import RenderableType

from ...domain.values import Response
from ...lib.utils import pipe
from ...domain.ports import Interface as DomainInterface


class DisplaySettings(TypedDict, total=False):
    box_style: Literal["single", "double", "rounded"]
    color: Literal[
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "bright_yellow",
        "bright_red",
        "bright_green",
        "bright_magenta",
    ]
    title: str
    markdown_disabled: bool
    padding: Tuple[int, int]
    collapsible: bool  # For traceback: show condensed version by default


# NOTE: LLM-written code
def _plain_make_box(content: str, title: Optional[str] = None, width: int = 70) -> str:
    """
    Create a simple ASCII box around content.
    Dependency-free alternative to Rich panels.
    """
    lines = content.split("\n")

    # Build the box
    top_border = "+" + "-" * (width - 2) + "+"
    if title:
        # Center the title in the top border
        title_str = f" {title} "
        padding = (width - 2 - len(title_str)) // 2
        top_border = (
            "+"
            + "-" * padding
            + title_str
            + "-" * (width - 2 - padding - len(title_str))
            + "+"
        )

    bottom_border = "+" + "-" * (width - 2) + "+"

    boxed_lines = [top_border]
    for line in lines:
        # Truncate or pad line to fit
        if len(line) > width - 4:
            line = line[: width - 7] + "..."
        boxed_lines.append(f"| {line:<{width - 4}} |")
    boxed_lines.append(bottom_border)

    return "\n".join(boxed_lines)


class PlainTextInterface(DomainInterface):
    @staticmethod
    def is_available() -> bool:
        return True

    def display(self, message: str, **kwargs: Unpack[DisplaySettings]) -> None:
        title = kwargs.get("title") if kwargs else None
        is_traceback = (
            title == "Traceback" or "Traceback (most recent call last)" in message
        )

        # Handle collapsible traceback
        if is_traceback and kwargs and kwargs.get("collapsible", False):
            condensed, _ = _extract_condensed_traceback(message)
            message = condensed
            # Add hint about full traceback
            message += "\n  (Use --full-traceback to see all frames)"

        # If we have display settings, use a box
        if kwargs and (title or kwargs.get("box_style")):
            print(_plain_make_box(message, title))
        else:
            print(message)

        # Add visual separation after boxed content
        if kwargs and (title or kwargs.get("box_style")):
            print()

    def display_stream(
        self, chunks: Iterator[str], **kwargs: Unpack[DisplaySettings]
    ) -> Response:
        title = kwargs.get("title") if kwargs else None

        # Show a simple progress indicator
        print("  * Analyzing...", end="", flush=True)

        full_text = ""
        for chunk in chunks:
            full_text += chunk

        # Clear the progress indicator and show result
        print("\r" + " " * 20 + "\r", end="")

        # Print the result in a box if we have settings
        if kwargs and title:
            print(_plain_make_box(full_text, title))
        else:
            print(full_text)

        # Add visual separation
        print()

        return Response(full_text)

    def prompt(self, prompt: str, **kwargs: Unpack[DisplaySettings]) -> Optional[str]:
        label = kwargs.get("title", prompt) if kwargs else prompt
        if label:
            print(f"+-- {label} --+")
            user_input = input("> ")
        else:
            user_input = input("> " if not prompt else f"{prompt}: ")

        if user_input in ["/quit", "/q"]:
            return None
        return user_input


def _make_panel(content: RenderableType, **kwargs: Unpack[DisplaySettings]):
    from rich.panel import Panel
    from rich import box

    box_map = {"single": box.SQUARE, "double": box.DOUBLE, "rounded": box.ROUNDED}

    if kwargs:
        box = pipe(
            kwargs.get("box_style"),
            box_map.get,
            lambda b: b or box.ROUNDED,
        )
        return Panel(
            content,
            border_style=kwargs.get("color") or "white",
            box=box,
            title=kwargs.get("title"),
        )
    else:
        return Panel(content)


# NOTE: LLM-written code
def _extract_condensed_traceback(traceback_text: str) -> Tuple[str, str]:
    """
    Extract a condensed version of the traceback showing only the last frame.
    Returns (condensed_traceback, full_traceback).
    """
    lines = traceback_text.strip().split("\n")

    # Find all "File " lines (frame starts)
    frame_starts: List[int] = []
    for i, line in enumerate(lines):
        if line.strip().startswith("File "):
            frame_starts.append(i)

    if len(frame_starts) <= 1:
        # Only one frame or none, return as-is
        return (traceback_text, traceback_text)

    # Get the header (first line, usually "Traceback (most recent call last):")
    header_end = frame_starts[0]
    header = "\n".join(lines[:header_end])

    # Get the last frame (from last "File " to the exception line)
    last_frame_start = frame_starts[-1]

    # Find the exception line (typically starts with ExceptionName:)
    exception_line_idx = len(lines) - 1
    for i in range(last_frame_start, len(lines)):
        if re.match(r"^\w+Error:|^\w+Exception:|^\w+Warning:", lines[i]):
            exception_line_idx = i
            break

    # Build condensed version
    last_frame = "\n".join(lines[last_frame_start : exception_line_idx + 1])
    frames_hidden = len(frame_starts) - 1

    condensed_parts = [header]
    if frames_hidden > 0:
        condensed_parts.append(
            f"  ... ({frames_hidden} frames hidden, see full traceback)"
        )
    condensed_parts.append(last_frame)

    condensed = "\n".join(condensed_parts)

    return (condensed, traceback_text)


# NOTE: LLM-written code
def _render_enhanced_traceback(traceback_text: str) -> RenderableType:
    """
    Render a traceback with syntax highlighting and emphasis on error lines.
    This is a display-layer enhancement only - no domain logic.
    """
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.console import Group

    lines = traceback_text.strip().split("\n")
    parts: List[RenderableType] = []
    code_block: List[str] = []
    in_code_block = False

    for i, line in enumerate(lines):
        # Detect the start of a file reference
        if line.strip().startswith("File "):
            # Flush any pending code block
            if code_block:
                code_text = "\n".join(code_block)
                parts.append(
                    Syntax(code_text, "python", theme="monokai", line_numbers=False)
                )
                code_block = []

            # File reference line - style it
            text = Text(line)
            text.stylize("dim cyan")
            parts.append(text)
            in_code_block = True

        elif (
            in_code_block
            and line.strip()
            and not line.strip().startswith(("Traceback", "File ", ">>>"))
        ):
            # This is a code line within the traceback
            # Check if next line is an error indicator (^) or this is the error line
            is_error_line = i + 1 < len(lines) and lines[i + 1].strip().startswith("^")

            if is_error_line or ">>>" in line:
                # Highlight the error line with a background
                text = Text(line)
                text.stylize("bold white on red")
                parts.append(text)
            else:
                code_block.append(line)

        elif line.strip().startswith("^"):
            # Caret line pointing to error - style it
            text = Text(line)
            text.stylize("bold red")
            parts.append(text)

        elif line.strip().startswith(("Traceback", "During handling")):
            # Flush code block
            if code_block:
                code_text = "\n".join(code_block)
                parts.append(
                    Syntax(code_text, "python", theme="monokai", line_numbers=False)
                )
                code_block = []
            in_code_block = False

            text = Text(line)
            text.stylize("dim")
            parts.append(text)

        elif re.match(r"^\w+Error:|^\w+Exception:|^\w+Warning:", line):
            # Flush code block
            if code_block:
                code_text = "\n".join(code_block)
                parts.append(
                    Syntax(code_text, "python", theme="monokai", line_numbers=False)
                )
                code_block = []
            in_code_block = False

            # Exception type and message - make it prominent
            text = Text(line)
            text.stylize("bold red")
            parts.append(text)

        else:
            # Other lines - keep as-is
            if code_block:
                code_block.append(line)
            else:
                parts.append(Text(line))

    # Flush any remaining code block
    if code_block:
        code_text = "\n".join(code_block)
        parts.append(Syntax(code_text, "python", theme="monokai", line_numbers=False))

    return Group(*parts)


class RichTextInterface(DomainInterface):
    @staticmethod
    def is_available() -> bool:
        try:
            from rich.console import Console  # type: ignore
            from rich.markdown import Markdown  # type: ignore

            return True
        except ImportError:
            return False

    def display(self, message: str, **kwargs: Unpack[DisplaySettings]) -> None:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.text import Text

        console = Console()

        if kwargs and kwargs.get("markdown_disabled"):
            # Check if this looks like a traceback for enhanced rendering
            title = kwargs.get("title", "")
            is_traceback = (
                title == "Traceback" or "Traceback (most recent call last)" in message
            )

            if is_traceback:
                # Check if collapsible mode is enabled
                if kwargs.get("collapsible", False):
                    condensed, _ = _extract_condensed_traceback(message)
                    enhanced = _render_enhanced_traceback(condensed)
                    console.print(_make_panel(enhanced, **kwargs))

                    # Add hint about viewing full traceback
                    hint = Text("  Use --full-traceback to see all frames", style="dim")
                    console.print(hint)
                else:
                    enhanced = _render_enhanced_traceback(message)
                    console.print(_make_panel(enhanced, **kwargs))

                # Add visual separation after traceback
                console.print()
            else:
                console.print(_make_panel(message, **kwargs))
        elif kwargs and not kwargs.get("markdown_enabled"):
            md = Markdown(message)
            console.print(_make_panel(md, **kwargs))
        else:
            console.print(message)

    def display_stream(
        self, chunks: Iterator[str], **kwargs: Unpack[DisplaySettings]
    ) -> Response:
        from rich.markdown import Markdown
        from rich.console import Console
        from rich.status import Status

        console = Console()
        full_text = ""

        # Show a subtle status indicator while streaming
        # This preserves scrollback history unlike Live
        with Status(
            "  ✧ Analyzing...",
            console=console,
            spinner="dots",
            spinner_style="bright_yellow",
        ):
            for chunk in chunks:
                full_text += chunk

        # Now print the final formatted result (this goes into scrollback)
        md = Markdown(full_text)
        content = _make_panel(md, **kwargs) if kwargs else md
        console.print(content)

        # Add visual separation after analysis
        console.print()

        return Response(full_text)

    def prompt(self, prompt: str, **kwargs: Unpack[DisplaySettings]) -> Optional[str]:
        from rich.prompt import Prompt as RichPrompt
        from rich.console import Console
        from rich.text import Text

        console = Console()

        if kwargs:
            color = kwargs.get("color", "white")
            title = kwargs.get("title", "")
            header = Text()
            header.append("╭─ ", style=f"bold {color}")
            header.append(title or prompt, style=f"bold {color}")
            console.print(header)
            user_input = console.input(Text("╰─▶ ", style=f"bold {color}"))
        else:
            user_input = RichPrompt.ask(prompt)

        if user_input in ["/quit", "/q"]:
            return None
        return user_input
