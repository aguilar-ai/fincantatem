from ...domain.errors import RedactionFailedError
from ...domain.ports import FileSystem as DomainFileSystem
from ...domain.values import (
    LineNumberOffset,
    SourceCode,
    SourceCodePath,
    SourceCodeSnippet,
)


def _redaction_routine(source_code: SourceCode) -> SourceCode:
    from ...lib.cautious.secrets import scan_secrets, strip_secrets
    from ...lib.cautious.pii import strip_pii

    secrets = scan_secrets(source_code)
    source_code = strip_secrets(source_code, secrets)
    source_code = strip_pii(source_code)
    return source_code


class FileSystem(DomainFileSystem):
    def __init__(self, cautious: bool = False):
        self.cautious = cautious

    def fetch_source_code_from_path(self, path: SourceCodePath) -> SourceCode:
        path_str = str(path)

        # Strategy 1: direct filesystem read
        try:
            with open(path_str, "r", encoding="utf-8") as f:
                return SourceCode(f.read())
        except OSError:
            pass

        # Strategy 2: linecache (works for some non-filesystem sources too)
        import linecache

        cached_lines = linecache.getlines(path_str)
        if cached_lines:
            if self.cautious:
                try:
                    source_code = SourceCode("".join(cached_lines))
                    source_code = _redaction_routine(source_code)
                except Exception as e:
                    raise RedactionFailedError(
                        "You've enabled cautious mode, but we failed to redact secrets and PII from the source code."
                    ) from e

            return SourceCode("".join(cached_lines))
        raise FileNotFoundError(f"Could not fetch source code for {path_str}")

    def fetch_source_code_snippet_from_path(
        self,
        path: SourceCodePath,
        line_number_offset: LineNumberOffset,
        *,
        before: int,
        after: int,
    ) -> SourceCodeSnippet:
        path_str = str(path)

        lines: list[str] = []

        # Strategy 1: direct filesystem read
        try:
            with open(path_str, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            lines = []

        # Strategy 2: linecache fallback
        if not lines:
            import linecache

            lines = linecache.getlines(path_str)

        if not lines:
            raise FileNotFoundError(f"Could not fetch source snippet for {path_str}")

        idx = int(line_number_offset) - 1
        start = max(0, idx - before)
        end = min(len(lines), idx + after + 1)
        if self.cautious:
            try:
                source_code = SourceCode("".join(lines[start:end]))
                source_code = _redaction_routine(source_code)
            except Exception as e:
                raise RedactionFailedError(
                    "You've enabled cautious mode, but we failed to redact secrets and PII from the source code snippet."
                ) from e
        return SourceCodeSnippet("".join(lines[start:end]))

    def fetch_source_code_from_frame(
        self, frame: object
    ) -> tuple[SourceCode, LineNumberOffset]:
        import inspect

        source_lines, start_line = inspect.getsourcelines(frame)  # type: ignore
        if self.cautious:
            try:
                source_code = SourceCode("".join(source_lines))
                source_code = _redaction_routine(source_code)
            except Exception as e:
                raise RedactionFailedError(
                    "You've enabled cautious mode, but we failed to redact secrets and PII from the source code from the frame."
                ) from e
        return SourceCode("".join(source_lines)), LineNumberOffset(start_line)

    def fetch_source_code_snippet_from_frame(self, frame: object) -> SourceCodeSnippet:
        import inspect

        source_lines, _ = inspect.getsourcelines(frame)  # type: ignore
        if self.cautious:
            try:
                source_code = SourceCode("".join(source_lines))
                source_code = _redaction_routine(source_code)
            except Exception as e:
                raise RedactionFailedError(
                    "You've enabled cautious mode, but we failed to redact secrets and PII from the source code snippet from the frame."
                ) from e
        return SourceCodeSnippet("".join(source_lines))
