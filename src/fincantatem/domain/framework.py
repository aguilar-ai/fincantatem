from typing import Optional, Protocol, Sequence, runtime_checkable

from .aggs import S, SourceCodeBundle
from .values import FrameworkName


@runtime_checkable
class Framework(Protocol[S]):
    @property
    def name(self) -> FrameworkName: ...

    @staticmethod
    def detect() -> bool: ...

    def extract_context(
        self, source_code_bundles: Sequence[SourceCodeBundle]
    ) -> Optional[S]: ...
