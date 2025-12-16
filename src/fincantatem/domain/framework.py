from typing import List, Protocol, TypeVar, runtime_checkable, Optional
from .values import *
from .aggs import SourceCodeBundle

T = TypeVar("T", covariant=True)


@runtime_checkable
class Framework(Protocol[T]):
    def __init__(self, imports: List[FrameworkName]): ...

    @property
    def name(self) -> FrameworkName: ...

    @staticmethod
    def detect() -> bool: ...

    def extract_context(
        self, source_code_bundles: List[SourceCodeBundle]
    ) -> Optional[T]: ...
