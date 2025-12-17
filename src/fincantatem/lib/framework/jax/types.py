import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Literal, NewType, Optional, Sequence, Tuple, cast

from ....domain.aggs import (
    ArrayMetadata,
    FrameContext,
    FrameworkContext,
    StringableContext,
    TransformationType,
)
from ....domain.values import (
    Documentation,
    FnName,
)

TracerLevel = NewType("TracerLevel", int)


@dataclass(frozen=True)
class JaxArrayMetadata(ArrayMetadata):
    """Enriched metadata for a JAX array variable, extends domain ArrayMetadata"""

    is_tracer: bool
    tracer_level: Optional[TracerLevel]


@dataclass(frozen=True)
class PRNGKeyMetadata:
    """Metadata for a JAX PRNGKey - JAX-specific concept"""

    name: str
    key_type: Literal["key", "subkey"]
    was_split: bool


@dataclass(frozen=True)
class JaxFrameContext(FrameContext):
    """Framework-specific context for a single stack frame in JAX code.

    Extends domain FrameContext with JAX-specific fields and types.
    """

    arrays: Sequence[JaxArrayMetadata]
    prng_keys: List[PRNGKeyMetadata]
    transformation_context: Optional[TransformationType]
    is_tracing: bool


@dataclass(frozen=True)
class JaxContext(FrameworkContext, StringableContext):
    """Complete JAX framework context for exception analysis.

    Extends domain FrameworkContext with JAX-specific fields.
    """

    relevant_documentation: List[Tuple[FnName, Documentation]]
    device_count: int
    default_backend: str

    def frame_context_string(self, index: int) -> Optional[str]:
        """Render framework context for a single frame at the given index."""
        if index < 0 or index >= len(self.frame_contexts):
            return None

        # Cast to JaxFrameContext since JaxContext always stores JaxFrameContext instances
        ctx = cast(JaxFrameContext, self.frame_contexts[index])

        # Skip frames with no relevant context
        if not ctx.arrays and not ctx.prng_keys and not ctx.transformation_context:
            return None

        frame = ET.Element("jax_frame_context")
        frame.set("version", self.version)
        frame.set("default_backend", self.default_backend)

        if ctx.is_tracing:
            frame.set("is_tracing", "true")

        for array in ctx.arrays:
            array_elem = ET.SubElement(frame, "array")
            array_elem.set("name", array.name)
            array_elem.set("shape", str(array.shape))
            array_elem.set("dtype", array.dtype)
            array_elem.set("device", array.device)
            if array.is_tracer:
                array_elem.set("is_tracer", "true")
                if array.tracer_level is not None:
                    array_elem.set("tracer_level", str(array.tracer_level))

        for prng_key in ctx.prng_keys:
            prng_elem = ET.SubElement(frame, "prng_key")
            prng_elem.set("name", prng_key.name)
            prng_elem.set("key_type", prng_key.key_type)
            if prng_key.was_split:
                prng_elem.set("was_split", "true")

        if ctx.transformation_context:
            transformation_elem = ET.SubElement(frame, "transformation_context")
            transformation_elem.text = ctx.transformation_context

        ET.indent(frame, space="  ")
        return ET.tostring(frame, encoding="utf-8").decode("utf-8")
