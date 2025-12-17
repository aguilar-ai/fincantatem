from importlib.util import find_spec
from typing import List, Sequence

from ....domain.aggs import SourceCodeBundle
from ....domain.framework import Framework
from ....domain.values import FrameworkName
from .helpers import (
    detect_transformation,
    extract_array_metadata,
    extract_prng_metadata,
    is_tracing,
)
from .types import JaxArrayMetadata, JaxContext, JaxFrameContext, PRNGKeyMetadata


class JaxFramework(Framework[JaxContext]):
    @property
    def name(self) -> FrameworkName:
        return FrameworkName("jax")

    @staticmethod
    def detect() -> bool:
        return find_spec("jax") is not None

    def extract_context(
        self, source_code_bundles: Sequence[SourceCodeBundle]
    ) -> JaxContext:
        """Extract JAX-specific context from exception stack frames."""
        import jax

        _jax_version = jax.__version__
        _device_count = len(jax.devices())
        _default_backend = jax.default_backend()

        frame_contexts: List[JaxFrameContext] = []

        for bundle in source_code_bundles:
            arrays: List[JaxArrayMetadata] = []
            prng_keys: List[PRNGKeyMetadata] = []

            if bundle.local_vars:
                for var_name, var_value in bundle.local_vars.items():
                    if var_name.startswith("__"):
                        continue

                    prng_meta = extract_prng_metadata(var_name, var_value)
                    if prng_meta is not None:
                        prng_keys.append(prng_meta)
                        continue

                    # Extract arrays, including those nested in containers (tuples/lists)
                    array_metas = extract_array_metadata(var_name, var_value)
                    if array_metas is not None:
                        arrays.append(array_metas)

            transformation = detect_transformation(bundle.function_name)

            frame_contexts.append(
                JaxFrameContext(
                    arrays=arrays,
                    prng_keys=prng_keys,
                    transformation_context=transformation,
                    is_tracing=is_tracing(),
                )
            )

        return JaxContext(
            name=self.name,
            version=_jax_version,
            frame_contexts=frame_contexts,
            relevant_documentation=[],
            device_count=_device_count,
            default_backend=_default_backend,
        )
