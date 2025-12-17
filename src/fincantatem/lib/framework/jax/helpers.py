from typing import Any, Literal, Optional

from ....domain.values import (
    ArrayDtype,
    ArrayShape,
    DeviceName,
    FnName,
    TransformationType,
)
from .types import JaxArrayMetadata, PRNGKeyMetadata, TracerLevel


def extract_array_metadata(name: str, value: Any) -> Optional[JaxArrayMetadata]:
    """Extract metadata from a JAX array or tracer."""
    import jax

    # Check for JAX array types (includes both Array and Tracer)
    if not (hasattr(value, "shape") and hasattr(value, "dtype")):
        return None

    # Verify it's actually a JAX type (not numpy)
    type_module = type(value).__module__
    if not type_module.startswith("jax"):
        return None

    is_tracer = isinstance(value, jax.core.Tracer)  # ty:ignore[possibly-missing-attribute]
    tracer_level: Optional[TracerLevel] = None
    if is_tracer and hasattr(value, "level"):
        tracer_level = TracerLevel(value.level)

    # Get device - tracers may not have device()
    device_str = "unknown"
    if hasattr(value, "device") and callable(value.device):
        try:
            device_str = str(value.device())
        except Exception:
            device_str = "unavailable"
    elif hasattr(value, "devices"):
        try:
            devices = value.devices()
            device_str = str(devices) if devices else "unknown"
        except Exception:
            device_str = "unavailable"

    return JaxArrayMetadata(
        name=name,
        shape=ArrayShape(tuple(value.shape)),
        dtype=ArrayDtype(str(value.dtype)),
        device=DeviceName(device_str),
        is_tracer=is_tracer,
        tracer_level=tracer_level,
    )


def extract_prng_metadata(name: str, value: Any) -> Optional[PRNGKeyMetadata]:
    """Extract metadata from a JAX PRNG key."""

    if not (hasattr(value, "shape") and hasattr(value, "dtype")):
        return None

    # PRNGKey detection: check dtype and shape patterns
    # Modern JAX uses typed keys with specific dtypes
    dtype_str = str(value.dtype)

    # Check for PRNG key patterns
    is_prng_key = False
    if "key" in dtype_str.lower():
        # Typed PRNG keys (jax.random.key)
        is_prng_key = True
    elif dtype_str == "uint32" and value.shape == (2,):
        # Legacy PRNGKey format
        is_prng_key = True
    elif dtype_str == "uint32" and len(value.shape) == 2 and value.shape[-1] == 2:
        # Batched/split keys
        is_prng_key = True

    if not is_prng_key:
        return None

    # Heuristics for key vs subkey based on variable name
    name_lower = name.lower()
    if "sub" in name_lower:
        key_type: Literal["key", "subkey"] = "subkey"
    else:
        key_type = "key"

    # Check if key appears to have been split (has batch dimension)
    was_split = len(value.shape) > 1 or "split" in name_lower

    return PRNGKeyMetadata(
        name=name,
        key_type=key_type,
        was_split=was_split,
    )


def detect_transformation(
    function_name: FnName,
) -> Optional[TransformationType]:
    """Detect JAX transformation from function name in stack frame."""
    fn_lower = function_name.lower()

    # Check for transformation markers in function names
    # JAX internal names often contain these patterns
    # Map JAX-specific names to framework-agnostic TransformationType
    if "vmap" in fn_lower or "batched" in fn_lower:
        return "vectorize"
    elif "pmap" in fn_lower or "parallel" in fn_lower:
        return "parallelize"
    elif "jit" in fn_lower or "compiled" in fn_lower or "xla" in fn_lower:
        return "compile"
    elif "grad" in fn_lower or "vjp" in fn_lower or "jvp" in fn_lower:
        return "differentiate"

    return None


def is_tracing() -> bool:
    """Check if JAX is currently in a tracing context."""
    import jax

    # Check the tracer stack
    if hasattr(jax.core, "cur_sublevel"):  # type: ignore[attr-defined]
        try:
            sublevel = jax.core.cur_sublevel()  # type: ignore[attr-defined]
            return sublevel.level > 0
        except Exception:
            pass

    # Alternative: check trace_state
    if hasattr(jax, "_src") and hasattr(jax._src, "core"):
        try:
            trace_state = jax._src.core.thread_local_state.trace_state
            return len(trace_state.trace_stack.stack) > 1
        except Exception:
            pass

    return False
