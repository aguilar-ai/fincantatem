# pyright: strict

from ...domain.ports import DecoratorEnv as DomainDecoratorEnv
from ...domain.aggs import Invocation, InferenceSettings
from ...domain.values import (
    ApiKey,
    InferenceApiIdentifier,
    InferenceApiUrl,
    ModelId,
    PresetIdentifier,
)
from ...domain.constants import INFERENCE_PRESETS
from typing import TypedDict, Optional, Unpack, cast
from ...lib.utils import pipe
from ...lib.constants import *
import os

Kwargs = TypedDict(
    "Kwargs",
    {
        "filename": str,
        "preset": str,
        "snippets": bool,
        "locals": bool,
    },
)


class DecoratorEnv(DomainDecoratorEnv):
    def read_args(self, **kwargs: Unpack[Kwargs]) -> Invocation:
        # Priority 1: Command-line arguments
        filename_cli, preset_cli, snippets_cli, locals_cli, cautious_cli = pipe(
            ["filename", "preset", "snippets", "locals", "cautious"],
            lambda kw: map(lambda k: kwargs.get(k), kw),
            tuple,
        )

        return Invocation(
            filename=filename_cli,
            preset=preset_cli,
            snippets=snippets_cli,
            locals=locals_cli,
            cautious=cautious_cli,
        )

    def read_env(self, preset: Optional[str] = None) -> InferenceSettings:
        api_url, api_key, model, env_preset = pipe(
            [
                INFERENCE_API_URL_ENV_VAR,
                INFERENCE_API_KEY_ENV_VAR,
                INFERENCE_MODEL_ENV_VAR,
                INFERENCE_PRESET_ENV_VAR,
            ],
            lambda env_vars: map(lambda env_var: os.getenv(env_var), env_vars),
            tuple,
        )

        # Decorator args take priority over environment variables
        effective_preset = preset or env_preset

        # If nothing specified, default to openrouter.
        if effective_preset is None:
            effective_preset = "openrouter"

        # Custom endpoint support: (FI_PRESET is treated as an identifier label)
        if effective_preset not in INFERENCE_PRESETS.keys():
            if api_url is None:
                raise ValueError(
                    f"Custom preset '{effective_preset}' requires {INFERENCE_API_URL_ENV_VAR}."
                )
            return InferenceSettings.custom(
                identifier=InferenceApiIdentifier(effective_preset),
                url=InferenceApiUrl(api_url),
                model=ModelId(model) if model is not None else None,
                api_key=ApiKey(api_key) if api_key is not None else None,
            )

        # Preset support, but allow env overrides for url/model/api_key.
        settings = InferenceSettings.preset(cast(PresetIdentifier, effective_preset))
        if api_url is not None:
            settings.url = InferenceApiUrl(api_url)
        if model is not None:
            settings.model = ModelId(model)
        if api_key is not None:
            settings.api_key = ApiKey(api_key)

        return settings
