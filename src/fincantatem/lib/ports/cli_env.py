# pyright: strict

import argparse
from typing import Optional, cast

from ...domain.ports import CLIEnv as DomainCLIEnv
from ...domain.aggs import Invocation, InferenceSettings
from ...domain.values import (
    ApiKey,
    InferenceApiIdentifier,
    InferenceApiUrl,
    ModelId,
    PresetIdentifier,
)
from ...domain.constants import INFERENCE_PRESETS
from ...lib.utils import pipe
from ...lib.constants import *
import os


class CLIEnv(DomainCLIEnv):
    def __init__(self, cautious: bool = False):
        self.cautious = cautious

    def read_args(self) -> Invocation:
        parser = argparse.ArgumentParser(
            prog="F.Incantatem",
        )
        parser.add_argument(
            "filename",
            type=str,
            help="The python file to run with F.Incantatem.",
        )
        parser.add_argument(
            "-p",
            "--preset",
            type=str,
            default="openrouter",
            help="The preset to use for the inference API.",
        )
        parser.add_argument(
            "-s",
            "--snippets",
            type=bool,
            default=True,
            help="Whether to include snippets of your code in the prompt or the full source code.",
        )
        parser.add_argument(
            "-l",
            "--locals",
            type=bool,
            default=False,
            help="Whether to include the local variables in the prompt.",
        )
        parser.add_argument(
            "-c",
            "--cautious",
            type=bool,
            default=False,
            help="Enable cautious mode to attempt redacting secrets and PII from the prompt.",
        )
        args = parser.parse_args()
        return Invocation(
            filename=args.filename,
            preset=args.preset,
            snippets=args.snippets,
            locals=args.locals,
            cautious=args.cautious,
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

        # CLI/decorator args take priority over environment variables
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
