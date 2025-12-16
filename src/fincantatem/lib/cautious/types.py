from dataclasses import dataclass

from ...domain.values import LineNumberOffset, Secret


@dataclass(frozen=True)
class SecretFinding:
    type: str
    secret_value: Secret
    is_verified: bool
    line_number: LineNumberOffset
