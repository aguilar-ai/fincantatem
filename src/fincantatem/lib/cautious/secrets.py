import hashlib
import tempfile
from typing import List

from bip_utils import Bip39Languages, Bip39MnemonicGenerator  # pyright: ignore
from detect_secrets.core.secrets_collection import SecretsCollection
from detect_secrets.settings import default_settings

from ...domain.values import LineNumberOffset, RedactedSecret, Secret, SourceCode
from ...lib.utils import pipe
from .types import SecretFinding


def redaction_policy(secret: Secret) -> RedactedSecret:
    h: bytes = pipe(secret.encode("utf-8"), hashlib.sha256, lambda h: h.digest())
    mn24 = Bip39MnemonicGenerator(Bip39Languages.ENGLISH).FromEntropy(h)
    mnemonic = pipe(
        mn24.ToList(),
        lambda x: map(lambda y: y.upper(), x),
        list,
    )

    return RedactedSecret(f"<REDACTED_SECRET_{mnemonic[0]}_{mnemonic[12]}>")


def scan_secrets(source_code: SourceCode) -> List[SecretFinding]:
    secrets = SecretsCollection()
    with (
        default_settings(),
        tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as temp_file,
    ):
        temp_file.write(source_code)
        temp_file.flush()
        secrets.scan_file(temp_file.name)

        return [
            SecretFinding(
                type=v.type,
                secret_value=Secret(v.secret_value),
                is_verified=v.is_verified,
                line_number=LineNumberOffset(v.line_number),
            )
            for _, v in secrets
            if v.secret_value is not None
        ]


def strip_secrets(source_code: SourceCode, secrets: List[SecretFinding]) -> SourceCode:
    for s in secrets:
        source_code = pipe(
            s.secret_value,
            redaction_policy,
            lambda r: source_code.replace(s.secret_value, r),
        )

    return SourceCode(source_code)
