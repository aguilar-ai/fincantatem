import json
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any, Dict, Final, Iterator, List, Optional

from ...domain.aggs import InferenceSettings, Message
from ...domain.ports import InferenceApi as DomainInferenceApi
from ...domain.values import ApiKey, ModelId, Prompt, Response
from ...lib.utils import get_in, pipe


def _extract_inference_response_content(
    r: urllib.request.Request,
) -> str:
    with urllib.request.urlopen(r) as resp:
        raw: str = resp.read().decode("utf-8")
        json_body = json.loads(raw)
        content: Optional[str] = get_in(["choices", 0, "message", "content"], json_body)

        if content is None:
            raise RuntimeError(f"Unexpected inference API response shape: {json_body}")
        return content


_DEFAULT_HEADERS: Final[Dict[str, str]] = {"Content-Type": "application/json"}


def _build_headers(api_key: Optional[ApiKey]) -> Dict[str, str]:
    return {
        **_DEFAULT_HEADERS,
        **({"Authorization": f"Bearer {api_key}"} if api_key is not None else {}),
    }


def _build_payload(
    model: ModelId, messages: List[Message[Prompt | Response]], stream: bool = False
) -> Dict[str, Any]:
    payload = {
        "model": str(model),
        "messages": [asdict(m) for m in messages],
    }
    if stream:
        payload["stream"] = stream
    return payload


def _handle_http_error(e: urllib.error.HTTPError) -> None:
    body = ""
    try:
        body = e.read().decode("utf-8")
    except Exception:
        body = ""
    raise RuntimeError(
        f"Inference API HTTP error {e.code} {e.reason}: {body or '<no body>'}"
    ) from e


class InferenceApi(DomainInferenceApi):
    def call(
        self,
        settings: InferenceSettings,
        prompt: Prompt,
        system_prompt: Optional[Prompt] = None,
    ) -> Response:
        if settings.model is None:
            raise ValueError("InferenceSettings.model is required")

        payload = pipe(
            _build_payload(settings.model, [Message(role="user", content=prompt)]),
            json.dumps,
            lambda p: p.encode("utf-8"),
        )

        headers = _build_headers(settings.api_key)

        req = urllib.request.Request(
            str(settings.url),
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            content = _extract_inference_response_content(req)
        except urllib.error.HTTPError as e:
            _handle_http_error(e)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Inference API request failed: {e.reason}") from e

        return Response(content)

    def call_stream(
        self, settings: InferenceSettings, messages: List[Message[Prompt | Response]]
    ) -> Iterator[str]:
        if settings.model is None:
            raise ValueError("InferenceSettings.model is required")

        payload = pipe(
            _build_payload(settings.model, messages, stream=True),
            json.dumps,
            lambda p: p.encode("utf-8"),
        )

        headers = _build_headers(settings.api_key)

        req = urllib.request.Request(
            str(settings.url),
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            _handle_http_error(e)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Inference API request failed: {e.reason}") from e

        try:
            for line_bytes in resp:
                line = line_bytes.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    data: Dict[str, Any] = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
        finally:
            resp.close()
