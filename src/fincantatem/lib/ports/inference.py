from ...domain.ports import InferenceApi as DomainInferenceApi
from ...domain.aggs import InferenceSettings, Message
from ...domain.values import Prompt, Response
from typing import Any, Dict, Iterator, List, Optional

import json
import urllib.error
import urllib.request


class InferenceApi(DomainInferenceApi):
    def call(
        self,
        settings: InferenceSettings,
        prompt: Prompt,
        system_prompt: Optional[Prompt] = None,
    ) -> Response:
        if settings.model is None:
            raise ValueError("InferenceSettings.model is required")

        payload = json.dumps(
            {
                "model": str(settings.model),
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if settings.api_key is not None:
            headers["Authorization"] = f"Bearer {settings.api_key}"

        req = urllib.request.Request(
            str(settings.url),
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            raise RuntimeError(
                f"Inference API HTTP error {e.code} {e.reason}: {body or '<no body>'}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Inference API request failed: {e.reason}") from e

        data: Dict[str, Any] = json.loads(raw)
        content: Optional[str] = None
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"Unexpected inference API response shape: {data}"
            ) from e

        if not isinstance(content, str):
            raise RuntimeError(
                f"Unexpected inference API content type: {type(content)}"
            )

        return Response(content)

    def call_stream(
        self, settings: InferenceSettings, messages: List[Message[Prompt | Response]]
    ) -> Iterator[str]:
        if settings.model is None:
            raise ValueError("InferenceSettings.model is required")

        payload = json.dumps(
            {
                "model": str(settings.model),
                "messages": [
                    {"role": m.role, "content": str(m.content)} for m in messages
                ],
                "stream": True,
            }
        ).encode("utf-8")

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if settings.api_key is not None:
            headers["Authorization"] = f"Bearer {settings.api_key}"

        req = urllib.request.Request(
            str(settings.url),
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            raise RuntimeError(
                f"Inference API HTTP error {e.code} {e.reason}: {body or '<no body>'}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Inference API request failed: {e.reason}") from e

        try:
            for line_bytes in resp:
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                if not line.startswith("data: "):
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
