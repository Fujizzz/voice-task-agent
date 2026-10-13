"""OpenAI-compatible model transport and an offline demo provider."""

import json
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from .contracts import Decision, Plan
from .errors import InvalidModelOutput, ModelError


class CompatibleModel:
    """Chat Completions adapter with local response-schema validation."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        response_format: Literal["json_object", "json_schema", "text"] = "json_object",
        timeout: float = 30,
        client: httpx.AsyncClient | None = None,
    ):
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) API root without credentials/query/fragment"
            )
        if not model.strip() or response_format not in {"json_object", "json_schema", "text"}:
            raise ValueError("A model name and supported response format are required")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self._api_key = model, api_key
        self.response_format = response_format
        self.timeout = timeout
        self._owned = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    async def generate_structured(self, *, prompt, payload, schema):
        json_schema = schema.model_json_schema()
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                    + "\nReturn only a JSON object matching: "
                    + json.dumps(json_schema),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 2048,
        }
        if self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif self.response_format == "json_schema":
            # strict:false permits schemas with optional fields across compatible servers.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": False},
            }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = await self.client.post(
                self.url, json=body, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelError(f"Model transport failed: {type(exc).__name__}") from None
        try:
            choice = response.json()["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") != "stop" or message.get("refusal"):
                raise InvalidModelOutput("Model did not return a complete non-refused response")
            content = message["content"]
            if not isinstance(content, str) or len(content) > 64000:
                raise InvalidModelOutput("Invalid or oversized model output")
            return schema.model_validate_json(content)
        except (KeyError, IndexError, TypeError, ValueError, ValidationError):
            raise InvalidModelOutput("Response did not match the requested schema") from None

    async def aclose(self):
        if self._owned:
            await self.client.aclose()


class DemoModel:
    """Deterministic offline responses for examples and tests."""

    async def generate_structured(self, *, prompt, payload, schema):
        if schema is Plan:
            return Plan(
                steps=["Read relevant application context", "Return a demonstration response"]
            )
        if not payload["observations"] and payload["tools"] and payload["tools_remaining"]:
            name = payload["tools"][0]["name"]
            arguments = {"query": payload["user_input"]} if name == "search_notes" else {}
            return Decision(action="tool", tool=name, arguments=arguments)
        if "interview" in prompt.lower():
            text = "[Offline demo] What design trade-off did you make in your project, and why?"
        else:
            result = payload["observations"][-1]["result"] if payload["observations"] else {}
            text = (
                "[Offline demo] Retrieved context: " + json.dumps(result, ensure_ascii=False)[:2000]
            )
        return Decision(action="ask" if "interview" in prompt.lower() else "respond", text=text)
