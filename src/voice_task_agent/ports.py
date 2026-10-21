"""Provider-neutral asynchronous model and speech interfaces."""

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class ModelPort(Protocol):
    async def generate_structured(
        self, *, prompt: str, payload: dict[str, Any], schema: type[StructuredModel]
    ) -> StructuredModel: ...


class TranscriberPort(Protocol):
    async def transcribe(self, path: str) -> str: ...
