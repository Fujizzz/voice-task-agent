"""Explicit, typed read-only tools for the bounded executor."""

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import ToolError


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[BaseModel, dict[str, Any]], Awaitable[Any]]
    read_only: bool = True


class ToolRegistry:
    """Register and dispatch application-owned async read-only tools."""

    def __init__(self, tools: list[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if not tool.read_only:
                raise ValueError("v0.1 supports read-only tools only")
            if not inspect.iscoroutinefunction(tool.handler):
                raise ValueError("Tool handlers must be async and cooperate with cancellation")
            if tool.arguments.model_config.get("extra") != "forbid":
                raise ValueError("Tool argument models must forbid extra fields")
            if not tool.name or tool.name in self._tools:
                raise ValueError("Tool names must be nonempty and unique")
            self._tools[tool.name] = tool

    def describe(self, allowed: list[str]) -> list[dict]:
        unknown = set(allowed) - self._tools.keys()
        if unknown:
            raise ValueError(f"Unknown configured tools: {sorted(unknown)}")
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.arguments.model_json_schema(),
            }
            for name in allowed
            for t in [self._tools[name]]
        ]

    async def execute(self, name: str, arguments: dict, context: dict, allowed: list[str]):
        if name not in allowed or name not in self._tools:
            raise ToolError("TOOL_NOT_ALLOWED")
        tool = self._tools[name]
        try:
            validated = tool.arguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError("INVALID_TOOL_ARGUMENTS") from exc
        # An application tool cannot mutate the persisted context through aliasing.
        isolated_context = json.loads(json.dumps(context, allow_nan=False))
        result = await tool.handler(validated, isolated_context)
        json.dumps(result, allow_nan=False)
        return result
