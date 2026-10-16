"""Explicit tool registry.

A tool is: a unique name, a description the model reads, a Pydantic input
model (which doubles as the JSON Schema sent to Groq), a Python handler, a
category and the permission it needs. Nothing runs unless it is registered.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from app.security.execution_policy import ToolPermission
from app.tools.common import ToolContext

TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolCategory(StrEnum):
    FILE = "basic_file"
    GIT = "git"
    STATIC_ANALYSIS = "static_analysis"
    TEST = "test"
    REPORT = "reporting"


class UnknownToolError(KeyError):
    """The model asked for a tool that is not registered."""


Handler = Callable[[ToolContext, Any], dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler
    permission: ToolPermission
    category: ToolCategory

    def json_schema(self) -> dict[str, Any]:
        """Function schema in the OpenAI-compatible format Groq expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _clean_schema(_inline_refs(self.input_model.model_json_schema())),
            },
        }


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace ``$ref`` pointers with their ``$defs`` bodies (flat, self-contained schema)."""
    definitions = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return resolve(definitions[ref.removeprefix("#/$defs/")])
            return {key: resolve(value) for key, value in node.items() if key != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def _clean_schema(schema: Any) -> Any:
    """Drop Pydantic's auto-generated ``title`` keys to save prompt tokens."""
    if isinstance(schema, dict):
        return {
            key: _clean_schema(value)
            for key, value in schema.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not TOOL_NAME_PATTERN.match(tool.name):
            raise ValueError(f"Invalid tool name: {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name!r}")
        if not tool.description.strip():
            raise ValueError(f"Tool {tool.name!r} needs a description")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(name) from None

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self, allowed: Callable[[ToolPermission], bool] | None = None) -> list[dict[str, Any]]:
        """Schemas for the model. Tools disabled by policy are not advertised."""
        return [tool.json_schema() for tool in self._tools.values() if allowed is None or allowed(tool.permission)]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
