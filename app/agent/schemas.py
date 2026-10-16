"""Data structures exchanged inside the agent loop.

Tool *input* schemas live next to each tool implementation (Pydantic models in
app/tools/*). This module holds the loop's own records and result types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

AgentStatus = Literal[
    "completed",
    "max_rounds",
    "api_error",
    "api_timeout",
    "empty_response",
    "malformed_response",
]


@dataclass
class ToolExecutionResult:
    tool_call_id: str
    name: str
    ok: bool
    payload: dict[str, Any]
    duration_seconds: float
    safe_arguments: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None

    def to_message_content(self, max_chars: int) -> str:
        """Serialize for the ``role: tool`` message, enforcing a size cap."""
        text = json.dumps(self.payload, default=str, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        return json.dumps(
            {
                "ok": self.ok,
                "truncated": True,
                "note": f"Result exceeded {max_chars} characters; narrow the request.",
                "preview": text[: max_chars - 300],
            },
            ensure_ascii=False,
        )


@dataclass
class AgentEvent:
    kind: Literal["model_call", "tool_start", "tool_end", "retry", "final", "error"]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    status: AgentStatus
    answer: str
    rounds: int
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    error: str | None = None

    @property
    def tools_used(self) -> list[str]:
        return [result.name for result in self.tool_results]
