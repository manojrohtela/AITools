"""Validate and execute one model-requested tool call.

This is the trust boundary: the model only proposes ``(name, arguments)``.
Python decides whether the tool exists, whether policy allows it, whether
the arguments are valid, and then runs a registered function with a timeout.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Any

from pydantic import ValidationError

from app.agent.registry import ToolRegistry, UnknownToolError
from app.agent.schemas import ToolExecutionResult
from app.security.execution_policy import ExecutionPolicy, PermissionDeniedError
from app.security.path_validator import PathSecurityError
from app.tools.common import ToolContext, ToolError

SECRET_KEY_HINTS = ("key", "token", "secret", "password", "credential")
MAX_DISPLAY_VALUE_CHARS = 120


def safe_display_arguments(arguments: Any) -> dict[str, Any]:
    """Arguments safe to print or log: redacts secret-looking keys, truncates."""
    if not isinstance(arguments, dict):
        return {"_raw": str(arguments)[:MAX_DISPLAY_VALUE_CHARS]}
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if any(hint in str(key).lower() for hint in SECRET_KEY_HINTS):
            safe[key] = "***redacted***"
        elif isinstance(value, str) and len(value) > MAX_DISPLAY_VALUE_CHARS:
            safe[key] = value[:MAX_DISPLAY_VALUE_CHARS] + "..."
        elif isinstance(value, (list, dict)):
            text = json.dumps(value, default=str)
            safe[key] = (
                value if len(text) <= MAX_DISPLAY_VALUE_CHARS else f"<{type(value).__name__} with {len(value)} items>"
            )
        else:
            safe[key] = value
    return safe


def parse_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string; parse it strictly."""
    if raw_arguments is None or raw_arguments == "":
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str):
        raise ValueError(f"Arguments must be a JSON object string, got {type(raw_arguments).__name__}")
    parsed = json.loads(raw_arguments)
    if not isinstance(parsed, dict):
        raise ValueError("Arguments must be a JSON object")
    return parsed


def _error(error_type: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": {"type": error_type, "message": message, **extra}}


def _run_with_timeout(func, timeout: float) -> dict[str, Any]:
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(func)
    try:
        return future.result(timeout=timeout)
    finally:
        # Never block on a hung tool. Python threads cannot be killed, so a
        # timed-out in-process tool may finish in the background; subprocess
        # tools (git, pytest) enforce their own hard timeouts.
        pool.shutdown(wait=False, cancel_futures=True)


def execute_tool_call(
    *,
    registry: ToolRegistry,
    context: ToolContext,
    policy: ExecutionPolicy,
    tool_call_id: str,
    name: str,
    raw_arguments: Any,
) -> ToolExecutionResult:
    started = time.monotonic()
    safe_args: dict[str, Any] = {}

    def finish(payload: dict[str, Any], error_type: str | None = None) -> ToolExecutionResult:
        duration = round(time.monotonic() - started, 3)
        ok = error_type is None
        context.tool_log.append(
            {"tool": name, "arguments": safe_args, "ok": ok, "error_type": error_type, "duration_seconds": duration}
        )
        return ToolExecutionResult(
            tool_call_id=tool_call_id,
            name=name,
            ok=ok,
            payload=payload,
            duration_seconds=duration,
            safe_arguments=safe_args,
            error_type=error_type,
        )

    try:
        tool = registry.get(name)
    except UnknownToolError:
        return finish(
            _error("unknown_tool", f"No tool named {name!r} is registered.", available_tools=registry.names()),
            "unknown_tool",
        )

    try:
        data = parse_arguments(raw_arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        return finish(_error("invalid_json", f"Could not parse tool arguments: {exc}"), "invalid_json")
    safe_args = safe_display_arguments(data)

    try:
        arguments = tool.input_model.model_validate(data)
    except ValidationError as exc:
        details = [
            {"field": ".".join(str(part) for part in err["loc"]) or "(root)", "message": err["msg"]}
            for err in exc.errors(include_url=False, include_input=False)
        ]
        return finish(_error("invalid_arguments", "Arguments failed validation.", details=details), "invalid_arguments")

    try:
        policy.authorize(tool.name, tool.permission)
    except PermissionDeniedError as exc:
        return finish(_error("permission_denied", str(exc)), "permission_denied")

    try:
        data_out = _run_with_timeout(lambda: tool.handler(context, arguments), policy.timeout_for(tool.permission))
    except concurrent.futures.TimeoutError:
        return finish(_error("timeout", f"Tool {name!r} exceeded its execution time limit."), "timeout")
    except PathSecurityError as exc:
        return finish(_error("security_violation", str(exc)), "security_violation")
    except PermissionDeniedError as exc:
        return finish(_error("permission_denied", str(exc)), "permission_denied")
    except ToolError as exc:
        return finish(_error("tool_error", str(exc)), "tool_error")
    except Exception as exc:  # noqa: BLE001 - report any tool crash back to the model
        return finish(_error("internal_error", f"{type(exc).__name__}: {exc}"), "internal_error")

    return finish({"ok": True, "data": data_out})
