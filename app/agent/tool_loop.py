"""The Groq tool-calling loop.

    user question -> Groq (messages + tool schemas)
      -> assistant message with tool_calls?  -- no -> final answer
      -> yes: for each call, validate + execute locally, append role=tool results
      -> send the whole conversation back to Groq, repeat (bounded)

The model never runs code. It only returns tool names and JSON arguments;
``execute_tool_call`` decides what actually happens.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import groq

from app.agent.executor import execute_tool_call
from app.agent.registry import ToolRegistry
from app.agent.schemas import AgentEvent, AgentResult, ToolExecutionResult
from app.config import Settings
from app.security.execution_policy import ExecutionPolicy
from app.tools.common import ToolContext

SYSTEM_PROMPT = """You are DevPilot, a read-only assistant that investigates a local code repository.

Rules:
- Use the provided tools to gather evidence. Never guess file contents, call relationships, git state or test results.
- Choose tools based on the question and on earlier tool results.
- Prefer search_code/list_files to locate code before read_file.
- You cannot modify the repository. You cannot run shell commands; run_tests is the only way to execute tests.
- Cite evidence as path:line wherever possible.
- Clearly separate confirmed facts (directly shown by tool output) from your hypotheses.
- Static-analysis tools may miss dynamic calls; say so when it matters.
- Only call create_report if the user asks for a report. Pass run_id values from run_tests instead of retyping results.
- If a tool returns an error, adapt (fix arguments or choose another tool) instead of repeating the same call.
- When you have enough information, stop calling tools and give a concise, well-structured final answer."""

MAX_MALFORMED_TOOL_RETRIES = 2
MAX_EMPTY_RETRIES = 1


class MalformedResponseError(Exception):
    """The API returned something that is not a usable chat completion."""


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str) -> Any:
        """Return the assistant message object (with .content and .tool_calls)."""


class GroqChatClient:
    """Thin wrapper over the official Groq SDK (OpenAI-compatible chat completions)."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self._settings = settings
        self._client = groq.Groq(
            api_key=settings.groq_api_key,
            timeout=settings.api_timeout_seconds,
            max_retries=2,
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str) -> Any:
        request: dict[str, Any] = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": self._settings.temperature,
            "max_completion_tokens": self._settings.max_completion_tokens,
        }
        if tools:
            request.update(
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=self._settings.parallel_tool_calls,
            )
        response = self._client.chat.completions.create(**request)
        if not getattr(response, "choices", None):
            raise MalformedResponseError("Response contained no choices")
        return response.choices[0].message


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from an SDK object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_tool_use_failed(exc: groq.BadRequestError) -> bool:
    """Groq reports unparsable model tool calls as HTTP 400 with code ``tool_use_failed``."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("code") == "tool_use_failed":
            return True
    return "tool_use_failed" in str(exc)


def normalize_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Convert SDK tool calls into plain dicts; raise on structurally broken calls."""
    calls = []
    for raw in _get(message, "tool_calls") or []:
        function = _get(raw, "function")
        call_id = _get(raw, "id")
        if not call_id or function is None:
            raise MalformedResponseError("Tool call without id or function")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": _get(function, "name") or "", "arguments": _get(function, "arguments") or ""},
            }
        )
    return calls


class ToolCallingAgent:
    def __init__(
        self,
        *,
        client: ChatClient,
        registry: ToolRegistry,
        context: ToolContext,
        policy: ExecutionPolicy,
        max_rounds: int,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.context = context
        self.policy = policy
        self.max_rounds = max_rounds
        self.on_event = on_event or (lambda event: None)

    def _emit(self, kind: str, **data: Any) -> None:
        self.on_event(AgentEvent(kind=kind, data=data))  # type: ignore[arg-type]

    def _call_model(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str = "auto"
    ) -> Any:
        self._emit("model_call", messages=len(messages))
        return self.client.complete(messages, tools, tool_choice)

    def run(self, question: str) -> AgentResult:
        self.policy.reset()
        tools = self.registry.schemas(allowed=self.policy.is_allowed)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        results: list[ToolExecutionResult] = []
        malformed_retries = 0
        empty_retries = 0
        max_chars = self.context.limits.max_tool_output_chars

        for round_number in range(1, self.max_rounds + 1):
            try:
                message = self._call_model(messages, tools)
                tool_calls = normalize_tool_calls(message)
            except groq.APITimeoutError as exc:
                return self._fail("api_timeout", f"Groq API timed out: {exc}", round_number, results)
            except groq.BadRequestError as exc:
                # Groq returns 400 "tool_use_failed" when the model emits an unparsable tool call.
                if _is_tool_use_failed(exc) and malformed_retries < MAX_MALFORMED_TOOL_RETRIES:
                    malformed_retries += 1
                    self._emit("retry", reason="model produced a malformed tool call")
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous tool call was malformed. Call a tool with a valid JSON "
                                "object that matches its schema, or answer directly."
                            ),
                        }
                    )
                    continue
                return self._fail("api_error", f"Groq rejected the request: {exc}", round_number, results)
            except groq.APIError as exc:
                return self._fail("api_error", f"Groq API error: {exc}", round_number, results)
            except MalformedResponseError as exc:
                return self._fail("malformed_response", str(exc), round_number, results)

            content = (_get(message, "content") or "").strip()
            if not tool_calls:
                if content:
                    self._emit("final", answer=content)
                    return AgentResult("completed", content, round_number, results)
                if empty_retries < MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    self._emit("retry", reason="empty model response")
                    messages.append(
                        {"role": "user", "content": "Please continue: call a tool or give your final answer."}
                    )
                    continue
                return self._fail("empty_response", "The model returned an empty response.", round_number, results)

            # The assistant turn (with its tool calls) must precede the tool results.
            messages.append({"role": "assistant", "content": content or None, "tool_calls": tool_calls})
            # Calls are executed one at a time, in the order the model listed them.
            for call in tool_calls:
                name = call["function"]["name"]
                self._emit("tool_start", name=name, call_id=call["id"])
                result = execute_tool_call(
                    registry=self.registry,
                    context=self.context,
                    policy=self.policy,
                    tool_call_id=call["id"],
                    name=name,
                    raw_arguments=call["function"]["arguments"],
                )
                results.append(result)
                self._emit(
                    "tool_end",
                    name=name,
                    ok=result.ok,
                    error_type=result.error_type,
                    duration=result.duration_seconds,
                    arguments=result.safe_arguments,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": result.to_message_content(max_chars),
                    }
                )

        return self._finish_after_limit(messages, tools, results)

    def _finish_after_limit(self, messages, tools, results) -> AgentResult:
        """Round limit reached: ask once more with tools disabled for a best-effort summary."""
        messages.append(
            {
                "role": "user",
                "content": (
                    "Tool-call round limit reached. Do not call more tools. Summarize what you found so far, "
                    "and state what remains unverified."
                ),
            }
        )
        try:
            message = self._call_model(messages, tools, tool_choice="none")
            answer = (_get(message, "content") or "").strip()
        except (groq.APIError, MalformedResponseError) as exc:
            answer = ""
            self._emit("error", message=str(exc))
        answer = answer or f"Stopped after {self.max_rounds} tool rounds without a final answer."
        self._emit("final", answer=answer)
        return AgentResult("max_rounds", answer, self.max_rounds, results)

    def _fail(self, status: str, error: str, rounds: int, results: list[ToolExecutionResult]) -> AgentResult:
        self._emit("error", message=error)
        return AgentResult(status, "", rounds, results, error=error)  # type: ignore[arg-type]
