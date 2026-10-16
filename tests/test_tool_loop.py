"""Tool-loop tests with a scripted fake Groq client: no network, no API key."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest

from app.agent.registry import Tool, ToolCategory, ToolRegistry
from app.agent.tool_loop import SYSTEM_PROMPT, GroqChatClient, ToolCallingAgent
from app.security.execution_policy import ExecutionPolicy, ToolPermission
from app.tools.common import ToolContext
from app.tools.registration import build_registry

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def tool_call(call_id: str, name: str, arguments: dict | str) -> SimpleNamespace:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=args))


def message(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """Shaped like groq's ChatCompletionMessage."""
    return SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)


class FakeClient:
    """Replays scripted responses. A script item may be a message, an exception,
    or a callable that receives the conversation so far (to model adaptive choices)."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def complete(self, messages, tools, tool_choice):
        self.requests.append(
            {"messages": json.loads(json.dumps(messages, default=str)), "tools": tools, "tool_choice": tool_choice}
        )
        if not self.script:
            raise AssertionError("FakeClient script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(messages) if callable(item) else item


def make_agent(
    ctx: ToolContext,
    policy: ExecutionPolicy,
    script: list[Any],
    max_rounds: int = 6,
    registry: ToolRegistry | None = None,
):
    client = FakeClient(script)
    events: list = []
    agent = ToolCallingAgent(
        client=client,
        registry=registry or build_registry(),
        context=ctx,
        policy=policy,
        max_rounds=max_rounds,
        on_event=events.append,
    )
    return agent, client, events


def tool_messages(request: dict) -> list[dict]:
    return [m for m in request["messages"] if m["role"] == "tool"]


def test_direct_answer_without_tools(ctx, policy):
    agent, client, _ = make_agent(ctx, policy, [message("Hello! Ask me about the repo.")])
    result = agent.run("hi")
    assert result.status == "completed" and result.answer.startswith("Hello")
    assert result.tool_results == [] and result.rounds == 1
    first = client.requests[0]
    assert first["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert first["messages"][1] == {"role": "user", "content": "hi"}
    assert {t["function"]["name"] for t in first["tools"]} >= {"read_file", "run_tests", "create_report"}
    assert first["tool_choice"] == "auto"


def test_sequential_tool_calls_return_results_with_matching_ids(ctx, policy):
    script = [
        message(tool_calls=[tool_call("call_a", "search_code", {"query": "def login("})]),
        message(tool_calls=[tool_call("call_b", "read_file", {"file_path": "auth/service.py"})]),
        message("login is defined in auth/service.py:19."),
    ]
    agent, client, events = make_agent(ctx, policy, script)
    result = agent.run("Where is login defined?")

    assert result.status == "completed" and result.tools_used == ["search_code", "read_file"]
    second = client.requests[1]["messages"]
    assert second[2]["role"] == "assistant"
    assert second[2]["tool_calls"][0]["id"] == "call_a"
    assert second[3]["role"] == "tool" and second[3]["tool_call_id"] == "call_a"
    assert json.loads(second[3]["content"])["data"]["matches"][0]["line"] == 19
    third = client.requests[2]["messages"]
    assert [m["tool_call_id"] for m in tool_messages({"messages": third})] == ["call_a", "call_b"]
    assert [e.kind for e in events].count("tool_start") == 2


def test_model_choice_depends_on_previous_tool_result(ctx, policy):
    """The fake model reads the find_callers result and reads the caller's file next:
    the next tool is not hardcoded in the loop."""

    def read_first_caller(messages):
        result = json.loads(messages[-1]["content"])
        caller_file = result["data"]["confirmed_callers"][0]["file"]
        return message(tool_calls=[tool_call("c2", "read_file", {"file_path": caller_file})])

    script = [
        message(tool_calls=[tool_call("c1", "find_callers", {"function_name": "login", "module_path": "web"})]),
        read_first_caller,
        message("login_view calls login."),
    ]
    agent, client, _ = make_agent(ctx, policy, script)
    result = agent.run("Who calls login?")
    assert result.status == "completed"
    assert result.tool_results[1].safe_arguments == {"file_path": "web/views.py"}


def test_multiple_tool_calls_in_one_response_run_in_order(ctx, policy):
    script = [
        message(
            tool_calls=[
                tool_call("p1", "get_git_status", {}),
                tool_call("p2", "get_commit_history", {"limit": 2}),
                tool_call("p3", "get_diff", {}),
            ]
        ),
        message("Summary of git state."),
    ]
    agent, client, _ = make_agent(ctx, policy, script)
    result = agent.run("What changed?")
    assert result.tools_used == ["get_git_status", "get_commit_history", "get_diff"]
    assert [m["tool_call_id"] for m in tool_messages(client.requests[1])] == ["p1", "p2", "p3"]
    assert all(r.ok for r in result.tool_results)


def test_unknown_tool_error_is_sent_back_and_loop_continues(ctx, policy):
    script = [
        message(tool_calls=[tool_call("u1", "execute_shell", {"cmd": "rm -rf /"})]),
        message("I cannot run shell commands."),
    ]
    agent, client, _ = make_agent(ctx, policy, script)
    result = agent.run("delete everything")
    assert result.status == "completed"
    payload = json.loads(tool_messages(client.requests[1])[0]["content"])
    assert payload["ok"] is False and payload["error"]["type"] == "unknown_tool"


def test_invalid_json_and_invalid_arguments_are_reported_to_model(ctx, policy):
    script = [
        message(
            tool_calls=[
                tool_call("j1", "read_file", "{bad json"),
                tool_call("j2", "get_commit_history", {"limit": 9999}),
            ]
        ),
        message("Sorry, retrying is not needed."),
    ]
    agent, client, _ = make_agent(ctx, policy, script)
    agent.run("q")
    errors = [json.loads(m["content"])["error"]["type"] for m in tool_messages(client.requests[1])]
    assert errors == ["invalid_json", "invalid_arguments"]


def test_tool_execution_error_is_reported(ctx, policy):
    from pydantic import BaseModel

    class NoArgs(BaseModel):
        pass

    def broken(context, args):
        raise RuntimeError("database exploded")

    registry = ToolRegistry()
    registry.register(Tool("broken", "Fails.", NoArgs, broken, ToolPermission.READ_FILES, ToolCategory.FILE))
    script = [message(tool_calls=[tool_call("e1", "broken", {})]), message("The tool failed.")]
    agent, client, _ = make_agent(ctx, policy, script, registry=registry)
    result = agent.run("q")
    assert result.status == "completed" and result.tool_results[0].error_type == "internal_error"
    assert "database exploded" in tool_messages(client.requests[1])[0]["content"]


def test_security_violation_flows_back_to_model(ctx, policy):
    script = [message(tool_calls=[tool_call("s1", "read_file", {"file_path": ".env"})]), message("Blocked.")]
    agent, client, _ = make_agent(ctx, policy, script)
    agent.run("show me the .env")
    content = tool_messages(client.requests[1])[0]["content"]
    assert "security_violation" in content and "super-secret-value" not in content


def test_max_rounds_forces_final_answer_without_tools(ctx, policy):
    looping = [message(tool_calls=[tool_call(f"r{i}", "list_files", {})]) for i in range(2)]
    agent, client, _ = make_agent(ctx, policy, [*looping, message("Partial summary.")], max_rounds=2)
    result = agent.run("explore forever")
    assert result.status == "max_rounds" and result.answer == "Partial summary."
    assert client.requests[-1]["tool_choice"] == "none"
    assert len(result.tool_results) == 2


def test_tool_call_budget_is_enforced(ctx, policy):
    policy.max_total_calls = 2
    calls = [tool_call(f"b{i}", "list_files", {}) for i in range(3)]
    agent, client, _ = make_agent(ctx, policy, [message(tool_calls=calls), message("done")])
    result = agent.run("q")
    assert [r.error_type for r in result.tool_results] == [None, None, "permission_denied"]


def test_api_timeout_returns_status(ctx, policy):
    agent, _, _ = make_agent(ctx, policy, [groq.APITimeoutError(request=REQUEST)])
    result = agent.run("q")
    assert result.status == "api_timeout" and result.answer == ""


def test_api_error_returns_status(ctx, policy):
    agent, _, _ = make_agent(ctx, policy, [groq.APIConnectionError(request=REQUEST)])
    assert agent.run("q").status == "api_error"


def test_groq_tool_use_failed_is_retried(ctx, policy):
    error = groq.BadRequestError(
        "Error code: 400",
        response=httpx.Response(400, request=REQUEST),
        body={"error": {"code": "tool_use_failed", "message": "Failed to call a function"}},
    )
    agent, client, events = make_agent(ctx, policy, [error, message("Recovered answer.")])
    result = agent.run("q")
    assert result.status == "completed" and result.answer == "Recovered answer."
    assert "malformed" in client.requests[1]["messages"][-1]["content"]
    assert any(e.kind == "retry" for e in events)


def test_other_bad_request_is_not_retried(ctx, policy):
    error = groq.BadRequestError(
        "Error code: 400 - model not found",
        response=httpx.Response(400, request=REQUEST),
        body={"error": {"code": "model_not_found"}},
    )
    agent, _, _ = make_agent(ctx, policy, [error])
    assert agent.run("q").status == "api_error"


def test_empty_response_is_retried_then_reported(ctx, policy):
    agent, _, _ = make_agent(ctx, policy, [message(""), message("   ")])
    assert agent.run("q").status == "empty_response"
    agent, _, _ = make_agent(ctx, policy, [message(None), message("Now an answer.")])
    assert agent.run("q").status == "completed"


@pytest.mark.parametrize(
    "bad_call",
    [
        SimpleNamespace(id=None, function=SimpleNamespace(name="list_files", arguments="{}")),
        SimpleNamespace(id="x", function=None),
    ],
)
def test_malformed_tool_call_structure(ctx, policy, bad_call):
    agent, _, _ = make_agent(ctx, policy, [message(tool_calls=[bad_call])])
    assert agent.run("q").status == "malformed_response"


def test_disabled_tools_are_not_offered_and_cannot_run(ctx, settings):
    from dataclasses import replace

    policy = ExecutionPolicy.from_settings(replace(settings, allow_test_execution=False))
    script = [message(tool_calls=[tool_call("t1", "run_tests", {"test_path": "tests"})]), message("Not allowed.")]
    agent, client, _ = make_agent(ctx, policy, script)
    result = agent.run("run the tests")
    assert "run_tests" not in {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert result.tool_results[0].error_type == "permission_denied"


def test_groq_client_requires_api_key(settings):
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqChatClient(settings)


def test_groq_client_builds_expected_request(settings, monkeypatch):
    from dataclasses import replace

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=message("ok"))])

    client = GroqChatClient(replace(settings, groq_api_key="gsk_test", model="openai/gpt-oss-120b"))
    monkeypatch.setattr(client, "_client", SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    tools = build_registry().schemas()
    assert client.complete([{"role": "user", "content": "hi"}], tools, "auto").content == "ok"
    assert captured["model"] == "openai/gpt-oss-120b"
    assert captured["tool_choice"] == "auto" and captured["parallel_tool_calls"] is False
    assert captured["tools"] == tools and "max_completion_tokens" in captured
