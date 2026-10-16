import json

import pytest
from pydantic import BaseModel

from app.agent.executor import execute_tool_call, safe_display_arguments
from app.agent.registry import Tool, ToolCategory, ToolRegistry, UnknownToolError
from app.security.execution_policy import ExecutionPolicy, ToolPermission
from app.tools.common import ToolContext
from app.tools.registration import build_registry

EXPECTED_TOOLS = {
    "list_files": ToolCategory.FILE,
    "search_code": ToolCategory.FILE,
    "read_file": ToolCategory.FILE,
    "get_git_status": ToolCategory.GIT,
    "get_commit_history": ToolCategory.GIT,
    "get_diff": ToolCategory.GIT,
    "analyze_dependencies": ToolCategory.STATIC_ANALYSIS,
    "find_callers": ToolCategory.STATIC_ANALYSIS,
    "trace_function_flow": ToolCategory.STATIC_ANALYSIS,
    "run_tests": ToolCategory.TEST,
    "analyze_test_results": ToolCategory.TEST,
    "create_report": ToolCategory.REPORT,
}


class EmptyArgs(BaseModel):
    pass


def _tool(name: str = "demo") -> Tool:
    return Tool(name, "A demo tool.", EmptyArgs, lambda ctx, args: {}, ToolPermission.READ_FILES, ToolCategory.FILE)


def test_all_twelve_tools_are_registered_with_categories():
    registry = build_registry()
    assert {tool.name: tool.category for tool in registry.tools()} == EXPECTED_TOOLS


def test_schemas_are_valid_function_definitions():
    for schema in build_registry().schemas():
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] in EXPECTED_TOOLS
        assert len(function["description"]) > 20
        params = function["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        text = json.dumps(params)
        assert "$ref" not in text and "$defs" not in text  # self-contained
        assert '"title": "' not in text  # Pydantic's auto titles removed; a property named "title" is fine


def test_required_fields_are_declared():
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in build_registry().schemas()}
    assert schemas["read_file"]["required"] == ["file_path"]
    assert schemas["search_code"]["required"] == ["query"]
    assert schemas["find_callers"]["required"] == ["function_name"]
    assert schemas["get_commit_history"]["properties"]["limit"]["maximum"] == 50


def test_duplicate_and_invalid_names_are_rejected():
    registry = ToolRegistry()
    registry.register(_tool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_tool())
    with pytest.raises(ValueError, match="Invalid tool name"):
        registry.register(_tool("bad name!"))


def test_unknown_tool_lookup_raises():
    with pytest.raises(UnknownToolError):
        build_registry().get("delete_everything")


def test_disabled_tools_are_not_advertised(policy: ExecutionPolicy):
    policy.allowed_permissions = frozenset({ToolPermission.READ_FILES})
    names = {s["function"]["name"] for s in build_registry().schemas(allowed=policy.is_allowed)}
    assert names == {"list_files", "search_code", "read_file"}


def _execute(ctx, policy, name, raw):
    return execute_tool_call(
        registry=build_registry(), context=ctx, policy=policy, tool_call_id="call_1", name=name, raw_arguments=raw
    )


def test_dispatch_runs_registered_tool(ctx: ToolContext, policy: ExecutionPolicy):
    result = _execute(ctx, policy, "read_file", '{"file_path": "auth/tokens.py"}')
    assert result.ok and result.payload["data"]["path"] == "auth/tokens.py"
    assert result.tool_call_id == "call_1"
    assert ctx.tool_log[-1]["tool"] == "read_file"


def test_unknown_tool_is_rejected_with_available_names(ctx, policy):
    result = _execute(ctx, policy, "rm_rf", "{}")
    assert not result.ok and result.error_type == "unknown_tool"
    assert "read_file" in result.payload["error"]["available_tools"]


@pytest.mark.parametrize("raw", ["{not json", "[1, 2]", '"string"'])
def test_invalid_json_arguments(ctx, policy, raw):
    result = _execute(ctx, policy, "read_file", raw)
    assert result.error_type == "invalid_json"


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("read_file", "{}"),  # missing required field
        ("read_file", '{"file_path": "a.py", "mode": "w"}'),  # unexpected field
        ("get_commit_history", '{"limit": 500}'),  # above maximum
        ("find_callers", '{"function_name": "os.system(\'ls\')"}'),  # pattern violation
        ("run_tests", '{"test_path": ""}'),
    ],
)
def test_argument_validation_errors(ctx, policy, name, raw):
    result = _execute(ctx, policy, name, raw)
    assert result.error_type == "invalid_arguments"
    assert result.payload["error"]["details"]


def test_validation_failures_do_not_consume_budget(ctx, policy):
    policy.max_total_calls = 1
    _execute(ctx, policy, "read_file", "{}")
    assert _execute(ctx, policy, "read_file", '{"file_path": "auth/tokens.py"}').ok


def test_security_violation_is_reported_not_raised(ctx, policy):
    result = _execute(ctx, policy, "read_file", '{"file_path": "../../etc/passwd"}')
    assert result.error_type == "security_violation"


def test_tool_crash_becomes_internal_error(ctx, policy):
    def boom(context, args):
        raise RuntimeError("kaboom")

    registry = ToolRegistry()
    registry.register(Tool("boom", "Always fails.", EmptyArgs, boom, ToolPermission.READ_FILES, ToolCategory.FILE))
    result = execute_tool_call(
        registry=registry, context=ctx, policy=policy, tool_call_id="c", name="boom", raw_arguments="{}"
    )
    assert result.error_type == "internal_error" and "kaboom" in result.payload["error"]["message"]


def test_tool_timeout(ctx, policy):
    import time

    registry = ToolRegistry()
    registry.register(
        Tool(
            "slow", "Sleeps.", EmptyArgs, lambda c, a: time.sleep(2) or {}, ToolPermission.READ_FILES, ToolCategory.FILE
        )
    )
    policy.tool_timeout_seconds = 0.2
    result = execute_tool_call(
        registry=registry, context=ctx, policy=policy, tool_call_id="c", name="slow", raw_arguments=None
    )
    assert result.error_type == "timeout"


def test_safe_display_arguments_redacts_and_truncates():
    safe = safe_display_arguments({"api_key": "gsk_123", "query": "x" * 500, "limit": 3})
    assert safe["api_key"] == "***redacted***"
    assert len(safe["query"]) < 200
    assert safe["limit"] == 3


def test_oversized_results_are_truncated_for_the_model(ctx, policy):
    result = _execute(ctx, policy, "read_file", '{"file_path": "auth/service.py"}')
    content = json.loads(result.to_message_content(max_chars=500))
    assert content["truncated"] is True and len(json.dumps(content)) < 1000
