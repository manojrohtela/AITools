from dataclasses import replace
from pathlib import Path

import pytest

from app.tools.call_graph import FindCallersArgs, TraceFunctionFlowArgs, find_callers, trace_function_flow
from app.tools.common import ToolContext, ToolError


def test_find_callers_confirmed_direct_calls(ctx: ToolContext):
    result = find_callers(ctx, FindCallersArgs(function_name="login"))
    assert result["definitions"] == [{"function": "auth.service:login", "file": "auth/service.py", "line": 19}]
    confirmed = {(c["caller"], c["line"]) for c in result["confirmed_callers"]}
    assert ("web.views:login_view", 5) in confirmed
    assert ("tests.test_auth:test_login_success", 9) in confirmed


def test_find_callers_separates_uncertain_and_references(ctx: ToolContext):
    result = find_callers(ctx, FindCallersArgs(function_name="login"))
    uncertain = result["uncertain_callers"]
    assert [(u["caller"], u["call_kind"]) for u in uncertain] == [("web.views:dynamic_dispatch", "getattr")]
    assert [r["function"] for r in result["references_without_call"]] == ["web.views:route_table"]
    confirmed_callers = {c["caller"] for c in result["confirmed_callers"]}
    assert "web.views:dynamic_dispatch" not in confirmed_callers


def test_find_callers_module_path_scope(ctx: ToolContext):
    result = find_callers(ctx, FindCallersArgs(function_name="login", module_path="web"))
    assert {c["file"] for c in result["confirmed_callers"]} == {"web/views.py"}


def test_self_method_call_is_confirmed(ctx: ToolContext):
    result = find_callers(ctx, FindCallersArgs(function_name="AuthService._issue_token"))
    assert [(c["caller"], c["call_kind"]) for c in result["confirmed_callers"]] == [
        ("auth.service:AuthService.authenticate", "self_method")
    ]


def test_method_call_on_unknown_receiver_is_uncertain(ctx: ToolContext):
    result = find_callers(ctx, FindCallersArgs(function_name="authenticate"))
    assert result["confirmed_callers"] == []
    assert [u["caller"] for u in result["uncertain_callers"]] == ["auth.service:login"]
    assert "unknown" in result["uncertain_callers"][0]["reason"]


def test_module_attribute_call_is_confirmed(ctx: ToolContext):
    result = find_callers(ctx, FindCallersArgs(function_name="create_token"))
    assert [c["caller"] for c in result["confirmed_callers"]] == ["auth.service:AuthService._issue_token"]


def test_same_name_different_function_is_not_confirmed(ctx: ToolContext, sample_repo: Path):
    (sample_repo / "other.py").write_text(
        "def hash_password(x):\n    return x\n\ndef use():\n    return hash_password(1)\n"
    )
    result = find_callers(ctx, FindCallersArgs(function_name="hash_password"))
    by_caller = {c["caller"]: c for c in result["confirmed_callers"]}
    assert "auth.passwords:verify_password" in by_caller and "other:use" in by_caller
    assert len(result["definitions"]) == 2


def test_find_callers_unknown_function_suggests(ctx: ToolContext):
    result = find_callers(ctx, FindCallersArgs(function_name="loginn"))
    assert result["definition_found"] is False and "login" in result["suggestions"]


def test_trace_follows_confirmed_calls(ctx: ToolContext):
    result = trace_function_flow(ctx, TraceFunctionFlowArgs(start_function="AuthService.authenticate", maximum_depth=3))
    assert set(result["paths"]) == {
        "auth.service:AuthService.authenticate -> auth.passwords:verify_password -> auth.passwords:hash_password",
        "auth.service:AuthService.authenticate -> auth.service:AuthService._issue_token -> auth.tokens:create_token",
    }
    assert result["deepest_level_reached"] == 2 and result["cycles"] == []
    edge = next(e for e in result["edges"] if e["callee"] == "auth.passwords:verify_password")
    assert edge == {
        "caller": "auth.service:AuthService.authenticate",
        "callee": "auth.passwords:verify_password",
        "file": "auth/service.py",
        "line": 11,
        "depth": 1,
    }


def test_trace_reports_unresolved_dynamic_calls(ctx: ToolContext):
    result = trace_function_flow(ctx, TraceFunctionFlowArgs(start_function="login"))
    assert any(u["call"] == "authenticate" and u["status"] == "uncertain" for u in result["unresolved_references"])
    assert "auth.service:AuthService.__init__" in {n["function"] for n in result["nodes"]}
    assert "Static" in result["limitations"]


def test_trace_detects_direct_recursion(ctx: ToolContext):
    result = trace_function_flow(ctx, TraceFunctionFlowArgs(start_function="factorial"))
    assert result["cycles"] == [
        {"cycle": ["utils.mathx:factorial", "utils.mathx:factorial"], "kind": "direct recursion"}
    ]


def test_trace_detects_mutual_recursion(ctx: ToolContext):
    result = trace_function_flow(ctx, TraceFunctionFlowArgs(start_function="is_even", maximum_depth=6))
    assert result["cycles"][0]["kind"] == "indirect recursion (cycle)"
    assert result["cycles"][0]["cycle"] == ["utils.mathx:is_even", "utils.mathx:is_odd", "utils.mathx:is_even"]


def _chain_repo(sample_repo: Path, length: int) -> None:
    lines = [f"def f{i}():\n    return f{i + 1}()\n" for i in range(length)] + [f"def f{length}():\n    return 0\n"]
    (sample_repo / "chain.py").write_text("\n".join(lines))


def test_trace_enforces_max_depth(ctx: ToolContext, sample_repo: Path):
    _chain_repo(sample_repo, 10)
    result = trace_function_flow(ctx, TraceFunctionFlowArgs(start_function="f0", maximum_depth=2))
    assert result["depth_limit_hit"] is True
    assert result["paths"] == ["chain:f0 -> chain:f1 -> chain:f2"]


def test_trace_enforces_node_limit(ctx: ToolContext, sample_repo: Path):
    _chain_repo(sample_repo, 10)
    small = ToolContext(settings=replace(ctx.settings, limits=replace(ctx.limits, max_graph_nodes=3)))
    result = trace_function_flow(small, TraceFunctionFlowArgs(start_function="f0", maximum_depth=6))
    assert result["node_limit_hit"] is True and len(result["nodes"]) == 3


def test_trace_unknown_function_raises_with_hint(ctx: ToolContext):
    with pytest.raises(ToolError, match="factorial"):
        trace_function_flow(ctx, TraceFunctionFlowArgs(start_function="factorail"))
