from pathlib import Path

import pytest

from app.tools.common import ToolContext
from app.tools.dependency_analysis import (
    AnalyzeDependenciesArgs,
    analyze_dependencies,
    find_cycles,
    strongly_connected_components,
)
from app.tools.python_ast import module_name_for


def _module(result, name):
    return next(m for m in result["modules"] if m["module"] == name)


def test_internal_stdlib_and_external_imports_are_classified(ctx: ToolContext, sample_repo: Path):
    (sample_repo / "auth" / "jwt_helper.py").write_text("import os\nimport requests\nfrom . import tokens\n")
    result = analyze_dependencies(ctx, AnalyzeDependenciesArgs(module_path="auth"))
    helper = _module(result, "auth.jwt_helper")
    assert helper["stdlib_imports"] == ["os"]
    assert helper["external_imports"] == ["requests"]
    assert helper["internal_imports"] == [{"module": "auth.tokens", "line": 3, "scope": "module"}]
    assert "requests" in result["external_packages"]


def test_dependency_edges(ctx: ToolContext):
    result = analyze_dependencies(ctx, AnalyzeDependenciesArgs(module_path="auth/service.py"))
    assert result["modules_analyzed"] == 1
    targets = {(e["to"], e["line"]) for e in result["dependency_edges"]}
    assert targets == {("auth.tokens", 1), ("auth.passwords", 2)}


def test_module_level_cycle_is_reported(ctx: ToolContext):
    cycles = analyze_dependencies(ctx, AnalyzeDependenciesArgs())["circular_dependencies"]
    assert ["utils.cycle_a", "utils.cycle_b", "utils.cycle_a"] in cycles["module_level_cycles"]


def test_deferred_import_cycle_is_only_a_candidate(ctx: ToolContext):
    cycles = analyze_dependencies(ctx, AnalyzeDependenciesArgs())["circular_dependencies"]
    flat = [set(c) for c in cycles["module_level_cycles"]]
    assert {"utils.lazy_c", "utils.lazy_d"} not in flat
    assert ["utils.lazy_c", "utils.lazy_d", "utils.lazy_c"] in cycles[
        "cycles_only_via_deferred_or_type_checking_imports"
    ]


def test_type_checking_imports_are_not_module_level(ctx: ToolContext, sample_repo: Path):
    (sample_repo / "web" / "typed.py").write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from web import views\n"
    )
    result = analyze_dependencies(ctx, AnalyzeDependenciesArgs(module_path="web/typed.py"))
    assert _module(result, "web.typed")["internal_imports"][0]["scope"] == "type_checking"


def test_no_cycles_claimed_for_acyclic_scope(ctx: ToolContext):
    result = analyze_dependencies(ctx, AnalyzeDependenciesArgs(module_path="auth"))
    assert result["circular_dependencies"]["module_level_cycles"] == []


def test_syntax_errors_are_reported_not_fatal(ctx: ToolContext, sample_repo: Path):
    (sample_repo / "broken.py").write_text("def oops(:\n")
    result = analyze_dependencies(ctx, AnalyzeDependenciesArgs())
    assert any(e["path"] == "broken.py" for e in result["parse_errors"])
    assert result["modules_analyzed"] > 5


def test_rejects_non_python_file_scope(ctx: ToolContext):
    from app.tools.common import ToolError

    with pytest.raises(ToolError):
        analyze_dependencies(ctx, AnalyzeDependenciesArgs(module_path="README.md"))


@pytest.mark.parametrize(
    ("path", "expected"),
    [("pkg/mod.py", ("pkg.mod", False)), ("pkg/__init__.py", ("pkg", True)), ("src/pkg/a.py", ("pkg.a", False))],
)
def test_module_names(path, expected):
    assert module_name_for(path) == expected


def test_tarjan_scc_and_cycle_detection():
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"d"}, "e": {"a"}}
    components = {tuple(c) for c in strongly_connected_components(graph)}
    assert ("a", "b", "c") in components and ("e",) in components
    cycles = find_cycles(graph)
    assert ["a", "b", "c", "a"] in cycles and ["d", "d"] in cycles
    assert len(cycles) == 2
