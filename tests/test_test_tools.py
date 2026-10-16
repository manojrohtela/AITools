from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.security.path_validator import PathSecurityError
from app.tools.common import ToolContext, ToolError
from app.tools.test_tools import (
    AnalyzeTestResultsArgs,
    RunTestsArgs,
    analyze_test_results,
    build_pytest_command,
    counts_from_summary,
    run_tests,
)

RAW_OUTPUT = """\
....F.E                                                                  [100%]
=========================== short test summary info ============================
FAILED tests/test_x.py::test_math - AssertionError: assert 1 == 2
ERROR tests/test_y.py::test_db - ConnectionRefusedError: [Errno 61] refused
=================== 1 failed, 4 passed, 1 skipped, 1 error in 0.12s ===================
"""


def test_run_tests_reports_counts_and_stores_run(ctx: ToolContext):
    result = run_tests(ctx, RunTestsArgs(test_path="tests"))
    assert result["exit_code"] == 1 and result["exit_meaning"] == "some tests failed"
    assert result["counts"] == {"total": 5, "passed": 4, "failed": 1, "errors": 0, "skipped": 0}
    assert "test_parse_token_with_colon_in_username" in result["stdout_tail"]
    assert result["run_id"] in ctx.test_runs


def test_run_tests_single_node_id(ctx: ToolContext):
    result = run_tests(ctx, RunTestsArgs(test_path="tests/test_known_bug.py::test_factorial"))
    assert result["exit_code"] == 0 and result["counts"]["passed"] == 1 and result["counts"]["total"] == 1


@pytest.mark.parametrize("test_path", ["-x", "--co", " -p evil", "tests/test_auth.py::test_a; rm -rf /"])
def test_run_tests_rejects_options_and_injection(ctx: ToolContext, test_path: str):
    with pytest.raises((ToolError, PathSecurityError)):
        run_tests(ctx, RunTestsArgs(test_path=test_path))


@pytest.mark.parametrize("test_path", ["../", "/tmp", ".env", "README.md"])
def test_run_tests_rejects_paths_outside_or_non_python(ctx: ToolContext, test_path: str):
    with pytest.raises((ToolError, PathSecurityError)):
        run_tests(ctx, RunTestsArgs(test_path=test_path))


def test_pytest_command_is_fixed_argv():
    cmd = build_pytest_command("/usr/bin/python3", "tests", Path("/tmp/r.xml"))
    assert cmd[:4] == ["/usr/bin/python3", "-m", "pytest", "tests"]
    assert "-p" in cmd and "no:cacheprovider" in cmd


def test_run_tests_timeout_kills_process(ctx: ToolContext, sample_repo: Path):
    (sample_repo / "tests" / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(30)\n")
    fast = ToolContext(settings=replace(ctx.settings, test_timeout_seconds=1))
    result = run_tests(fast, RunTestsArgs(test_path="tests/test_slow.py"))
    assert result["timed_out"] is True and result["exit_code"] is None
    assert result["duration_seconds"] < 10


def test_run_tests_does_not_leak_secrets_or_write_caches(ctx, sample_repo: Path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_leak_check")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--maxfail=0 -p evil")
    (sample_repo / "tests" / "test_env.py").write_text(
        "import os\n\ndef test_env():\n    assert 'GROQ_API_KEY' not in os.environ\n"
        "    assert 'PYTEST_ADDOPTS' not in os.environ\n"
    )
    result = run_tests(ctx, RunTestsArgs(test_path="tests/test_env.py"))
    assert result["exit_code"] == 0, result["stdout_tail"]
    assert not (sample_repo / ".pytest_cache").exists()
    assert not list(sample_repo.rglob("__pycache__"))


def test_analyze_results_from_run_id(ctx: ToolContext):
    run_id = run_tests(ctx, RunTestsArgs(test_path="tests"))["run_id"]
    analysis = analyze_test_results(ctx, AnalyzeTestResultsArgs(run_id=run_id))
    observed = analysis["observed"]
    assert (observed["total"], observed["passed"], observed["failed"]) == (5, 4, 1)
    failure = observed["failures"][0]
    assert failure["test"] == "tests/test_known_bug.py::test_parse_token_with_colon_in_username"
    assert failure["file"] == "tests/test_known_bug.py" and failure["line"] == 9
    assert failure["exception_type"] == "AssertionError"
    assert analysis["possible_explanations"]["category_counts"] == {"assertion_failure": 1}
    assert "hypotheses" in analysis["possible_explanations"]["disclaimer"]
    assert analysis["coverage"].startswith("Not measured")


def test_analyze_collection_error(ctx: ToolContext, sample_repo: Path):
    (sample_repo / "tests" / "test_broken_import.py").write_text("import does_not_exist_pkg\n")
    run_id = run_tests(ctx, RunTestsArgs(test_path="tests/test_broken_import.py"))["run_id"]
    analysis = analyze_test_results(ctx, AnalyzeTestResultsArgs(run_id=run_id))
    assert analysis["observed"]["errors"] == 1
    assert analysis["possible_explanations"]["category_counts"] == {"collection_error": 1}


def test_analyze_raw_output():
    analysis = analyze_test_results(None, AnalyzeTestResultsArgs(raw_output=RAW_OUTPUT))  # type: ignore[arg-type]
    observed = analysis["observed"]
    assert (observed["passed"], observed["failed"], observed["errors"], observed["skipped"]) == (4, 1, 1, 1)
    assert [f["exception_type"] for f in observed["failures"]] == ["AssertionError", "ConnectionRefusedError"]
    categories = analysis["possible_explanations"]["category_counts"]
    assert categories == {"assertion_failure": 1, "setup_or_fixture_error": 1}


def test_analyze_does_not_invent_results(ctx: ToolContext):
    with pytest.raises(ToolError, match="Unknown run_id"):
        analyze_test_results(ctx, AnalyzeTestResultsArgs(run_id="made-up"))
    with pytest.raises(ToolError, match="Could not find"):
        analyze_test_results(ctx, AnalyzeTestResultsArgs(raw_output="everything is fine, trust me"))


def test_analyze_requires_exactly_one_source():
    with pytest.raises(ValidationError):
        AnalyzeTestResultsArgs()
    with pytest.raises(ValidationError):
        AnalyzeTestResultsArgs(run_id="a", raw_output="b")


def test_counts_from_summary():
    assert counts_from_summary("== 3 passed, 2 warnings in 1.0s ==") == {
        "passed": 3,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "total": 3,
    }
    assert counts_from_summary("no summary") is None
