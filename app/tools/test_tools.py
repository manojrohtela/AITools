"""Category D: restricted pytest execution and test-result analysis."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.common import TestRunRecord, ToolContext, ToolError, tail_text

NODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\[\]\-.:,/ ]{1,300}$")
SECRET_ENV_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)
# PYTEST_ADDOPTS could inject arbitrary options; strip it and DevPilot's own config.
BLOCKED_ENV_VARS = frozenset({"PYTEST_ADDOPTS", "PYTEST_PLUGINS"})

EXIT_CODE_MEANINGS = {
    0: "all tests passed",
    1: "some tests failed",
    2: "test execution was interrupted",
    3: "internal pytest error",
    4: "pytest usage error",
    5: "no tests were collected",
}
SUMMARY_COUNT_PATTERN = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
SUMMARY_LINE_PATTERN = re.compile(r"^=+ (.+?) in [\d.]+s(?: \([^)]*\))? =+$", re.MULTILINE)
SHORT_SUMMARY_PATTERN = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$", re.MULTILINE)
EXCEPTION_PATTERN = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception|Exit|Failed|Interrupt|Timeout))\b")


class RunTestsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_path: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Test file/directory relative to the repo root, optionally with '::test_name'. No options.",
    )


class AnalyzeTestResultsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str | None = Field(None, max_length=64, description="run_id returned by run_tests (preferred).")
    raw_output: str | None = Field(None, max_length=50_000, description="Raw pytest console output to parse instead.")

    @model_validator(mode="after")
    def exactly_one_source(self) -> AnalyzeTestResultsArgs:
        if (self.run_id is None) == (self.raw_output is None):
            raise ValueError("Provide exactly one of run_id or raw_output")
        return self


def _split_target(ctx: ToolContext, test_path: str) -> tuple[Path, str]:
    """Validate the target; returns (resolved path, pytest target string)."""
    if test_path.lstrip().startswith("-"):
        raise ToolError("test_path must be a path, not a command-line option")
    path_part, sep, node_part = test_path.partition("::")
    if sep and not NODE_ID_PATTERN.match(node_part):
        raise ToolError("Invalid test node id after '::'")
    resolved = ctx.resolve_readable(path_part)
    if resolved.is_file() and resolved.suffix != ".py":
        raise ToolError("test_path must be a Python file or a directory")
    if sep and not resolved.is_file():
        raise ToolError("'::' node ids require a test file path")
    target = ctx.rel(resolved) + (f"::{node_part}" if sep else "")
    return resolved, target


def _test_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in BLOCKED_ENV_VARS
        and not key.startswith(("DEVPILOT_", "GROQ_"))
        and not SECRET_ENV_PATTERN.search(key)
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # keep the target repository unmodified
    return env


def build_pytest_command(python_executable: str, target: str, junit_path: Path) -> list[str]:
    """The only command shape run_tests can ever execute."""
    return [
        python_executable,
        "-m",
        "pytest",
        target,
        "-q",
        "-rfE",
        "--tb=short",
        "--color=no",
        "-p",
        "no:cacheprovider",
        "-o",
        "junit_family=xunit1",
        f"--junitxml={junit_path}",
    ]


def _run_process(command: list[str], cwd: Path, timeout: int) -> tuple[int | None, str, str, bool]:
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        command,
        cwd=cwd,
        env=_test_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)  # kill pytest and any children
        stdout, stderr = process.communicate()
        return None, stdout or "", stderr or "", True


def _node_id(classname: str, name: str, file: str | None) -> str:
    """Rebuild a pytest-style node id (path::Class::test) from junit attributes."""
    if not file:
        return f"{classname}::{name}" if classname else name
    if not classname:  # module-level entry, e.g. a collection error
        return file
    module = file.removesuffix(".py").replace("/", ".")
    class_part = classname[len(module) + 1 :] if classname.startswith(module + ".") else ""
    return "::".join(part for part in (file, class_part, name) if part)


def parse_junit_xml(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    root = ET.parse(path).getroot()  # noqa: S314 - file produced by our own pytest run
    cases = []
    for case in root.iter("testcase"):
        outcome, message, details = "passed", "", ""
        for child_tag, child_outcome in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            child = case.find(child_tag)
            if child is not None:
                outcome = child_outcome
                message = child.get("message", "") or ""
                details = child.text or ""
                break
        cases.append(
            {
                "test": _node_id(case.get("classname", ""), case.get("name", ""), case.get("file")),
                "file": case.get("file"),
                "line": int(case.get("line")) + 1 if case.get("line", "").isdigit() else None,
                "outcome": outcome,
                "message": message[:500],
                "details": details[-1500:],
                "time_seconds": float(case.get("time", 0) or 0),
            }
        )
    return cases


def counts_from_cases(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"total": len(cases), "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    key_for = {"passed": "passed", "failed": "failed", "error": "errors", "skipped": "skipped"}
    for case in cases:
        counts[key_for[case["outcome"]]] += 1
    return counts


def counts_from_summary(output: str) -> dict[str, int] | None:
    summaries = SUMMARY_LINE_PATTERN.findall(output)
    if not summaries:
        return None
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for number, label in SUMMARY_COUNT_PATTERN.findall(summaries[-1]):
        if label in {"error", "errors"}:
            counts["errors"] += int(number)
        elif label in counts:
            counts[label] += int(number)
    counts["total"] = counts["passed"] + counts["failed"] + counts["errors"] + counts["skipped"]
    return counts


def run_tests(ctx: ToolContext, args: RunTestsArgs) -> dict[str, Any]:
    _resolved, target = _split_target(ctx, args.test_path)
    timeout = ctx.settings.test_timeout_seconds
    max_chars = ctx.limits.max_test_output_chars
    with tempfile.TemporaryDirectory(prefix="devpilot-") as tmp:
        junit_path = Path(tmp) / "results.xml"
        command = build_pytest_command(ctx.settings.python_executable, target, junit_path)
        started = time.monotonic()
        exit_code, stdout, stderr, timed_out = _run_process(command, ctx.repo_root, timeout)
        duration = round(time.monotonic() - started, 2)
        try:
            cases = parse_junit_xml(junit_path)
        except ET.ParseError:
            cases = []

    counts = (
        counts_from_cases(cases)
        if cases
        else (
            counts_from_summary(stdout)
            or {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
            }
        )
    )
    run_id = uuid.uuid4().hex[:12]
    ctx.test_runs[run_id] = TestRunRecord(
        run_id=run_id,
        target=target,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_seconds=duration,
        counts=counts,
        testcases=cases,
        stdout_tail=tail_text(stdout, max_chars),
        stderr_tail=tail_text(stderr, max_chars // 2),
        stdout_full=stdout[-50_000:],
    )
    return {
        "run_id": run_id,
        "target": target,
        "command": f"python -m pytest {target} -q -rfE --tb=short (fixed options)",
        "exit_code": exit_code,
        "exit_meaning": "timed out and was killed" if timed_out else EXIT_CODE_MEANINGS.get(exit_code, "unknown"),
        "timed_out": timed_out,
        "duration_seconds": duration,
        "counts": counts,
        "stdout_tail": tail_text(stdout, max_chars),
        "stderr_tail": tail_text(stderr, max_chars // 2),
        "next_step": "Call analyze_test_results with this run_id for structured failure details.",
    }


def _exception_type(message: str, details: str) -> str | None:
    if message.startswith("assert ") or "AssertionError" in message:
        return "AssertionError"
    for text in (message, details):
        match = EXCEPTION_PATTERN.search(text)
        if match:
            return match.group(1).split(".")[-1]
    return None


CATEGORY_RULES: list[tuple[set[str], str, str]] = [
    (
        {"AssertionError"},
        "assertion_failure",
        "The code returned a value the test did not expect: either a behaviour regression or an outdated expectation.",
    ),
    (
        {"ImportError", "ModuleNotFoundError"},
        "import_error",
        "A module or dependency could not be imported: missing package, wrong environment, or a renamed module.",
    ),
    (
        {"AttributeError", "TypeError", "NameError"},
        "code_or_api_mismatch",
        "A name, attribute or call signature does not match: possibly a refactor that callers or tests did not follow.",
    ),
    (
        {"TimeoutError", "Timeout", "Failed"},
        "timeout_or_explicit_fail",
        "The test timed out or called pytest.fail explicitly.",
    ),
    (
        {"ConnectionError", "ConnectionRefusedError", "OSError", "PermissionError", "FileNotFoundError"},
        "environment",
        "The failure depends on the environment (network, files, permissions) rather than pure logic.",
    ),
]


def _categorize(exception: str | None, outcome: str, test: str, message: str = "") -> tuple[str, str]:
    if outcome == "error" and ("collection failure" in message or "::" not in test):
        return "collection_error", "pytest could not collect this module, often due to an import or syntax error."
    if outcome == "error":
        return "setup_or_fixture_error", "The test errored outside its body, typically in a fixture or setup/teardown."
    for exceptions, category, explanation in CATEGORY_RULES:
        if exception in exceptions:
            return category, explanation
    return "uncategorized", "No rule matched this exception type; read the failure details."


def _failures_from_record(record: TestRunRecord) -> list[dict[str, Any]]:
    return [
        {
            "test": case["test"],
            "file": case["file"],
            "line": case["line"],
            "outcome": case["outcome"],
            "exception_type": _exception_type(case["message"], case["details"]),
            "message": case["message"][:300],
        }
        for case in record.testcases
        if case["outcome"] in {"failed", "error"}
    ]


def _failures_from_raw(output: str) -> list[dict[str, Any]]:
    failures = []
    for kind, test, message in SHORT_SUMMARY_PATTERN.findall(output):
        message = message or ""
        failures.append(
            {
                "test": test,
                "file": test.split("::")[0],
                "line": None,
                "outcome": "failed" if kind == "FAILED" else "error",
                "exception_type": _exception_type(message, ""),
                "message": message[:300],
            }
        )
    return failures


def analyze_test_results(ctx: ToolContext, args: AnalyzeTestResultsArgs) -> dict[str, Any]:
    if args.run_id is not None:
        record = ctx.test_runs.get(args.run_id)
        if record is None:
            raise ToolError(f"Unknown run_id {args.run_id!r}. Known: {sorted(ctx.test_runs)}")
        counts = dict(record.counts)
        failures = _failures_from_record(record)
        if not record.testcases:
            failures = _failures_from_raw(record.stdout_full)
        source = {
            "type": "run_tests",
            "run_id": record.run_id,
            "target": record.target,
            "exit_code": record.exit_code,
            "timed_out": record.timed_out,
        }
    else:
        raw = args.raw_output or ""
        counts = counts_from_summary(raw)
        failures = _failures_from_raw(raw)
        if counts is None and not failures:
            raise ToolError("Could not find a pytest summary line or FAILED/ERROR lines in raw_output")
        counts = counts or {"total": None, "passed": None, "failed": None, "errors": None, "skipped": None}
        source = {"type": "raw_output"}

    hypotheses = []
    category_counts: dict[str, int] = {}
    for failure in failures:
        category, explanation = _categorize(
            failure["exception_type"], failure["outcome"], failure["test"], failure["message"]
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        hypotheses.append(
            {
                "test": failure["test"],
                "category": category,
                "possible_explanation": explanation,
                "based_on": f"observed outcome={failure['outcome']}, exception_type={failure['exception_type']}",
            }
        )

    return {
        "source": source,
        "observed": {
            "total": counts.get("total"),
            "passed": counts.get("passed"),
            "failed": counts.get("failed"),
            "errors": counts.get("errors"),
            "skipped": counts.get("skipped"),
            "failures": failures[:50],
            "failures_truncated": len(failures) > 50,
        },
        "possible_explanations": {
            "disclaimer": "Heuristic categories from exception types. These are hypotheses, not verified root causes.",
            "category_counts": category_counts,
            "per_failure": hypotheses[:50],
        },
        "coverage": "Not measured. DevPilot does not collect coverage data.",
    }
