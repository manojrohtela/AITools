"""Category E: Markdown report generation.

The model supplies content only. The file name and directory are decided by
Python, test results are copied from recorded runs (never from model text),
and code references are checked against the repository.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.security.path_validator import PathSecurityError
from app.tools.common import ToolContext, ToolError

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

STANDARD_LIMITATIONS = (
    "Call-graph and dependency results come from static AST analysis and may miss dynamic dispatch, "
    "reflection, decorators and runtime-generated calls.",
    "Findings marked 'hypothesis' are model-generated interpretations and have not been verified.",
    "Code coverage was not measured.",
)


class CodeReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., min_length=1, max_length=500)
    line: int | None = Field(None, ge=1)
    note: str = Field("", max_length=300)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=1, max_length=200)
    detail: str = Field(..., min_length=1, max_length=3000)
    kind: Literal["confirmed", "hypothesis"] = Field(
        ..., description="'confirmed' only if directly supported by tool output; otherwise 'hypothesis'."
    )
    references: list[CodeReference] = Field(default_factory=list, max_length=20)


class CreateReportArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=3, max_length=120)
    summary: str = Field(..., min_length=1, max_length=5000)
    findings: list[Finding] = Field(default_factory=list, max_length=40)
    test_run_ids: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="run_id values from run_tests; results are copied from the recorded runs.",
    )
    code_references: list[CodeReference] = Field(default_factory=list, max_length=60)
    limitations: list[str] = Field(default_factory=list, max_length=10)


def _clean(text: str) -> str:
    return CONTROL_CHARS.sub("", text).strip()


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return (slug or "report")[:60]


def _verify_reference(ctx: ToolContext, ref: CodeReference) -> tuple[str, str]:
    """Return (rendered reference, verification status)."""
    location = f"{ref.path}:{ref.line}" if ref.line else ref.path
    try:
        path = ctx.resolve_readable(ref.path)
    except PathSecurityError:
        return f"`{_clean(location)}`", "not found / not permitted"
    if not path.is_file():
        return f"`{ctx.rel(path)}`", "verified (directory)" if path.is_dir() else "not a file"
    rel_location = f"{ctx.rel(path)}:{ref.line}" if ref.line else ctx.rel(path)
    if ref.line:
        try:
            line_count = sum(1 for _ in path.open("rb"))
        except OSError:
            return f"`{rel_location}`", "unreadable"
        if ref.line > line_count:
            return f"`{rel_location}`", f"line out of range (file has {line_count} lines)"
    return f"`{rel_location}`", "verified"


def _render_test_section(ctx: ToolContext, run_ids: list[str]) -> list[str]:
    lines = ["## Test Results", ""]
    if not run_ids:
        lines.append("_No tests were run for this report._")
        return lines
    for run_id in run_ids:
        record = ctx.test_runs.get(run_id)
        if record is None:
            raise ToolError(f"Unknown test run_id {run_id!r}; only results from run_tests can be reported")
        c = record.counts
        status = "timed out" if record.timed_out else f"exit code {record.exit_code}"
        lines += [
            f"### `{record.target}` (run `{run_id}`, {status}, {record.duration_seconds}s)",
            "",
            "| Total | Passed | Failed | Errors | Skipped |",
            "|---|---|---|---|---|",
            f"| {c.get('total')} | {c.get('passed')} | {c.get('failed')} | {c.get('errors')} | {c.get('skipped')} |",
            "",
        ]
        failed = [case for case in record.testcases if case["outcome"] in {"failed", "error"}]
        for case in failed[:20]:
            where = f" ({case['file']}:{case['line']})" if case.get("file") else ""
            lines.append(f"- **{case['outcome'].upper()}** `{case['test']}`{where}: {case['message'][:200]}")
        if failed:
            lines.append("")
    lines.append("_Test results are copied verbatim from recorded runs, not written by the model._")
    return lines


def render_report(ctx: ToolContext, args: CreateReportArgs, generated_at: datetime) -> str:
    lines = [
        f"# {_clean(args.title)}",
        "",
        f"- **Repository:** `{ctx.repo_root.name}`",
        f"- **Generated:** {generated_at.isoformat(timespec='seconds')}",
        f"- **Model:** `{ctx.settings.model}`",
        "",
        "## Summary",
        "",
        _clean(args.summary),
        "",
    ]
    for kind, heading in (
        ("confirmed", "## Confirmed Findings"),
        ("hypothesis", "## Hypotheses (model-generated, unverified)"),
    ):
        selected = [f for f in args.findings if f.kind == kind]
        lines += [heading, ""]
        if not selected:
            lines += ["_None._", ""]
        for finding in selected:
            lines += [f"### {_clean(finding.title)}", "", _clean(finding.detail), ""]
            for ref in finding.references:
                rendered, status = _verify_reference(ctx, ref)
                note = f" - {_clean(ref.note)}" if ref.note else ""
                lines.append(f"- {rendered} [{status}]{note}")
            if finding.references:
                lines.append("")

    lines += _render_test_section(ctx, args.test_run_ids)
    lines += ["", "## Code References", ""]
    if not args.code_references:
        lines.append("_None provided._")
    else:
        lines += ["| Reference | Verification | Note |", "|---|---|---|"]
        for ref in args.code_references:
            rendered, status = _verify_reference(ctx, ref)
            lines.append(f"| {rendered} | {status} | {_clean(ref.note).replace('|', '/')} |")

    lines += ["", "## Tools Executed", ""]
    for entry in ctx.tool_log:
        flag = "ok" if entry["ok"] else f"error: {entry['error_type']}"
        lines.append(f"- `{entry['tool']}` {entry['arguments']} ({flag})")

    lines += ["", "## Limitations", ""]
    lines += [f"- {_clean(item)}" for item in (*args.limitations, *STANDARD_LIMITATIONS)]
    return "\n".join(lines) + "\n"


def create_report(ctx: ToolContext, args: CreateReportArgs) -> dict[str, Any]:
    reports_dir: Path = ctx.settings.reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC)
    content = render_report(ctx, args, generated_at)
    if len(content) > ctx.limits.max_report_chars:
        raise ToolError(f"Report exceeds {ctx.limits.max_report_chars} characters; shorten the findings")

    filename = f"{generated_at.strftime('%Y%m%d-%H%M%S')}-{_slugify(args.title)}.md"
    path = (reports_dir / filename).resolve()
    if path.parent != reports_dir:  # defence in depth; the slug cannot contain separators
        raise ToolError("Refusing to write outside the reports directory")
    counter = 1
    while path.exists():
        path = reports_dir / f"{path.stem}-{counter}.md"
        counter += 1
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return {
        "report_path": str(path),
        "characters": len(content),
        "content": content[:8000],
        "content_truncated": len(content) > 8000,
    }
