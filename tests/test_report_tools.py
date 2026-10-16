from pathlib import Path

import pytest
from pydantic import ValidationError

from app.tools.common import ToolContext, ToolError
from app.tools.report_tools import CreateReportArgs, create_report
from app.tools.test_tools import RunTestsArgs, run_tests


def _args(**overrides) -> CreateReportArgs:
    data = {
        "title": "Login flow review",
        "summary": "The login flow goes through AuthService.",
        "findings": [
            {
                "title": "login is called by login_view",
                "detail": "Confirmed by find_callers.",
                "kind": "confirmed",
                "references": [{"path": "web/views.py", "line": 5}],
            },
            {"title": "Tokens are not signed", "detail": "Possibly insecure.", "kind": "hypothesis"},
        ],
        "code_references": [
            {"path": "auth/service.py", "line": 19, "note": "login definition"},
            {"path": "auth/service.py", "line": 999},
            {"path": "../../etc/passwd"},
            {"path": ".env"},
        ],
    }
    data.update(overrides)
    return CreateReportArgs.model_validate(data)


def test_report_is_written_inside_reports_dir(ctx: ToolContext):
    result = create_report(ctx, _args())
    path = Path(result["report_path"])
    assert path.parent == ctx.settings.reports_dir.resolve()
    assert path.name.endswith("-login-flow-review.md")
    assert path.read_text() == result["content"]


def test_report_title_cannot_choose_destination(ctx: ToolContext):
    result = create_report(ctx, _args(title="../../../etc/evil"))
    path = Path(result["report_path"])
    assert path.parent == ctx.settings.reports_dir.resolve()
    assert "/" not in path.name.removesuffix(".md")


def test_report_separates_confirmed_and_hypotheses(ctx: ToolContext):
    content = create_report(ctx, _args())["content"]
    confirmed = content.index("## Confirmed Findings")
    hypotheses = content.index("## Hypotheses (model-generated, unverified)")
    assert confirmed < content.index("login is called by login_view") < hypotheses
    assert content.index("Tokens are not signed") > hypotheses


def test_report_verifies_code_references(ctx: ToolContext):
    content = create_report(ctx, _args())["content"]
    assert "| `auth/service.py:19` | verified | login definition |" in content
    assert "line out of range" in content
    assert content.count("not found / not permitted") == 2
    assert "`web/views.py:5` [verified]" in content


def test_report_includes_recorded_test_results_and_limitations(ctx: ToolContext):
    run_id = run_tests(ctx, RunTestsArgs(test_path="tests"))["run_id"]
    content = create_report(ctx, _args(test_run_ids=[run_id]))["content"]
    assert "| 5 | 4 | 1 | 0 | 0 |" in content
    assert "test_parse_token_with_colon_in_username" in content
    assert "## Limitations" in content and "Code coverage was not measured" in content


def test_report_without_tests_says_so(ctx: ToolContext):
    assert "_No tests were run for this report._" in create_report(ctx, _args())["content"]


def test_report_rejects_unknown_test_run(ctx: ToolContext):
    with pytest.raises(ToolError, match="Unknown test run_id"):
        create_report(ctx, _args(test_run_ids=["fabricated"]))


def test_report_input_validation():
    with pytest.raises(ValidationError):
        _args(title="")
    with pytest.raises(ValidationError):
        _args(output_path="/tmp/x.md")
    with pytest.raises(ValidationError):
        _args(findings=[{"title": "x", "detail": "y", "kind": "definitely-true"}])


def test_reports_do_not_overwrite(ctx: ToolContext):
    first = create_report(ctx, _args())["report_path"]
    second = create_report(ctx, _args())["report_path"]
    assert first != second and Path(first).exists() and Path(second).exists()
