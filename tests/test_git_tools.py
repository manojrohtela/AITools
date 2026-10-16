import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.security.execution_policy import PermissionDeniedError
from app.security.path_validator import PathSecurityError
from app.tools.common import ToolContext, ToolError
from app.tools.git_tools import (
    CommitHistoryArgs,
    DiffArgs,
    GitStatusArgs,
    get_commit_history,
    get_diff,
    get_git_status,
    parse_log,
    parse_porcelain_status,
    run_git,
)


def _snapshot(repo: Path) -> tuple[str, bytes, str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    return head, (repo / ".git" / "index").read_bytes(), status


def test_git_status_reports_branch_staged_modified_untracked(ctx: ToolContext):
    status = get_git_status(ctx, GitStatusArgs())
    assert status["branch"] == "main"
    assert status["staged"] == [{"path": "web/staged.py", "status": "A"}]
    assert status["modified"] == [{"path": "auth/passwords.py", "status": "M"}]
    assert "notes.txt" in status["untracked"]
    assert status["clean"] is False


def test_git_status_withholds_secret_and_excluded_paths(ctx: ToolContext):
    status = get_git_status(ctx, GitStatusArgs())
    assert ".env" not in status["untracked"] and "id_rsa" not in status["untracked"]
    assert ".venv/lib.py" not in status["untracked"]
    assert ".env.example" in status["untracked"]
    assert status["withheld_sensitive_or_excluded_paths"] == 3


def test_git_status_rejects_paths_outside_repo(ctx: ToolContext):
    with pytest.raises(PathSecurityError):
        get_git_status(ctx, GitStatusArgs(repository_path="../"))


def test_commit_history_fields_and_limit(ctx: ToolContext):
    history = get_commit_history(ctx, CommitHistoryArgs(limit=1))
    assert history["count"] == 1
    commit = history["commits"][0]
    assert commit["message"] == "Add revoke_token"
    assert commit["author"] == "Test User"
    assert len(commit["hash"]) == 40 and commit["date"].startswith("20")
    assert "email" not in str(commit).lower() and "test@example.com" not in str(commit)


def test_commit_history_limit_is_validated():
    with pytest.raises(ValidationError):
        CommitHistoryArgs(limit=51)
    with pytest.raises(ValidationError):
        CommitHistoryArgs(limit=0)


def test_diff_working_tree(ctx: ToolContext):
    diff = get_diff(ctx, DiffArgs())
    assert [f["path"] for f in diff["changed_files"]] == ["auth/passwords.py"]
    assert diff["summary"]["lines_added"] == 2
    assert "+# TODO: use bcrypt" in diff["unified_diff"]


def test_diff_staged_and_single_file(ctx: ToolContext):
    staged = get_diff(ctx, DiffArgs(staged=True))
    assert [f["path"] for f in staged["changed_files"]] == ["web/staged.py"]
    single = get_diff(ctx, DiffArgs(file_path="auth/tokens.py"))
    assert single["changed_files"] == []


def test_diff_withholds_sensitive_files(ctx: ToolContext, sample_repo: Path):
    subprocess.run(["git", "add", "-f", ".env"], cwd=sample_repo, check=True)
    diff = get_diff(ctx, DiffArgs(staged=True))
    assert ".env" in diff["withheld_sensitive_files"]
    assert "super-secret-value" not in diff["unified_diff"]
    with pytest.raises(PathSecurityError):
        get_diff(ctx, DiffArgs(file_path=".env"))


def test_diff_size_limit(ctx: ToolContext, sample_repo: Path):
    from dataclasses import replace

    (sample_repo / "auth" / "service.py").write_text("changed\n" * 500)
    small = ToolContext(settings=replace(ctx.settings, limits=replace(ctx.limits, max_diff_chars=200)))
    diff = get_diff(small, DiffArgs())
    assert diff["diff_truncated"] is True and len(diff["unified_diff"]) < 300


def test_git_tools_do_not_modify_repository_state(ctx: ToolContext, sample_repo: Path):
    before = _snapshot(sample_repo)
    get_git_status(ctx, GitStatusArgs())
    get_commit_history(ctx, CommitHistoryArgs(limit=5))
    get_diff(ctx, DiffArgs())
    get_diff(ctx, DiffArgs(staged=True))
    assert _snapshot(sample_repo) == before


def test_run_git_refuses_mutating_commands(ctx: ToolContext):
    with pytest.raises(PermissionDeniedError):
        run_git(ctx, ["commit", "-m", "nope"])


def test_repository_hooks_and_fsmonitor_cannot_execute(ctx: ToolContext, sample_repo: Path, tmp_path: Path):
    marker = tmp_path / "pwned"
    subprocess.run(["git", "config", "core.fsmonitor", f"touch {marker}"], cwd=sample_repo, check=True)
    get_git_status(ctx, GitStatusArgs())
    get_diff(ctx, DiffArgs())
    assert not marker.exists()


def test_non_git_directory_errors(ctx: ToolContext, tmp_path: Path):
    from dataclasses import replace

    plain = tmp_path / "plain"
    plain.mkdir()
    other = ToolContext(settings=replace(ctx.settings, repo_root=plain))
    with pytest.raises(ToolError, match="not a Git repository"):
        get_commit_history(other, CommitHistoryArgs())


def test_parse_porcelain_handles_renames_and_branches():
    output = "## feature...origin/feature [ahead 2]\0R  new.py\0old.py\0 M mod.py\0?? new file.txt\0"
    parsed = parse_porcelain_status(output)
    assert parsed["branch"] == "feature" and parsed["upstream"] == "origin/feature"
    assert parsed["tracking"] == "ahead 2"
    assert parsed["staged"] == [{"path": "new.py", "original_path": "old.py", "status": "R"}]
    assert parsed["modified"] == [{"path": "mod.py", "status": "M"}]
    assert parsed["untracked"] == ["new file.txt"]
    assert parse_porcelain_status("## No commits yet on main\0")["has_commits"] is False


def test_parse_log():
    output = "abc\x1fAda\x1f2026-01-01T00:00:00+00:00\x1fFix: a|b\x1e\n"
    assert parse_log(output) == [
        {
            "hash": "abc",
            "short_hash": "abc",
            "author": "Ada",
            "date": "2026-01-01T00:00:00+00:00",
            "message": "Fix: a|b",
        }
    ]
