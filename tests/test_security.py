from pathlib import Path

import pytest

from app.config import Settings
from app.security.execution_policy import (
    ExecutionPolicy,
    PermissionDeniedError,
    ToolPermission,
    validate_git_args,
)
from app.security.path_validator import PathSecurityError, is_sensitive_name, resolve_in_repo
from app.tools.common import ToolContext


@pytest.mark.parametrize(
    "raw",
    ["../outside.txt", "auth/../../outside.txt", "/etc/passwd", "~/.ssh/id_rsa", "auth/\x00.py", "", "   "],
)
def test_path_traversal_and_bad_paths_are_rejected(sample_repo: Path, raw: str):
    with pytest.raises(PathSecurityError):
        resolve_in_repo(sample_repo, raw)


def test_absolute_path_inside_repo_is_allowed(sample_repo: Path):
    target = sample_repo / "auth" / "service.py"
    assert resolve_in_repo(sample_repo, str(target)) == target.resolve()


def test_symlink_escaping_repo_is_rejected(sample_repo: Path, tmp_path: Path):
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("secret")
    (sample_repo / "link.txt").symlink_to(outside)
    with pytest.raises(PathSecurityError, match="outside the repository"):
        resolve_in_repo(sample_repo, "link.txt")


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.production",
        "prod.env",
        "id_rsa",
        "id_ed25519.pub",
        "server.pem",
        "tls.key",
        "credentials.json",
        "secrets.yaml",
        ".netrc",
        ".pypirc",
        "service-account-prod.json",
    ],
)
def test_sensitive_names_are_detected(name: str):
    assert is_sensitive_name(name, Settings(repo_root=Path("."), reports_dir=Path(".")).sensitive_patterns)


@pytest.mark.parametrize("name", [".env.example", "service.py", "README.md", "keyboard.py", "tokens.py"])
def test_ordinary_names_are_not_sensitive(name: str):
    assert not is_sensitive_name(name, Settings(repo_root=Path("."), reports_dir=Path(".")).sensitive_patterns)


@pytest.mark.parametrize("raw", [".env", "id_rsa", ".git/config", ".git/HEAD", ".venv/lib.py"])
def test_readable_check_blocks_secrets_and_excluded_dirs(ctx: ToolContext, raw: str):
    with pytest.raises(PathSecurityError):
        ctx.resolve_readable(raw)


def test_env_example_is_readable(ctx: ToolContext):
    assert ctx.resolve_readable(".env.example").name == ".env.example"


@pytest.mark.parametrize(
    "args",
    [
        ["commit", "-m", "x"],
        ["push"],
        ["checkout", "main"],
        ["reset", "--hard"],
        ["add", "."],
        ["config", "user.name"],
        ["clean", "-fd"],
        ["fetch"],
        [],
    ],
)
def test_git_allowlist_rejects_mutating_subcommands(args: list[str]):
    with pytest.raises(PermissionDeniedError):
        validate_git_args(args)


@pytest.mark.parametrize("args", [["diff", "--output=/tmp/x"], ["log", "--exec=sh"], ["diff", "--ext-diff"]])
def test_git_allowlist_rejects_dangerous_options(args: list[str]):
    with pytest.raises(PermissionDeniedError):
        validate_git_args(args)


def test_git_allowlist_accepts_read_only_commands():
    validate_git_args(["status", "--porcelain=v1"])
    validate_git_args(["log", "--max-count=5"])
    validate_git_args(["diff", "--numstat", "--", "--output-looking-filename.py"])  # after "--" is a path


def test_policy_blocks_disabled_permission(settings: Settings):
    from dataclasses import replace

    policy = ExecutionPolicy.from_settings(replace(settings, allow_test_execution=False))
    assert not policy.is_allowed(ToolPermission.EXECUTE_TESTS)
    with pytest.raises(PermissionDeniedError, match="disabled"):
        policy.authorize("run_tests", ToolPermission.EXECUTE_TESTS)


def test_policy_enforces_total_budget_and_per_tool_limits(policy: ExecutionPolicy):
    policy.max_total_calls = 3
    for _ in range(3):
        policy.authorize("read_file", ToolPermission.READ_FILES)
    with pytest.raises(PermissionDeniedError, match="budget"):
        policy.authorize("read_file", ToolPermission.READ_FILES)

    policy.reset()
    policy.max_total_calls = 100
    for _ in range(policy.per_tool_limits["run_tests"]):
        policy.authorize("run_tests", ToolPermission.EXECUTE_TESTS)
    with pytest.raises(PermissionDeniedError, match="at most"):
        policy.authorize("run_tests", ToolPermission.EXECUTE_TESTS)


def test_settings_repr_never_contains_api_key(settings: Settings):
    from dataclasses import replace

    secret = "gsk_should_not_appear"
    assert secret not in repr(replace(settings, groq_api_key=secret))


def test_symlink_alias_to_secret_is_not_searched_or_read(ctx: ToolContext, sample_repo: Path):
    from app.tools.file_tools import ListFilesArgs, ReadFileArgs, SearchCodeArgs, list_files, read_file, search_code

    (sample_repo / "innocent.txt").symlink_to(sample_repo / ".env")
    assert search_code(ctx, SearchCodeArgs(query="super-secret-value"))["total_matches"] == 0
    assert "innocent.txt" not in {e["path"] for e in list_files(ctx, ListFilesArgs())["entries"]}
    with pytest.raises(PathSecurityError):
        read_file(ctx, ReadFileArgs(file_path="innocent.txt"))


def test_git_pathspec_magic_is_literal(ctx: ToolContext):
    from app.tools.git_tools import DiffArgs, get_diff

    result = get_diff(ctx, DiffArgs(file_path=":(glob)**"))
    assert result["changed_files"] == []  # treated as a literal (non-existent) file name
