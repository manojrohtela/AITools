import json
from dataclasses import replace

import pytest

from app.tools.common import ToolContext, ToolError
from app.tools.file_tools import ListFilesArgs, ReadFileArgs, SearchCodeArgs, list_files, read_file, search_code


def test_list_files_excludes_git_venv_and_secrets(ctx: ToolContext):
    paths = {entry["path"] for entry in list_files(ctx, ListFilesArgs())["entries"]}
    assert "auth/service.py" in paths and "auth" in paths
    assert ".env.example" in paths
    assert not any(p.startswith((".git", ".venv")) for p in paths)
    assert ".env" not in paths and "id_rsa" not in paths


def test_list_files_extension_filter_and_types(ctx: ToolContext):
    result = list_files(ctx, ListFilesArgs(directory_path="auth", file_extension="py"))
    assert {e["path"] for e in result["entries"]} == {
        "auth/__init__.py",
        "auth/passwords.py",
        "auth/service.py",
        "auth/tokens.py",
    }
    assert all(e["type"] == "file" and e["language"] == "python" for e in result["entries"])
    assert result["total_count"] == 4


def test_list_files_respects_result_limit(ctx: ToolContext):
    small = ToolContext(settings=replace(ctx.settings, limits=replace(ctx.limits, max_list_results=3)))
    result = list_files(small, ListFilesArgs())
    assert result["returned_count"] == 3 and result["truncated"] is True and result["total_count"] > 3


def test_list_files_rejects_files_and_outside_dirs(ctx: ToolContext):
    with pytest.raises(ToolError):
        list_files(ctx, ListFilesArgs(directory_path="auth/service.py"))
    from app.security.path_validator import PathSecurityError

    with pytest.raises(PathSecurityError):
        list_files(ctx, ListFilesArgs(directory_path=".."))


def test_search_code_returns_lines_and_snippets(ctx: ToolContext):
    result = search_code(ctx, SearchCodeArgs(query="def login(", file_extension=".py"))
    assert result["matches"] == [
        {"path": "auth/service.py", "line": 19, "snippet": "def login(username: str, password: str, user_repository):"}
    ]
    assert result["total_matches"] == 1


def test_search_code_never_reads_secret_or_excluded_files(ctx: ToolContext):
    assert search_code(ctx, SearchCodeArgs(query="super-secret-value"))["total_matches"] == 0
    hits = search_code(ctx, SearchCodeArgs(query="def login"))["matches"]
    assert all(not m["path"].startswith(".venv") for m in hits)


def test_search_code_case_sensitivity_and_match_limit(ctx: ToolContext):
    assert search_code(ctx, SearchCodeArgs(query="AUTHSERVICE"))["total_matches"] > 0
    assert search_code(ctx, SearchCodeArgs(query="AUTHSERVICE", case_sensitive=True))["total_matches"] == 0
    small = ToolContext(settings=replace(ctx.settings, limits=replace(ctx.limits, max_search_matches=2)))
    result = search_code(small, SearchCodeArgs(query="login"))
    assert result["returned_matches"] == 2 and result["truncated"] is True


def test_search_code_skips_large_files(ctx: ToolContext):
    (ctx.repo_root / "big.py").write_text("needle_xyz\n" * 1000)
    small = ToolContext(settings=replace(ctx.settings, limits=replace(ctx.limits, max_search_file_bytes=100)))
    result = search_code(small, SearchCodeArgs(query="needle_xyz"))
    assert result["total_matches"] == 0 and result["files_skipped_too_large"] >= 1


def test_read_file_returns_numbered_content(ctx: ToolContext):
    result = read_file(ctx, ReadFileArgs(file_path="auth/tokens.py"))
    assert result["path"] == "auth/tokens.py"
    assert result["content"].splitlines()[0] == "    1| import secrets"
    assert result["line_count"] == len(result["content"].splitlines())


@pytest.mark.parametrize("path", [".env", "id_rsa", ".git/config", "../outside.py"])
def test_read_file_blocks_sensitive_and_outside_paths(ctx: ToolContext, path: str):
    from app.security.path_validator import PathSecurityError

    with pytest.raises(PathSecurityError):
        read_file(ctx, ReadFileArgs(file_path=path))


def test_read_file_enforces_size_and_binary_limits(ctx: ToolContext):
    (ctx.repo_root / "large.py").write_text("x = 1\n" * 100)
    (ctx.repo_root / "blob.py").write_bytes(b"abc\x00def")
    small = ToolContext(settings=replace(ctx.settings, limits=replace(ctx.limits, max_file_bytes=50)))
    with pytest.raises(ToolError, match="limit"):
        read_file(small, ReadFileArgs(file_path="large.py"))
    with pytest.raises(ToolError, match="Binary"):
        read_file(ctx, ReadFileArgs(file_path="blob.py"))
    with pytest.raises(ToolError, match="regular file"):
        read_file(ctx, ReadFileArgs(file_path="auth"))


def test_tool_outputs_are_json_serializable(ctx: ToolContext):
    for output in (
        list_files(ctx, ListFilesArgs()),
        search_code(ctx, SearchCodeArgs(query="token")),
        read_file(ctx, ReadFileArgs(file_path="web/views.py")),
    ):
        json.dumps(output)
