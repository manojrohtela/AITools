"""Category B: read-only Git tools.

Every command is built here from a fixed template. Model input only ever
becomes a validated path placed after ``--`` or a bounded integer.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.security.execution_policy import GIT_SAFETY_FLAGS, validate_git_args
from app.security.path_validator import check_readable, is_excluded_dir_name, is_sensitive_name
from app.tools.common import ToolContext, ToolError, truncate_text


class GitStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository_path: str | None = Field(
        None, max_length=500, description="Optional sub-directory of the configured repository."
    )


class CommitHistoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(10, ge=1, le=50, description="Number of recent commits to return (1-50).")


class DiffArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str | None = Field(None, max_length=500, description="Optional file to restrict the diff to.")
    staged: bool = Field(False, description="Show staged changes instead of unstaged working-tree changes.")


def _git_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "SYSTEMROOT", "TMPDIR"}}
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_OPTIONAL_LOCKS="0",  # read-only: do not refresh the index
        GIT_PAGER="cat",
        LC_ALL="C",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_LITERAL_PATHSPECS="1",  # ":(glob)", ":/" etc. are treated as plain file names
    )
    return env


def run_git(ctx: ToolContext, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run an allow-listed git subcommand without a shell."""
    validate_git_args(args)
    command = ["git", *GIT_SAFETY_FLAGS, *args]
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            cwd=cwd or ctx.repo_root,
            env=_git_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=ctx.limits.git_timeout_seconds,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise ToolError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"git {args[0]} timed out") from exc


def _require_git_repo(ctx: ToolContext, cwd: Path) -> Path:
    result = run_git(ctx, ["rev-parse", "--show-toplevel"], cwd=cwd)
    if result.returncode != 0:
        raise ToolError("The configured repository is not a Git repository")
    return Path(result.stdout.strip()).resolve()


def _parse_branch(header: str) -> dict[str, Any]:
    text = header[3:] if header.startswith("## ") else header
    if text.startswith("No commits yet on "):
        return {"branch": text.removeprefix("No commits yet on ").strip(), "has_commits": False}
    if text.startswith("HEAD (no branch)"):
        return {"branch": None, "detached": True, "has_commits": True}
    info: dict[str, Any] = {"has_commits": True}
    branch_part, _, tracking = text.partition(" [")
    local, _, upstream = branch_part.partition("...")
    info["branch"] = local.strip()
    if upstream:
        info["upstream"] = upstream.strip()
    if tracking:
        info["tracking"] = tracking.rstrip("]")
    return info


def parse_porcelain_status(output: str) -> dict[str, Any]:
    """Parse ``git status --porcelain=v1 -b -z`` output."""
    parts = output.split("\0")
    result: dict[str, Any] = {"branch": None, "staged": [], "modified": [], "untracked": []}
    index = 0
    while index < len(parts):
        entry = parts[index]
        index += 1
        if not entry:
            continue
        if entry.startswith("## "):
            result.update(_parse_branch(entry))
            continue
        x, y, path = entry[0], entry[1], entry[3:]
        record: dict[str, Any] = {"path": path}
        if x in "RC":
            record["original_path"] = parts[index]
            index += 1
        if x == "?":
            result["untracked"].append(path)
            continue
        if x == "!":
            continue
        if x != " ":
            result["staged"].append({**record, "status": x})
        if y != " ":
            result["modified"].append({**record, "status": y})
    return result


def get_git_status(ctx: ToolContext, args: GitStatusArgs) -> dict[str, Any]:
    cwd = ctx.resolve_readable(args.repository_path) if args.repository_path else ctx.repo_root
    if not cwd.is_dir():
        raise ToolError("repository_path must be a directory")
    toplevel = _require_git_repo(ctx, cwd)
    result = run_git(ctx, ["status", "--porcelain=v1", "--branch", "-z", "--untracked-files=all"], cwd=cwd)
    if result.returncode != 0:
        raise ToolError(f"git status failed: {result.stderr.strip()[:300]}")
    status = parse_porcelain_status(result.stdout)
    withheld = 0
    for key in ("staged", "modified", "untracked"):
        kept = []
        for item in status[key]:
            path = item if isinstance(item, str) else item["path"]
            if _is_permitted_diff_path(ctx, path):
                kept.append(item)
            else:
                withheld += 1
        status[key] = kept
    status["withheld_sensitive_or_excluded_paths"] = withheld
    status["git_root_is_configured_root"] = toplevel == ctx.repo_root
    status["path_note"] = "Paths are relative to the Git top-level directory."
    status["clean"] = withheld == 0 and not (status["staged"] or status["modified"] or status["untracked"])
    return status


def parse_log(output: str) -> list[dict[str, str]]:
    commits = []
    for record in output.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        fields = record.split("\x1f")
        if len(fields) != 4:
            continue
        commit_hash, author, date, subject = fields
        commits.append(
            {"hash": commit_hash, "short_hash": commit_hash[:8], "author": author, "date": date, "message": subject}
        )
    return commits


def get_commit_history(ctx: ToolContext, args: CommitHistoryArgs) -> dict[str, Any]:
    _require_git_repo(ctx, ctx.repo_root)
    limit = min(args.limit, ctx.limits.max_commits)
    # Author e-mail is deliberately not requested.
    result = run_git(ctx, ["log", f"--max-count={limit}", "--no-color", "--format=%H%x1f%an%x1f%aI%x1f%s%x1e"])
    if result.returncode != 0:
        if "does not have any commits" in result.stderr:
            return {"commits": [], "count": 0, "note": "The repository has no commits yet."}
        raise ToolError(f"git log failed: {result.stderr.strip()[:300]}")
    commits = parse_log(result.stdout)
    return {"commits": commits, "count": len(commits), "limit": limit}


def _is_permitted_diff_path(ctx: ToolContext, rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if not parts:
        return False
    if any(is_excluded_dir_name(part, ctx.settings.excluded_dirs) for part in parts[:-1]):
        return False
    return not is_sensitive_name(parts[-1], ctx.settings.sensitive_patterns)


def get_diff(ctx: ToolContext, args: DiffArgs) -> dict[str, Any]:
    _require_git_repo(ctx, ctx.repo_root)
    base = ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--relative"]
    if args.staged:
        base.append("--cached")

    pathspec: list[str] = []
    if args.file_path:
        target = ctx.resolve(args.file_path, must_exist=False)
        if target.exists():
            check_readable(
                ctx.repo_root,
                target,
                excluded_dirs=ctx.settings.excluded_dirs,
                sensitive_patterns=ctx.settings.sensitive_patterns,
            )
        pathspec = [ctx.rel(target)]

    numstat = run_git(ctx, [*base, "--numstat", "--", *pathspec])
    if numstat.returncode != 0:
        raise ToolError(f"git diff failed: {numstat.stderr.strip()[:300]}")

    changed: list[dict[str, Any]] = []
    withheld: list[str] = []
    for line in numstat.stdout.splitlines():
        added, deleted, path = (line.split("\t", 2) + ["", ""])[:3]
        if not path:
            continue
        if not _is_permitted_diff_path(ctx, path):
            withheld.append(path)
            continue
        changed.append(
            {
                "path": path,
                "added_lines": None if added == "-" else int(added),
                "deleted_lines": None if deleted == "-" else int(deleted),
                "binary": added == "-",
            }
        )

    diff_text = ""
    truncated = False
    permitted = [item["path"] for item in changed if not item["binary"]]
    if permitted:
        unified = run_git(ctx, [*base, "-U3", "--", *permitted])
        if unified.returncode != 0:
            raise ToolError(f"git diff failed: {unified.stderr.strip()[:300]}")
        diff_text, truncated = truncate_text(unified.stdout, ctx.limits.max_diff_chars)

    return {
        "mode": "staged" if args.staged else "working_tree",
        "changed_files": changed,
        "summary": {
            "files_changed": len(changed),
            "lines_added": sum(item["added_lines"] or 0 for item in changed),
            "lines_deleted": sum(item["deleted_lines"] or 0 for item in changed),
        },
        "withheld_sensitive_files": withheld,
        "unified_diff": diff_text,
        "diff_truncated": truncated,
        "note": "Untracked files are not part of git diff; use get_git_status to see them.",
    }
