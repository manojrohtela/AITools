"""Category A: basic, read-only file tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.security.path_validator import iter_repo_entries, iter_repo_files
from app.tools.common import (
    LANGUAGE_BY_EXTENSION,
    SEARCHABLE_EXTENSIONS,
    ToolContext,
    ToolError,
    is_probably_binary,
    normalize_extension,
)

EXTENSION_PATTERN = r"^\.?[A-Za-z0-9_+-]{1,12}$"


class ListFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory_path: str = Field(".", description="Directory relative to the repository root.", max_length=500)
    file_extension: str | None = Field(
        None, description="Only include files with this extension, e.g. '.py'.", pattern=EXTENSION_PATTERN
    )


class SearchCodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1, max_length=200, description="Literal text to find (not a regex).")
    file_extension: str | None = Field(
        None, description="Only search files with this extension, e.g. '.py'.", pattern=EXTENSION_PATTERN
    )
    case_sensitive: bool = Field(False, description="Match case exactly.")


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(..., min_length=1, max_length=500, description="File path relative to the repository root.")


def list_files(ctx: ToolContext, args: ListFilesArgs) -> dict[str, Any]:
    start = ctx.resolve_readable(args.directory_path)
    if not start.is_dir():
        raise ToolError(f"Not a directory: {args.directory_path!r}")
    extension = normalize_extension(args.file_extension)
    limit = ctx.limits.max_list_results

    entries: list[dict[str, str]] = []
    total = 0
    for path, is_dir in iter_repo_entries(
        ctx.repo_root,
        start,
        excluded_dirs=ctx.settings.excluded_dirs,
        sensitive_patterns=ctx.settings.sensitive_patterns,
    ):
        if is_dir and extension:
            continue
        if not is_dir and extension and path.suffix.lower() != extension:
            continue
        total += 1
        if len(entries) < limit:
            entries.append(
                {
                    "path": ctx.rel(path),
                    "type": "directory" if is_dir else "file",
                    "language": "" if is_dir else LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "other"),
                }
            )
    file_count = sum(1 for entry in entries if entry["type"] == "file")
    return {
        "directory": ctx.rel(start),
        "entries": entries,
        "total_count": total,
        "returned_count": len(entries),
        "returned_file_count": file_count,
        "truncated": total > limit,
    }


def search_code(ctx: ToolContext, args: SearchCodeArgs) -> dict[str, Any]:
    extension = normalize_extension(args.file_extension)
    if extension is not None and extension not in SEARCHABLE_EXTENSIONS:
        raise ToolError(f"Extension {extension!r} is not a searchable text/source type")
    needle = args.query if args.case_sensitive else args.query.lower()
    limits = ctx.limits

    matches: list[dict[str, Any]] = []
    total_matches = 0
    files_searched = 0
    skipped_large = 0
    for path in iter_repo_files(
        ctx.repo_root,
        ctx.repo_root,
        excluded_dirs=ctx.settings.excluded_dirs,
        sensitive_patterns=ctx.settings.sensitive_patterns,
        extension=extension,
    ):
        if path.suffix.lower() not in SEARCHABLE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > limits.max_search_file_bytes:
                skipped_large += 1
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if is_probably_binary(raw):
            continue
        files_searched += 1
        text = raw.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            haystack = line if args.case_sensitive else line.lower()
            if needle not in haystack:
                continue
            total_matches += 1
            if len(matches) < limits.max_search_matches:
                matches.append(
                    {
                        "path": ctx.rel(path),
                        "line": line_number,
                        "snippet": line.strip()[: limits.max_snippet_chars],
                    }
                )
    return {
        "query": args.query,
        "matches": matches,
        "total_matches": total_matches,
        "returned_matches": len(matches),
        "files_searched": files_searched,
        "files_skipped_too_large": skipped_large,
        "truncated": total_matches > len(matches),
    }


def read_file(ctx: ToolContext, args: ReadFileArgs) -> dict[str, Any]:
    path = ctx.resolve_readable(args.file_path)
    if not path.is_file():
        raise ToolError(f"Not a regular file: {args.file_path!r}")
    size = path.stat().st_size
    if size > ctx.limits.max_file_bytes:
        raise ToolError(
            f"File is {size} bytes; the limit is {ctx.limits.max_file_bytes}. "
            "Use search_code to locate the relevant lines instead."
        )
    raw = path.read_bytes()
    if is_probably_binary(raw):
        raise ToolError("Binary files cannot be read")
    lines = raw.decode("utf-8", errors="replace").splitlines()
    numbered = "\n".join(f"{number:>5}| {line}" for number, line in enumerate(lines, start=1))
    return {
        "path": ctx.rel(path),
        "line_count": len(lines),
        "size_bytes": size,
        "content_format": "each line is prefixed with its 1-based line number and '| '",
        "content": numbered,
    }
