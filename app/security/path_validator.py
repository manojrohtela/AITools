"""Path validation: every model-supplied path is untrusted input."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from app.config import SAFE_ENV_TEMPLATES


class PathSecurityError(ValueError):
    """Raised when a path violates the repository sandbox or secret rules."""


def resolve_in_repo(repo_root: Path, raw_path: str, *, must_exist: bool = True) -> Path:
    """Resolve ``raw_path`` and guarantee it stays inside ``repo_root``.

    Symlinks are resolved before the containment check, so a link pointing
    outside the repository is rejected as well.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PathSecurityError("Path must be a non-empty string")
    if "\x00" in raw_path:
        raise PathSecurityError("Path contains a null byte")

    root = repo_root.resolve()
    candidate = Path(raw_path.strip()).expanduser()
    target = candidate if candidate.is_absolute() else root / candidate
    resolved = target.resolve(strict=False)

    if not resolved.is_relative_to(root):
        raise PathSecurityError(f"Path is outside the repository: {raw_path!r}")
    if must_exist and not resolved.exists():
        raise PathSecurityError(f"Path does not exist in the repository: {raw_path!r}")
    return resolved


def to_relative(repo_root: Path, path: Path) -> str:
    """Return a POSIX-style path relative to the repository root."""
    rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
    return rel or "."


def is_sensitive_name(name: str, patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    if lowered in SAFE_ENV_TEMPLATES:
        return False
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in patterns)


def is_excluded_dir_name(name: str, excluded_dirs: Iterable[str]) -> bool:
    return name in set(excluded_dirs) or name.endswith(".egg-info")


def check_readable(
    repo_root: Path,
    path: Path,
    *,
    excluded_dirs: Iterable[str],
    sensitive_patterns: Iterable[str],
) -> None:
    """Reject secret files and anything inside excluded directories (e.g. .git)."""
    rel_parts = path.resolve().relative_to(repo_root.resolve()).parts
    excluded = set(excluded_dirs)
    for part in rel_parts[:-1]:
        if is_excluded_dir_name(part, excluded):
            raise PathSecurityError(f"Access to excluded directory {part!r} is not allowed")
    if rel_parts and is_sensitive_name(rel_parts[-1], sensitive_patterns):
        raise PathSecurityError(f"Access to sensitive file {rel_parts[-1]!r} is blocked")
    if rel_parts and path.is_dir() and is_excluded_dir_name(rel_parts[-1], excluded):
        raise PathSecurityError(f"Access to excluded directory {rel_parts[-1]!r} is not allowed")


def _is_safe_symlink_target(root: Path, link: Path, excluded: set[str], patterns: tuple[str, ...]) -> bool:
    """A symlink is only followed if its *target* passes the same rules as a real file
    (inside the repo, not a secret, not inside an excluded directory)."""
    target = link.resolve()
    if not target.is_relative_to(root):
        return False
    parts = target.relative_to(root).parts
    if any(is_excluded_dir_name(part, excluded) for part in parts[:-1]):
        return False
    return not (parts and is_sensitive_name(parts[-1], patterns))


def iter_repo_entries(
    repo_root: Path,
    start: Path,
    *,
    excluded_dirs: Iterable[str],
    sensitive_patterns: Iterable[str],
) -> Iterator[tuple[Path, bool]]:
    """Yield ``(path, is_dir)`` pairs below ``start``, in a stable order.

    Excluded directories are pruned, sensitive files are skipped, and
    symlinks that leave the repository are ignored.
    """
    root = repo_root.resolve()
    excluded = set(excluded_dirs)
    patterns = tuple(sensitive_patterns)
    for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not is_excluded_dir_name(d, excluded))
        current = Path(dirpath)
        for dirname in dirnames:
            yield current / dirname, True
        for filename in sorted(filenames):
            if is_sensitive_name(filename, patterns):
                continue
            file_path = current / filename
            if file_path.is_symlink() and not _is_safe_symlink_target(root, file_path, excluded, patterns):
                continue
            yield file_path, False


def iter_repo_files(
    repo_root: Path,
    start: Path,
    *,
    excluded_dirs: Iterable[str],
    sensitive_patterns: Iterable[str],
    extension: str | None = None,
) -> Iterator[Path]:
    for path, is_dir in iter_repo_entries(
        repo_root, start, excluded_dirs=excluded_dirs, sensitive_patterns=sensitive_patterns
    ):
        if is_dir:
            continue
        if extension and path.suffix.lower() != extension:
            continue
        yield path
