"""Shared context and helpers used by tool implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.security.path_validator import check_readable, resolve_in_repo, to_relative


class ToolError(Exception):
    """A tool failed in an expected way; the message is safe to show the model."""


LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".rst": "text",
    ".txt": "text",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
}

SEARCHABLE_EXTENSIONS: frozenset[str] = frozenset(LANGUAGE_BY_EXTENSION)


@dataclass
class TestRunRecord:
    """Observed result of one run_tests call, kept for later analysis."""

    run_id: str
    target: str
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    counts: dict[str, int]
    testcases: list[dict[str, Any]]
    stdout_tail: str
    stderr_tail: str
    stdout_full: str


@dataclass
class ToolContext:
    """Per-session state shared by tools. Holds no model-controlled config."""

    settings: Settings
    test_runs: dict[str, TestRunRecord] = field(default_factory=dict)
    tool_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def repo_root(self) -> Path:
        return self.settings.repo_root.resolve()

    @property
    def limits(self):
        return self.settings.limits

    def resolve(self, raw_path: str, *, must_exist: bool = True) -> Path:
        return resolve_in_repo(self.repo_root, raw_path, must_exist=must_exist)

    def resolve_readable(self, raw_path: str) -> Path:
        """Resolve a path and enforce the secret-file and excluded-dir rules."""
        path = self.resolve(raw_path)
        check_readable(
            self.repo_root,
            path,
            excluded_dirs=self.settings.excluded_dirs,
            sensitive_patterns=self.settings.sensitive_patterns,
        )
        return path

    def rel(self, path: Path) -> str:
        return to_relative(self.repo_root, path)


def normalize_extension(extension: str | None) -> str | None:
    if extension is None or not extension.strip():
        return None
    ext = extension.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def is_probably_binary(sample: bytes) -> bool:
    return b"\x00" in sample[:4096]


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n... [truncated]", True


def tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "[... earlier output truncated ...]\n" + text[-max_chars:]
