"""Execution policy: which tools may run, how often, and for how long."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from app.config import Settings


class ToolPermission(StrEnum):
    READ_FILES = "read_files"
    READ_GIT = "read_git"
    ANALYZE = "analyze"
    EXECUTE_TESTS = "execute_tests"
    WRITE_REPORTS = "write_reports"


class PermissionDeniedError(Exception):
    """Raised when policy forbids a tool call."""


# Git is only ever invoked with these read-only subcommands.
ALLOWED_GIT_SUBCOMMANDS: frozenset[str] = frozenset({"status", "log", "diff", "rev-parse"})

# Options that could write files or run programs; rejected defensively even
# though DevPilot builds every git argument list itself.
FORBIDDEN_GIT_OPTION_PREFIXES: tuple[str, ...] = (
    "--output",
    "--exec",
    "--upload-pack",
    "--receive-pack",
    "--ext-diff",
    "--textconv",
    "-o",
)

# Hardening flags placed before every git subcommand. A repository's own
# config can define commands (fsmonitor, hooks, pager, external diff); these
# overrides stop a hostile repository from executing code through git.
GIT_SAFETY_FLAGS: tuple[str, ...] = (
    "--no-pager",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.pager=cat",
    "-c",
    "diff.external=",
    "-c",
    "protocol.allow=never",
)


def validate_git_args(args: list[str]) -> None:
    """Allow only known read-only subcommands and block dangerous options."""
    if not args:
        raise PermissionDeniedError("Empty git command")
    if args[0] not in ALLOWED_GIT_SUBCOMMANDS:
        raise PermissionDeniedError(f"git subcommand {args[0]!r} is not allowed")
    for arg in args[1:]:
        if arg == "--":
            break
        if arg.startswith(FORBIDDEN_GIT_OPTION_PREFIXES):
            raise PermissionDeniedError(f"git option {arg!r} is not allowed")


@dataclass
class ExecutionPolicy:
    allowed_permissions: frozenset[ToolPermission]
    max_total_calls: int
    tool_timeout_seconds: float
    test_timeout_seconds: float
    per_tool_limits: dict[str, int] = field(default_factory=dict)
    _counts: Counter = field(default_factory=Counter, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> ExecutionPolicy:
        allowed = {ToolPermission.READ_FILES, ToolPermission.READ_GIT, ToolPermission.ANALYZE}
        if settings.allow_test_execution:
            allowed.add(ToolPermission.EXECUTE_TESTS)
        if settings.allow_reports:
            allowed.add(ToolPermission.WRITE_REPORTS)
        return cls(
            allowed_permissions=frozenset(allowed),
            max_total_calls=settings.max_tool_calls,
            tool_timeout_seconds=settings.tool_timeout_seconds,
            test_timeout_seconds=settings.test_timeout_seconds,
            per_tool_limits={"run_tests": settings.max_test_runs, "create_report": 3},
        )

    def is_allowed(self, permission: ToolPermission) -> bool:
        return permission in self.allowed_permissions

    def authorize(self, tool_name: str, permission: ToolPermission) -> None:
        """Check permission and budgets, then count the call."""
        if not self.is_allowed(permission):
            raise PermissionDeniedError(
                f"Tool {tool_name!r} requires permission {permission.value!r}, which is disabled by configuration"
            )
        if sum(self._counts.values()) >= self.max_total_calls:
            raise PermissionDeniedError(
                f"Tool-call budget exhausted ({self.max_total_calls} calls). "
                "Answer with the information already collected."
            )
        limit = self.per_tool_limits.get(tool_name)
        if limit is not None and self._counts[tool_name] >= limit:
            raise PermissionDeniedError(f"Tool {tool_name!r} may run at most {limit} times per request")
        self._counts[tool_name] += 1

    def timeout_for(self, permission: ToolPermission) -> float:
        if permission is ToolPermission.EXECUTE_TESTS:
            # The test runner enforces its own subprocess timeout; leave headroom.
            return self.test_timeout_seconds + 15
        return self.tool_timeout_seconds

    def reset(self) -> None:
        self._counts.clear()
