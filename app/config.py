"""Runtime configuration for DevPilot.

All tunable limits live here so security boundaries are easy to audit.
Values come from environment variables (optionally loaded from a .env file).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Groq's OpenAI-compatible endpoint. The official Groq SDK uses this host by
# default; it is kept here for documentation and for OpenAI-compatible clients.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Production model on Groq with tool-use support (verified against the Groq
# model catalog on 2026-10-16). Override with DEVPILOT_MODEL.
DEFAULT_MODEL = "openai/gpt-oss-120b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env",
        "virtualenv",
        "node_modules",
        "__pycache__",
        "site-packages",
        "build",
        "dist",
        ".eggs",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        ".ssh",
        ".aws",
        ".gnupg",
    }
)

# Matched case-insensitively against the file name with fnmatch.
DEFAULT_SENSITIVE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.crt",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "credentials",
    "credentials.*",
    "*credentials*.json",
    "secrets.*",
    "*secret*.json",
    "*secret*.yaml",
    "*secret*.yml",
    "service-account*.json",
    ".netrc",
    ".pypirc",
    ".npmrc",
    ".htpasswd",
    "*.kdbx",
    ".git-credentials",
)

# Template files that are conventionally safe to read.
SAFE_ENV_TEMPLATES: frozenset[str] = frozenset({".env.example", ".env.sample", ".env.template"})


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Limits:
    """Hard limits applied to every tool, regardless of model arguments."""

    max_file_bytes: int = 200_000
    max_list_results: int = 300
    max_search_matches: int = 100
    max_search_file_bytes: int = 500_000
    max_snippet_chars: int = 200
    max_commits: int = 50
    max_diff_chars: int = 20_000
    max_tool_output_chars: int = 24_000
    max_ast_files: int = 2_000
    max_graph_nodes: int = 200
    max_trace_depth: int = 6
    max_test_output_chars: int = 4_000
    max_report_chars: int = 60_000
    git_timeout_seconds: int = 15


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    reports_dir: Path
    model: str = DEFAULT_MODEL
    groq_api_key: str | None = field(default=None, repr=False)
    api_timeout_seconds: float = 60.0
    temperature: float = 0.2
    max_completion_tokens: int = 4096
    max_tool_rounds: int = 10
    max_tool_calls: int = 30
    tool_timeout_seconds: float = 60.0
    test_timeout_seconds: int = 120
    max_test_runs: int = 3
    parallel_tool_calls: bool = False
    allow_test_execution: bool = True
    allow_reports: bool = True
    python_executable: str = sys.executable
    excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS
    sensitive_patterns: tuple[str, ...] = DEFAULT_SENSITIVE_PATTERNS
    limits: Limits = field(default_factory=Limits)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}")


def _env_number(name: str, default: float, cast: type, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def load_settings(
    repo_path: str | None = None,
    model: str | None = None,
    env_file: str | Path | None = None,
) -> Settings:
    """Build Settings from CLI overrides, environment variables and defaults."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    repo_raw = repo_path or os.getenv("DEVPILOT_REPO_PATH") or os.getcwd()
    repo_root = Path(repo_raw).expanduser().resolve()
    if not repo_root.is_dir():
        raise ConfigError(f"Repository path is not a directory: {repo_root}")

    reports_raw = os.getenv("DEVPILOT_REPORTS_DIR")
    reports_dir = Path(reports_raw).expanduser().resolve() if reports_raw else PROJECT_ROOT / "reports"

    python_exe = os.getenv("DEVPILOT_PYTHON") or sys.executable
    if not Path(python_exe).is_file():
        raise ConfigError(f"DEVPILOT_PYTHON does not point to a file: {python_exe}")

    return Settings(
        repo_root=repo_root,
        reports_dir=reports_dir,
        model=model or os.getenv("DEVPILOT_MODEL") or DEFAULT_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        api_timeout_seconds=_env_number("DEVPILOT_API_TIMEOUT", 60.0, float, 1),
        max_tool_rounds=int(_env_number("DEVPILOT_MAX_TOOL_ROUNDS", 10, int, 1)),
        max_tool_calls=int(_env_number("DEVPILOT_MAX_TOOL_CALLS", 30, int, 1)),
        tool_timeout_seconds=_env_number("DEVPILOT_TOOL_TIMEOUT", 60.0, float, 1),
        test_timeout_seconds=int(_env_number("DEVPILOT_TEST_TIMEOUT", 120, int, 1)),
        parallel_tool_calls=_env_bool("DEVPILOT_PARALLEL_TOOL_CALLS", False),
        allow_test_execution=_env_bool("DEVPILOT_ALLOW_TEST_EXECUTION", True),
        allow_reports=_env_bool("DEVPILOT_ALLOW_REPORTS", True),
        python_executable=python_exe,
        excluded_dirs=DEFAULT_EXCLUDED_DIRS | frozenset(_env_list("DEVPILOT_EXTRA_EXCLUDED_DIRS")),
        sensitive_patterns=DEFAULT_SENSITIVE_PATTERNS + _env_list("DEVPILOT_EXTRA_SENSITIVE_PATTERNS"),
    )
