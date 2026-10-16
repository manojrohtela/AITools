"""Shared fixtures: a throw-away Git repository built from examples/sample_app."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import Settings
from app.security.execution_policy import ExecutionPolicy
from app.tools.common import ToolContext

SAMPLE_APP = Path(__file__).resolve().parent.parent / "examples" / "sample_app"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """sample_app as a git repo with: 2 commits, an unstaged edit, a staged file,
    an untracked file, and planted secret files that must never be exposed."""
    repo = tmp_path / "repo"
    shutil.copytree(SAMPLE_APP, repo, ignore=shutil.ignore_patterns("__pycache__"))
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "Initial commit")
    (repo / "auth" / "tokens.py").write_text(
        (repo / "auth" / "tokens.py").read_text() + "\n\ndef revoke_token(token: str) -> None:\n    pass\n"
    )
    git(repo, "add", "auth/tokens.py")
    git(repo, "commit", "-q", "-m", "Add revoke_token")
    # Working-tree state
    passwords = repo / "auth" / "passwords.py"
    passwords.write_text(passwords.read_text() + "\n# TODO: use bcrypt\n")
    (repo / "web" / "staged.py").write_text("STAGED = True\n")
    git(repo, "add", "web/staged.py")
    (repo / "notes.txt").write_text("untracked notes\n")
    # Secrets that must stay hidden
    (repo / ".env").write_text("API_TOKEN=super-secret-value\n")
    (repo / ".env.example").write_text("API_TOKEN=\n")
    (repo / "id_rsa").write_text("-----BEGIN PRIVATE KEY-----\nsuper-secret-value\n")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "lib.py").write_text("def login():\n    pass\n")
    return repo


@pytest.fixture
def settings(sample_repo: Path, tmp_path: Path) -> Settings:
    return Settings(
        repo_root=sample_repo,
        reports_dir=tmp_path / "reports",
        groq_api_key=None,
        python_executable=sys.executable,
        test_timeout_seconds=60,
    )


@pytest.fixture
def ctx(settings: Settings) -> ToolContext:
    return ToolContext(settings=settings)


@pytest.fixture
def policy(settings: Settings) -> ExecutionPolicy:
    return ExecutionPolicy.from_settings(settings)
