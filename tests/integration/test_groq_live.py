"""Manual integration test against the real Groq API.

Skipped unless both are set:
    GROQ_API_KEY=...  DEVPILOT_RUN_INTEGRATION=1

Run:
    DEVPILOT_RUN_INTEGRATION=1 pytest tests/integration -m integration -v -s
"""

from __future__ import annotations

import os

import pytest

from app.agent.tool_loop import GroqChatClient, ToolCallingAgent
from app.config import load_settings
from app.security.execution_policy import ExecutionPolicy
from app.tools.common import ToolContext
from app.tools.registration import build_registry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.getenv("GROQ_API_KEY") and os.getenv("DEVPILOT_RUN_INTEGRATION") == "1"),
        reason="set GROQ_API_KEY and DEVPILOT_RUN_INTEGRATION=1 to call the real Groq API",
    ),
]


def test_live_groq_tool_calling(sample_repo, tmp_path):
    settings = load_settings(repo_path=str(sample_repo))
    from dataclasses import replace

    settings = replace(settings, reports_dir=tmp_path / "reports")
    events = []
    agent = ToolCallingAgent(
        client=GroqChatClient(settings),
        registry=build_registry(),
        context=ToolContext(settings=settings),
        policy=ExecutionPolicy.from_settings(settings),
        max_rounds=settings.max_tool_rounds,
        on_event=events.append,
    )
    result = agent.run(
        "Which functions call the login function? Use the find_callers tool, then answer with file:line references."
    )
    print(f"\nmodel={settings.model} status={result.status} tools={result.tools_used}\n{result.answer}")
    assert result.status in {"completed", "max_rounds"}, result.error
    assert "find_callers" in result.tools_used
    assert "views.py" in result.answer
