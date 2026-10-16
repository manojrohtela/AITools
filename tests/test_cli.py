from app import main as cli
from app.agent.schemas import AgentEvent, AgentResult


def test_cli_requires_api_key(monkeypatch, capsys, sample_repo):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda **kw: _settings(sample_repo, None))
    assert cli.main(["--repo", str(sample_repo), "-q", "hi"]) == 2
    assert "GROQ_API_KEY is not set" in capsys.readouterr().err


def test_cli_rejects_missing_repo(capsys, tmp_path):
    assert cli.main(["--repo", str(tmp_path / "missing"), "-q", "hi"]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_print_event_shows_tool_and_safe_arguments(capsys):
    cli.print_event(
        AgentEvent(
            "tool_end",
            {
                "name": "read_file",
                "ok": True,
                "error_type": None,
                "duration": 0.01,
                "arguments": {"file_path": "a.py", "api_key": "***redacted***"},
            },
        )
    )
    out = capsys.readouterr().out
    assert "read_file" in out and "a.py" in out and "***redacted***" in out


def test_print_result_formats_errors(capsys):
    cli.print_result(AgentResult("api_timeout", "", 1, error="Groq API timed out"))
    assert "[api_timeout] Groq API timed out" in capsys.readouterr().out


def test_interactive_exits_on_eof_and_ctrl_c(monkeypatch, capsys, sample_repo):
    settings = _settings(sample_repo, "gsk_x")
    for exc in (EOFError, KeyboardInterrupt):

        def raise_exc(prompt, exc=exc):
            raise exc

        monkeypatch.setattr("builtins.input", raise_exc)
        cli.interactive(cli.build_agent(settings), settings)
    output = capsys.readouterr().out
    assert output.count("Bye.") == 2 and "gsk_x" not in output


def _settings(repo, key):
    import sys

    from app.config import Settings

    return Settings(repo_root=repo, reports_dir=repo / "r", groq_api_key=key, python_executable=sys.executable)
