"""DevPilot command-line interface.

Usage:
    python -m app.main                       # interactive, repo = DEVPILOT_REPO_PATH or cwd
    python -m app.main --repo ../some-project
    python -m app.main --repo . -q "Where is the login flow implemented?"
"""

from __future__ import annotations

import argparse
import json
import sys

from app.agent.schemas import AgentEvent, AgentResult
from app.agent.tool_loop import GroqChatClient, ToolCallingAgent
from app.config import ConfigError, Settings, load_settings
from app.security.execution_policy import ExecutionPolicy
from app.tools.common import ToolContext
from app.tools.registration import build_registry

BANNER = """DevPilot - AI developer assistant (read-only)
Repository : {repo}
Model      : {model}
Type a question, '/tools' to list tools, or '/exit' to quit. Ctrl+C cancels."""


def print_event(event: AgentEvent) -> None:
    data = event.data
    if event.kind == "tool_start":
        print(f"  -> calling {data['name']}", flush=True)
    elif event.kind == "tool_end":
        status = "ok" if data["ok"] else f"error: {data['error_type']}"
        args = json.dumps(data.get("arguments", {}), ensure_ascii=False)
        print(f"     {data['name']}({args}) [{status}, {data['duration']}s]", flush=True)
    elif event.kind == "retry":
        print(f"  !! retrying: {data['reason']}", flush=True)
    elif event.kind == "error":
        print(f"  !! {data['message']}", flush=True)


def print_result(result: AgentResult) -> None:
    print()
    if result.status in {"completed", "max_rounds"}:
        if result.status == "max_rounds":
            print("[stopped at the tool-round limit; partial answer]")
        print(result.answer)
    else:
        print(f"[{result.status}] {result.error}")
    print(f"\n({len(result.tool_results)} tool calls, {result.rounds} model rounds)")


def build_agent(settings: Settings) -> ToolCallingAgent:
    registry = build_registry()
    return ToolCallingAgent(
        client=GroqChatClient(settings),
        registry=registry,
        context=ToolContext(settings=settings),
        policy=ExecutionPolicy.from_settings(settings),
        max_rounds=settings.max_tool_rounds,
        on_event=print_event,
    )


def ask(agent: ToolCallingAgent, question: str) -> None:
    try:
        print_result(agent.run(question))
    except KeyboardInterrupt:
        print("\n[cancelled]")


def interactive(agent: ToolCallingAgent, settings: Settings) -> None:
    print(BANNER.format(repo=settings.repo_root, model=settings.model))
    while True:
        try:
            question = input("\ndevpilot> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not question:
            continue
        if question in {"/exit", "/quit"}:
            return
        if question == "/tools":
            for tool in agent.registry.tools():
                enabled = "" if agent.policy.is_allowed(tool.permission) else " (disabled)"
                print(f"  {tool.name:<22} [{tool.category.value}]{enabled}")
            continue
        ask(agent, question)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="devpilot", description="AI developer assistant using Groq tool calling.")
    parser.add_argument("--repo", help="Repository to analyze (default: DEVPILOT_REPO_PATH or current directory).")
    parser.add_argument("--model", help="Groq model id (default: DEVPILOT_MODEL or openai/gpt-oss-120b).")
    parser.add_argument("-q", "--question", help="Ask one question and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_settings(repo_path=args.repo, model=args.model)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        return 2

    agent = build_agent(settings)
    if args.question:
        ask(agent, args.question)
    else:
        interactive(agent, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
