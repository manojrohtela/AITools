"""Tool registration: every tool DevPilot can run is listed here explicitly."""

from app.agent.registry import Tool, ToolCategory, ToolRegistry
from app.security.execution_policy import ToolPermission as P
from app.tools import call_graph, dependency_analysis, file_tools, git_tools, report_tools, test_tools


def build_registry() -> ToolRegistry:
    """Register every tool explicitly. Adding a tool = one entry here."""
    registry = ToolRegistry()
    tools = [
        Tool(
            "list_files",
            "List files and directories (recursively) inside the repository. Excludes .git, virtualenvs, "
            "build output and secret files. Use first to orient yourself.",
            file_tools.ListFilesArgs,
            file_tools.list_files,
            P.READ_FILES,
            ToolCategory.FILE,
        ),
        Tool(
            "search_code",
            "Case-insensitive literal text search across source files. Returns file paths, line numbers and "
            "snippets. Use to locate where a concept or identifier appears.",
            file_tools.SearchCodeArgs,
            file_tools.search_code,
            P.READ_FILES,
            ToolCategory.FILE,
        ),
        Tool(
            "read_file",
            "Read one source file. Returns the content with 1-based line numbers. Secret files are blocked.",
            file_tools.ReadFileArgs,
            file_tools.read_file,
            P.READ_FILES,
            ToolCategory.FILE,
        ),
        Tool(
            "get_git_status",
            "Read-only git status: current branch, staged, modified and untracked files.",
            git_tools.GitStatusArgs,
            git_tools.get_git_status,
            P.READ_GIT,
            ToolCategory.GIT,
        ),
        Tool(
            "get_commit_history",
            "Recent commits (hash, author, ISO date, subject), newest first.",
            git_tools.CommitHistoryArgs,
            git_tools.get_commit_history,
            P.READ_GIT,
            ToolCategory.GIT,
        ),
        Tool(
            "get_diff",
            "Uncommitted changes: changed files, line counts and a unified diff (optionally one file, "
            "or staged changes).",
            git_tools.DiffArgs,
            git_tools.get_diff,
            P.READ_GIT,
            ToolCategory.GIT,
        ),
        Tool(
            "analyze_dependencies",
            "AST-based Python import analysis: per-module internal/stdlib/external imports, dependency "
            "edges and circular-import candidates.",
            dependency_analysis.AnalyzeDependenciesArgs,
            dependency_analysis.analyze_dependencies,
            P.ANALYZE,
            ToolCategory.STATIC_ANALYSIS,
        ),
        Tool(
            "find_callers",
            "AST-based search for functions that call a given Python function. Separates confirmed callers "
            "from uncertain matches (unknown receiver types, dynamic lookups).",
            call_graph.FindCallersArgs,
            call_graph.find_callers,
            P.ANALYZE,
            ToolCategory.STATIC_ANALYSIS,
        ),
        Tool(
            "trace_function_flow",
            "Trace the static call tree starting from a Python function, up to a maximum depth. Reports "
            "call paths, cycles/recursion and unresolved calls.",
            call_graph.TraceFunctionFlowArgs,
            call_graph.trace_function_flow,
            P.ANALYZE,
            ToolCategory.STATIC_ANALYSIS,
        ),
        Tool(
            "run_tests",
            "Run pytest on a test file or directory (optionally 'file.py::test_name') with fixed options and "
            "a timeout. Returns a run_id, exit code and pass/fail counts.",
            test_tools.RunTestsArgs,
            test_tools.run_tests,
            P.EXECUTE_TESTS,
            ToolCategory.TEST,
        ),
        Tool(
            "analyze_test_results",
            "Analyze results of a run_tests call (by run_id) or raw pytest output: counts, failure details, "
            "and heuristic failure categories kept separate from observed facts.",
            test_tools.AnalyzeTestResultsArgs,
            test_tools.analyze_test_results,
            P.ANALYZE,
            ToolCategory.TEST,
        ),
        Tool(
            "create_report",
            "Write a Markdown report to DevPilot's reports directory. Only call when the user asks for a "
            "report. Mark each finding 'confirmed' (backed by tool output) or 'hypothesis'. Reference test "
            "runs by run_id.",
            report_tools.CreateReportArgs,
            report_tools.create_report,
            P.WRITE_REPORTS,
            ToolCategory.REPORT,
        ),
    ]
    for tool in tools:
        registry.register(tool)
    return registry
