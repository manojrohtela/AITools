"""Shared AST loading for the static-analysis tools.

Parses every permitted ``.py`` file in the repository once per tool call and
maps file paths to dotted module names (supports flat and ``src/`` layouts).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from app.security.path_validator import iter_repo_files
from app.tools.common import ToolContext, ToolError

ImportScope = Literal["module", "function", "type_checking"]


@dataclass
class ParsedModule:
    name: str
    path: str  # relative to repo root
    is_package: bool
    tree: ast.Module
    lines: list[str]

    def snippet(self, line: int, max_chars: int = 200) -> str:
        if 1 <= line <= len(self.lines):
            return self.lines[line - 1].strip()[:max_chars]
        return ""


@dataclass
class PythonProject:
    modules: dict[str, ParsedModule] = field(default_factory=dict)
    parse_errors: list[dict[str, str]] = field(default_factory=list)
    files_truncated: bool = False

    def module_for_path(self, rel_path: str) -> ParsedModule | None:
        for module in self.modules.values():
            if module.path == rel_path:
                return module
        return None


@dataclass(frozen=True)
class ImportRecord:
    target: str | None  # absolute dotted module, None if relative import escapes the root
    names: tuple[tuple[str, str | None], ...]  # (imported name, alias) for "from" imports
    alias: str | None  # alias for "import x as y"
    is_from: bool
    line: int
    scope: ImportScope


def module_name_for(rel_path: str) -> tuple[str, bool]:
    parts = list(PurePosixPath(rel_path).with_suffix("").parts)
    if len(parts) > 1 and parts[0] == "src":
        parts = parts[1:]
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return (".".join(parts) if parts else "__init__"), is_package


def load_project(ctx: ToolContext) -> PythonProject:
    project = PythonProject()
    for index, path in enumerate(
        iter_repo_files(
            ctx.repo_root,
            ctx.repo_root,
            excluded_dirs=ctx.settings.excluded_dirs,
            sensitive_patterns=ctx.settings.sensitive_patterns,
            extension=".py",
        )
    ):
        if index >= ctx.limits.max_ast_files:
            project.files_truncated = True
            break
        rel_path = ctx.rel(path)
        try:
            if path.stat().st_size > ctx.limits.max_search_file_bytes:
                project.parse_errors.append({"path": rel_path, "error": "file too large to analyze"})
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, ValueError, OSError) as exc:
            project.parse_errors.append({"path": rel_path, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        name, is_package = module_name_for(rel_path)
        if name in project.modules:  # e.g. duplicate under src/ and root
            name = f"{name}@{rel_path}"
        project.modules[name] = ParsedModule(name, rel_path, is_package, tree, source.splitlines())
    return project


def resolve_relative(module: ParsedModule, level: int, target: str | None) -> str | None:
    """Turn ``from ..x import y`` into an absolute dotted module name."""
    if level == 0:
        return target
    package_parts = module.name.split(".") if module.is_package else module.name.split(".")[:-1]
    drop = level - 1
    if drop > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - drop]
    if target:
        base = base + target.split(".")
    return ".".join(base) or None


def _is_type_checking_test(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    return isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"


def iter_imports(module: ParsedModule) -> Iterator[ImportRecord]:
    """Yield imports with the scope they run in (module, function, TYPE_CHECKING)."""

    def walk(node: ast.AST, scope: ImportScope) -> Iterator[ImportRecord]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    yield ImportRecord(alias.name, (), alias.asname, False, child.lineno, scope)
            elif isinstance(child, ast.ImportFrom):
                target = resolve_relative(module, child.level, child.module)
                names = tuple((alias.name, alias.asname) for alias in child.names)
                yield ImportRecord(target, names, None, True, child.lineno, scope)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                yield from walk(child, "function" if scope == "module" else scope)
            elif isinstance(child, ast.If) and _is_type_checking_test(child.test):
                for stmt in child.body:
                    yield from walk(ast.Module(body=[stmt], type_ignores=[]), "type_checking")
                for stmt in child.orelse:
                    yield from walk(ast.Module(body=[stmt], type_ignores=[]), scope)
            else:
                yield from walk(child, scope)

    yield from walk(module.tree, "module")


def in_scope(module: ParsedModule, scope_path: str | None, scope_is_dir: bool) -> bool:
    if scope_path is None or scope_path == ".":
        return True
    if scope_is_dir:
        return module.path.startswith(scope_path.rstrip("/") + "/")
    return module.path == scope_path


def resolve_scope(ctx: ToolContext, module_path: str | None) -> tuple[str | None, bool]:
    if not module_path:
        return None, True
    path: Path = ctx.resolve_readable(module_path)
    if path.is_file() and path.suffix != ".py":
        raise ToolError("module_path must be a Python file or a directory")
    return ctx.rel(path), path.is_dir()
