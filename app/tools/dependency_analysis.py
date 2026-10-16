"""Category C: Python import/dependency analysis using the AST."""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.tools.common import ToolContext
from app.tools.python_ast import ImportRecord, PythonProject, in_scope, iter_imports, load_project, resolve_scope

MAX_MODULES_IN_OUTPUT = 150
MAX_EDGES_IN_OUTPUT = 300


class AnalyzeDependenciesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module_path: str | None = Field(
        None, max_length=500, description="Optional Python file or directory to focus on (repo-relative)."
    )


def _internal_targets(record: ImportRecord, modules: set[str]) -> list[str]:
    """Internal modules an import refers to (empty if not internal)."""
    if record.target is None:
        return []
    found: list[str] = []
    if record.is_from:
        for name, _alias in record.names:
            candidate = f"{record.target}.{name}"
            if candidate in modules:  # "from pkg import submodule"
                found.append(candidate)
        if not found and record.target in modules:
            found.append(record.target)
        return found
    parts = record.target.split(".")
    for end in range(len(parts), 0, -1):  # "import a.b.c" -> nearest known module
        candidate = ".".join(parts[:end])
        if candidate in modules:
            return [candidate]
    return []


def strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    """Iterative Tarjan's algorithm."""
    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in sorted(graph):
        if root in index_of:
            continue
        work = [(root, iter(sorted(graph.get(root, ()))))]
        index_of[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index_of:
                    index_of[child] = lowlink[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_of[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return components


def _cycle_path(graph: dict[str, set[str]], component: list[str]) -> list[str]:
    """A concrete path start -> ... -> start inside one strongly connected component."""
    members = set(component)
    start = component[0]
    queue = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        for nxt in sorted(graph.get(path[-1], ())):
            if nxt == start:
                return path + [start]
            if nxt in members and nxt not in seen:
                seen.add(nxt)
                queue.append(path + [nxt])
    return component + [start]


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles = []
    for component in strongly_connected_components(graph):
        node = component[0]
        if len(component) > 1 or node in graph.get(node, set()):
            cycles.append(_cycle_path(graph, component))
    return cycles


def _classify_external(target: str | None, internal_tops: set[str]) -> str:
    if target is None:
        return "unresolved"
    top = target.split(".")[0]
    if top in internal_tops:
        return "unresolved"
    if top in sys.stdlib_module_names:
        return "stdlib"
    return "external"


def build_import_graph(project: PythonProject) -> tuple[dict[str, set[str]], dict[str, set[str]], list[dict[str, Any]]]:
    """Return (module-level graph, graph incl. deferred imports, edge list)."""
    module_names = set(project.modules)
    strict: dict[str, set[str]] = defaultdict(set)
    full: dict[str, set[str]] = defaultdict(set)
    edges: list[dict[str, Any]] = []
    for name, module in project.modules.items():
        strict.setdefault(name, set())
        full.setdefault(name, set())
        for record in iter_imports(module):
            for target in _internal_targets(record, module_names):
                full[name].add(target)
                if record.scope == "module":
                    strict[name].add(target)
                edges.append(
                    {"from": name, "to": target, "line": record.line, "scope": record.scope, "path": module.path}
                )
    return dict(strict), dict(full), edges


def analyze_dependencies(ctx: ToolContext, args: AnalyzeDependenciesArgs) -> dict[str, Any]:
    scope_path, scope_is_dir = resolve_scope(ctx, args.module_path)
    project = load_project(ctx)
    module_names = set(project.modules)
    internal_tops = {name.split(".")[0] for name in module_names}
    strict_graph, full_graph, edges = build_import_graph(project)

    scoped = [m for m in project.modules.values() if in_scope(m, scope_path, scope_is_dir)]
    scoped_names = {m.name for m in scoped}

    modules_out = []
    external_counts: dict[str, int] = defaultdict(int)
    for module in sorted(scoped, key=lambda m: m.path)[:MAX_MODULES_IN_OUTPUT]:
        internal, stdlib, external, unresolved = [], set(), set(), set()
        for record in iter_imports(module):
            targets = _internal_targets(record, module_names)
            if targets:
                internal.extend({"module": t, "line": record.line, "scope": record.scope} for t in targets)
                continue
            kind = _classify_external(record.target, internal_tops)
            label = record.target or f"<relative import at line {record.line}>"
            if kind == "stdlib":
                stdlib.add(label.split(".")[0])
            elif kind == "external":
                external.add(label.split(".")[0])
                external_counts[label.split(".")[0]] += 1
            else:
                unresolved.add(label)
        modules_out.append(
            {
                "module": module.name,
                "path": module.path,
                "internal_imports": internal,
                "external_imports": sorted(external),
                "stdlib_imports": sorted(stdlib),
                "unresolved_imports": sorted(unresolved),
            }
        )

    module_cycles = [c for c in find_cycles(strict_graph) if scoped_names & set(c)]
    module_cycle_sets = {frozenset(c) for c in module_cycles}
    deferred_candidates = [
        c for c in find_cycles(full_graph) if scoped_names & set(c) and frozenset(c) not in module_cycle_sets
    ]
    scoped_edges = [e for e in edges if e["from"] in scoped_names]
    imported_by: dict[str, int] = defaultdict(int)
    for edge in edges:
        imported_by[edge["to"]] += 1

    return {
        "scope": scope_path or ".",
        "modules_analyzed": len(scoped),
        "modules": modules_out,
        "modules_truncated": len(scoped) > MAX_MODULES_IN_OUTPUT,
        "dependency_edges": scoped_edges[:MAX_EDGES_IN_OUTPUT],
        "edges_truncated": len(scoped_edges) > MAX_EDGES_IN_OUTPUT,
        "most_imported_internal_modules": sorted(imported_by.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
        "external_packages": sorted(external_counts),
        "circular_dependencies": {
            "module_level_cycles": module_cycles,
            "cycles_only_via_deferred_or_type_checking_imports": deferred_candidates,
            "interpretation": (
                "module_level_cycles are cycles present in the static graph of module-level imports. "
                "They are cycle candidates: whether they fail at runtime depends on import order and "
                "which names are used at import time. Cycles that only exist through function-level "
                "or TYPE_CHECKING imports usually do not break imports."
            ),
        },
        "parse_errors": project.parse_errors,
        "files_truncated": project.files_truncated,
    }
