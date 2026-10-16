"""Category C: AST-based caller search and bounded call-flow tracing.

Static analysis only. Each call site is resolved to one of:

* ``confirmed``  - the callee is provably a specific function in this repo
  (same-module function, ``from x import f``, ``module.f()``, ``self.method()``
  where the enclosing class defines the method, ``ClassName.method()``).
* ``uncertain``  - the name matches but the target cannot be proven
  (``obj.method()`` on an unknown type, inherited methods, ``getattr``).
* ``builtin`` / ``external`` - Python builtins or third-party/stdlib code.
* ``unresolved`` - nothing in the repository matches.
"""

from __future__ import annotations

import ast
import builtins
import difflib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.common import ToolContext, ToolError
from app.tools.python_ast import ParsedModule, PythonProject, in_scope, iter_imports, load_project, resolve_scope

BUILTIN_NAMES = frozenset(dir(builtins))
MODULE_LEVEL = "<module>"
MAX_CALLERS_IN_OUTPUT = 100
MAX_START_MATCHES = 5
MAX_PATHS_IN_OUTPUT = 50
MAX_UNRESOLVED_IN_OUTPUT = 100

CallKind = Literal["name", "self_method", "attribute", "getattr", "dynamic"]
ResolutionStatus = Literal["confirmed", "uncertain", "builtin", "external", "unresolved"]

NAME_DESCRIPTION = "Function name, e.g. 'login', or 'ClassName.method' to target a method."


class FindCallersArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    function_name: str = Field(
        ..., min_length=1, max_length=200, pattern=r"^[A-Za-z_][\w.]*$", description=NAME_DESCRIPTION
    )
    module_path: str | None = Field(
        None, max_length=500, description="Optional file/directory to restrict where callers are searched."
    )


class TraceFunctionFlowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_function: str = Field(
        ..., min_length=1, max_length=200, pattern=r"^[A-Za-z_][\w.]*$", description=NAME_DESCRIPTION
    )
    maximum_depth: int = Field(3, ge=1, le=6, description="How many call levels to follow (1-6).")


@dataclass
class CallSite:
    name: str | None
    kind: CallKind
    receiver: str | None
    line: int


@dataclass
class FunctionInfo:
    qualname: str  # "module:Class.method" / "module:func" / "module:<module>"
    name: str
    module: str
    class_name: str | None
    path: str
    line: int
    calls: list[CallSite] = field(default_factory=list)
    references: list[tuple[str, int]] = field(default_factory=list)  # name used but not called


@dataclass
class ModuleSymbols:
    functions: set[str] = field(default_factory=set)  # top-level function names
    classes: dict[str, set[str]] = field(default_factory=dict)  # class -> method names
    imported_names: dict[str, tuple[str, str]] = field(default_factory=dict)  # local -> (module, attr)
    imported_modules: dict[str, str] = field(default_factory=dict)  # local -> module


@dataclass
class Resolution:
    status: ResolutionStatus
    targets: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class CallIndex:
    project: PythonProject
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    by_name: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    symbols: dict[str, ModuleSymbols] = field(default_factory=dict)

    def module(self, name: str) -> ParsedModule:
        return self.project.modules[name]


class _Collector(ast.NodeVisitor):
    def __init__(self, module: ParsedModule, index: CallIndex, symbols: ModuleSymbols) -> None:
        self.module = module
        self.index = index
        self.symbols = symbols
        self.class_stack: list[str] = []
        self.function_stack: list[FunctionInfo] = []
        self.call_func_nodes: set[int] = set()
        self.module_function = self._add_function(MODULE_LEVEL, None, 1, register_name=False)

    def _add_function(self, name: str, class_name: str | None, line: int, register_name: bool = True) -> FunctionInfo:
        outer = [f.name for f in self.function_stack if f.name != MODULE_LEVEL]
        local = ".".join([*([class_name] if class_name else []), *outer, name])
        qualname = f"{self.module.name}:{local}"
        if qualname in self.index.functions:  # redefinition: keep both, distinguish by line
            qualname = f"{qualname}@{line}"
        info = FunctionInfo(qualname, name, self.module.name, class_name, self.module.path, line)
        self.index.functions[qualname] = info
        if register_name:
            self.index.by_name[name].append(qualname)
        return info

    @property
    def current(self) -> FunctionInfo:
        return self.function_stack[-1] if self.function_stack else self.module_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not self.class_stack and not self.function_stack:
            self.symbols.classes[node.name] = {
                item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # class_stack is cleared inside function bodies, so a non-empty stack
        # means this def sits directly in a class body: it is a method.
        class_name = self.class_stack[-1] if self.class_stack else None
        if not self.class_stack and not self.function_stack:
            self.symbols.functions.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        info = self._add_function(node.name, class_name, node.lineno)
        self.function_stack.append(info)
        saved_classes = self.class_stack
        self.class_stack = []
        for stmt in node.body:
            self.visit(stmt)
        self.class_stack = saved_classes
        self.function_stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        self.call_func_nodes.add(id(func))
        site: CallSite
        if isinstance(func, ast.Name):
            if (
                func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                site = CallSite(node.args[1].value, "getattr", _unparse(node.args[0]), node.lineno)
            else:
                site = CallSite(func.id, "name", None, node.lineno)
        elif isinstance(func, ast.Attribute):
            receiver = _unparse(func.value)
            kind: CallKind = "self_method" if receiver in {"self", "cls"} else "attribute"
            site = CallSite(func.attr, kind, receiver, node.lineno)
        else:
            site = CallSite(None, "dynamic", _unparse(func), node.lineno)
        self.current.calls.append(site)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and id(node) not in self.call_func_nodes:
            self.current.references.append((node.id, node.lineno))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and id(node) not in self.call_func_nodes:
            self.current.references.append((node.attr, node.lineno))
        self.generic_visit(node)


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)[:80]
    except Exception:  # noqa: BLE001
        return type(node).__name__


def build_call_index(project: PythonProject) -> CallIndex:
    index = CallIndex(project=project)
    for module in project.modules.values():
        symbols = ModuleSymbols()
        for record in iter_imports(module):
            if record.target is None:
                continue
            if record.is_from:
                for name, alias in record.names:
                    if name != "*":
                        symbols.imported_names[alias or name] = (record.target, name)
            elif record.alias:
                symbols.imported_modules[record.alias] = record.target
            else:
                symbols.imported_modules[record.target.split(".")[0]] = record.target.split(".")[0]
                symbols.imported_modules[record.target] = record.target
        index.symbols[module.name] = symbols
        _Collector(module, index, symbols).visit(module.tree)
    return index


def _function_in_module(index: CallIndex, module: str, name: str, hops: int = 2) -> str | None:
    """qualname of a top-level function ``name`` defined in (or re-exported by) ``module``."""
    symbols = index.symbols.get(module)
    if symbols is None:
        return None
    if name in symbols.functions:
        return f"{module}:{name}"
    if name in symbols.classes and "__init__" in symbols.classes[name]:
        return f"{module}:{name}.__init__"
    if hops > 0 and name in symbols.imported_names:  # re-export, e.g. package __init__
        source_module, source_name = symbols.imported_names[name]
        return _function_in_module(index, source_module, source_name, hops - 1)
    return None


def _method_candidates(index: CallIndex, name: str) -> list[str]:
    return [q for q in index.by_name.get(name, []) if index.functions[q].class_name]


def resolve_call(index: CallIndex, caller: FunctionInfo, site: CallSite) -> Resolution:
    symbols = index.symbols[caller.module]
    name = site.name
    if name is None:
        return Resolution("unresolved", reason=f"dynamic call target: {site.receiver}")

    if site.kind == "name":
        local = _function_in_module(index, caller.module, name, hops=0)
        if local and name not in symbols.imported_names:
            return Resolution("confirmed", [local], "defined in the same module")
        if name in symbols.imported_names:
            source_module, source_name = symbols.imported_names[name]
            target = _function_in_module(index, source_module, source_name)
            if target:
                return Resolution("confirmed", [target], f"imported from {source_module}")
            if source_module.split(".")[0] not in {m.split(".")[0] for m in index.symbols}:
                return Resolution("external", reason=f"imported from external module {source_module}")
            return Resolution("unresolved", reason=f"imported from {source_module}, definition not found")
        if name in symbols.classes:
            return Resolution("confirmed", [], "class instantiation (no __init__ defined)")
        if name in BUILTIN_NAMES:
            return Resolution("builtin")
        same_module = [q for q in index.by_name.get(name, []) if index.functions[q].module == caller.module]
        if same_module:
            return Resolution("uncertain", same_module, "matches a nested/local function name in this module")
        return Resolution("unresolved", reason="name not defined or imported in this module")

    if site.kind == "self_method":
        if caller.class_name and name in symbols.classes.get(caller.class_name, set()):
            return Resolution(
                "confirmed", [f"{caller.module}:{caller.class_name}.{name}"], "method of the enclosing class"
            )
        return Resolution(
            "uncertain",
            _method_candidates(index, name),
            "method not defined on the enclosing class (inherited or dynamic)",
        )

    if site.kind == "attribute":
        receiver = site.receiver or ""
        if receiver in symbols.imported_modules or receiver in symbols.imported_names:
            module_name = symbols.imported_modules.get(receiver)
            if module_name is None:
                source_module, source_name = symbols.imported_names[receiver]
                source_symbols = index.symbols.get(source_module)
                if source_symbols and name in source_symbols.classes.get(source_name, set()):
                    # "from pkg.mod import Klass" then Klass.method()
                    return Resolution(
                        "confirmed",
                        [f"{source_module}:{source_name}.{name}"],
                        f"method of imported class {source_name}",
                    )
                module_name = f"{source_module}.{source_name}"  # "from pkg import module" then module.f()
            if module_name in index.symbols:
                target = _function_in_module(index, module_name, name)
                if target:
                    return Resolution("confirmed", [target], f"function of module {module_name}")
            elif module_name.split(".")[0] not in {m.split(".")[0] for m in index.symbols}:
                return Resolution("external", reason=f"attribute of external module {module_name}")
        if receiver in symbols.classes and name in symbols.classes[receiver]:
            return Resolution("confirmed", [f"{caller.module}:{receiver}.{name}"], "class attribute call")
        candidates = _method_candidates(index, name) or index.by_name.get(name, [])
        if candidates:
            return Resolution(
                "uncertain", list(candidates), f"method call on '{receiver}' whose type is unknown statically"
            )
        return Resolution("unresolved", reason=f"no definition named {name!r} in the repository")

    # getattr(obj, "name")
    return Resolution("uncertain", list(index.by_name.get(name, [])), "dynamic getattr() lookup")


def _matching_definitions(index: CallIndex, raw_name: str) -> list[str]:
    parts = raw_name.split(".")
    name = parts[-1]
    class_filter = parts[-2] if len(parts) >= 2 else None
    matches = []
    for qualname in index.by_name.get(name, []):
        info = index.functions[qualname]
        if class_filter and info.class_name != class_filter and not info.module.endswith(class_filter):
            continue
        matches.append(qualname)
    return matches


def _suggest(index: CallIndex, name: str) -> list[str]:
    return difflib.get_close_matches(name.split(".")[-1], list(index.by_name), n=5, cutoff=0.6)


def _location(index: CallIndex, qualname: str) -> dict[str, Any]:
    info = index.functions[qualname]
    return {"function": qualname, "file": info.path, "line": info.line}


def find_callers(ctx: ToolContext, args: FindCallersArgs) -> dict[str, Any]:
    scope_path, scope_is_dir = resolve_scope(ctx, args.module_path)
    index = build_call_index(load_project(ctx))
    target_name = args.function_name.split(".")[-1]
    definitions = _matching_definitions(index, args.function_name)
    definition_set = set(definitions)

    confirmed: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for caller in index.functions.values():
        module = index.module(caller.module)
        if not in_scope(module, scope_path, scope_is_dir):
            continue
        called_lines = set()
        for site in caller.calls:
            if site.name != target_name:
                continue
            resolution = resolve_call(index, caller, site)
            entry = {
                "caller": caller.qualname,
                "file": caller.path,
                "line": site.line,
                "snippet": module.snippet(site.line, ctx.limits.max_snippet_chars),
                "call_kind": site.kind,
                "reason": resolution.reason,
            }
            called_lines.add(site.line)
            if resolution.status == "confirmed" and definition_set & set(resolution.targets):
                confirmed.append(entry)
            elif resolution.status in {"uncertain", "unresolved"} and (
                not resolution.targets or definition_set & set(resolution.targets) or not definitions
            ):
                uncertain.append({**entry, "status": resolution.status})
        for name, line in caller.references:
            if name == target_name and line not in called_lines:
                references.append(
                    {
                        "function": caller.qualname,
                        "file": caller.path,
                        "line": line,
                        "snippet": module.snippet(line, ctx.limits.max_snippet_chars),
                    }
                )

    # A function's own definition line is not a reference to itself.
    def_lines = {(index.functions[q].path, index.functions[q].line) for q in definitions}
    references = [r for r in references if (r["file"], r["line"]) not in def_lines]

    return {
        "function_name": args.function_name,
        "definitions": [_location(index, q) for q in definitions],
        "definition_found": bool(definitions),
        "suggestions": [] if definitions else _suggest(index, args.function_name),
        "confirmed_callers": confirmed[:MAX_CALLERS_IN_OUTPUT],
        "uncertain_callers": uncertain[:MAX_CALLERS_IN_OUTPUT],
        "references_without_call": references[:MAX_CALLERS_IN_OUTPUT],
        "counts": {"confirmed": len(confirmed), "uncertain": len(uncertain), "references": len(references)},
        "notes": (
            "confirmed = call statically resolved to the definition. uncertain = name matches but the "
            "target type is unknown (e.g. obj.method()) or the lookup is dynamic. references_without_call "
            "may be callbacks/registrations. Calls via reflection, decorators or dependency injection "
            "may be missed."
        ),
        "parse_errors": index.project.parse_errors,
    }


def trace_function_flow(ctx: ToolContext, args: TraceFunctionFlowArgs) -> dict[str, Any]:
    index = build_call_index(load_project(ctx))
    starts = _matching_definitions(index, args.start_function)
    if not starts:
        suggestions = _suggest(index, args.start_function)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolError(f"No function definition named {args.start_function!r} found.{hint}")
    max_depth = min(args.maximum_depth, ctx.limits.max_trace_depth)
    max_nodes = ctx.limits.max_graph_nodes

    edges: list[dict[str, Any]] = []
    paths: list[list[str]] = []
    cycles: list[dict[str, Any]] = []
    unresolved: dict[tuple[str, str], dict[str, Any]] = {}
    external_calls: set[str] = set()
    nodes: set[str] = set(starts[:MAX_START_MATCHES])
    expanded: set[str] = set()
    state = {"depth_limit_hit": False, "node_limit_hit": False, "deepest": 0}

    def visit(qualname: str, path: list[str]) -> None:
        depth = len(path) - 1
        state["deepest"] = max(state["deepest"], depth)
        info = index.functions[qualname]
        if depth >= max_depth:
            if info.calls:
                state["depth_limit_hit"] = True
            paths.append(path)
            return
        if qualname in expanded:
            paths.append(path + ["(already expanded above)"])
            return
        expanded.add(qualname)

        children: dict[str, int] = {}
        for site in info.calls:
            resolution = resolve_call(index, info, site)
            if resolution.status == "confirmed":
                for target in resolution.targets:
                    if target in index.functions:
                        children.setdefault(target, site.line)
            elif resolution.status in {"builtin", "external"}:
                external_calls.add(site.name or "?")
            else:
                key = (qualname, site.name or site.receiver or "?")
                unresolved.setdefault(
                    key,
                    {
                        "from_function": qualname,
                        "file": info.path,
                        "line": site.line,
                        "call": site.name or site.receiver,
                        "status": resolution.status,
                        "reason": resolution.reason,
                        "candidates": resolution.targets[:5],
                    },
                )
        if not children:
            paths.append(path)
            return
        for target, line in children.items():
            edges.append({"caller": qualname, "callee": target, "file": info.path, "line": line, "depth": depth + 1})
            if target in path:
                kind = "direct recursion" if target == qualname else "indirect recursion (cycle)"
                cycles.append({"cycle": path[path.index(target) :] + [target], "kind": kind})
                paths.append(path + [f"{target} (cycle)"])
                continue
            if target not in nodes and len(nodes) >= max_nodes:
                state["node_limit_hit"] = True
                continue
            nodes.add(target)
            visit(target, path + [target])

    for start in starts[:MAX_START_MATCHES]:
        visit(start, [start])

    return {
        "start_function": args.start_function,
        "start_definitions": [_location(index, q) for q in starts[:MAX_START_MATCHES]],
        "ambiguous_start": len(starts) > 1,
        "maximum_depth": max_depth,
        "deepest_level_reached": state["deepest"],
        "depth_limit_hit": state["depth_limit_hit"],
        "node_limit_hit": state["node_limit_hit"],
        "paths": [" -> ".join(p) for p in paths[:MAX_PATHS_IN_OUTPUT]],
        "paths_truncated": len(paths) > MAX_PATHS_IN_OUTPUT,
        "edges": edges[:300],
        "nodes": [_location(index, q) for q in sorted(nodes)],
        "cycles": cycles,
        "unresolved_references": list(unresolved.values())[:MAX_UNRESOLVED_IN_OUTPUT],
        "builtin_or_external_calls": sorted(external_calls)[:100],
        "limitations": (
            "Static AST analysis only. Only calls resolved with certainty are followed. Dynamic dispatch, "
            "inheritance, decorators, reflection, monkey-patching and runtime-generated calls may be missed; "
            "see unresolved_references for calls that could not be resolved."
        ),
    }
