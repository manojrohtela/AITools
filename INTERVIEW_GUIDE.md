# DevPilot: Interview Guide

This guide explains the concepts behind DevPilot and the decisions in its actual code. Every answer points at a real file, so you can open it during a walkthrough.

---

## 1. What is AI tool calling?

Tool calling, also called function calling, lets an LLM ask the application to run a function on its behalf.

1. The application sends the model a **JSON Schema description** of each available function.
2. When the model decides a function would help, it does not answer in prose. It returns a **structured request**: a function name plus JSON arguments.
3. The application runs the function and sends the **result** back as a new message.
4. The model uses that result to continue, which may mean another call or a final answer.

The model only ever produces text: a name and a JSON string. **The application does all execution.**

## 2. Tool calling vs a normal function call

| Normal function call | Tool call |
|---|---|
| The programmer decides *which* function runs and *when*, at write time. | The model decides at runtime, from natural language and earlier results. |
| Arguments come from typed code. | Arguments are model-generated JSON and must be treated as untrusted input. |
| Errors are raised as exceptions. | Errors are returned *to the model* as data, so it can adapt. In DevPilot that is `{"ok": false, "error": {...}}`. |
| One call site. | A loop: request, execute, append the result, request again. |

## 3. Why tools are useful

- **Grounding.** The model reads the real `auth/service.py` instead of guessing what a login function usually looks like.
- **Fresh and private data.** Git state and local files are not in the training data.
- **Deterministic capabilities.** AST parsing, graph traversal and pytest are exact. LLMs are not.
- **Composition.** The model can chain small tools into a plan that nobody hardcoded. For example: search, then read, then find callers, then run tests, then report.

## 4. The Groq request and response flow in DevPilot

DevPilot uses the official `groq` SDK, which wraps Groq's OpenAI-compatible endpoint at `https://api.groq.com/openai/v1/chat/completions`. See `GroqChatClient` in [`app/agent/tool_loop.py`](app/agent/tool_loop.py).

**Request:**

```python
client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "system", ...}, {"role": "user", ...}, ...],
    tools=[{"type": "function", "function": {"name", "description", "parameters": <JSON Schema>}}, ...],
    tool_choice="auto",            # "none" on the final forced-summary call
    parallel_tool_calls=False,
    temperature=0.2,
    max_completion_tokens=4096,    # max_tokens is deprecated in Groq's API reference
)
```

**Response with a tool request:**

```text
choices[0].message.tool_calls = [
  { id: "call_abc", type: "function",
    function: { name: "find_callers", arguments: "{\"function_name\": \"login\"}" } }
]
```

`arguments` is a **string**, so it must be parsed with `json.loads` and validated.

**Returning the result:** first append the assistant message with its `tool_calls`, then append one message per call:

```python
{"role": "tool", "tool_call_id": "call_abc", "name": "find_callers", "content": "<JSON string>"}
```

Then send the whole history again. The API matches each result to its request through `tool_call_id`.

## 5. Why the LLM does not execute Python functions directly

- The model runs on Groq's servers and has no access to your machine. It can only emit text.
- Even if it could, letting it execute arbitrary code would make prompt injection equivalent to remote code execution.
- The application keeps control. In [`app/agent/executor.py`](app/agent/executor.py), the model's output is only a proposal. Python checks that the tool is registered, that the JSON parses, that Pydantic accepts the arguments, and that policy allows the call. Then it runs a function *we* wrote, with a timeout.

## 6. Tool schema design

DevPilot's rules for schemas, from [`app/agent/registry.py`](app/agent/registry.py) and the tool modules:

- **One Pydantic model per tool.** It generates the JSON Schema and also validates the arguments, so the two cannot drift apart.
- **`extra="forbid"`**, which becomes `additionalProperties: false`. Hallucinated parameters are rejected rather than ignored.
- **Constraints live in the schema:** `limit` has `ge=1, le=50`, `maximum_depth` is 1 to 6, and function names must match `^[A-Za-z_][\w.]*$`. The model sees the limits, and Pydantic enforces them.
- **Schemas are flattened** with `$defs` inlined, and Pydantic's auto-generated `title` keys are removed. The schema is self-contained and uses fewer tokens.
- **Descriptions are written for the model.** They say *when* to use a tool, not just what it does. For example, `create_report` says "Only call when the user asks for a report".
- **Narrow inputs, not general ones.** `run_tests` takes a `test_path`, not a command. `create_report` takes a title and findings, not a file path.
- **Prefer references over copies.** `analyze_test_results` and `create_report` take a `run_id` rather than test output typed by the model, so the model cannot invent results.

## 7. Tool validation

The executor runs these checks in order:

1. **Registry lookup.** Unknown names return `unknown_tool` plus the valid names, so the model can recover.
2. **JSON parsing.** Arguments must be a JSON object. Anything else returns `invalid_json`.
3. **Pydantic validation.** Failures return `invalid_arguments` with field-level messages. Input values are not echoed back.
4. **Policy authorization.** Permission enabled? Budget left? Otherwise `permission_denied`. Validation happens *before* authorization, so malformed calls do not burn the budget.
5. **Execution with a timeout.** Each exception type maps to a distinct error type: `security_violation`, `tool_error`, `internal_error` or `timeout`.

Inside the tools there is a second layer of validation that checks things the schema cannot express. Examples: the path must resolve inside the repository, the file must not be a secret, and the test target must not start with `-`.

## 8. Sequential tool calling

`parallel_tool_calls=False` is sent by default. The default model, `openai/gpt-oss-120b`, does not support parallel calls according to Groq's docs.

Some models may still return several calls in one response. The loop handles this: it appends the assistant message once, then executes the calls **one by one in order**, appending a `role: tool` reply for every `tool_call_id`. The API rejects a conversation where any tool call has no reply.

Why sequential? It gives deterministic ordering, simpler budgets and readable logs. In this domain the next call usually depends on the previous result anyway, for example finding callers before reading the caller's file.

## 9. Error handling

| Failure | Where | Handling |
|---|---|---|
| Unknown tool, bad JSON, invalid args, policy denial, tool crash, tool timeout | `executor.py` | Returned to the model as a structured error. The loop continues so the model can adapt. |
| Groq `tool_use_failed`, meaning the model's tool call was unparsable | `tool_loop.py` | Detected through the error body `code` and retried up to twice with a corrective message. |
| API timeout or connection error | `tool_loop.py` | The SDK retries twice, then the loop stops with `api_timeout` or `api_error`. |
| Empty response | `tool_loop.py` | One nudge, then `empty_response`. |
| Tool call without `id` or `function` | `normalize_tool_calls` | `malformed_response`. A call without an id cannot be answered correctly. |
| Round limit | `_finish_after_limit` | One last call with `tool_choice="none"` produces a best-effort summary and the status `max_rounds`. |
| Oversized tool output | `ToolExecutionResult.to_message_content` | Replaced with a truncated preview and a "narrow the request" note. |

Design principle: **errors the model can fix go back to the model. Errors only the operator can fix stop the run.**

## 10. Security controls

- **Untrusted model output:** registry allowlist, schema validation, and no `eval` or `exec` anywhere.
- **Filesystem:** `resolve_in_repo` resolves symlinks *then* checks containment. A naive `startswith` check on the unresolved path would let a symlink escape. `check_readable` blocks secrets and `.git/`, where `.git/config` can hold credentials in remote URLs.
- **Symlink aliases:** the directory walker follows a symlink only if its *target* passes the secret and excluded-directory rules. Without this, `search_code` could read `.env` through a harmless-looking link; a regression test covers it.
- **Git:** an allowlist of read-only subcommands with fixed argument templates, paths always after `--`, and `GIT_LITERAL_PATHSPECS=1` so a path like `:(glob)**` cannot widen a diff. Hardening flags matter because **a repository's own `.git/config` can execute commands**. A `core.fsmonitor` value runs on a plain `git status`; this was verified during development. DevPilot overrides fsmonitor, hooks, pager and external diff on every call, and the test `test_repository_hooks_and_fsmonitor_cannot_execute` covers it.
- **Tests:** a fixed pytest argv, no shell, a scrubbed environment so the Groq key never reaches test code, `PYTEST_ADDOPTS` removed because it could inject options, a process-group kill on timeout, and no cache or bytecode written into the repository.
- **Writes:** only reports, only into the configured directory, with a generated filename and exclusive create.
- **Budgets:** rounds, total calls, per-tool caps, timeouts and output-size limits.
- **Honest residual risk:** `run_tests` executes repository code by design, and file contents can attempt prompt injection. Both are documented in the README.

## 11. AST-based static analysis

Python's `ast` module parses source into a syntax tree without running it. That makes it safe to use on untrusted code, unlike `import`.

- **Imports** ([`python_ast.py`](app/tools/python_ast.py), [`dependency_analysis.py`](app/tools/dependency_analysis.py)): DevPilot walks `Import` and `ImportFrom` nodes and resolves relative imports from the module's package. It records the *scope* of each import: module level, inside a function, or inside `if TYPE_CHECKING`. Edges form a graph. **Tarjan's strongly connected components** algorithm, implemented iteratively, finds cycles, and a BFS extracts a concrete cycle path. Cycles through module-level imports are reported separately from cycles that exist only through deferred imports, because the latter usually do not break.
- **Calls** ([`call_graph.py`](app/tools/call_graph.py)): a `NodeVisitor` records every function with its class context and every `Call` node. Each call site is classified as a plain name, `self.method`, `obj.attr`, `getattr(obj, "name")` or dynamic. Resolution uses the module's symbol table: its functions, classes, imported names and imported modules.
  - **confirmed** means the call provably targets this repo's definition. Examples: a same-module function, `from x import f`, `module.f()`, `self.method()` when the class defines it, and `Klass.method()`.
  - **uncertain** means the name matches but the target cannot be proven. Examples: `obj.authenticate()` where `obj`'s type is unknown, inherited methods, and `getattr`.
  - Names that are referenced without being called are reported separately, because they are often callbacks.
- **Traversal** (`trace_function_flow`): a DFS over confirmed edges only, with a `max_depth` limit, a node limit and an `expanded` set to avoid re-walking. A cycle is detected when a callee is already on the *current path*, which distinguishes recursion from a function that is merely reached twice.

## 12. Limitations of static analysis

- There is no runtime type information. `service = AuthService(repo); service.authenticate()` is *uncertain* because DevPilot does not track variable types. In the sample app, this is exactly why `login` to `authenticate` appears under `unresolved_references`.
- It does not handle dynamic dispatch, inheritance, decorators that wrap or replace functions, `getattr` with computed names, monkey-patching, plugin registries, dependency injection or `importlib`.
- Import cycles in the static graph are **candidates**. Whether they fail depends on import order and on which names are used at import time.
- Python only.

DevPilot never hides this. Every analysis result includes `limitations` or `notes` text, and uncertain matches are listed separately rather than dropped.

## 13. Deterministic code vs the LLM

| Decision | Who decides | Why |
|---|---|---|
| Which tool to call next and how to explain results | **LLM** | Requires language understanding and planning. |
| Whether a path is inside the repo, or a file is a secret | **Code** | Security must be exact and cannot be talked out of. |
| Who calls `login` and whether imports form a cycle | **Code (AST and graph algorithms)** | Exact, repeatable and testable. LLMs hallucinate call graphs. |
| Test pass and fail counts | **Code (JUnit XML parsing)** | Facts must not be generated text. |
| The report's file name and location | **Code** | The model must not choose filesystem destinations. |
| Test results in the report | **Code**, copied from the recorded `run_id` | Prevents fabricated results. |
| Why a test probably failed | **Heuristic code, labelled as hypothesis**, plus LLM interpretation | Useful, but explicitly separated from observed facts. |

Rule of thumb: **the LLM chooses and explains, and code measures and enforces.**

---

## Likely interview questions

**Q: Walk me through what happens when I ask "Who calls login?"**
A: `ToolCallingAgent.run` sends the system prompt, the question and 12 tool schemas to Groq with `tool_choice="auto"`. The model returns a `tool_calls` entry for `find_callers` with `{"function_name": "login"}`. The executor looks it up in the registry, runs `json.loads`, validates with `FindCallersArgs`, checks the `analyze` permission and the budget, and runs `find_callers` with a timeout. That function parses the repo's Python files with `ast` and resolves each call. The result goes back as a `role: tool` message with the same `tool_call_id`. The model then either calls another tool or answers.

**Q: How do you stop the model from reading `.env`?**
A: There are three layers. The path is resolved, including symlinks, and must stay under the repository root. `check_readable` then blocks secret filename patterns and anything under `.git`, `.venv` and similar directories. Finally, the listing and search walkers skip those files entirely, so the model never learns they exist. Git status and diff withhold them too. Tests plant a real `.env` and `id_rsa` and assert their contents never appear.

**Q: The model can run tests. Isn't that code execution?**
A: Yes, and it is the one deliberate exception. The *command* is fixed: the model only supplies a validated path or node ID, which cannot start with `-`, and the call uses no shell. The *code* that runs is the repository's own tests, so you should only use DevPilot on trusted repositories, or disable the tool with `DEVPILOT_ALLOW_TEST_EXECUTION=false`. In that case the tool is not even advertised to the model. Secrets are scrubbed from the test environment, and a timeout kills the process group.

**Q: Why not just let the model generate shell commands?**
A: Then a prompt injection in a README could become `rm -rf` or exfiltration. Narrow, typed tools shrink the attack surface to what each tool can do, and every tool here is read-only except report writing to a fixed folder.

**Q: What surprised you while building it?**
A: Two things. First, `git status` on an untrusted repository can execute code through `core.fsmonitor`. I verified this and added hardening flags plus a regression test. Second, my own registration code had a circular import: the tools package imported the registry, which imported the tools package. That is the kind of issue `analyze_dependencies` reports, and I fixed it by moving registration into its own module.

**Q: How do you test an LLM loop without calling the LLM?**
A: The loop depends on a small `ChatClient` protocol with one method, `complete(messages, tools, tool_choice)`. Tests inject a `FakeClient` that replays scripted responses shaped like Groq's message objects, raises real `groq` exceptions, or runs a callable that inspects the conversation. That last option is how I prove the next tool depends on the previous result. Tests assert on the exact messages sent: ordering, `tool_call_id` matching and `tool_choice`. A separate integration test calls the real API only when explicitly enabled.

**Q: What does `find_callers` do with `getattr(module, "login")()`?**
A: It records it as a `getattr` call site with the name `"login"` and classifies it as **uncertain**. The string could resolve to anything at runtime, so it is reported but never presented as a confirmed caller.

**Q: How do you detect recursion in `trace_function_flow`?**
A: The DFS carries the current path. If a callee is already on the path, that is a cycle: *direct recursion* if the callee is the caller itself, *indirect recursion* otherwise. A separate `expanded` set stops re-walking nodes that were reached through different branches. That case is not a cycle, so the two are kept distinct.

**Q: How do you find circular imports?**
A: Build a directed graph from module-level imports and run Tarjan's SCC algorithm. Any component with more than one node, or a node with a self-edge, contains a cycle, and a BFS within the component prints a concrete path. A second graph that includes function-level and `TYPE_CHECKING` imports finds the cycles that exist *only* through deferred imports, and those are reported separately as lower-risk candidates.

**Q: What happens if the model loops forever?**
A: It cannot. There is a round limit (default 10), a total tool-call budget (default 30) and per-tool caps, with `run_tests` and `create_report` at 3 each per question. When the round limit is hit, one last request with `tool_choice="none"` asks the model to summarize what it found and what remains unverified.

**Q: What happens if a tool hangs?**
A: In-process tools run in a worker thread with a timeout, and the model gets a `timeout` error immediately. Python threads cannot be force-killed, so the work may finish in the background; that limitation is documented. Subprocess tools have hard timeouts: pytest runs in its own session and the whole process group gets `SIGKILL`, and git has its own timeout.

**Q: Why Groq, and why this model?**
A: Groq offers fast inference and an OpenAI-compatible tool-calling API through an official SDK. `openai/gpt-oss-120b` is a *production* model on Groq with tool use and a 131k context. Its lack of parallel tool calls fits the sequential design. The model ID is configurable, and `llama-3.3-70b-versatile` is the documented alternative.

**Q: How would you add a new tool?**
A: Write a Pydantic input model and a function `(ToolContext, args) -> dict` in `app/tools/`. Add one `Tool(...)` entry in `app/tools/registration.py` with a description, permission and category. The schema, validation, policy checks, timeout, error handling and CLI display then work automatically. Add tests.

**Q: How would you make this production-grade?**
A: Run tests in a sandbox such as a container with no network. Cache the AST index. Add structured logging and token-usage metrics. Add multi-turn memory with summarization. Add type inference so more callers are confirmed. Run the integration test in CI with a secret key.
