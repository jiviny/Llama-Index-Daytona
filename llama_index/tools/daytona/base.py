"""LlamaIndex tool spec exposing a real Daytona sandbox to an agent.

This module bundles two things:

* The verified Daytona **execution spine** (smoke-tested against the ``daytona``
  SDK 0.149-0.184): a small, well-behaved API for running Python/shell inside a
  real, isolated Daytona sandbox.
* The LlamaIndex **tool surface**:

  - :class:`DaytonaCodeInterpreterToolSpec` — a ``BaseToolSpec`` exposing
    ``run_python`` and ``run_shell`` tools. Call ``.to_tool_list()`` to get the
    ``FunctionTool`` list an agent consumes.
  - :func:`make_daytona_function_tools` — a plain ``FunctionTool`` factory, an
    alternative for callers who do not want the spec abstraction.

Each tool call runs in a *fresh, isolated, ephemeral* sandbox (created and torn
down by ``run_*_ephemeral``), which is the safe default for agent tool calls:
one tool call cannot see another's state, and cleanup is guaranteed even on
error.

Statefulness note (verified against daytona 0.184.0): each ``process.code_run``
runs in a FRESH Python interpreter, so in-memory variables do NOT survive
between calls. What persists across calls on a reused :class:`DaytonaSession` is
the sandbox **filesystem** (and anything installed in it) — to carry data from
one snippet to the next, write it to a file.

For convenience, ``run_python`` auto-echoes a trailing bare expression
(REPL-style): a snippet ending in ``result`` prints its value without an
explicit ``print(...)``.

Credentials are read from **ambient environment variables**, never from a
bundled ``.env``:

* the ``daytona`` SDK reads ``DAYTONA_API_KEY`` / ``DAYTONA_API_URL`` /
  ``DAYTONA_TARGET`` from the environment;
* the OpenAI client (used by the workflow example, not by this library) reads
  ``OPENAI_API_KEY``.

This library intentionally does NOT load a ``.env``. Set the variables in your
environment, or call ``dotenv.load_dotenv()`` yourself in your app/examples.

Verified call shape (do not change without re-running the smoke test)::

    from daytona import Daytona
    d = Daytona()                            # auto-reads DAYTONA_API_KEY/_URL/_TARGET
    sb = d.create(timeout=180)
    r = sb.process.code_run("print(2+2)")    # r.exit_code == 0, r.result == "4"
    r2 = sb.process.exec("echo hi")          # r2.result == "hi"
    d.delete(sb)
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List

from daytona import Daytona

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec


# --------------------------------------------------------------------------- #
# Execution spine (verified) — no .env auto-loading; uses ambient env vars.
# --------------------------------------------------------------------------- #
@dataclass
class ExecResult:
    """Normalized result of a sandbox execution."""

    ok: bool
    exit_code: int
    stdout: str

    def __str__(self) -> str:  # so tools can return the object directly
        return self.stdout


def _normalize(resp) -> ExecResult:
    stdout = resp.result
    if stdout is None and getattr(resp, "artifacts", None) is not None:
        stdout = resp.artifacts.stdout
    exit_code = resp.exit_code if resp.exit_code is not None else 0
    return ExecResult(ok=(exit_code == 0), exit_code=exit_code, stdout=stdout or "")


def _echo_last_expression(code: str) -> str:
    """Make a trailing bare expression print its value, REPL-style.

    Daytona's ``code_run`` runs the snippet as a *script*, so a final bare
    expression (e.g. ``result`` instead of ``print(result)``) produces no
    output — and an agent that wrote REPL-style code may then hallucinate a
    value. We mirror a REPL: if the last top-level statement is a bare
    expression, echo its value when it is not ``None``. An explicit
    ``print(...)`` is unaffected (its value is ``None``, so it is not echoed
    twice). Unparseable code is returned untouched so the real ``SyntaxError``
    surfaces in the sandbox.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    head = ast.unparse(ast.Module(body=tree.body[:-1], type_ignores=[]))
    echo = (
        f"_daytona_last = ({ast.unparse(tree.body[-1].value)})\n"
        "if _daytona_last is not None:\n"
        "    print(repr(_daytona_last))"
    )
    return f"{head}\n{echo}" if head else echo


def _client() -> Daytona:
    # Daytona() with no args auto-reads DAYTONA_API_KEY / DAYTONA_API_URL /
    # DAYTONA_TARGET from the ambient environment.
    return Daytona()


class DaytonaSession:
    """A reused sandbox for running several snippets without paying the
    cold-start cost each time.

    What persists across calls is the sandbox **filesystem** (and anything you
    install in it) — NOT in-memory Python state. Each ``run_python`` is a fresh
    interpreter (Daytona's stateless ``code_run``), so to carry data from one
    snippet to the next, write it to a file. Always use as a context manager so
    the sandbox is deleted even on error::

        with DaytonaSession() as s:
            s.run_python("open('/tmp/v', 'w').write('41')")                      # interpreter #1
            print(s.run_python("print(int(open('/tmp/v').read()) + 1)").stdout)  # -> "42"
    """

    def __init__(self, create_timeout: int = 180) -> None:
        self._create_timeout = create_timeout
        self._client = _client()
        self._sandbox = None

    def __enter__(self) -> "DaytonaSession":
        self._sandbox = self._client.create(timeout=self._create_timeout)
        return self

    def run_python(self, code: str, timeout: int = 120) -> ExecResult:
        # Auto-echo a trailing bare expression (REPL-style) so a snippet ending
        # in `result` instead of `print(result)` still returns its value.
        code = _echo_last_expression(code)
        return _normalize(self._sandbox.process.code_run(code, timeout=timeout))

    def run_shell(self, command: str, timeout: int = 120) -> ExecResult:
        return _normalize(self._sandbox.process.exec(command, timeout=timeout))

    @property
    def sandbox_id(self):
        return getattr(self._sandbox, "id", None) or getattr(
            self._sandbox, "sandbox_id", None
        )

    def __exit__(self, *exc) -> None:
        if self._sandbox is not None:
            try:
                self._client.delete(self._sandbox)
            finally:
                self._sandbox = None


def run_python_ephemeral(code: str, timeout: int = 120) -> ExecResult:
    """Create a fresh sandbox, run one Python snippet, then delete it.

    Stateless and safe — the default for agent *tools*, where each tool call
    should be isolated. Cleanup happens even if execution raises.
    """
    with DaytonaSession() as session:
        return session.run_python(code, timeout=timeout)


def run_shell_ephemeral(command: str, timeout: int = 120) -> ExecResult:
    with DaytonaSession() as session:
        return session.run_shell(command, timeout=timeout)


# --------------------------------------------------------------------------- #
# LlamaIndex tool surface.
# --------------------------------------------------------------------------- #
def _format(result) -> str:
    """Render an ``ExecResult`` as a compact, LLM-friendly string.

    The agent reads this verbatim, so surface the exit code and any output
    (or an explicit empty marker) rather than returning a bare object.
    """
    status = "ok" if result.ok else f"error (exit_code={result.exit_code})"
    body = result.stdout if result.stdout != "" else "<no output>"
    return f"[{status}]\n{body}"


class DaytonaCodeInterpreterToolSpec(BaseToolSpec):
    """A LlamaIndex tool spec that executes code in a real Daytona sandbox.

    Usage::

        from llama_index.tools.daytona import DaytonaCodeInterpreterToolSpec

        tools = DaytonaCodeInterpreterToolSpec().to_tool_list()
        agent = FunctionAgent(tools=tools, llm=...)
    """

    spec_functions: List[str] = ["run_python", "run_shell"]

    def __init__(self, timeout: int = 120) -> None:
        """``timeout`` bounds each individual sandbox execution (seconds)."""
        self._timeout = timeout

    def run_python(self, code: str) -> str:
        """Execute a Python 3 snippet in a fresh, isolated Daytona sandbox and
        return its output.

        Use this to actually run Python and observe real results: do
        arithmetic, parse data, call libraries, etc. ``print(...)`` whatever you
        want returned (a trailing bare expression is also auto-printed,
        REPL-style). Each call gets a clean sandbox, so state does NOT persist
        between calls — put everything you need in one snippet.

        Args:
            code: A self-contained Python program. Print the result you want.

        Returns:
            The captured stdout (and an ``[ok]``/``[error ...]`` status line).
        """
        return _format(run_python_ephemeral(code, timeout=self._timeout))

    def run_shell(self, command: str) -> str:
        """Run a shell command in a fresh, isolated Daytona sandbox and return
        its output.

        Use this for non-Python tasks: inspecting files, running CLI tools, or
        installing packages with ``pip``. Each call gets a clean sandbox, so
        state does NOT persist between calls.

        Note: only **stdout** is captured. A command that writes its result
        solely to stderr yields ``[ok]\\n<no output>`` even on success; redirect
        with ``2>&1`` (or print to stdout) if you need that text back.

        Args:
            command: A shell command line to execute (e.g. ``echo hi``).

        Returns:
            The captured stdout (and an ``[ok]``/``[error ...]`` status line).
        """
        return _format(run_shell_ephemeral(command, timeout=self._timeout))


def make_daytona_function_tools(timeout: int = 120) -> List[FunctionTool]:
    """Alternative factory returning plain ``FunctionTool`` objects.

    Equivalent to ``DaytonaCodeInterpreterToolSpec(timeout).to_tool_list()`` but
    built directly from functions, for callers who prefer not to use the spec.
    """
    spec = DaytonaCodeInterpreterToolSpec(timeout=timeout)
    return [
        FunctionTool.from_defaults(
            fn=spec.run_python,
            name="run_python",
            description=(
                "Execute a self-contained Python 3 snippet in a fresh, isolated "
                "Daytona sandbox and return its stdout. print() the result you "
                "want back (a trailing bare expression is auto-printed too). "
                "State does not persist between calls."
            ),
        ),
        FunctionTool.from_defaults(
            fn=spec.run_shell,
            name="run_shell",
            description=(
                "Run a shell command in a fresh, isolated Daytona sandbox and "
                "return its stdout. State does not persist between calls."
            ),
        ),
    ]
