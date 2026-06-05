"""Live integration tests — these create REAL Daytona sandboxes and (one test)
call the REAL OpenAI API. Marked ``live`` so they can be skipped with
``-m "not live"`` for a fast offline run.

Credentials are read from ambient environment variables (DAYTONA_API_KEY,
OPENAI_API_KEY). The library never loads a .env; set them in your shell or load
them yourself before running these tests.
"""
from __future__ import annotations

import os

import pytest

from llama_index.tools.daytona import DaytonaCodeInterpreterToolSpec
from llama_index.tools.daytona.base import DaytonaSession, run_python_ephemeral

# run_workflow lives in the example (see conftest for sys.path wiring).
from workflow_demo import run_workflow

pytestmark = pytest.mark.live


def _require_daytona():
    if not os.environ.get("DAYTONA_API_KEY"):
        pytest.skip("DAYTONA_API_KEY not set; skipping live test")


def _require_openai():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping live test")


def test_real_sandbox_runs_trivial_python():
    _require_daytona()
    result = run_python_ephemeral("print(2 + 2)")
    assert result.ok is True
    assert result.exit_code == 0
    assert result.stdout.strip() == "4"


def test_tool_runs_trivial_python_in_real_sandbox():
    """The LlamaIndex tool surface (what the agent calls) executes real code."""
    _require_daytona()
    spec = DaytonaCodeInterpreterToolSpec()
    out = spec.run_python("print(6 * 7)")
    assert out.startswith("[ok]")
    assert "42" in out


def test_stateful_session_persists_value_across_calls():
    """Each code_run is a fresh interpreter, but the sandbox FILESYSTEM persists:
    a value written to a file in call #1 is read back (incremented) in call #2."""
    _require_daytona()
    with DaytonaSession() as session:
        first = session.run_python("open('/tmp/state.txt', 'w').write('41')")
        assert first.ok is True, first.stdout
        second = session.run_python(
            "print(int(open('/tmp/state.txt').read()) + 1)"
        )
        assert second.ok is True, second.stdout
        assert second.stdout.strip() == "42"


def test_workflow_solves_computational_task_end_to_end():
    """Real OpenAI-driven agent writes Python, runs it in a real Daytona sandbox,
    and returns the sum of squares 1..20 == 2870."""
    _require_daytona()
    _require_openai()
    task = (
        "What is the sum of the squares of the integers from 1 to 20? "
        "Compute it by running Python and report only the number."
    )
    answer = run_workflow(task, model="gpt-4o-mini")
    assert "2870" in answer
