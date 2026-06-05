"""Fast, deterministic wiring tests — no network, no tokens, no real sandbox.

These prove the integration is wired correctly without paying for an LLM or a
Daytona sandbox:

* the LlamaIndex tool spec exposes the right tools and formats sandbox output
  in an LLM-friendly way (executor stubbed);
* a real ``FunctionAgent`` driven by a *scripted* ``FunctionCallingLLM`` solves
  a computational task end-to-end through the genuine tool-calling loop, with a
  deterministic numeric assertion (2870) — the executor is stubbed so no real
  sandbox runs, but the agent -> tool -> agent path is exercised for real;
* the ``DaytonaCodeWorkflow`` graph transitions StartEvent -> agent_step ->
  AgentResultEvent -> finish -> StopEvent and returns the agent's answer.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Sequence

import pytest

import llama_index.tools.daytona.base as base
from llama_index.tools.daytona import DaytonaCodeInterpreterToolSpec
from llama_index.tools.daytona.base import ExecResult

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import MockLLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection

# Imported from the example (see conftest for sys.path wiring).
from workflow_demo import DaytonaCodeWorkflow


# --------------------------------------------------------------------------- #
# (1) Tool layer: tool list + output formatting (executor stubbed)
# --------------------------------------------------------------------------- #
def test_tool_spec_exposes_expected_tools():
    """The spec publishes exactly run_python and run_shell, each with a docstring
    the LLM can use to decide when to call it."""
    tools = DaytonaCodeInterpreterToolSpec().to_tool_list()
    names = [t.metadata.name for t in tools]
    assert names == ["run_python", "run_shell"]
    for t in tools:
        assert t.metadata.description  # non-empty description for the LLM


def test_run_python_tool_formats_ok_output(monkeypatch):
    """A successful execution is rendered '[ok]\\n<stdout>' for the agent."""
    monkeypatch.setattr(
        base,
        "run_python_ephemeral",
        lambda code, timeout=120: ExecResult(ok=True, exit_code=0, stdout="2870"),
    )
    spec = DaytonaCodeInterpreterToolSpec()
    assert spec.run_python("print(2870)") == "[ok]\n2870"


def test_run_python_tool_formats_error_and_empty_output(monkeypatch):
    """Errors surface the exit code; empty stdout surfaces an explicit marker."""
    monkeypatch.setattr(
        base,
        "run_python_ephemeral",
        lambda code, timeout=120: ExecResult(ok=False, exit_code=1, stdout=""),
    )
    spec = DaytonaCodeInterpreterToolSpec()
    out = spec.run_python("raise SystemExit(1)")
    assert out == "[error (exit_code=1)]\n<no output>"


def test_function_tool_is_callable_with_stubbed_executor(monkeypatch):
    """The FunctionTool the agent actually invokes routes kwargs through to the
    (stubbed) executor and returns the formatted result."""
    captured = {}

    def fake_run(code, timeout=120):
        captured["code"] = code
        return ExecResult(ok=True, exit_code=0, stdout="hello")

    monkeypatch.setattr(base, "run_python_ephemeral", fake_run)
    tools = {t.metadata.name: t for t in DaytonaCodeInterpreterToolSpec().to_tool_list()}
    out = tools["run_python"].call(code="print('hello')")
    assert captured["code"] == "print('hello')"
    assert str(out) == "[ok]\nhello"


def test_make_daytona_function_tools_factory(monkeypatch):
    """The plain-FunctionTool factory yields the same two named tools and routes
    through the (stubbed) executor."""
    monkeypatch.setattr(
        base,
        "run_python_ephemeral",
        lambda code, timeout=120: ExecResult(ok=True, exit_code=0, stdout="ok-out"),
    )
    from llama_index.tools.daytona import make_daytona_function_tools

    tools = {t.metadata.name: t for t in make_daytona_function_tools()}
    assert set(tools) == {"run_python", "run_shell"}
    assert str(tools["run_python"].call(code="print('ok-out')")) == "[ok]\nok-out"


# --------------------------------------------------------------------------- #
# Executor normalization (deterministic, no sandbox).
# --------------------------------------------------------------------------- #
def test_normalize_handles_none_result_without_crashing():
    """_normalize must never produce a None stdout (the wrappers index/format it
    as a string). A response with result=None and no artifacts -> ''."""

    class _Resp:
        result = None
        exit_code = 0
        artifacts = None

    res = base._normalize(_Resp())
    assert res.stdout == ""
    assert res.ok is True
    assert res.exit_code == 0


def test_session_exit_deletes_even_if_body_raises(monkeypatch):
    """A user-raised exception inside a `with DaytonaSession()` body must not leak
    the sandbox: __exit__ still deletes it (try/finally)."""
    created = []
    deleted = []

    class _FakeSandbox:
        id = "sb-fake"

    class _FakeClient:
        def create(self, timeout=180):
            sb = _FakeSandbox()
            created.append(sb)
            return sb

        def delete(self, sb):
            deleted.append(sb)

    monkeypatch.setattr(base, "_client", lambda: _FakeClient())

    with pytest.raises(ValueError, match="user boom"):
        with base.DaytonaSession():
            raise ValueError("user boom")

    assert created and deleted and created == deleted


def test_sandbox_deleted_even_when_run_raises(monkeypatch):
    """No leak even when code_run raises a transport error: __exit__ deletes."""
    created = []
    deleted = []

    class _FakeProc:
        def code_run(self, code, timeout=120):
            raise RuntimeError("transport blew up mid-exec")

    class _FakeSandbox:
        id = "sb-fake-1"
        process = _FakeProc()

    class _FakeClient:
        def create(self, timeout=180):
            sb = _FakeSandbox()
            created.append(sb)
            return sb

        def delete(self, sb):
            deleted.append(sb)

    monkeypatch.setattr(base, "_client", lambda: _FakeClient())

    with pytest.raises(RuntimeError, match="transport blew up"):
        base.run_python_ephemeral("anything")

    assert len(created) == 1
    assert len(deleted) == 1, "sandbox leaked: created but never deleted"
    assert created == deleted


def test_tool_renders_runtime_error_without_raising(monkeypatch):
    """A failed execution is surfaced as an [error ...] block containing the
    traceback — the tool itself never raises."""
    monkeypatch.setattr(
        base,
        "run_python_ephemeral",
        lambda code, timeout=120: ExecResult(
            ok=False,
            exit_code=1,
            stdout="Traceback (most recent call last):\nValueError: boom",
        ),
    )
    spec = DaytonaCodeInterpreterToolSpec()
    out = spec.run_python("raise ValueError('boom')")
    assert out.startswith("[error (exit_code=1)]")
    assert "ValueError: boom" in out


# --------------------------------------------------------------------------- #
# Scripted deterministic LLM: turn 1 -> call run_python, turn 2 -> final answer
# --------------------------------------------------------------------------- #
class ScriptedToolLLM(FunctionCallingLLM):
    """A minimal deterministic FunctionCallingLLM used to drive a *real*
    FunctionAgent without any network call.

    On the first turn (no tool message in history yet) it emits a single tool
    call to ``run_python``. On the next turn (after the tool result is in the
    history) it emits a plain-text final answer.
    """

    tool_code: str = "print(sum(i * i for i in range(1, 21)))"
    final_answer: str = "The sum of the squares from 1 to 20 is 2870."

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True, model_name="scripted-stub")

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Optional[Any] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict:
        messages: List[ChatMessage] = list(chat_history or [])
        if user_msg is not None:
            messages.append(
                ChatMessage(role=MessageRole.USER, content=str(user_msg))
                if isinstance(user_msg, str)
                else user_msg
            )
        return {"messages": messages, "tools": tools}

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> List[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls", [])

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        tool_result_seen = any(
            getattr(m, "role", None) == MessageRole.TOOL for m in messages
        )
        if not tool_result_seen:
            selection = ToolSelection(
                tool_id="call_run_python_1",
                tool_name="run_python",
                tool_kwargs={"code": self.tool_code},
            )
            msg = ChatMessage(role=MessageRole.ASSISTANT, content="")
            msg.additional_kwargs["tool_calls"] = [selection]
            return ChatResponse(message=msg)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self.final_answer)
        )

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        response = await self.achat(messages, **kwargs)

        async def _gen():
            yield response

        return _gen()

    # Remaining abstract methods are unused by the function-calling agent path.
    def chat(self, messages, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError

    def stream_chat(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def complete(self, prompt, formatted: bool = False, **kwargs) -> CompletionResponse:  # pragma: no cover
        return CompletionResponse(text="")

    async def acomplete(self, prompt, formatted: bool = False, **kwargs) -> CompletionResponse:  # pragma: no cover
        return CompletionResponse(text="")

    def stream_complete(self, prompt, formatted: bool = False, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream_complete(self, prompt, formatted: bool = False, **kwargs):  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# (2) Real FunctionAgent + scripted LLM + stubbed executor: numeric assertion
# --------------------------------------------------------------------------- #
def test_agent_solves_task_with_scripted_llm_and_stubbed_executor(monkeypatch):
    """End-to-end agent wiring with a deterministic numeric assertion (2870),
    no tokens and no real sandbox."""
    monkeypatch.setattr(
        base,
        "run_python_ephemeral",
        lambda code, timeout=120: ExecResult(ok=True, exit_code=0, stdout="2870"),
    )
    tools = DaytonaCodeInterpreterToolSpec().to_tool_list()
    agent = FunctionAgent(
        tools=tools, llm=ScriptedToolLLM(), system_prompt="Solve by running code."
    )

    async def _go():
        return await agent.run(user_msg="Sum of squares 1..20, run Python.")

    answer = asyncio.run(_go())
    assert "2870" in str(answer)


# --------------------------------------------------------------------------- #
# (3) Graph wiring: StartEvent -> agent_step -> finish -> StopEvent
# --------------------------------------------------------------------------- #
def test_workflow_graph_wiring_returns_agent_answer():
    """The Workflow graph passes the agent's answer through to the StopEvent."""

    class _StubAgent:
        async def run(self, user_msg):
            assert "squares" in user_msg  # the task is threaded through
            return "Computed in sandbox: 2870."

    async def _go():
        wf = DaytonaCodeWorkflow(llm=MockLLM(), timeout=30)
        wf._agent = _StubAgent()
        return await wf.run(task="sum of squares 1..20")

    result = asyncio.run(_go())
    assert str(result) == "Computed in sandbox: 2870."


def test_workflow_rejects_startevent_without_task():
    """agent_step raises a clear error if the StartEvent carries no task/input."""

    class _StubAgent:
        async def run(self, user_msg):  # pragma: no cover - should not be reached
            return "unused"

    async def _go():
        wf = DaytonaCodeWorkflow(llm=MockLLM(), timeout=30)
        wf._agent = _StubAgent()
        return await wf.run()

    with pytest.raises(Exception) as excinfo:
        asyncio.run(_go())
    assert "task" in str(excinfo.value).lower()
