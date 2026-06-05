"""A LlamaIndex Workflow ("graph") that solves tasks by writing & running code
inside a real Daytona sandbox.

The graph is intentionally explicit so the wiring is visible:

    StartEvent(task) --agent_step--> AgentResultEvent --finish--> StopEvent(answer)

Inside ``agent_step`` an OpenAI-backed ``FunctionAgent`` is given the Daytona
tools (``run_python`` / ``run_shell``). To answer, the agent must WRITE Python
and RUN it in a sandbox rather than computing in its head — that is the whole
point of the integration.

Run it (after `pip install llama-index-tools-daytona llama-index-llms-openai`)::

    python examples/workflow_demo.py

Credentials: this example *does* load a local ``.env`` (via python-dotenv) for
convenience. The library itself never loads ``.env``; it relies on ambient
environment variables: DAYTONA_API_KEY (+ optional DAYTONA_API_URL /
DAYTONA_TARGET) and OPENAI_API_KEY.
"""
from __future__ import annotations

import asyncio

# Examples may load a .env for convenience; the library never does.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional for the example
    pass

from llama_index.tools.daytona import DaytonaCodeInterpreterToolSpec

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a precise problem-solver with access to a real, isolated code "
    "sandbox via the run_python and run_shell tools. To answer ANY question "
    "that involves computation, data, or code, you MUST write Python and run it "
    "with run_python instead of computing the answer yourself. Always print() "
    "the value you need. Each tool call runs in a fresh sandbox, so include "
    "everything needed in a single snippet. After the tool returns, give a short "
    "final answer that states the result."
)


class AgentResultEvent(Event):
    """Carries the agent's textual answer from the agent step to the finish."""

    answer: str


class DaytonaCodeWorkflow(Workflow):
    """Two-step graph: run the Daytona-equipped agent, then finish.

    The agent is built once per workflow instance from the Daytona tool list and
    an OpenAI LLM.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: str = DEFAULT_MODEL,
        sandbox_timeout: int = 120,
        timeout: float | None = 180.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(timeout=timeout, verbose=verbose, **kwargs)
        self._llm = llm or OpenAI(model=model)
        self._tools = DaytonaCodeInterpreterToolSpec(
            timeout=sandbox_timeout
        ).to_tool_list()
        self._agent = FunctionAgent(
            tools=self._tools,
            llm=self._llm,
            system_prompt=SYSTEM_PROMPT,
        )

    @step
    async def agent_step(self, ev: StartEvent) -> AgentResultEvent:
        """Hand the task to the Daytona-equipped agent and capture its answer."""
        task = getattr(ev, "task", None) or getattr(ev, "input", None)
        if task is None:
            raise ValueError("StartEvent must carry a 'task' (or 'input').")
        response = await self._agent.run(user_msg=str(task))
        return AgentResultEvent(answer=str(response))

    @step
    async def finish(self, ev: AgentResultEvent) -> StopEvent:
        """Emit the agent's answer as the workflow result."""
        return StopEvent(result=ev.answer)


async def arun_workflow(
    task: str,
    model: str = DEFAULT_MODEL,
    sandbox_timeout: int = 120,
    verbose: bool = False,
) -> str:
    """Async: run the workflow on ``task`` and return the final answer."""
    wf = DaytonaCodeWorkflow(
        model=model, sandbox_timeout=sandbox_timeout, verbose=verbose
    )
    result = await wf.run(task=task)
    return str(result)


def run_workflow(
    task: str,
    model: str = DEFAULT_MODEL,
    sandbox_timeout: int = 120,
    verbose: bool = False,
) -> str:
    """Run the Daytona code workflow on ``task`` and return the final answer.

    Synchronous wrapper around :func:`arun_workflow` for easy scripting.
    """
    return asyncio.run(
        arun_workflow(
            task,
            model=model,
            sandbox_timeout=sandbox_timeout,
            verbose=verbose,
        )
    )


SAMPLE_TASK = (
    "What is the sum of the squares of the integers from 1 to 20? "
    "Compute it by running Python and report only the number."
)


def main() -> None:
    print("=" * 70)
    print("llama-index-tools-daytona workflow demo")
    print("=" * 70)
    print(f"\nTask: {SAMPLE_TASK}")
    answer = run_workflow(SAMPLE_TASK)
    print("\n----- agent answer -----")
    print(answer)
    print("------------------------")


if __name__ == "__main__":
    main()
