# llama-index-tools-daytona

A [LlamaIndex](https://www.llamaindex.ai/) tool that runs the code an agent writes inside an isolated [Daytona](https://www.daytona.io/) sandbox.

## Install

```bash
pip install llama-index-tools-daytona llama-index-llms-openai
```

## Configure

Read from the environment (the package does not load a `.env`):

- `DAYTONA_API_KEY` — required ([dashboard](https://app.daytona.io/))
- `OPENAI_API_KEY` — required for the model
- `DAYTONA_API_URL`, `DAYTONA_TARGET` — optional

## Use

```python
import asyncio
from llama_index.tools.daytona import DaytonaCodeInterpreterToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(
    tools=DaytonaCodeInterpreterToolSpec().to_tool_list(),
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="Solve tasks by writing Python and running it with run_python.",
)

async def main():
    print(await agent.run(user_msg="Compute the 50th Fibonacci number."))

asyncio.run(main())
```

- `to_tool_list()` returns `run_python` and `run_shell`.
- `make_daytona_function_tools()` returns the same as plain `FunctionTool`s.
- Tool output is a string: `[ok]\n<stdout>` or `[error (exit_code=N)]\n<stdout>`.

## Notes

- Each tool call runs in a fresh sandbox; state does not carry between calls.
- A trailing bare expression is printed, like a REPL.
- For state across calls, reuse a `DaytonaSession` and write to its filesystem.
- Only stdout is captured; redirect with `2>&1` if a command writes to stderr.

## License

MIT
