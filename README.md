# LlamaIndex Tools Integration: Daytona

`llama-index-tools-daytona` gives a [LlamaIndex](https://www.llamaindex.ai/)
agent a tool to execute **Python or shell code inside a real, isolated
[Daytona](https://www.daytona.io/) sandbox** and get the output back. The agent
writes code, runs it in a clean cloud sandbox, and reads the result — instead of
"computing" answers in its head.

## Installation

```bash
pip install llama-index-tools-daytona
```

This is a [PEP 420 namespace package](https://peps.python.org/pep-0420/) under
`llama_index.tools`; it merges with your installed `llama-index-core`.

## Environment variables

This library reads credentials from **ambient environment variables** — it does
**not** load a `.env` file for you. Set these before running:

| Variable           | Used by            | Required                          |
| ------------------ | ------------------ | --------------------------------- |
| `DAYTONA_API_KEY`  | Daytona SDK        | Yes                               |
| `DAYTONA_API_URL`  | Daytona SDK        | Only for self-hosted / custom URL |
| `DAYTONA_TARGET`   | Daytona SDK        | Optional (target region)          |
| `OPENAI_API_KEY`   | your agent's LLM   | Yes, if using an OpenAI LLM       |

If you keep credentials in a `.env`, call `dotenv.load_dotenv()` yourself in
your app before constructing the tools.

## Usage

```python
from llama_index.tools.daytona import DaytonaCodeInterpreterToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

tools = DaytonaCodeInterpreterToolSpec().to_tool_list()
agent = FunctionAgent(
    tools=tools,
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="Solve tasks by writing Python and running it with run_python.",
)

answer = await agent.run(user_msg="What is the sum of the squares of 1..20? Run Python.")
print(answer)  # -> 2870
```

Prefer plain `FunctionTool` objects? Use the factory:

```python
from llama_index.tools.daytona import make_daytona_function_tools

tools = make_daytona_function_tools(timeout=120)
```

## Tools

The spec exposes two tools:

- **`run_python(code)`** — execute a self-contained Python 3 snippet and return
  its stdout.
- **`run_shell(command)`** — run a shell command and return its stdout.

Each returns an LLM-friendly string with an `[ok]` / `[error (exit_code=N)]`
status line followed by the captured output.

> Only **stdout** is captured. A command that writes its result solely to
> stderr yields `[ok]\n<no output>` even on success — redirect with `2>&1` or
> print to stdout if you need that text back.

## Statefulness (important)

Each tool call runs in a **fresh, isolated, ephemeral sandbox** that is created
and torn down per call — so one tool call cannot see another's state, and
cleanup is guaranteed even on error. This is the safe default for agent tools.

Under the hood, Daytona's `code_run` is **stateless per call**: every execution
gets a brand-new Python interpreter, so in-memory variables do **not** survive
between calls. What *does* persist on a reused sandbox is the **filesystem**
(and anything you install in it). To carry data from one snippet to the next on
a persistent session, write it to a file and read it back:

```python
from llama_index.tools.daytona import DaytonaSession

with DaytonaSession() as s:
    s.run_python("open('/tmp/v', 'w').write('41')")                      # interpreter #1
    print(s.run_python("print(int(open('/tmp/v').read()) + 1)").stdout)  # -> "42"
```

## License

MIT, to follow the LlamaIndex `llama-index-tools-*` ecosystem convention (those
integration packages are MIT). (Its sibling package `pydantic-ai-daytona` is
Apache-2.0 to match the `daytona` SDK — the two differ by design.)
