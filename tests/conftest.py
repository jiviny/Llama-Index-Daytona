"""Shared pytest configuration.

Registers the ``live`` marker that gates tests which hit a *real* Daytona
sandbox and/or the *real* OpenAI API. Run only the fast deterministic tests
with ``-m "not live"``; run everything to exercise the real integration.

The example ``examples/workflow_demo.py`` is made importable so the workflow
wiring test can import ``DaytonaCodeWorkflow`` from it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _PKG_ROOT / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: test that creates a REAL Daytona sandbox and/or calls the REAL "
        "OpenAI API (slower, needs network + credentials in the environment).",
    )
