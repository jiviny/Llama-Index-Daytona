"""Offline unit tests for the REPL-style trailing-expression auto-echo.

These need no Daytona sandbox or OpenAI key: they exec the *transformed* code in
a local namespace with a captured ``print`` and assert what would be emitted.
"""
from llama_index.tools.daytona.base import _echo_last_expression


def _captured_output(code: str) -> list:
    transformed = _echo_last_expression(code)
    captured: list = []
    namespace = {"print": lambda *args: captured.append(" ".join(str(a) for a in args))}
    exec(transformed, namespace)
    return captured


def test_trailing_bare_expression_is_echoed():
    assert _captured_output("x = 41\nx + 1") == ["42"]


def test_trailing_string_expression_is_echoed_as_repr():
    assert _captured_output("'2cf24dba'") == ["'2cf24dba'"]


def test_explicit_print_is_not_double_echoed():
    assert _captured_output("print('hi')") == ["hi"]


def test_statement_ending_code_is_unchanged():
    assert _captured_output("y = 5") == []


def test_none_valued_trailing_expression_is_not_echoed():
    assert _captured_output("None") == []


def test_syntax_error_passes_through_untouched():
    bad = "def f(:"  # not valid Python
    assert _echo_last_expression(bad) == bad
