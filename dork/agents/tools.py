"""Tools the research agent can call.

Each tool has a name, a human description (shown to the model for tool selection),
and a ``run`` callable taking a dict of arguments and returning a string
observation. Tools are intentionally small and safe: the calculator uses an AST
evaluator (never :func:`eval`), and the Python tool runs in a restricted
namespace. The Python tool is a *best-effort* sandbox, not a security boundary —
see ``docs/agent_design.md``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dork.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Tool:
    """A callable tool exposed to the agent."""

    name: str
    description: str
    func: Callable[[dict[str, Any]], str]

    def run(self, args: dict[str, Any]) -> str:
        try:
            return self.func(args)
        except Exception as exc:  # tools must never crash the agent loop
            return f"ERROR: {type(exc).__name__}: {exc}"


# ─────────────────────────── Calculator ──────────────────────────────
_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression safely via the AST (no :func:`eval`)."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree)


def _calculator(args: dict[str, Any]) -> str:
    expr = str(args.get("expression", args.get("query", ""))).strip()
    if not expr:
        return "ERROR: no expression provided"
    result = safe_eval(expr)
    # Render integers without a trailing .0 for clean output.
    if result == int(result):
        result = int(result)
    return str(result)


# ───────────────────────── Python sandbox ────────────────────────────
_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "round": round,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "print": print,
}
_FORBIDDEN = ("import", "open(", "__", "exec(", "eval(", "compile(", "globals(", "os.", "sys.")


def _python_exec(args: dict[str, Any]) -> str:
    code = str(args.get("code", "")).strip()
    if not code:
        return "ERROR: no code provided"
    low = code.lower()
    if any(tok in low for tok in _FORBIDDEN):
        return "ERROR: code contains a forbidden construct (imports/dunder/io disallowed)"
    buf = io.StringIO()
    sandbox_globals: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    sandbox_locals: dict[str, Any] = {}
    with contextlib.redirect_stdout(buf):
        exec(code, sandbox_globals, sandbox_locals)
    out = buf.getvalue().strip()
    result = sandbox_locals.get("result")
    if result is not None:
        out = f"{out}\nresult={result}".strip()
    return out or "(no output)"


def default_tools() -> dict[str, Tool]:
    """Return the standalone tools (search/summarize/compare are bound by the agent)."""
    return {
        "calculator": Tool(
            "calculator",
            "Evaluate an arithmetic expression. args: {expression: str}",
            _calculator,
        ),
        "python_exec": Tool(
            "python_exec",
            "Run a short, import-free Python snippet; set `result` for output. args: {code: str}",
            _python_exec,
        ),
    }
