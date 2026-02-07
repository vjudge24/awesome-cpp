"""Calculator tool for agent use — safe mathematical expression evaluation."""

import ast
import operator
from typing import Any

from langchain_core.tools import tool

# Allowed operations for safe evaluation
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node with only arithmetic operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, % with parentheses.
    Does NOT support variable assignment or function calls.

    Args:
        expression: A mathematical expression (e.g., "2 + 3 * 4")

    Returns:
        The result as a string
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)

        if result == int(result):
            return str(int(result))
        return f"{result:.6f}"
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Error evaluating expression: {e}"
