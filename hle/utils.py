"""
Shared utilities for HLE answer extraction, normalization, and comparison.
"""
from __future__ import annotations

import math
import re


def extract_answer(text: str) -> str | None:
    """Extract answer from ANSWER: ... line (last occurrence)."""
    matches = re.findall(r'ANSWER:\s*(.+)', text, re.IGNORECASE)
    return matches[-1].strip() if matches else None


# ── LaTeX normalization ───────────────────────────────────────────────────────

def _strip_latex_formatting(s: str) -> str:
    """
    Remove LaTeX formatting commands that don't affect mathematical value.
    e.g. \\left( \\frac{a}{b} \\right) → (a)/(b)
    """
    # Remove \\left and \\right (purely for sizing)
    s = re.sub(r'\\(?:left|right)\s*', '', s)
    # \\dfrac / \\tfrac → \\frac
    s = re.sub(r'\\[dt]frac', r'\\frac', s)
    # \\frac{a}{b} → (a)/(b)  — repeated to handle nesting
    for _ in range(5):
        s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    # Simplify redundant parens around single atoms: (\pi) → \pi, (1002) → 1002
    s = re.sub(r'\(([^()\s+\-*/^,]+)\)', r'\1', s)
    # Remove remaining LaTeX $ wrappers and whitespace
    s = s.strip('$').strip()
    s = ' '.join(s.split())
    return s


def normalize(s: str) -> str:
    """Normalize answer string for string comparison."""
    s = s.strip().strip('$').lower().strip('.,;: ')
    return ' '.join(s.split())


def _normalize_for_sympy(s: str) -> str:
    """Convert a math expression string to something sympy can parse."""
    s = _strip_latex_formatting(s)
    # ^ → ** (exponentiation)
    s = s.replace('^', '**')
    # n! → factorial(n)
    s = re.sub(r'(\d+)!', lambda m: f'factorial({m.group(1)})', s)
    # \cos, \sin, \pi etc.
    s = re.sub(r'\\(cos|sin|tan|exp|log|ln|pi|sqrt)\b', r'\1', s)
    # subscripts: F_1 → F1, V_m → Vm
    s = re.sub(r'([A-Za-z])_\{?(\w+)\}?', r'\1\2', s)
    # Remove remaining backslashes
    s = s.replace('\\', '')
    # Implied multiplication: 2p → 2*p, nF → n*F (digit followed by letter)
    s = re.sub(r'(\d)([A-Za-z])', r'\1*\2', s)
    return s.strip()


def _try_sympy_equal(s1: str, s2: str) -> bool | None:
    """
    Try to check algebraic equivalence using sympy.
    Returns True/False, or None if parsing fails.
    """
    try:
        from sympy import simplify, sympify, factorial  # noqa: F401
        from sympy.abc import p, n, x, y, z
        import sympy

        e1 = _normalize_for_sympy(s1)
        e2 = _normalize_for_sympy(s2)

        # Collect all single-letter variables that appear
        all_vars = set(re.findall(r'\b([A-Za-z])\b', e1 + ' ' + e2))
        # Build a local namespace with those symbols
        ns = {v: sympy.Symbol(v) for v in all_vars}
        ns['factorial'] = sympy.factorial
        ns['pi'] = sympy.pi
        ns['cos'] = sympy.cos
        ns['sin'] = sympy.sin
        ns['sqrt'] = sympy.sqrt

        expr1 = sympify(e1, locals=ns)
        expr2 = sympify(e2, locals=ns)

        diff = simplify(expr1 - expr2)
        return diff == 0
    except Exception:
        return None


# ── Numeric evaluation ────────────────────────────────────────────────────────

def _try_eval_math(s: str) -> str | None:
    """
    Try to evaluate a pure-numeric expression (no variables) to a canonical string.
    Handles: 9!^2, 3!, 2^10, etc.
    """
    try:
        expr = s.strip().replace('^', '**')
        expr = re.sub(r'(\d+)!', lambda m: str(math.factorial(int(m.group(1)))), expr)
        if re.search(r'[a-zA-Z_]', expr):
            return None
        val = eval(expr, {"__builtins__": {}})
        if isinstance(val, (int, float)):
            if isinstance(val, float) and val == int(val):
                val = int(val)
            return str(val)
    except Exception:
        pass
    return None


# ── Main comparison ───────────────────────────────────────────────────────────

def is_correct(predicted: str | None, ground_truth: str) -> bool:
    """
    Check if predicted matches ground truth. Tries in order:
    1. Exact match after basic normalization.
    2. Match after stripping LaTeX formatting (left, right, frac).
    3. Numeric evaluation (9!^2 == 131681894400).
    4. Algebraic equivalence via sympy (2p-1 == p^2-(1-p)^2).
    """
    if predicted is None:
        return False

    # 1. Basic normalization
    if normalize(predicted) == normalize(ground_truth):
        return True

    # 2. LaTeX formatting stripped (also compare without any spaces for math expressions)
    pred_stripped = normalize(_strip_latex_formatting(predicted))
    gt_stripped   = normalize(_strip_latex_formatting(ground_truth))
    if pred_stripped == gt_stripped:
        return True
    if pred_stripped.replace(' ', '') == gt_stripped.replace(' ', ''):
        return True

    # 3. Numeric evaluation (pure numbers / factorial expressions)
    pred_val = _try_eval_math(predicted)
    gt_val   = _try_eval_math(ground_truth)
    if pred_val is not None and gt_val is not None and pred_val == gt_val:
        return True

    # 4. Algebraic equivalence via sympy
    result = _try_sympy_equal(predicted, ground_truth)
    if result is True:
        return True

    return False
