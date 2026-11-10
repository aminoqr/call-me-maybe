"""Scoring helpers shared by the benchmark runner and the tests."""

import json
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from src.models import FunctionDefinition
from src.validation import validate_call

CATEGORIES = ("correct", "wrong_name", "wrong_params", "invalid")


def extract_json_object(text: str) -> Optional[str]:
    """Return the first *complete* top-level ``{...}`` in *text*.

    The scan is string-aware, so braces inside JSON strings (such as
    ``"{user}"`` in a template) do not confuse it.  Returns ``None`` if the
    object is missing or still open.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return None


def parse_call(raw: str) -> Optional[Dict[str, Any]]:
    """Parse free-form model output into ``{"name", "parameters"}``."""
    chunk = extract_json_object(raw)
    if chunk is None:
        return None
    try:
        obj = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    return {
        "name": obj.get("name"),
        "parameters": obj.get("parameters", obj.get("arguments")),
    }


def values_equal(got: Any, expected: Any) -> bool:
    """Compare one parameter value with numeric tolerance."""
    if isinstance(expected, bool) or isinstance(got, bool):
        return got is expected
    if isinstance(expected, (int, float)):
        if not isinstance(got, (int, float)):
            return False
        return math.isclose(
            float(got), float(expected), rel_tol=1e-9, abs_tol=1e-9
        )
    return bool(got == expected)


def params_match(got: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """True when both dicts have the same keys and equal values."""
    if set(got) != set(expected):
        return False
    return all(values_equal(got[k], v) for k, v in expected.items())


def classify(
    result: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
    functions: Sequence[FunctionDefinition],
) -> str:
    """Bucket one output as correct / wrong_name / wrong_params / invalid.

    ``invalid`` means the output is not a schema-valid call at all (no JSON,
    unknown function, wrong/missing parameters or wrong types).  A call can
    be valid yet wrong: that is ``wrong_name`` or ``wrong_params``.
    """
    if result is None:
        return "invalid"
    if validate_call(
        result.get("name"), result.get("parameters"), functions
    ):
        return "invalid"
    if result["name"] != expected["name"]:
        return "wrong_name"
    if not params_match(result["parameters"], expected["parameters"]):
        return "wrong_params"
    return "correct"


def wilson_interval(
    successes: int, n: int, z: float = 1.96
) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (in percent)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))
