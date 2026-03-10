"""Utility helpers for file I/O and prompt construction."""

import sys
import json
from typing import Any, List

from src.models import FunctionDefinition


def load_json_file(file_path: str) -> Any:
    """Load and parse a JSON file.

    Prints a human-readable error and exits with code 1 on
    any failure (missing file, invalid JSON, permission error,
    etc.).

    Args:
        file_path: Path to the JSON file.

    Returns:
        The parsed JSON data (typically a list or dict).
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)


def build_func_prompt(
    funcs: List[FunctionDefinition],
) -> str:
    """Build a multi-line description of available functions.

    Used as context inside the LLM prompt so the model knows
    which functions and parameter signatures exist.

    Args:
        funcs: Validated function definitions.

    Returns:
        A newline-separated string listing every function.
    """
    lines = []
    for fn in funcs:
        params = ", ".join(
            f"{k}: {v.type}"
            for k, v in fn.parameters.items()
        )
        lines.append(
            f"- {fn.name}({params}): {fn.description}"
        )
    return "\n".join(lines)
