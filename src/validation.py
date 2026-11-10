"""Schema validation of a generated function call.

Constrained decoding makes these checks pass by construction.  The same
checks are applied to *unconstrained* output in the benchmarks, so
"valid" means exactly the same thing for both.
"""

from typing import Any, Dict, List, Sequence

from src.models import FunctionDefinition


def _type_ok(value: Any, type_name: str) -> bool:
    """Return True if *value* satisfies the JSON-schema-ish *type_name*."""
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):  # bool is an int subclass; reject it
        return False
    if type_name == "integer":
        return isinstance(value, int) or (
            isinstance(value, float) and value.is_integer()
        )
    if type_name == "number":
        return isinstance(value, (int, float))
    return True  # unknown type names are not enforced


def validate_call(
    name: Any,
    parameters: Any,
    functions: Sequence[FunctionDefinition],
) -> List[str]:
    """Check a call against the function definitions.

    Args:
        name: The proposed function name.
        parameters: The proposed ``parameters`` object.
        functions: The available function definitions.

    Returns:
        A list of human-readable problems; empty when the call is valid.
    """
    by_name: Dict[str, FunctionDefinition] = {f.name: f for f in functions}
    if not isinstance(name, str) or name not in by_name:
        return [f"unknown function {name!r}"]
    if not isinstance(parameters, dict):
        return ["parameters is not an object"]

    problems: List[str] = []
    expected = by_name[name].parameters
    for missing in sorted(set(expected) - set(parameters)):
        problems.append(f"missing parameter {missing!r}")
    for extra in sorted(set(parameters) - set(expected)):
        problems.append(f"unexpected parameter {extra!r}")
    for key, spec in expected.items():
        if key in parameters and not _type_ok(parameters[key], spec.type):
            problems.append(
                f"parameter {key!r} is not a {spec.type}: "
                f"{parameters[key]!r}"
            )
    return problems
