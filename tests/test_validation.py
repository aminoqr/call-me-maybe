"""Tests for schema validation of generated calls."""

import pytest

from src.validation import validate_call


def test_valid_calls(functions):
    assert validate_call("fn_multiply_numbers", {"a": 1, "b": 2.5},
                         functions) == []
    assert validate_call("fn_is_even", {"n": 4}, functions) == []
    assert validate_call(
        "fn_execute_sql_query", {"query": "x", "database": "y"}, functions
    ) == []


def test_unknown_function(functions):
    assert "unknown function" in validate_call("fn_nope", {}, functions)[0]
    assert validate_call(None, {}, functions)


def test_parameters_must_be_object(functions):
    assert validate_call("fn_is_even", [4], functions) == [
        "parameters is not an object"
    ]


def test_missing_and_extra_parameters(functions):
    problems = validate_call("fn_is_even", {"x": 1}, functions)
    assert any("missing parameter 'n'" in p for p in problems)
    assert any("unexpected parameter 'x'" in p for p in problems)


@pytest.mark.parametrize(
    "name, params",
    [
        ("fn_is_even", {"n": "4"}),                   # str for integer
        ("fn_is_even", {"n": True}),                  # bool is not integer
        ("fn_is_even", {"n": 4.5}),                   # non-integral float
        ("fn_multiply_numbers", {"a": "3", "b": 5}),  # str for number
        ("fn_multiply_numbers", {"a": True, "b": 5}),
        ("fn_read_file", {"path": 1, "encoding": "utf-8"}),
    ],
)
def test_type_errors(functions, name, params):
    assert validate_call(name, params, functions)


def test_integral_float_accepted_for_integer(functions):
    assert validate_call("fn_is_even", {"n": 4.0}, functions) == []
