"""End-to-end pipeline tests with the oracle fake model."""

import pytest
from pydantic import ValidationError

from src.models import FunctionCallResult
from src.pipeline import build_prompt, build_token_tables
from src.validation import validate_call
from tests.conftest import FakeModel, json_completion, make_caller


def run(functions, name, params_json, distractor=None, request="req"):
    model = FakeModel(json_completion(name, params_json), distractor)
    outcome = make_caller(model, functions).call(request)
    return model, outcome


def test_number_function(functions):
    _, out = run(functions, "fn_multiply_numbers", '"a": 3.0, "b": 5.0')
    assert out.error is None
    assert out.result == {
        "prompt": "req", "name": "fn_multiply_numbers",
        "parameters": {"a": 3.0, "b": 5.0},
    }


def test_integer_parameter_is_int(functions):
    _, out = run(functions, "fn_is_even", '"n": 7')
    assert out.result["parameters"] == {"n": 7}
    assert type(out.result["parameters"]["n"]) is int


def test_mixed_types(functions):
    _, out = run(
        functions, "fn_calculate_compound_interest",
        '"principal": 1234567.89, "rate": 0.0375, "years": 23',
    )
    p = out.result["parameters"]
    assert p == {"principal": 1234567.89, "rate": 0.0375, "years": 23}
    assert type(p["years"]) is int


def test_strings_with_escapes(functions):
    _, out = run(
        functions, "fn_execute_sql_query",
        '"query": "SELECT * FROM t WHERE n = \\"Bob\\"", '
        '"database": "production"',
    )
    assert out.result["parameters"] == {
        "query": 'SELECT * FROM t WHERE n = "Bob"',
        "database": "production",
    }


def test_windows_path_backslashes(functions):
    _, out = run(
        functions, "fn_read_file",
        '"path": "C:\\\\Users\\\\john\\\\config.ini", '
        '"encoding": "latin-1"',
    )
    assert out.result["parameters"]["path"] == "C:\\Users\\john\\config.ini"


def test_template_braces_and_apostrophes(functions):
    _, out = run(
        functions, "fn_format_template", '"template": "Hello {user}\'s!"'
    )
    assert out.result["parameters"] == {"template": "Hello {user}'s!"}


@pytest.mark.parametrize("name,params", [
    ("fn_multiply_numbers", '"a": 3.0, "b": 5.0'),
    ("fn_is_even", '"n": 4'),
    ("fn_read_file", '"path": "/a/b", "encoding": "utf-8"'),
])
def test_result_always_schema_valid(functions, name, params):
    _, out = run(functions, name, params)
    assert validate_call(
        out.result["name"], out.result["parameters"], functions
    ) == []
    FunctionCallResult(**out.result)  # pydantic accepts it


def test_adversarial_model_cannot_break_number_schema(functions):
    """Model screams 'z' at every step; output must still be well-typed."""
    model, out = run(
        functions, "fn_multiply_numbers", '"a": 3.0, "b": 5.0',
        distractor="z",
    )
    assert out.result is not None
    assert out.result["name"] == "fn_multiply_numbers"
    assert validate_call(
        out.result["name"], out.result["parameters"], functions
    ) == []


def test_adversarial_model_cannot_pick_unknown_function(functions):
    _, out = run(functions, "fn_is_even", '"n": 4', distractor="q")
    assert out.result["name"] in {f.name for f in functions}


def test_structural_tokens_are_injected_not_sampled(functions):
    model, out = run(functions, "fn_is_even", '"n": 4')
    text = model.decode(out.tokens)
    assert text.startswith(' {"name": "fn_is_even", "parameters": {"n": ')


def test_model_calls_are_far_fewer_than_output_tokens(functions):
    """Injection means most output tokens never need a forward pass."""
    model, out = run(
        functions, "fn_calculate_compound_interest",
        '"principal": 1000.0, "rate": 0.05, "years": 10',
    )
    assert model.calls < len(out.tokens)


def test_build_prompt_keeps_user_braces_verbatim():
    text = build_prompt("- f()", "fill {user} and {0}")
    assert "fill {user} and {0}" in text
    assert text.rstrip().endswith("Function Call:")


def test_token_tables_exclude_control_chars_from_strings(functions):
    model = FakeModel()
    tables = build_token_tables(
        functions, model.id_to_token,
        lambda t: model.encode(t)[0].tolist(),
    )
    newline = model.char_to_id["\n"]
    assert newline not in set(tables.str_tids.tolist())
    # digits are number tokens; letters are not
    assert model.char_to_id["7"] in set(tables.num_first.tolist())
    assert model.char_to_id["a"] not in set(tables.num_first.tolist())
    # space-prefixed number tokens may start a number but not continue one
    sp12 = next(i for i, t in model.id_to_token.items() if t == "\u012012")
    assert sp12 in set(tables.num_first.tolist())
    assert sp12 not in set(tables.num_cont.tolist())
    assert {model.char_to_id[","], model.char_to_id["}"]} == tables.stop_tids


def test_models_reject_bad_result():
    with pytest.raises(ValidationError):
        FunctionCallResult(prompt="p", name="n")  # type: ignore
