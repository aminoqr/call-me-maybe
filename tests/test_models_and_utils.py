"""Tests for data models and small utilities."""

import json

import pytest
from pydantic import ValidationError

from src.models import FunctionCallResult, FunctionDefinition, PromptInput
from src.utils import build_func_prompt, load_json_file


def test_function_definition_requires_all_fields():
    with pytest.raises(ValidationError):
        FunctionDefinition(name="f", description="d")  # type: ignore


def test_prompt_input_requires_string():
    with pytest.raises(ValidationError):
        PromptInput(prompt=None)  # type: ignore


def test_function_call_result_roundtrip():
    r = FunctionCallResult(prompt="p", name="n", parameters={"a": 1})
    assert r.model_dump()["parameters"] == {"a": 1}


def test_build_func_prompt_lists_every_function(functions):
    text = build_func_prompt(functions)
    assert len(text.splitlines()) == len(functions)
    assert "- fn_is_even(n: integer):" in text


def test_load_json_file_ok(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps([1, 2]))
    assert load_json_file(str(p)) == [1, 2]


@pytest.mark.parametrize("content", [None, "{not json"])
def test_load_json_file_exits_cleanly(tmp_path, capsys, content):
    p = tmp_path / "x.json"
    if content is not None:
        p.write_text(content)
    with pytest.raises(SystemExit) as exc:
        load_json_file(str(p))
    assert exc.value.code == 1
    assert "Error loading" in capsys.readouterr().out
