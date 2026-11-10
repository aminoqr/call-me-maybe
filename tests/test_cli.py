"""CLI error paths. These fail before any model is loaded."""

import json
import sys

import pytest

from src.__main__ import main

GOOD_FN = {
    "name": "f", "description": "d",
    "parameters": {"x": {"type": "number"}},
    "returns": {"type": "number"},
}


def run_cli(monkeypatch, tmp_path, functions, prompts):
    fp, pp = tmp_path / "f.json", tmp_path / "p.json"
    fp.write_text(json.dumps(functions))
    pp.write_text(json.dumps(prompts))
    monkeypatch.setattr(sys, "argv", [
        "src", "--functions_definition", str(fp), "--input", str(pp),
        "--output", str(tmp_path / "out.json"),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def test_functions_file_must_be_array(monkeypatch, tmp_path, capsys):
    assert run_cli(monkeypatch, tmp_path, {"a": 1}, []) == 1
    assert "must be a JSON array" in capsys.readouterr().out


def test_functions_file_schema_error(monkeypatch, tmp_path, capsys):
    assert run_cli(monkeypatch, tmp_path, [{"name": "f"}], []) == 1
    assert "Validation error in functions file" in capsys.readouterr().out


def test_empty_function_list(monkeypatch, tmp_path, capsys):
    assert run_cli(monkeypatch, tmp_path, [], []) == 1
    assert "no function definitions" in capsys.readouterr().out


def test_prompts_file_must_be_array(monkeypatch, tmp_path, capsys):
    assert run_cli(monkeypatch, tmp_path, [GOOD_FN], {"prompt": "x"}) == 1
    assert "must be a JSON array" in capsys.readouterr().out


def test_prompts_schema_error(monkeypatch, tmp_path, capsys):
    assert run_cli(monkeypatch, tmp_path, [GOOD_FN], [{"nope": 1}]) == 1
    assert "Validation error in prompts file" in capsys.readouterr().out


def test_missing_functions_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [
        "src", "--functions_definition", str(tmp_path / "missing.json"),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
