"""Tests for benchmark scoring and baseline helpers."""

import numpy as np
import pytest

from benchmarks.baselines import few_shot_prompt, sample_token, zero_shot_prompt
from benchmarks.scoring import (
    classify,
    extract_json_object,
    parse_call,
    values_equal,
    wilson_interval,
)


@pytest.mark.parametrize("text, expected", [
    ('x {"a": 1} y', '{"a": 1}'),
    ('{"a": {"b": 2}}', '{"a": {"b": 2}}'),
    ('{"t": "{user}"}', '{"t": "{user}"}'),          # braces in strings
    ('{"t": "a\\"}"}', '{"t": "a\\"}"}'),            # escaped quote
    ('{"a": 1', None),                               # still open
    ('no json here', None),
])
def test_extract_json_object(text, expected):
    assert extract_json_object(text) == expected


def test_parse_call_variants():
    ok = parse_call('Sure: {"name": "f", "parameters": {"a": 1}}')
    assert ok == {"name": "f", "parameters": {"a": 1}}
    assert parse_call('{"name": "f", "arguments": {"a": 1}}')[
        "parameters"] == {"a": 1}
    assert parse_call('{"a": 1}') is None            # no "name"
    assert parse_call('{"name": }') is None          # broken JSON
    assert parse_call("fn_multiply_numbers(3, 5)") is None


def test_values_equal():
    assert values_equal(3, 3.0)
    assert values_equal(0.1 + 0.2, 0.3)
    assert not values_equal(True, 1)
    assert not values_equal("3", 3)
    assert values_equal("abc", "abc")
    assert not values_equal("abc", "abd")


def test_classify_buckets(functions):
    exp = {"name": "fn_is_even", "parameters": {"n": 4}}
    good = {"name": "fn_is_even", "parameters": {"n": 4}}
    assert classify(good, exp, functions) == "correct"
    assert classify(None, exp, functions) == "invalid"
    assert classify({"name": "fn_is_even", "parameters": {"x": 4}},
                    exp, functions) == "invalid"
    assert classify({"name": "fn_is_even", "parameters": {"n": 5}},
                    exp, functions) == "wrong_params"
    assert classify(
        {"name": "fn_multiply_numbers", "parameters": {"a": 4, "b": 1}},
        exp, functions) == "wrong_name"


def test_wilson_interval():
    lo, hi = wilson_interval(59, 59)
    assert hi == pytest.approx(100.0) and 90 < lo < 100
    lo, hi = wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = wilson_interval(5, 10)
    assert lo < 50 < hi


def test_sample_token_greedy_and_seeded():
    logits = np.array([0.0, 5.0, 1.0])
    rng = np.random.default_rng(0)
    assert sample_token(logits, 0.0, rng) == 1
    a = [sample_token(logits, 1.0, np.random.default_rng(7)) for _ in range(5)]
    assert a == [sample_token(logits, 1.0, np.random.default_rng(7))] * 5


def test_prompts_include_request_and_functions(functions):
    for build in (zero_shot_prompt, few_shot_prompt):
        text = build(functions, "REQ-123")
        assert "REQ-123" in text and "fn_is_even" in text
        assert text.rstrip().endswith("Function Call:")
    assert "fn_multiply_numbers\", \"parameters\"" in few_shot_prompt(
        functions, "x")
