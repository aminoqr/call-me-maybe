"""Integrity checks for the benchmark ground truth."""

import json

from benchmarks.baselines import FEW_SHOT_EXAMPLES
from benchmarks.scoring import params_match
from src.validation import validate_call
from tests.conftest import ROOT

ITEMS = json.loads((ROOT / "data/benchmarks/eval_set.json").read_text())


def test_ground_truth_is_schema_valid(functions):
    for it in ITEMS:
        assert validate_call(it["name"], it["parameters"], functions) == [], it


def test_prompts_unique():
    prompts = [i["prompt"] for i in ITEMS]
    assert len(prompts) == len(set(prompts))


def test_has_dev_and_heldout_splits():
    splits = {i["split"] for i in ITEMS}
    assert splits == {"dev", "heldout"}
    assert sum(i["split"] == "dev" for i in ITEMS) == 11


def test_every_function_covered_in_heldout(functions):
    held = {i["name"] for i in ITEMS if i["split"] == "heldout"}
    assert held == {f.name for f in functions}


def test_few_shot_examples_do_not_leak_eval_prompts():
    prompts = {i["prompt"] for i in ITEMS}
    assert all(q not in prompts for q, _ in FEW_SHOT_EXAMPLES)


def test_dev_set_matches_shipped_inputs():
    shipped = json.loads(
        (ROOT / "data/input/function_calling_tests.json").read_text()
    )
    dev = [i["prompt"] for i in ITEMS if i["split"] == "dev"]
    assert dev == [s["prompt"] for s in shipped]


def test_params_match_sanity():
    assert params_match({"a": 3}, {"a": 3.0})
    assert not params_match({"a": 3}, {"a": 4.0})
