"""Slow tests that load the real Qwen3-0.6B model.

Deselected by default.  Run with ``make test-slow`` (``pytest -m slow``).
"""

import json

import pytest

from tests.conftest import ROOT

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def caller(functions):
    pytest.importorskip("torch")
    from llm_sdk import Small_LLM_Model
    from src.kvcache import CachedModel
    from src.pipeline import FunctionCaller

    model = CachedModel(Small_LLM_Model(device="cpu"))
    return FunctionCaller.from_model(model, functions)


ITEMS = json.loads((ROOT / "data/benchmarks/eval_set.json").read_text())
# one dev prompt per function keeps this to ~10 s
PICKS = {}
for it in ITEMS:
    if it["split"] == "dev":
        PICKS.setdefault(it["name"], it)


@pytest.mark.parametrize("item", list(PICKS.values()),
                         ids=list(PICKS.keys()))
def test_real_model_end_to_end(caller, functions, item):
    from benchmarks.scoring import classify

    outcome = caller.call(item["prompt"])
    assert outcome.error is None
    assert classify(outcome.result, item, functions) == "correct"


def test_real_model_is_deterministic(caller):
    prompt = ITEMS[0]["prompt"]
    assert caller.call(prompt).result == caller.call(prompt).result
