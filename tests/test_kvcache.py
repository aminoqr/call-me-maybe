"""The KV cache must be a pure optimisation: same logits, less work."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.kvcache import CachedModel  # noqa: E402


class _Inner:
    """Minimal stand-in for Small_LLM_Model around a tiny random model."""

    def __init__(self):
        torch.manual_seed(0)
        cfg = transformers.Qwen3Config(
            vocab_size=120, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8,
            max_position_embeddings=256,
        )
        self._model = transformers.AutoModelForCausalLM.from_config(
            cfg).eval()
        self._device = "cpu"

    def get_logits_from_input_ids(self, ids):
        with torch.no_grad():
            out = self._model(input_ids=torch.tensor([ids]))
        return out.logits[0, -1].tolist()


@pytest.fixture(scope="module")
def inner():
    return _Inner()


def assert_same(model, inner, ids):
    got = model.get_logits_from_input_ids(ids)
    ref = np.asarray(inner.get_logits_from_input_ids(ids))
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)
    assert int(np.argmax(got)) == int(np.argmax(ref))


def test_growing_sequence_matches_full_forward(inner):
    m = CachedModel(inner)
    seq = [5, 9, 11, 3]
    for extra in [7, 22, 40, 41, 2]:
        assert_same(m, inner, seq)
        seq = seq + [extra]
    assert_same(m, inner, seq)


def test_growing_sequence_only_feeds_new_tokens(inner):
    m = CachedModel(inner)
    seq = list(range(1, 11))
    m.get_logits_from_input_ids(seq)
    m.get_logits_from_input_ids(seq + [50])
    m.get_logits_from_input_ids(seq + [50, 51])
    assert m.forward_tokens == 10 + 1 + 1
    assert m.forward_calls == 3


def test_diverging_sequence_crops_cache(inner):
    m = CachedModel(inner)
    a = [1, 2, 3, 4, 5, 6]
    m.get_logits_from_input_ids(a)
    b = [1, 2, 3, 9, 8]            # diverges after 3 shared tokens
    assert_same(m, inner, b)
    assert_same(m, inner, a)       # and back again


def test_shared_prefix_across_prompts_is_reused(inner):
    m = CachedModel(inner)
    preamble = list(range(10, 40))
    m.get_logits_from_input_ids(preamble + [1, 2])
    before = m.forward_tokens
    assert_same(m, inner, preamble + [3, 4, 5])
    assert m.forward_tokens - before == 3


def test_repeated_identical_request(inner):
    m = CachedModel(inner)
    ids = [4, 5, 6, 7]
    assert_same(m, inner, ids)
    assert_same(m, inner, ids)     # needs >=1 fresh token: re-feeds last


def test_shorter_request_than_cached(inner):
    m = CachedModel(inner)
    m.get_logits_from_input_ids([1, 2, 3, 4, 5, 6])
    assert_same(m, inner, [1, 2, 3])


def test_single_token_and_empty(inner):
    m = CachedModel(inner)
    assert_same(m, inner, [7])
    with pytest.raises(ValueError):
        m.get_logits_from_input_ids([])


def test_use_cache_false_matches_and_counts_full_work(inner):
    m = CachedModel(inner, use_cache=False)
    seq = [1, 2, 3]
    assert_same(m, inner, seq)
    assert_same(m, inner, seq + [4])
    assert m.forward_tokens == 3 + 4


def test_reset(inner):
    m = CachedModel(inner)
    m.get_logits_from_input_ids([1, 2, 3])
    m.reset()
    m.get_logits_from_input_ids([1, 2, 3, 4])
    assert m.forward_tokens == 3 + 4


def test_attribute_forwarding(inner):
    m = CachedModel(inner)
    assert m._device == "cpu"
