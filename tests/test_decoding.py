"""Unit tests for the constrained-decoding primitives."""

import numpy as np
import pytest

from src.decoding import (
    _find_unescaped_quote,
    _json_unescape,
    apply_logit_mask,
    gen_bool_value,
    gen_number_value,
    gen_string_value,
    masked_argmax,
    select_function,
)
from tests.conftest import FakeModel


# ---------------------------------------------------------------- escapes
@pytest.mark.parametrize(
    "text, expected",
    [
        ('abc', None),
        ('abc"', 3),
        ('"', 0),
        ('a\\"b"', 4),          # escaped quote skipped
        ('a\\\\"b', 3),         # escaped backslash, then a real quote
        ('a\\\\\\"b"', 6),      # \\ then \" then real quote
        ('', None),
        ('trailing\\', None),   # dangling backslash must not crash
    ],
)
def test_find_unescaped_quote(text, expected):
    assert _find_unescaped_quote(text) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('plain', 'plain'),
        ('say \\"hi\\"', 'say "hi"'),
        ('C:\\\\Users\\\\john', 'C:\\Users\\john'),
        ('line\\nbreak', 'line\nbreak'),
        ('bad \\x escape', 'bad \\x escape'),   # invalid JSON -> unchanged
        ('', ''),
    ],
)
def test_json_unescape(raw, expected):
    assert _json_unescape(raw) == expected


# ------------------------------------------------------------------ masks
def test_apply_logit_mask_keeps_only_valid():
    out = apply_logit_mask([1.0, 2.0, 3.0, 4.0], [1, 3])
    assert out[1] == 2.0 and out[3] == 4.0
    assert out[0] == out[2] == float("-inf")


def test_apply_logit_mask_ignores_out_of_range_ids():
    out = apply_logit_mask([1.0, 2.0], [1, 99])
    assert out == [float("-inf"), 2.0]


def test_masked_argmax_basic():
    assert masked_argmax([0.1, 9.0, 0.5, 7.0], [0, 2, 3]) == 3


def test_masked_argmax_ignores_out_of_vocab_ids():
    assert masked_argmax([1.0, 2.0, 3.0], [1, 50, -4]) == 1


def test_masked_argmax_empty_raises():
    with pytest.raises(ValueError):
        masked_argmax([1.0, 2.0], [])
    with pytest.raises(ValueError):
        masked_argmax([1.0, 2.0], [99])


@pytest.mark.parametrize("use_numpy", [False, True])
def test_masked_argmax_matches_reference_implementation(use_numpy):
    """The fast path must pick the same token as mask-then-argmax."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        logits = rng.normal(size=500)
        valid = rng.choice(500, size=rng.integers(1, 60), replace=False)
        ref_masked = apply_logit_mask(logits.tolist(), valid.tolist())
        ref = ref_masked.index(max(ref_masked))
        arg = logits if use_numpy else logits.tolist()
        assert masked_argmax(arg, valid.tolist()) == ref


# --------------------------------------------------------- select_function
def _seqs(model, names):
    return {n: model.encode(n)[0].tolist() for n in names}


def test_select_function_picks_target_among_shared_prefixes():
    names = ["fn_read_file", "fn_read_fish", "fn_write_file"]
    model = FakeModel(target=' {"name": "fn_read_fish')
    seqs = _seqs(model, names)
    ctx = model.encode('Function Call: {"name": "')[0].tolist()
    name, toks = select_function(model, ctx, [], seqs)
    assert name == "fn_read_fish"
    assert model.decode(toks) == "fn_read_fish"


def test_select_function_never_leaves_candidate_set():
    # The model "wants" an unrelated letter; masking must override it.
    names = ["fn_alpha", "fn_beta"]
    model = FakeModel(target="", distractor="z")
    seqs = _seqs(model, names)
    ctx = model.encode("Function Call:")[0].tolist()
    name, _ = select_function(model, ctx, [], seqs)
    assert name in names


def test_select_function_single_candidate_needs_no_model_calls():
    model = FakeModel()
    seqs = _seqs(model, ["fn_only"])
    name, toks = select_function(model, [0], [], seqs)
    assert name == "fn_only"
    assert model.calls == 0
    assert model.decode(toks) == "fn_only"


def test_select_function_stops_calling_once_unique():
    # Candidates diverge at the first character, so one call suffices.
    model = FakeModel(target=' {"name": "fn_aaaa')
    seqs = _seqs(model, ["fn_aaaa", "gn_bbbb"])
    ctx = model.encode('Function Call: {"name": "')[0].tolist()
    select_function(model, ctx, [], seqs)
    assert model.calls == 1


# ------------------------------------------------------------ gen_string
def _str_ids(model):
    return np.array(
        [i for i, t in model.id_to_token.items()
         if not any(ord(c) < 0x20 for c in t)], dtype=np.int64,
    )


def _ctx(model, tail):
    prompt = "Function Call:" + tail
    return model.encode(prompt)[0].tolist()


def test_gen_string_stops_at_closing_quote():
    model = FakeModel(target=' {"q": "hello world"')
    ids = _str_ids(model)
    ctx = _ctx(model, ' {"q": "')
    val, toks = gen_string_value(model, ctx[:0] + [0], ctx, ids)
    assert val == "hello world"
    assert model.decode(toks).endswith('"')


def test_gen_string_handles_escaped_quotes_and_backslashes():
    target = ' {"q": "say \\"hi\\" in C:\\\\tmp"'
    model = FakeModel(target=target)
    ctx = _ctx(model, ' {"q": "')
    val, _ = gen_string_value(model, [0], ctx, _str_ids(model))
    assert val == 'say "hi" in C:\\tmp'


def test_gen_string_never_emits_control_characters():
    model = FakeModel(target="", distractor="\n")
    ctx = _ctx(model, ' {"q": "')
    val, toks = gen_string_value(
        model, [0], ctx, _str_ids(model), max_tokens=10
    )
    assert "\n" not in model.decode(toks)
    assert len(toks) == 10  # hit the safety cap, never terminated


# ------------------------------------------------------------ gen_number
def _num_tables(model):
    sp = "\u0120"
    first, cont = [], []
    for tid, tok in model.id_to_token.items():
        cleaned = tok.replace(sp, "")
        if cleaned and all(c in "0123456789.+-eE" for c in cleaned):
            first.append(tid)
            if not tok.startswith(sp):
                cont.append(tid)
    stop = {
        tid for tid, tok in model.id_to_token.items()
        if tok.replace(sp, " ").strip() in (",", "}")
    }
    return first, cont, stop


@pytest.mark.parametrize(
    "literal, expected",
    [("3.5", 3.5), ("-12", -12.0), ("1234567.89", 1234567.89),
     ("0.0375", 0.0375), ("2e3", 2000.0)],
)
def test_gen_number_parses_target(literal, expected):
    model = FakeModel(target=f' {{"a": {literal}, "b"')
    first, cont, stop = _num_tables(model)
    ctx = _ctx(model, ' {"a": ')
    val, toks = gen_number_value(model, [0], ctx, first, cont, stop)
    assert val == pytest.approx(expected)
    assert model.decode(toks) == literal


def test_gen_number_only_emits_number_characters():
    model = FakeModel(target="", distractor="z")
    first, cont, stop = _num_tables(model)
    ctx = _ctx(model, ' {"a": ')
    val, toks = gen_number_value(
        model, [0], ctx, first, cont, stop, max_tokens=8
    )
    text = model.decode(toks).strip()
    assert set(text) <= set("0123456789.+-eE")
    assert isinstance(val, float)


def test_gen_number_falls_back_to_zero_on_garbage():
    model = FakeModel(target=' {"a": e,')
    first, cont, stop = _num_tables(model)
    ctx = _ctx(model, ' {"a": ')
    val, _ = gen_number_value(model, [0], ctx, first, cont, stop)
    assert val == 0.0


# --------------------------------------------------------------- gen_bool
@pytest.mark.parametrize("word, expected", [("true", True), ("false", False)])
def test_gen_bool(word, expected):
    model = FakeModel(target=f' {{"b": {word}')
    t, f = model.encode("true")[0].tolist(), model.encode("false")[0].tolist()
    ctx = _ctx(model, ' {"b": ')
    val, toks = gen_bool_value(model, [0], ctx, t, f)
    assert val is expected
    assert toks == (t if expected else f)
