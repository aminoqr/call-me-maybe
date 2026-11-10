"""Constrained decoding helpers for structured LLM generation.

Each function uses logit masking to force the model to produce
only tokens that are valid at the current position in the
JSON output, guaranteeing well-formed results.
"""

import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch optional
    from llm_sdk import Small_LLM_Model

Logits = Union[Sequence[float], npt.NDArray[Any]]
TokenIds = Union[Sequence[int], npt.NDArray[Any]]


def _find_unescaped_quote(s: str) -> Optional[int]:
    """Return the index of the first unescaped '"' in *s*, or None.

    Backslash-escaped quotes (``\\\"``), including doubly-escaped
    backslashes, are skipped so that JSON escape sequences inside a
    string value do not trigger premature termination.
    """
    i = 0
    while i < len(s):
        if s[i] == '\\':
            i += 2  # skip the character following the backslash
            continue
        if s[i] == '"':
            return i
        i += 1
    return None


def _json_unescape(s: str) -> str:
    """Decode JSON string escape sequences inside *s*.

    Wraps the raw value in quotes and round-trips it through
    ``json.loads`` so that sequences like ``\\\\``, ``\\\"``,
    ``\\n`` etc. are converted to their actual characters.
    Falls back to returning *s* unchanged on any error.
    """
    try:
        return str(json.loads('"' + s + '"'))
    except Exception:
        return s


def masked_argmax(logits: Logits, valid_token_ids: TokenIds) -> int:
    """Return the id of the highest-scoring token among *valid_token_ids*.

    Equivalent to ``argmax(apply_logit_mask(logits, valid))`` but never
    materialises a vocabulary-sized masked copy, which matters because it
    runs once per generated token.  Ties resolve to the first id in
    *valid_token_ids*.

    Raises:
        ValueError: If no id in *valid_token_ids* falls inside the vocab.
    """
    ids = np.asarray(valid_token_ids, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < len(logits))]
    if ids.size == 0:
        raise ValueError("no valid token ids inside the vocabulary")
    scores = np.asarray(logits)[ids]
    return int(ids[int(np.argmax(scores))])


def apply_logit_mask(
    logits: List[float], valid_token_ids: List[int]
) -> List[float]:
    """Mask logits so only *valid_token_ids* remain.

    Every position not listed in *valid_token_ids* is set to
    negative infinity, ensuring greedy / argmax decoding can
    only pick an allowed token.

    Args:
        logits: Raw logit scores for the full vocabulary.
        valid_token_ids: Token IDs that should stay unmasked.

    Returns:
        A new logits list with invalid positions set to ``-inf``.
    """
    masked = [-float('inf')] * len(logits)
    for tid in valid_token_ids:
        if tid < len(masked):
            masked[tid] = logits[tid]
    return masked


def select_function(
    model: "Small_LLM_Model",
    input_ids: List[int],
    prefix: List[int],
    func_seqs: Dict[str, List[int]],
) -> Tuple[str, List[int]]:
    """Pick the best function name via trie-constrained decoding.

    At each token position only tokens that are consistent with at
    least one remaining candidate function name are unmasked.  When
    a single candidate remains its suffix is appended directly.

    Args:
        model: The loaded LLM.
        input_ids: Tokenised prompt.
        prefix: Tokens already generated before the function name.
        func_seqs: Mapping of function name to its token sequence.

    Returns:
        ``(function_name, generated_tokens)`` tuple.
    """
    candidates = list(func_seqs.keys())
    pos = 0
    tokens: List[int] = []

    while candidates:
        valid: Set[int] = set()
        alive: List[str] = []
        for c in candidates:
            seq = func_seqs[c]
            if pos < len(seq):
                valid.add(seq[pos])
                alive.append(c)
        candidates = alive

        if not valid:
            break

        # Only one candidate left — emit its remaining tokens.
        if len(candidates) == 1:
            rest = func_seqs[candidates[0]][pos:]
            tokens.extend(rest)
            break

        logits = model.get_logits_from_input_ids(
            input_ids + prefix + tokens
        )
        pick = masked_argmax(logits, sorted(valid))
        tokens.append(pick)
        candidates = [
            c for c in candidates if func_seqs[c][pos] == pick
        ]
        pos += 1

    decoded = model.decode(tokens).strip()
    for name in func_seqs:
        if name == decoded:
            return name, tokens
    return list(func_seqs.keys())[0], tokens


def gen_string_value(
    model: "Small_LLM_Model",
    input_ids: List[int],
    ctx: List[int],
    str_tids: TokenIds,
    max_tokens: int = 50,
) -> Tuple[str, List[int]]:
    """Generate a JSON string value, stopping at a closing quote.

    Tokens are constrained to *str_tids* (characters safe for
    JSON strings).  Generation stops as soon as a ``"`` appears
    in the decoded output.

    Args:
        model: The loaded LLM.
        input_ids: Tokenised prompt.
        ctx: Context tokens including the opening quote.
        str_tids: Vocabulary IDs allowed inside a string.
        max_tokens: Safety cap on generated length.

    Returns:
        ``(string_value, generated_tokens)`` tuple.
    """
    vtokens: List[int] = []
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            input_ids + ctx + vtokens
        )
        pick = masked_argmax(logits, str_tids)
        vtokens.append(pick)

        decoded = model.decode(vtokens)
        idx = _find_unescaped_quote(decoded)
        if idx is not None:
            return _json_unescape(decoded[:idx]).strip(), vtokens

    decoded = model.decode(vtokens)
    idx = _find_unescaped_quote(decoded)
    if idx is not None:
        return _json_unescape(decoded[:idx]).strip(), vtokens
    return _json_unescape(decoded.strip()).strip(), vtokens


def gen_number_value(
    model: "Small_LLM_Model",
    input_ids: List[int],
    ctx: List[int],
    first_tids: TokenIds,
    cont_tids: TokenIds,
    stop_tids: Set[int],
    max_tokens: int = 20,
) -> Tuple[float, List[int]]:
    """Generate a numeric JSON value with constrained decoding.

    Only digit / sign / decimal / exponent characters are allowed.
    Stops when the model selects a structural stop token (``,``
    or ``}``).

    Args:
        model: The loaded LLM.
        input_ids: Tokenised prompt.
        ctx: Context tokens preceding the number.
        first_tids: Valid token IDs for the first position.
        cont_tids: Valid token IDs for continuation positions.
        stop_tids: Structural tokens that signal end of number.
        max_tokens: Safety cap on generated length.

    Returns:
        ``(float_value, generated_tokens)`` tuple.
    """
    vtokens: List[int] = []
    cont_or_stop = np.concatenate([
        np.asarray(cont_tids, dtype=np.int64),
        np.asarray(sorted(stop_tids), dtype=np.int64),
    ])
    for _ in range(max_tokens):
        valid = first_tids if not vtokens else cont_or_stop
        logits = model.get_logits_from_input_ids(
            input_ids + ctx + vtokens
        )
        pick = masked_argmax(logits, valid)
        if pick in stop_tids:
            break
        vtokens.append(pick)

    decoded = model.decode(vtokens).strip()
    try:
        return float(decoded), vtokens
    except ValueError:
        cleaned = ''.join(
            c for c in decoded if c in '0123456789.+-eE'
        )
        try:
            return float(cleaned), vtokens
        except ValueError:
            return 0.0, vtokens


def gen_bool_value(
    model: "Small_LLM_Model",
    input_ids: List[int],
    ctx: List[int],
    true_toks: List[int],
    false_toks: List[int],
) -> Tuple[bool, List[int]]:
    """Generate a boolean value via constrained decoding.

    Compares the model's score for the first token of
    ``true`` versus ``false`` and returns the winner.

    Args:
        model: The loaded LLM.
        input_ids: Tokenised prompt.
        ctx: Context tokens preceding the boolean.
        true_toks: Token sequence for the literal ``true``.
        false_toks: Token sequence for the literal ``false``.

    Returns:
        ``(bool_value, generated_tokens)`` tuple.
    """
    logits = model.get_logits_from_input_ids(input_ids + ctx)
    t_score = logits[true_toks[0]]
    f_score = logits[false_toks[0]]
    if t_score >= f_score:
        return True, true_toks
    return False, false_toks
