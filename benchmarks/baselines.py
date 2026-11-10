"""Unconstrained baselines the constrained decoder is compared against.

Both baselines let the model write free-form text and parse it afterwards,
which is what most people do without constrained decoding:

* ``zero_shot``       - one greedy attempt, instructions only.
* ``few_shot_retry``  - two worked examples in the prompt, and if the output
  is not a schema-valid call, up to two more *sampled* attempts
  (temperature 0.7), keeping the first valid one.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from benchmarks.scoring import extract_json_object, parse_call
from src.models import FunctionDefinition
from src.utils import build_func_prompt
from src.validation import validate_call

FORMAT_HINT = (
    "Respond with ONLY a JSON object like "
    '{"name": "...", "parameters": {...}}\n'
)

# Deliberately disjoint from the evaluation prompts.
FEW_SHOT_EXAMPLES = (
    ("What is 9 times 6?",
     '{"name": "fn_multiply_numbers", "parameters": {"a": 9, "b": 6}}'),
    ("Open /srv/app.log using ascii encoding",
     '{"name": "fn_read_file", "parameters": '
     '{"path": "/srv/app.log", "encoding": "ascii"}}'),
)


def zero_shot_prompt(functions: Sequence[FunctionDefinition],
                     request: str) -> str:
    """Instruction-only prompt."""
    return (
        "Available functions:\n"
        f"{build_func_prompt(list(functions))}\n\n"
        f"User request: {request}\n"
        "Select the single most appropriate function "
        "and copy ALL parameter values verbatim from the request.\n"
        f"{FORMAT_HINT}"
        "Function Call: "
    )


def few_shot_prompt(functions: Sequence[FunctionDefinition],
                    request: str) -> str:
    """Instruction prompt with worked examples."""
    shots = "".join(
        f"User request: {q}\nFunction Call: {a}\n\n"
        for q, a in FEW_SHOT_EXAMPLES
    )
    return (
        "Available functions:\n"
        f"{build_func_prompt(list(functions))}\n\n"
        "Select the single most appropriate function "
        "and copy ALL parameter values verbatim from the request.\n"
        f"{FORMAT_HINT}\n"
        f"{shots}"
        f"User request: {request}\n"
        "Function Call: "
    )


def sample_token(logits: Any, temperature: float,
                 rng: np.random.Generator) -> int:
    """Greedy when ``temperature <= 0``, otherwise softmax sampling."""
    arr = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        return int(np.argmax(arr))
    z = (arr - arr.max()) / temperature
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate_free(
    model: Any,
    prompt: str,
    stop_ids: Set[int],
    max_tokens: int = 120,
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[str, int]:
    """Generate unconstrained text until a full JSON object or EOS.

    Returns:
        ``(text, number_of_generated_tokens)``.
    """
    rng = rng or np.random.default_rng(0)
    inp = model.encode(prompt)[0].tolist()
    gen: List[int] = []
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(inp + gen)
        pick = sample_token(logits, temperature, rng)
        if pick in stop_ids:
            break
        gen.append(pick)
        if extract_json_object(model.decode(gen)) is not None:
            break
    return model.decode(gen), len(gen)


def run_baseline(
    model: Any,
    functions: Sequence[FunctionDefinition],
    request: str,
    stop_ids: Set[int],
    *,
    few_shot: bool,
    max_attempts: int,
    seed: int,
) -> Dict[str, Any]:
    """Run one baseline on one request and return result + cost."""
    build = few_shot_prompt if few_shot else zero_shot_prompt
    prompt = build(functions, request)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    tokens = 0
    raw = ""
    parsed: Optional[Dict[str, Any]] = None
    attempts = 0
    for attempt in range(max_attempts):
        attempts += 1
        raw, n = generate_free(
            model, prompt, stop_ids,
            temperature=0.0 if attempt == 0 else 0.7, rng=rng,
        )
        tokens += n
        parsed = parse_call(raw)
        if parsed is not None and not validate_call(
            parsed["name"], parsed["parameters"], functions
        ):
            break
    result = (
        {"prompt": request, **parsed} if parsed is not None else None
    )
    return {
        "result": result,
        "raw": raw[:300],
        "seconds": time.perf_counter() - t0,
        "tokens": tokens,
        "attempts": attempts,
    }
