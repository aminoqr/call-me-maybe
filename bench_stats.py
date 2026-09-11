"""Benchmark constrained vs unconstrained function calling.

Collects accuracy, validity, timing, and token-count stats for README.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from llm_sdk import Small_LLM_Model
from src.decoding import (
    gen_bool_value,
    gen_number_value,
    gen_string_value,
    select_function,
)
from src.models import FunctionCallResult, FunctionDefinition, PromptInput
from src.utils import build_func_prompt, load_json_file


EXPECTED: Dict[str, Dict[str, Any]] = {
    "What is the product of 3 and 5?": {
        "name": "fn_multiply_numbers",
        "parameters": {"a": 3.0, "b": 5.0},
    },
    "What is the product of 12 and 4?": {
        "name": "fn_multiply_numbers",
        "parameters": {"a": 12.0, "b": 4.0},
    },
    "Is 4 an even number?": {
        "name": "fn_is_even",
        "parameters": {"n": 4},
    },
    "Is 7 an even number?": {
        "name": "fn_is_even",
        "parameters": {"n": 7},
    },
    "Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years": {
        "name": "fn_calculate_compound_interest",
        "parameters": {
            "principal": 1234567.89,
            "rate": 0.0375,
            "years": 23,
        },
    },
    "Execute SQL query 'SELECT * FROM users' on the production database": {
        "name": "fn_execute_sql_query",
        "parameters": {
            "query": "SELECT * FROM users",
            "database": "production",
        },
    },
    "Run the query 'INSERT INTO logs VALUES (1, 2, 3)' on the system database": {
        "name": "fn_execute_sql_query",
        "parameters": {
            "query": "INSERT INTO logs VALUES (1, 2, 3)",
            "database": "system",
        },
    },
    "Read the file at /home/user/data.json with utf-8 encoding": {
        "name": "fn_read_file",
        "parameters": {
            "path": "/home/user/data.json",
            "encoding": "utf-8",
        },
    },
    "Read C:\\Users\\john\\config.ini with latin-1 encoding": {
        "name": "fn_read_file",
        "parameters": {
            "path": "C:\\Users\\john\\config.ini",
            "encoding": "latin-1",
        },
    },
    "Format template: Hello {user}'s profile!": {
        "name": "fn_format_template",
        "parameters": {"template": "Hello {user}'s profile!"},
    },
    "Format template: Say \"hello\" to {name}": {
        "name": "fn_format_template",
        "parameters": {"template": 'Say "hello" to {name}'},
    },
}


def params_match(got: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    if set(got.keys()) != set(expected.keys()):
        return False
    for k, exp in expected.items():
        g = got[k]
        if isinstance(exp, float):
            try:
                if abs(float(g) - exp) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif isinstance(exp, int) and not isinstance(exp, bool):
            try:
                if int(g) != exp:
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if str(g) != str(exp):
                return False
    return True


def score_result(prompt: str, result: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    exp = EXPECTED.get(prompt)
    if result is None or exp is None:
        return {
            "valid_json_schema": False,
            "correct_name": False,
            "correct_params": False,
            "end_to_end": False,
        }
    try:
        FunctionCallResult(**result)
        valid = True
    except Exception:
        valid = False
    correct_name = result.get("name") == exp["name"]
    correct_params = params_match(result.get("parameters", {}), exp["parameters"])
    return {
        "valid_json_schema": valid,
        "correct_name": correct_name,
        "correct_params": correct_params,
        "end_to_end": valid and correct_name and correct_params,
    }


def try_parse_unconstrained(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of free-form model JSON output."""
    text = raw.strip()
    # strip markdown fences if any
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find first {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start : end + 1]
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict) and "name" in obj:
            return {
                "name": obj.get("name"),
                "parameters": obj.get("parameters", obj.get("arguments", {})),
            }
    except json.JSONDecodeError:
        pass
    return None


def run_unconstrained(
    model: Small_LLM_Model,
    prompt_text: str,
    max_tokens: int = 80,
) -> Tuple[str, int, float]:
    """Greedy decode without logit masking. Returns raw text, tokens, seconds."""
    inp = model.encode(prompt_text)[0].tolist()
    gen: List[int] = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(inp + gen)
        pick = max(range(len(logits)), key=lambda i: logits[i])
        gen.append(pick)
        decoded = model.decode(gen)
        opens = decoded.count("{")
        closes = decoded.count("}")
        if opens > 0 and opens == closes and (
            '"name"' in decoded or "parameters" in decoded or "fn_" in decoded
        ):
            break
    elapsed = time.perf_counter() - t0
    return model.decode(gen), len(gen), elapsed


def run_constrained_one(
    model: Small_LLM_Model,
    functions: List[FunctionDefinition],
    prompt: str,
    name_key: List[int],
    params_pfx: List[int],
    func_seqs: Dict[str, List[int]],
    true_toks: List[int],
    false_toks: List[int],
    str_tids: List[int],
    num_first: List[int],
    num_cont: List[int],
    stop_tids: Set[int],
    func_desc: str,
) -> Tuple[Optional[Dict[str, Any]], int, float]:
    prompt_text = (
        "Available functions:\n"
        f"{func_desc}\n\n"
        f"User request: {prompt}\n"
        "Select the single most appropriate function "
        "and copy ALL parameter values verbatim from "
        "the request.\n"
        'Function Call: {"'
    )
    inp = model.encode(prompt_text)[0].tolist()
    t0 = time.perf_counter()
    gen = list(name_key)
    fname, fn_toks = select_function(model, inp, gen, func_seqs)
    gen.extend(fn_toks)
    gen.extend(params_pfx)

    sel_fn = next((fn for fn in functions if fn.name == fname), None)
    if sel_fn is None:
        return None, len(gen), time.perf_counter() - t0

    values: Dict[str, Any] = {}
    pitems = list(sel_fn.parameters.items())
    for i, (pname, pdef) in enumerate(pitems):
        if i > 0:
            gen.extend(model.encode(", ")[0].tolist())
        if pdef.type == "string":
            gen.extend(model.encode(f'"{pname}": "')[0].tolist())
            val, vt = gen_string_value(model, inp, gen, str_tids)
            gen.extend(vt)
            values[pname] = val
        elif pdef.type == "boolean":
            gen.extend(model.encode(f'"{pname}": ')[0].tolist())
            bval, bvt = gen_bool_value(model, inp, gen, true_toks, false_toks)
            gen.extend(bvt)
            values[pname] = bval
        elif pdef.type == "integer":
            gen.extend(model.encode(f'"{pname}": ')[0].tolist())
            nval, nvt = gen_number_value(
                model, inp, gen, num_first, num_cont, stop_tids
            )
            gen.extend(nvt)
            values[pname] = int(nval)
        else:
            gen.extend(model.encode(f'"{pname}": ')[0].tolist())
            nval, nvt = gen_number_value(
                model, inp, gen, num_first, num_cont, stop_tids
            )
            gen.extend(nvt)
            values[pname] = nval

    result = {"prompt": prompt, "name": fname, "parameters": values}
    elapsed = time.perf_counter() - t0
    try:
        FunctionCallResult(**result)
        return result, len(gen), elapsed
    except ValidationError:
        return None, len(gen), elapsed


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    keys = [
        "valid_json_schema",
        "correct_name",
        "correct_params",
        "end_to_end",
    ]
    counts = {k: sum(1 for r in rows if r["scores"][k]) for k in keys}
    times = [r["seconds"] for r in rows]
    toks = [r["tokens"] for r in rows]
    return {
        "n": n,
        "counts": counts,
        "pct": {k: (100.0 * counts[k] / n if n else 0) for k in keys},
        "total_seconds": sum(times),
        "avg_seconds": sum(times) / n if n else 0,
        "total_tokens": sum(toks),
        "avg_tokens": sum(toks) / n if n else 0,
        "min_seconds": min(times) if times else 0,
        "max_seconds": max(times) if times else 0,
    }


def main() -> None:
    functions = [
        FunctionDefinition(**fn)
        for fn in load_json_file("data/input/functions_definition.json")
    ]
    prompts = [
        PromptInput(**p)
        for p in load_json_file("data/input/function_calling_tests.json")
    ]
    prompts = [p for p in prompts if p.prompt and p.prompt.strip()]

    print(f"Loading model... ({len(functions)} funcs, {len(prompts)} prompts)")
    t_load = time.perf_counter()
    # Force CPU: host GPU (GTX 1070 / sm_61) is not supported by this torch build.
    model = Small_LLM_Model(device="cpu")
    load_s = time.perf_counter() - t_load
    print(f"Model loaded in {load_s:.1f}s on {model._device}")

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {int(v): k for k, v in vocab.items()}

    name_key = model.encode('"name": "')[0].tolist()
    params_pfx = model.encode('", "parameters": {')[0].tolist()
    func_seqs = {fn.name: model.encode(fn.name)[0].tolist() for fn in functions}
    true_toks = model.encode("true")[0].tolist()
    false_toks = model.encode("false")[0].tolist()
    sp = "\u0120"
    str_tids = [
        tid
        for tid, tok in id_to_token.items()
        if not any(ord(c) < 0x20 for c in tok.replace(sp, " "))
    ]
    num_chars = "0123456789.+-eE"
    num_first: List[int] = []
    num_cont: List[int] = []
    for tid, tok in id_to_token.items():
        cleaned = tok.replace(sp, "")
        if cleaned and all(c in num_chars for c in cleaned):
            num_first.append(tid)
            if not tok.startswith(sp):
                num_cont.append(tid)
    stop_tids = {
        tid
        for tid, tok in id_to_token.items()
        if tok.replace(sp, " ").strip() in (",", "}")
    }
    func_desc = build_func_prompt(functions)

    constrained_rows: List[Dict[str, Any]] = []
    unconstrained_rows: List[Dict[str, Any]] = []
    constrained_results: List[Dict[str, Any]] = []

    print("\n=== CONSTRAINED ===")
    for p in prompts:
        print(f"\n[C] {p.prompt[:70]}")
        result, ntok, secs = run_constrained_one(
            model,
            functions,
            p.prompt,
            name_key,
            params_pfx,
            func_seqs,
            true_toks,
            false_toks,
            str_tids,
            num_first,
            num_cont,
            stop_tids,
            func_desc,
        )
        scores = score_result(p.prompt, result)
        print(f"  -> {result}  ({secs:.2f}s, {ntok} toks) e2e={scores['end_to_end']}")
        if result:
            constrained_results.append(result)
        constrained_rows.append(
            {
                "prompt": p.prompt,
                "result": result,
                "seconds": secs,
                "tokens": ntok,
                "scores": scores,
            }
        )

    print("\n=== UNCONSTRAINED BASELINE ===")
    for p in prompts:
        prompt_text = (
            "Available functions:\n"
            f"{func_desc}\n\n"
            f"User request: {p.prompt}\n"
            "Select the single most appropriate function "
            "and copy ALL parameter values verbatim from "
            "the request.\n"
            "Respond with ONLY a JSON object like "
            '{"name": "...", "parameters": {...}}\n'
            "Function Call: "
        )
        print(f"\n[U] {p.prompt[:70]}")
        raw, ntok, secs = run_unconstrained(model, prompt_text)
        parsed = try_parse_unconstrained(raw)
        if parsed is not None:
            parsed = {"prompt": p.prompt, **parsed}
        scores = score_result(p.prompt, parsed)
        print(f"  raw[:120]={raw[:120]!r}")
        print(
            f"  -> {parsed}  ({secs:.2f}s, {ntok} toks) "
            f"e2e={scores['end_to_end']} valid={scores['valid_json_schema']}"
        )
        unconstrained_rows.append(
            {
                "prompt": p.prompt,
                "raw": raw,
                "result": parsed,
                "seconds": secs,
                "tokens": ntok,
                "scores": scores,
            }
        )

    c_agg = aggregate(constrained_rows)
    u_agg = aggregate(unconstrained_rows)

    out = {
        "device": model._device,
        "model": "Qwen/Qwen3-0.6B",
        "model_load_seconds": load_s,
        "n_functions": len(functions),
        "n_prompts": len(prompts),
        "n_prompt_file_entries": len(
            load_json_file("data/input/function_calling_tests.json")
        ),
        "constrained": {
            "aggregate": c_agg,
            "rows": constrained_rows,
        },
        "unconstrained": {
            "aggregate": u_agg,
            "rows": [
                {
                    **r,
                    "raw": r["raw"][:500],
                }
                for r in unconstrained_rows
            ],
        },
    }

    Path("data/output").mkdir(parents=True, exist_ok=True)
    with open("data/output/bench_stats.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open("data/output/function_calling_results.json", "w") as f:
        json.dump(constrained_results, f, indent=4)

    print("\n========== SUMMARY ==========")
    print(json.dumps({"constrained": c_agg, "unconstrained": u_agg, "load_s": load_s}, indent=2))
    print("Wrote data/output/bench_stats.json and function_calling_results.json")


if __name__ == "__main__":
    main()
