"""Benchmark constrained decoding against unconstrained baselines.

Usage::

    uv run python -m benchmarks.run_benchmark
    uv run python -m benchmarks.run_benchmark --limit 10      # smoke test

Writes ``data/benchmarks/results.json`` (every row, every method) and
prints a summary.  All methods share one model instance and the same KV
cache implementation, so latency differences are due to the method.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from benchmarks.baselines import run_baseline
from benchmarks.scoring import CATEGORIES, classify, wilson_interval
from llm_sdk import Small_LLM_Model
from src import pipeline
from src.kvcache import CachedModel
from src.models import FunctionDefinition
from src.pipeline import FunctionCaller
from src.utils import load_json_file

EVAL_SET = "data/benchmarks/eval_set.json"
FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_OUT = "data/benchmarks/results.json"
METHODS = (
    "constrained",
    "constrained_legacy_prefix",
    "zero_shot",
    "few_shot_retry",
)
SEED = 1234


def _timed(
    model: CachedModel, fn: Callable[[], Dict[str, Any]]
) -> Dict[str, Any]:
    calls0, toks0 = model.forward_calls, model.forward_tokens
    t0 = time.perf_counter()
    out = fn()
    out.setdefault("seconds", time.perf_counter() - t0)
    out["forward_calls"] = model.forward_calls - calls0
    out["forward_tokens"] = model.forward_tokens - toks0
    return out


def constrained_runner(
    caller: FunctionCaller,
) -> Callable[[str], Dict[str, Any]]:
    """Adapt :class:`FunctionCaller` to the benchmark row format."""
    def run(request: str) -> Dict[str, Any]:
        outcome = caller.call(request)
        return {
            "result": outcome.result,
            "raw": "",
            "tokens": len(outcome.tokens),
            "attempts": 1,
            "error": outcome.error,
        }
    return run


def legacy_caller(
    model: CachedModel, functions: List[FunctionDefinition]
) -> FunctionCaller:
    """Caller using the original prompt tail ``Function Call: {"`` plus
    the forced prefix ``"name": "`` (which yields ``{""name"``)."""
    return FunctionCaller.from_model(
        model, functions,
        prompt_template=pipeline.PROMPT_TEMPLATE + ' {{"',
        forced_prefix='"name": "',
    )


def summarise(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate accuracy and cost, overall and per split."""
    def block(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(subset)
        counts = {
            c: sum(1 for r in subset if r["category"] == c)
            for c in CATEGORIES
        }
        lo, hi = wilson_interval(counts["correct"], n)
        secs = [r["seconds"] for r in subset]
        return {
            "n": n,
            "counts": counts,
            "valid_pct": 100 * (n - counts["invalid"]) / n if n else 0,
            "accuracy_pct": 100 * counts["correct"] / n if n else 0,
            "accuracy_ci95": [round(lo, 1), round(hi, 1)],
            "total_seconds": sum(secs),
            "mean_seconds": statistics.fmean(secs) if secs else 0,
            "median_seconds": statistics.median(secs) if secs else 0,
            "max_seconds": max(secs) if secs else 0,
            "mean_output_tokens": (
                statistics.fmean(r["tokens"] for r in subset)
                if subset else 0
            ),
            "mean_forward_calls": (
                statistics.fmean(r["forward_calls"] for r in subset)
                if subset else 0
            ),
            "mean_attempts": (
                statistics.fmean(r["attempts"] for r in subset)
                if subset else 0
            ),
        }

    out = {"all": block(rows)}
    for split in ("dev", "heldout"):
        sub = [r for r in rows if r["split"] == split]
        if sub:
            out[split] = block(sub)
    funcs = sorted({r["expected"]["name"] for r in rows})
    out["per_function"] = {
        f: block([r for r in rows if r["expected"]["name"] == f])
        for f in funcs
    }
    return out


def evaluate(
    name: str,
    model: CachedModel,
    runner: Callable[[str], Dict[str, Any]],
    items: List[Dict[str, Any]],
    functions: List[FunctionDefinition],
) -> Dict[str, Any]:
    """Run *runner* over *items* and score every output."""
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        out = _timed(model, lambda: runner(item["prompt"]))
        cat = classify(out["result"], item, functions)
        rows.append({
            "prompt": item["prompt"], "split": item["split"],
            "expected": {"name": item["name"],
                         "parameters": item["parameters"]},
            "category": cat, **out,
        })
        print(
            f"[{name}] {i}/{len(items)} {cat:<12} "
            f"{out['seconds']:5.1f}s  {item['prompt'][:56]}",
            flush=True,
        )
    return {"summary": summarise(rows), "rows": rows}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N prompts (smoke test)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--nocache-n", type=int, default=11,
                    help="prompts for the no-KV-cache speed comparison")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    functions = [FunctionDefinition(**f) for f in load_json_file(FUNCTIONS)]
    items: List[Dict[str, Any]] = load_json_file(EVAL_SET)
    if args.limit:
        items = items[:args.limit]
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"{len(functions)} functions, {len(items)} prompts", flush=True)
    t0 = time.perf_counter()
    base = Small_LLM_Model(device="cpu")  # GTX 1070 (sm_61) unsupported
    load_s = time.perf_counter() - t0
    model = CachedModel(base)
    tok = base._tokenizer
    stop_ids = {
        i for i in (
            tok.eos_token_id,
            tok.convert_tokens_to_ids("<|im_end|>"),
            tok.convert_tokens_to_ids("<|endoftext|>"),
        ) if i is not None
    }

    caller = FunctionCaller.from_model(model, functions)
    caller.call(items[0]["prompt"])  # warm-up, not timed or scored

    results: Dict[str, Any] = {
        "model": "Qwen/Qwen3-0.6B", "device": "cpu",
        "model_load_seconds": load_s,
        "n_prompts": len(items), "n_functions": len(functions),
        "methods": {},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2, default=str))

    for name in wanted:
        if name == "constrained":
            runner = constrained_runner(caller)
        elif name == "constrained_legacy_prefix":
            runner = constrained_runner(legacy_caller(model, functions))
        elif name in ("zero_shot", "few_shot_retry"):
            few = name == "few_shot_retry"

            def runner(req: str, few: bool = few) -> Dict[str, Any]:
                return run_baseline(
                    model, functions, req, stop_ids, few_shot=few,
                    max_attempts=3 if few else 1, seed=SEED,
                )
        else:
            raise SystemExit(f"unknown method {name!r}")
        results["methods"][name] = evaluate(
            name, model, runner, items, functions
        )
        save()

    # KV-cache speed-up: same constrained pipeline, cache on vs off.
    n = min(args.nocache_n, len(items))
    if n and "constrained" in wanted:
        slow = CachedModel(base, use_cache=False)
        slow_caller = FunctionCaller.from_model(slow, functions)
        nocache = evaluate(
            "constrained_nocache", slow, constrained_runner(slow_caller),
            items[:n], functions,
        )
        fast_rows = results["methods"]["constrained"]["rows"][:n]
        same = all(
            a["result"] == b["result"]
            for a, b in zip(fast_rows, nocache["rows"])
        )
        results["kv_cache_comparison"] = {
            "n": n,
            "outputs_identical": same,
            "cached_total_seconds": sum(r["seconds"] for r in fast_rows),
            "uncached_total_seconds": nocache["summary"]["all"][
                "total_seconds"],
            "cached_forward_tokens": sum(
                r["forward_tokens"] for r in fast_rows),
            "uncached_forward_tokens": sum(
                r["forward_tokens"] for r in nocache["rows"]),
        }
        save()

    print("\n=== SUMMARY ===")
    for name, data in results["methods"].items():
        s = data["summary"]["all"]
        print(f"{name:<28} acc {s['accuracy_pct']:5.1f}%  "
              f"valid {s['valid_pct']:5.1f}%  "
              f"{s['mean_seconds']:.2f}s/prompt")
    print(json.dumps(results.get("kv_cache_comparison"), indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
