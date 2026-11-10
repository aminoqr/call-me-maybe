"""Entry point for the constrained-decoding function calling system.

Reads function definitions and user prompts from JSON files,
uses an LLM with logit masking to generate structured function
calls, and writes validated results to an output JSON file.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from pydantic import ValidationError

from src.models import FunctionDefinition, PromptInput
from src.utils import load_json_file
from src.pipeline import FunctionCaller


def main() -> None:
    """Run the full function-calling pipeline."""
    # ── 1. Argument Parsing ──────────────────────────────
    parser = argparse.ArgumentParser(
        description="Constrained Decoding Function Caller"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="disable the KV cache (slower; for benchmarking)",
    )
    args = parser.parse_args()

    # ── 2. Data Loading & Validation ─────────────────────
    raw_functions = load_json_file(args.functions_definition)
    if not isinstance(raw_functions, list):
        print(
            "Error: functions definition file "
            "must be a JSON array."
        )
        sys.exit(1)
    try:
        functions = [
            FunctionDefinition(**fn) for fn in raw_functions
        ]
        print(
            f"Successfully loaded {len(functions)} "
            "function definitions."
        )
    except (ValidationError, TypeError) as e:
        print(f"Validation error in functions file: {e}")
        sys.exit(1)
    if not functions:
        print(
            "Error: no function definitions found "
            "\u2014 cannot proceed."
        )
        sys.exit(1)

    raw_prompts = load_json_file(args.input)
    if not isinstance(raw_prompts, list):
        print("Error: input prompts file must be a JSON array.")
        sys.exit(1)
    try:
        prompts = [PromptInput(**p) for p in raw_prompts]
        print(f"Successfully loaded {len(prompts)} prompts.")
    except (ValidationError, TypeError) as e:
        print(f"Validation error in prompts file: {e}")
        sys.exit(1)

    # ── 3. LLM Initialisation & Vocabulary Setup ─────────
    # Imported lazily: torch is heavy and argument/data errors above
    # should fail fast without it.
    from llm_sdk import Small_LLM_Model
    from src.kvcache import CachedModel

    print("Initializing the LLM model...")
    model = CachedModel(
        Small_LLM_Model(), use_cache=not args.no_kv_cache
    )
    caller = FunctionCaller.from_model(model, functions)

    # ── 4. Processing Loop ───────────────────────────────
    results: List[Dict[str, Any]] = []
    try:
        for p in tqdm(prompts, desc="Processing prompts", unit="prompt"):
            # Skip empty / whitespace-only prompts
            if not p.prompt or not p.prompt.strip():
                tqdm.write("\nSkipping empty prompt.")
                continue

            tqdm.write(f"\nProcessing: {p.prompt}")
            try:
                outcome = caller.call(p.prompt)
            except Exception as e:
                tqdm.write(
                    "  Error processing prompt, "
                    f"skipping: {e}"
                )
                continue
            if outcome.result is None:
                tqdm.write(f"  Skipping: {outcome.error}")
                continue
            results.append(outcome.result)
            tqdm.write(f"  Function: {outcome.result['name']}")
            tqdm.write(f"  Result: {outcome.result}")

    except KeyboardInterrupt:
        tqdm.write("\nInterrupted. Saving partial results...")

    # ── 5. Save Results ──────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
