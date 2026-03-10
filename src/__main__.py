"""Entry point for the constrained-decoding function calling system.

Reads function definitions and user prompts from JSON files,
uses an LLM with logit masking to generate structured function
calls, and writes validated results to an output JSON file.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

from pydantic import ValidationError

from src.models import (
    FunctionDefinition,
    FunctionCallResult,
    PromptInput,
)
from src.utils import load_json_file, build_func_prompt
from src.decoding import (
    select_function,
    gen_string_value,
    gen_number_value,
    gen_bool_value,
)
from llm_sdk import Small_LLM_Model


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
    print("Initializing the LLM model...")
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_token: Dict[int, str] = {
        int(v): k for k, v in vocab.items()
    }

    # Structural JSON token sequences
    name_key = model.encode('"name": "')[0].tolist()
    params_pfx = model.encode(
        '", "parameters": {'
    )[0].tolist()

    # Function-name token sequences (for the trie)
    func_seqs: Dict[str, List[int]] = {
        fn.name: model.encode(fn.name)[0].tolist()
        for fn in functions
    }

    # Boolean literals
    true_toks = model.encode('true')[0].tolist()
    false_toks = model.encode('false')[0].tolist()

    # String-safe token IDs (no control characters)
    sp = '\u0120'
    str_tids: List[int] = [
        tid for tid, tok in id_to_token.items()
        if not any(
            ord(c) < 0x20 for c in tok.replace(sp, ' ')
        )
    ]

    # Number-safe token IDs
    num_chars = '0123456789.+-eE'
    num_first: List[int] = []
    num_cont: List[int] = []
    for tid, tok in id_to_token.items():
        cleaned = tok.replace(sp, '')
        if cleaned and all(c in num_chars for c in cleaned):
            num_first.append(tid)
            if not tok.startswith(sp):
                num_cont.append(tid)

    # Structural stop tokens (, and })
    stop_tids: Set[int] = {
        tid for tid, tok in id_to_token.items()
        if tok.replace(sp, ' ').strip() in (',', '}')
    }

    func_desc = build_func_prompt(functions)

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
                prompt_text = (
                    "Available functions:\n"
                    f"{func_desc}\n\n"
                    f"User request: {p.prompt}\n"
                    "Select the single most appropriate function "
                    "and copy ALL parameter values verbatim from "
                    "the request.\n"
                    'Function Call: {"'
                )
                inp = model.encode(
                    prompt_text
                )[0].tolist()

                # Phase 1 – force "name": "
                gen = list(name_key)

                # Phase 2 – constrained function selection
                fname, fn_toks = select_function(
                    model, inp, gen, func_seqs
                )
                gen.extend(fn_toks)
                tqdm.write(f"  Function: {fname}")

                # Phase 3 – force ", "parameters": {
                gen.extend(params_pfx)

                # Phase 4 – constrained parameter generation
                sel_fn = next(
                    (fn for fn in functions
                     if fn.name == fname),
                    None,
                )
                if sel_fn is None:
                    tqdm.write(
                        f"  Warning: '{fname}' "
                        "not found, skipping."
                    )
                    continue

                pitems = list(sel_fn.parameters.items())
                values: dict[str, Any] = {}

                for i, (pname, pdef) in enumerate(pitems):
                    if i > 0:
                        sep = model.encode(
                            ', '
                        )[0].tolist()
                        gen.extend(sep)

                    if pdef.type == "string":
                        kt = model.encode(
                            f'"{pname}": "'
                        )[0].tolist()
                        gen.extend(kt)
                        val, vt = gen_string_value(
                            model, inp, gen, str_tids
                        )
                        gen.extend(vt)
                        values[pname] = val

                    elif pdef.type == "boolean":
                        kt = model.encode(
                            f'"{pname}": '
                        )[0].tolist()
                        gen.extend(kt)
                        bval, bvt = gen_bool_value(
                            model, inp, gen,
                            true_toks, false_toks,
                        )
                        gen.extend(bvt)
                        values[pname] = bval

                    elif pdef.type == "integer":
                        kt = model.encode(
                            f'"{pname}": '
                        )[0].tolist()
                        gen.extend(kt)
                        nval, nvt = gen_number_value(
                            model, inp, gen,
                            num_first, num_cont,
                            stop_tids,
                        )
                        gen.extend(nvt)
                        values[pname] = int(nval)

                    else:  # number (default)
                        kt = model.encode(
                            f'"{pname}": '
                        )[0].tolist()
                        gen.extend(kt)
                        nval, nvt = gen_number_value(
                            model, inp, gen,
                            num_first, num_cont,
                            stop_tids,
                        )
                        gen.extend(nvt)
                        values[pname] = nval

                # Build and validate result
                result: dict[str, Any] = {
                    "prompt": p.prompt,
                    "name": fname,
                    "parameters": values,
                }
                try:
                    FunctionCallResult(**result)
                    results.append(result)
                    tqdm.write(f"  Result: {result}")
                except ValidationError as e:
                    tqdm.write(
                        "  Validation error, "
                        f"skipping: {e}"
                    )

            except Exception as e:
                tqdm.write(
                    "  Error processing prompt, "
                    f"skipping: {e}"
                )

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
