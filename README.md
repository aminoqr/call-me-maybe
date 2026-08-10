*This project has been created as part of the 42 curriculum by aasylbye.*

# call-me-maybe

## Description

**call-me-maybe** is a function-calling system for Large Language Models (LLMs) that uses **constrained decoding** to guarantee structurally valid JSON output. Instead of hoping the model produces correct JSON and then trying to parse it, this project forces every generated token to be valid at its position in the output schema — making malformed output impossible by construction.

Given a set of function definitions (with typed parameters) and natural-language prompts, the system:

1. Presents the available functions to a local LLM (Qwen3-0.6B).
2. Uses **logit masking** at each decoding step so only structurally valid tokens can be selected.
3. Outputs a JSON array where every entry contains the original prompt, the selected function name, and correctly-typed argument values.

The result is a lightweight, fully offline pipeline that turns free-form user questions into deterministic, schema-conforming function calls — no retries, no regex post-processing, no external API needed.

## Instructions

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone <repository-url>
cd call-me-maybe
make install
```

This runs `uv sync`, which creates a virtual environment and installs all dependencies (PyTorch, Transformers, Pydantic, etc.) from `pyproject.toml`.

### Running

```bash
make run
```

This executes `uv run python -m src` with the default input/output paths:

| File | Purpose |
|------|---------|
| `data/input/functions_definition.json` | Available function schemas |
| `data/input/function_calling_tests.json` | User prompts to process |
| `data/output/function_calling_results.json` | Generated results |

You can override paths via CLI arguments:

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/results.json
```

### Linting

```bash
make lint
```

Runs `flake8` (style) and `mypy` (type checking) across the codebase.

### Example Usage

**Input prompt file** (`data/input/function_calling_tests.json`):
```json
[
  {"prompt": "What is the product of 3 and 5?"},
  {"prompt": "Execute SQL query 'SELECT * FROM users' on the production database"}
]
```

**Output:**
```json
[
  {
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": {"a": 3.0, "b": 5.0}
  },
  {
    "prompt": "Execute SQL query 'SELECT * FROM users' on the production database",
    "name": "fn_execute_sql_query",
    "parameters": {"query": "SELECT * FROM users", "database": "production"}
  }
]
```

Empty prompts are skipped. The constrained decoder always selects a valid function from the definitions file.

## Algorithm Explanation

The system uses **constrained (structured) decoding** — a technique where, at every token-generation step, the set of allowed next tokens is restricted to those that keep the output on a valid path through the target schema.

### Phase 1 — Structural Prefix

The JSON skeleton tokens `"name": "` are injected directly (not generated), guaranteeing the output starts with the correct key.

### Phase 2 — Function Name Selection (Trie-Constrained)

All function names are pre-tokenised into sequences. At each position, a **trie** of remaining candidates determines which token IDs are valid:

- If multiple candidates share a prefix, the model picks among the diverging tokens (with all others masked to `-inf`).
- Once a single candidate remains, its suffix is emitted deterministically.

This means the model **can only output a function name that exists** in the definitions file.

### Phase 3 — Structural Separator

The tokens for `", "parameters": {` are again injected, not generated.

### Phase 4 — Parameter Value Generation

For each parameter in the selected function's schema, the key tokens (`"param_name": `) are injected, then the value is generated with type-specific constraints:

| Type | Constraint |
|------|-----------|
| **string** | Only tokens without control characters are allowed; generation stops at the first `"` in the decoded output |
| **number** | First token restricted to digit/sign characters; continuation tokens also allow `.`, `e`, `E`; stops at `,` or `}` |
| **boolean** | Compares model logit scores for `true` vs `false` first tokens; emits the winning literal |

Every logit-masking step uses `apply_logit_mask()`, which sets disallowed positions to negative infinity before argmax selection.

## Design Decisions

- **Greedy argmax decoding** — No sampling or temperature. Produces deterministic, reproducible results and avoids the need for random seeds or retry logic.
- **Pre-tokenised vocabularies** — All structural tokens, function names, and valid-character sets are computed once before the processing loop, keeping per-prompt overhead to just the LLM forward passes.
- **Pydantic validation** — Every result is validated against `FunctionCallResult` before being added to the output. Malformed entries are logged and skipped rather than silently included.
- **Modular file layout** — Decoding logic (`decoding.py`), utility helpers (`utils.py`), data models (`models.py`), and the CLI entry point (`__main__.py`) are separated for readability and testability.
- **Graceful error handling** — Invalid JSON inputs, empty prompts, empty function lists, and per-prompt runtime errors are all caught and reported without crashing the entire run. `KeyboardInterrupt` saves partial results.

## Performance Analysis

Benchmarks were run on the full 11-prompt test set in `data/input/function_calling_tests.json` using **Qwen3-0.6B** on CUDA. Three stages are compared:

| Stage | Description |
|-------|-------------|
| **Initial** | Stub implementation with no decoding loop (`TODO` in `__main__.py`) — produces zero results |
| **Baseline** | Same prompt and greedy argmax, but **no** logit masking — the model generates JSON freely |
| **Constrained** | Full pipeline with trie-based function selection, per-type logit masking, and Pydantic validation |

### Accuracy (11 prompts)

| Metric | Initial | Baseline (unconstrained) | Constrained decoding |
|--------|---------|--------------------------|----------------------|
| Usable output | **0 / 11 (0%)** | **0 / 11 (0%)** | **11 / 11 (100%)** |
| Valid JSON | 0% | 0% | **100%** |
| Correct function name | 0% | 0% | **100%** (11/11) |
| Correct parameters | 0% | 0% | **100%** (11/11) |
| End-to-end accuracy | 0% | 0% | **100%** (11/11) |

The unconstrained baseline failed on **every** prompt. Typical raw outputs looked like:

```json
{"fn_multiply_numbers", "a": 3, "b": 5}
```

That is not valid JSON (comma-separated values instead of `"key": value` pairs), so `json.loads` rejects it. On harder prompts the model hallucinated multiple function names in one object or invented parameter keys (`"name": "Alice"` instead of copying literals from the prompt).

Constrained decoding fixed all of these failure modes:

- **Structural validity** — JSON keys (`"name"`, `"parameters"`) and braces are injected, not generated.
- **Function selection** — trie masking restricts tokens to real function names from the schema.
- **Typed parameters** — per-type masks enforce valid strings, numbers, booleans, and integers.
- **Escaped strings** — paths like `C:\Users\john\config.ini` and templates with embedded quotes decode correctly.

### Speed

| Approach | Total time (11 prompts) | Avg per prompt |
|----------|-------------------------|----------------|
| Baseline (unconstrained) | ~762 s | ~69 s |
| Constrained decoding | ~88 s | ~8 s |

Constrained decoding is roughly **8.7× faster** despite doing more work per token, because structural tokens are injected instead of generated and generation stops as soon as each value is complete. The main bottleneck remains the LLM forward pass (one per generated token).

### Reliability

| Property | Baseline | Constrained |
|----------|----------|-------------|
| Guaranteed valid JSON schema | No | **Yes** |
| Guaranteed known function name | No | **Yes** |
| Guaranteed correct parameter types | No | **Yes** |
| Deterministic (greedy, no sampling) | Yes | **Yes** |

### Summary

Starting from a non-functional stub (0% success), adding constrained decoding raised end-to-end accuracy from **0% to 100%** on the benchmark set — a complete turnaround on a 0.6B model that cannot reliably produce structured JSON on its own. The system now handles arithmetic, booleans, high-precision floats, SQL queries, file paths with backslashes, and template strings with nested quotes.

## Challenges Faced

1. **Vocabulary format differences** — The Qwen tokenizer provides both `vocab.json` and `tokenizer.json` with different structures. The solution uses `get_path_to_vocab_file()` which returns the straightforward `{token: id}` mapping.
2. **Multi-token function names** — Names like `fn_substitute_string_with_regex` span many tokens. A naive single-token approach fails; the trie-based `select_function` handles arbitrary-length names correctly.
3. **String termination** — Detecting when a generated string value is "done" is tricky because tokenizers can merge the closing `"` with preceding text. The solution decodes accumulated tokens after each step and checks for the quote character in the decoded string.
4. **Number parsing** — The model sometimes produces tokens with leading spaces (the `Ġ` / `\u0120` prefix). These need to be stripped when building the valid-character sets, while still allowing space-prefixed tokens in the right contexts.

## Testing Strategy

- **Lint validation**: `make lint` runs `flake8` and `mypy --disallow-untyped-defs` on every source file, enforcing style and type safety.
- **Pydantic schema validation**: Every generated result is validated against `FunctionCallResult` before inclusion. This catches type mismatches, missing keys, and extra fields automatically.
- **Diverse prompt coverage**: The test set includes arithmetic, string reversal, square roots, regex substitution, empty prompts, and ambiguous prompts (e.g. "multiply" when no multiply function is defined) to exercise all code paths.
- **Error-path testing**: The input set includes an empty prompt (`""`) to verify it is skipped gracefully. Non-list JSON, missing files, and keyboard interrupts are also handled.

## Resources

- [Structured Generation (Outlines)](https://dottxt-ai.github.io/outlines/latest/) — Reference library for constrained decoding; inspired the logit-masking approach used here.
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/) — For the model and tokenizer APIs.
- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — The LLM used for inference.
- [Pydantic Documentation](https://docs.pydantic.dev/latest/) — For input/output validation models.
- [How LLM Function Calling Works](https://gorilla.cs.berkeley.edu/blogs/7_function_calling.html) — Background on function calling in LLMs.

### AI Usage

AI (GitHub Copilot) was used as a development assistant throughout this project for:

- **Code structuring**: Helped organise the codebase into modules (`decoding.py`, `utils.py`, `models.py`, `__main__.py`) and write docstrings.
- **Debugging**: Assisted in diagnosing issues with vocabulary loading (tokenizer.json vs vocab.json formats) and the constrained decoding state machine.
- **Error handling**: Helped implement graceful handling of edge cases (empty prompts, invalid JSON inputs, keyboard interrupts).

All algorithmic decisions (trie-constrained function selection, per-type logit masking, greedy decoding) were made by the developer; AI was used to accelerate implementation and catch issues.