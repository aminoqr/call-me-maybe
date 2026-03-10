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

**Input prompt file:**
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

**Output:**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

Empty prompts are skipped. Ambiguous prompts (e.g. "multiply 3 by 5" when no multiply function exists) are mapped to the best available match — the constrained decoder guarantees a valid function is always selected.

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

- **Accuracy**: The constrained decoder correctly maps straightforward prompts to the right function and extracts literal values from the prompt text. Ambiguous prompts (no matching function exists) gracefully fall back to the closest available function.
- **Speed**: The main bottleneck is the LLM forward pass (one per generated token). Pre-computing token sets and injecting structural tokens (instead of generating them) reduces the number of forward passes significantly — typically ~5–15 per prompt depending on parameter count and string length.
- **Reliability**: Because every token is constrained, the output is **guaranteed** to be valid JSON with the correct keys and types. There is no possibility of malformed output, unclosed brackets, or hallucinated function names.

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