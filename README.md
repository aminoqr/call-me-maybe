# call-me-maybe

I got tired of tiny LLMs inventing broken JSON and calling it a day.

**call-me-maybe** is a local function-calling pipeline for **Qwen3-0.6B**. At every decoding step I mask illegal logits so the model *cannot* leave the JSON schema. No retries. No regex cleanup. If a token would break the structure, it never gets chosen.

You give it function definitions + natural language prompts. It returns schema-valid function calls with typed arguments — offline, deterministic, greedy argmax.

```bash
make install
make run
```

---

## Why I built this

Small models are fun until you ask them for structured output. Mine would spit out pseudo-code, invent parameter names, or produce something that *looked* like JSON and then died in `json.loads`.

I wanted the opposite of "hope and post-process": **make invalid output impossible while the tokens are being generated**.

So I implemented constrained decoding from scratch — trie-based function name selection, type-aware value generation (string / number / integer / boolean), and structural tokens injected instead of generated.

---

## Results (measured on the full test set)

Setup: **Qwen/Qwen3-0.6B**, **CPU**, **6** function definitions, **11** prompts in `data/input/function_calling_tests.json`.

I compared three stages:

| Stage | What it is |
|-------|------------|
| **Initial** | Stub / no real decoding loop — nothing usable comes out |
| **Baseline** | Same prompts + greedy decoding, **no** logit masking (free-form JSON) |
| **Constrained** | Full system (this repo) |

### Accuracy — before → after

| Metric | Initial | Baseline (unconstrained) | Constrained |
|--------|---------|--------------------------|-------------|
| End-to-end correct | **0 / 11 (0%)** | **2 / 11 (18.2%)** | **11 / 11 (100%)** |
| Valid JSON schema | 0% | 2 / 11 (18.2%) | **11 / 11 (100%)** |
| Correct function name | 0% | 2 / 11 (18.2%) | **11 / 11 (100%)** |
| Correct parameters | 0% | 2 / 11 (18.2%) | **11 / 11 (100%)** |

Unconstrained failed hard: **9 / 11** outputs were not even parseable as a single `{"name", "parameters"}` object. Typical junk:

- `fn_multiply_numbers(3, 5)` instead of JSON
- wrong keys (`principal` / `rate` / `years` on an `fn_is_even` call)
- half-finished strings / truncated templates
- picking the wrong function for compound interest and stuffing the numbers into a fake SQL query

Constrained decoding fixed **all 11**. That's **0% → 18.2% → 100%** end-to-end.

### Speed — before → after

| Approach | Total (11 prompts) | Avg / prompt | Min | Max |
|----------|--------------------|--------------|-----|-----|
| Baseline (unconstrained) | **693.7 s** | **63.1 s** | 11.1 s | 131.5 s |
| Constrained | **173.8 s** | **15.8 s** | 5.4 s | 40.4 s |
| Official `make run` wall time | **173.4 s** | **~14.8 s** (tqdm) | — | — |

**~4.0× faster** after constraints (693.7 → 173.8). Same model, same CPU, fewer wasted tokens and early stops on typed values.

### Token budget — before → after

| Approach | Total generated tokens | Avg / prompt |
|----------|------------------------|--------------|
| Baseline | **419** | **38.1** |
| Constrained | **297** | **27.0** |

**29% fewer tokens** generated. Structural keys like `"name": "` and `", "parameters": {` are injected, not sampled.

### Per-prompt constrained timings (this run)

| Prompt (short) | Function | Seconds | Tokens |
|----------------|----------|---------|--------|
| product of 3 and 5 | `fn_multiply_numbers` | 9.2 | 22 |
| product of 12 and 4 | `fn_multiply_numbers` | 10.3 | 23 |
| is 4 even | `fn_is_even` | 5.6 | 16 |
| is 7 even | `fn_is_even` | 5.4 | 16 |
| compound interest 1234567.89… | `fn_calculate_compound_interest` | 40.4 | 48 |
| SQL SELECT users / production | `fn_execute_sql_query` | 12.2 | 29 |
| SQL INSERT logs / system | `fn_execute_sql_query` | 25.8 | 37 |
| read `/home/user/data.json` | `fn_read_file` | 16.4 | 29 |
| read `C:\Users\john\config.ini` | `fn_read_file` | 21.1 | 32 |
| template with `{user}'s` | `fn_format_template` | 12.4 | 22 |
| template with nested `"hello"` | `fn_format_template` | 15.1 | 23 |

Model load was **~3.9 s**. Hardest case was the long float (`1234567.89`) — still correct.

### What “100%” actually covers

- **numbers** (`3.0`, `12.0`, `0.0375`, `1234567.89`)
- **integers** (`4`, `7`, `23`)
- **booleans** (even/odd via function selection + typed args)
- **strings** with SQL, Unix paths, Windows backslashes, apostrophes, and embedded quotes

Reproduce the comparison:

```bash
uv run python bench_stats.py
# writes data/output/bench_stats.json + function_calling_results.json
```

---

## How it works (short version)

1. **Prefix injection** — force `"name": "` so the model starts inside the schema.
2. **Trie over function names** — only tokens that continue a real function name stay unmasked. When one candidate remains, emit its suffix deterministically.
3. **Inject** `", "parameters": {`.
4. **Per-parameter generation** with type masks:
   - string → no control chars; stop on unescaped `"`
   - number / integer → digits / signs / `.` / `e`; stop on `,` or `}`
   - boolean → compare logits for `true` vs `false`, emit the winner
5. **Pydantic** validates every result before it hits the output file.

Core idea: `apply_logit_mask()` sets disallowed vocab positions to `-inf` before argmax. Greedy decoding, no temperature, same input → same output.

---

## Run it

**Needs:** Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repository-url>
cd call-me-maybe
make install   # uv sync
make run       # uv run python -m src
```

Defaults:

| Path | Role |
|------|------|
| `data/input/functions_definition.json` | schemas (**6** functions) |
| `data/input/function_calling_tests.json` | prompts (**11** in the shipped set) |
| `data/output/function_calling_results.json` | results |

Custom paths:

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/results.json
```

Lint: `make lint` (`flake8` + `mypy`).

Example:

```json
{"prompt": "What is the product of 3 and 5?"}
```

→

```json
{
  "prompt": "What is the product of 3 and 5?",
  "name": "fn_multiply_numbers",
  "parameters": {"a": 3.0, "b": 5.0}
}
```

---

## Stuff that actually hurt (and what I did)

1. **Multi-token names** — `fn_calculate_compound_interest` is not one token. Single-step masking fails; the trie in `select_function` fixed it.
2. **String termination** — tokenizers merge `"` into weird pieces. I decode after each step and stop on the first *unescaped* quote (so `\"` inside templates doesn't end early).
3. **Numbers with weird spaces** — Qwen's `Ġ` / `\u0120` prefixes. Stripped when building digit sets; still allowed where they belong.
4. **Vocab file format** — `vocab.json` vs `tokenizer.json`. Used `get_path_to_vocab_file()` for the flat `{token: id}` map.
5. **Old GPU** — host GTX 1070 (sm_61) isn't supported by current PyTorch CUDA wheels, so runs here fall back to CPU. Numbers above are CPU measurements.

Layout: `src/decoding.py` (masks + generators), `src/utils.py`, `src/models.py`, `src/__main__.py`, plus `llm_sdk/` for the local model wrapper.

---

## Resources

- [Outlines — structured generation](https://dottxt-ai.github.io/outlines/latest/) — inspiration for logit masking
- [Transformers](https://huggingface.co/docs/transformers/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [Gorilla — function calling background](https://gorilla.cs.berkeley.edu/blogs/7_function_calling.html)

### AI usage

I used GitHub Copilot as a pair programmer for boilerplate, docstrings, and hunting tokenizer/vocab bugs. The decoding design (trie selection, per-type masks, greedy constrained loop) is mine — Copilot did not invent the algorithm.

---

**Bottom line:** on a 0.6B model that only got **2 / 11** unconstrained calls right, constrained decoding got **11 / 11**, ran **~4×** faster (**693.7 s → 173.8 s**), and used **~29%** fewer generated tokens (**419 → 297**). That's the whole pitch.
