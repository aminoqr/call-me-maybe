"""The constrained function-calling pipeline, independent of any CLI.

``__main__`` and the benchmarks both go through :class:`FunctionCaller`, so
the code that is measured is the code that ships.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError

from src.decoding import (
    gen_bool_value,
    gen_number_value,
    gen_string_value,
    select_function,
)
from src.models import FunctionCallResult, FunctionDefinition
from src.utils import build_func_prompt

# GPT-2 style byte-level BPE marks a leading space with this character.
SPACE_MARK = "Ġ"
NUMBER_CHARS = "0123456789.+-eE"

PROMPT_TEMPLATE = (
    "Available functions:\n"
    "{functions}\n\n"
    "User request: {request}\n"
    "Select the single most appropriate function "
    "and copy ALL parameter values verbatim from "
    "the request.\n"
    "Function Call:"
)
# Appended to the prompt as *forced* tokens (never sampled).
FORCED_PREFIX = ' {"name": "'
PARAMS_SEPARATOR = '", "parameters": {'


def build_prompt(
    functions_desc: str,
    request: str,
    template: str = PROMPT_TEMPLATE,
) -> str:
    """Render the instruction prompt for one user request."""
    return template.format(functions=functions_desc, request=request)


@dataclass(frozen=True)
class TokenTables:
    """Pre-computed token-id sets that drive the logit masks."""

    name_key: List[int]
    params_pfx: List[int]
    separator: List[int]
    func_seqs: Dict[str, List[int]]
    true_toks: List[int]
    false_toks: List[int]
    str_tids: npt.NDArray[np.int64]
    num_first: npt.NDArray[np.int64]
    num_cont: npt.NDArray[np.int64]
    stop_tids: Set[int]


def build_token_tables(
    functions: Sequence[FunctionDefinition],
    id_to_token: Dict[int, str],
    encode: Callable[[str], List[int]],
    forced_prefix: str = FORCED_PREFIX,
) -> TokenTables:
    """Derive every mask the decoder needs from the tokenizer vocabulary.

    Args:
        functions: Available function definitions.
        id_to_token: Mapping from token id to its vocabulary string.
        encode: Text -> token ids, without special tokens.
        forced_prefix: Text injected before the function name.
    """
    str_tids = [
        tid for tid, tok in id_to_token.items()
        if not any(
            ord(c) < 0x20 for c in tok.replace(SPACE_MARK, " ")
        )
    ]

    num_first: List[int] = []
    num_cont: List[int] = []
    for tid, tok in id_to_token.items():
        cleaned = tok.replace(SPACE_MARK, "")
        if cleaned and all(c in NUMBER_CHARS for c in cleaned):
            num_first.append(tid)
            if not tok.startswith(SPACE_MARK):
                num_cont.append(tid)

    stop_tids = {
        tid for tid, tok in id_to_token.items()
        if tok.replace(SPACE_MARK, " ").strip() in (",", "}")
    }

    def arr(ids: List[int]) -> npt.NDArray[np.int64]:
        return np.asarray(sorted(ids), dtype=np.int64)

    return TokenTables(
        name_key=encode(forced_prefix),
        params_pfx=encode(PARAMS_SEPARATOR),
        separator=encode(", "),
        func_seqs={fn.name: encode(fn.name) for fn in functions},
        true_toks=encode("true"),
        false_toks=encode("false"),
        str_tids=arr(str_tids),
        num_first=arr(num_first),
        num_cont=arr(num_cont),
        stop_tids=stop_tids,
    )


@dataclass
class CallOutcome:
    """Result of running the pipeline on a single prompt."""

    result: Optional[Dict[str, Any]]
    tokens: List[int] = field(default_factory=list)
    error: Optional[str] = None


class FunctionCaller:
    """Turn natural-language requests into schema-valid function calls."""

    def __init__(
        self,
        model: Any,
        functions: Sequence[FunctionDefinition],
        tables: TokenTables,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        """Bind a model, its function set and the derived mask tables."""
        self.prompt_template = prompt_template
        self.model = model
        self.functions = list(functions)
        self.tables = tables
        self.func_desc = build_func_prompt(self.functions)

    @classmethod
    def from_model(
        cls,
        model: Any,
        functions: Sequence[FunctionDefinition],
        *,
        prompt_template: str = PROMPT_TEMPLATE,
        forced_prefix: str = FORCED_PREFIX,
    ) -> "FunctionCaller":
        """Build a caller by reading *model*'s vocabulary from disk."""
        with open(model.get_path_to_vocab_file(), encoding="utf-8") as fh:
            vocab = json.load(fh)
        id_to_token = {int(v): k for k, v in vocab.items()}

        def encode(text: str) -> List[int]:
            return list(model.encode(text)[0].tolist())

        return cls(
            model, functions,
            build_token_tables(
                functions, id_to_token, encode, forced_prefix
            ),
            prompt_template,
        )

    def _encode(self, text: str) -> List[int]:
        return list(self.model.encode(text)[0].tolist())

    def call(self, request: str) -> CallOutcome:
        """Generate one function call for *request*.

        Never raises for model-side problems; failures are reported in
        :attr:`CallOutcome.error` so a batch run can continue.
        """
        t = self.tables
        inp = self._encode(
            build_prompt(self.func_desc, request, self.prompt_template)
        )

        # Phase 1 - force ` {"name": "`
        gen: List[int] = list(t.name_key)

        # Phase 2 - trie-constrained function name
        fname, fn_toks = select_function(
            self.model, inp, gen, t.func_seqs
        )
        gen.extend(fn_toks)

        # Phase 3 - force `", "parameters": {`
        gen.extend(t.params_pfx)

        fn = next((f for f in self.functions if f.name == fname), None)
        if fn is None:
            return CallOutcome(None, gen, f"unknown function {fname!r}")

        # Phase 4 - typed, constrained parameter values
        values: Dict[str, Any] = {}
        for i, (pname, pdef) in enumerate(fn.parameters.items()):
            if i > 0:
                gen.extend(t.separator)
            if pdef.type == "string":
                gen.extend(self._encode(f'"{pname}": "'))
                sval, vt = gen_string_value(
                    self.model, inp, gen, t.str_tids
                )
                values[pname] = sval
            elif pdef.type == "boolean":
                gen.extend(self._encode(f'"{pname}": '))
                bval, vt = gen_bool_value(
                    self.model, inp, gen, t.true_toks, t.false_toks
                )
                values[pname] = bval
            else:  # "number" and "integer"
                gen.extend(self._encode(f'"{pname}": '))
                nval, vt = gen_number_value(
                    self.model, inp, gen,
                    t.num_first, t.num_cont, t.stop_tids,
                )
                values[pname] = int(nval) if pdef.type == "integer" else nval
            gen.extend(vt)

        result = {"prompt": request, "name": fname, "parameters": values}
        try:
            FunctionCallResult(**result)
        except ValidationError as exc:
            return CallOutcome(None, gen, f"validation error: {exc}")
        return CallOutcome(result, gen)
