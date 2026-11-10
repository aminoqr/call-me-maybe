"""Shared fixtures: a tiny deterministic fake LLM and the real schemas.

The fake model is a character-level "oracle": it has a target completion
string and, at every step, gives the next target character the highest
logit.  That lets tests exercise the real masking/pipeline code in
milliseconds with no weights, no torch and no network.
"""

import json
import string
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from src.models import FunctionDefinition
from src.pipeline import FunctionCaller, TokenTables, build_token_tables

ROOT = Path(__file__).resolve().parent.parent
MARKER = "Function Call:"


class FakeModel:
    """Character-level stand-in for ``Small_LLM_Model``."""

    def __init__(
        self,
        target: str = "",
        distractor: Optional[str] = None,
        target_logit: float = 5.0,
        distractor_logit: float = 20.0,
    ) -> None:
        chars = [c for c in string.printable if c not in "\t\r\x0b\x0c"]
        tokens = ["\u0120" if c == " " else c for c in chars]
        tokens += ["12", "\u012012", ".5"]  # a few multi-char tokens
        self.id_to_token: Dict[int, str] = dict(enumerate(tokens))
        self.char_to_id: Dict[str, int] = {
            c: i for i, c in enumerate(chars)
        }
        self.target = target
        self.distractor = distractor
        self.target_logit = target_logit
        self.distractor_logit = distractor_logit
        self.calls = 0

    # -- tokenizer-ish API ------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        return np.array([[self.char_to_id[c] for c in text]])

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids).replace(
            "\u0120", " "
        )

    # -- model API --------------------------------------------------------
    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        self.calls += 1
        logits = [0.0] * len(self.id_to_token)
        text = self.decode(input_ids)
        idx = text.rfind(MARKER)
        produced = text[idx + len(MARKER):] if idx >= 0 else text
        if self.distractor is not None:
            logits[self.char_to_id[self.distractor]] = self.distractor_logit
        if self.target.startswith(produced) and len(produced) < len(
            self.target
        ):
            nxt = self.target[len(produced)]
            logits[self.char_to_id[nxt]] = self.target_logit
        return logits


@pytest.fixture(scope="session")
def functions() -> List[FunctionDefinition]:
    raw = json.loads(
        (ROOT / "data/input/functions_definition.json").read_text()
    )
    return [FunctionDefinition(**f) for f in raw]


def make_caller(
    model: FakeModel, functions: List[FunctionDefinition]
) -> FunctionCaller:
    def encode(text: str) -> List[int]:
        return model.encode(text)[0].tolist()

    tables: TokenTables = build_token_tables(
        functions, model.id_to_token, encode
    )
    return FunctionCaller(model, functions, tables)


def json_completion(name: str, params_json: str) -> str:
    """The exact text the pipeline should reproduce after the prompt."""
    return (
        f' {{"name": "{name}", "parameters": {{{params_json}}}}}'
    )
