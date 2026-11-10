"""KV-cached wrapper around :class:`llm_sdk.Small_LLM_Model`.

The stock SDK re-runs the *entire* sequence through the network for every
generated token, so a prompt of ``P`` tokens that produces ``G`` tokens costs
``G`` forward passes over ``~P`` tokens each.  Autoregressive transformers do
not need that: the keys/values of every already-seen token can be reused and
only the new token has to be processed.

:class:`CachedModel` adds that without touching the SDK or the decoding
code.  It exposes the same ``get_logits_from_input_ids`` method, but keeps a
``DynamicCache`` between calls and works out how much of the new request is
already cached:

* request extends the cached sequence  -> feed only the new tokens;
* request diverges part-way through    -> crop the cache, feed the rest;
* request is unrelated (new prompt)    -> reuse the shared prefix (the
  function-definition preamble is identical across prompts) or start over.

Greedy decoding therefore produces the same tokens as the uncached path up to
floating-point noise (verified in ``tests/test_kvcache.py``).
"""

from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import torch


class CachedModel:
    """Drop-in replacement for ``Small_LLM_Model`` with a KV cache.

    Any attribute not defined here (``encode``, ``decode``,
    ``get_path_to_vocab_file`` ...) is forwarded to the wrapped model.

    Attributes:
        forward_tokens: Total number of token positions pushed through the
            network so far (a cost proxy; handy for benchmarks).
        forward_calls: Number of forward passes executed.
    """

    def __init__(self, inner: Any, use_cache: bool = True) -> None:
        """Wrap *inner*, which must expose ``_model`` and ``_device``.

        With ``use_cache=False`` the wrapper only counts work and defers to
        the SDK's own (uncached) method; this is the pre-optimisation
        behaviour, kept so benchmarks can measure the speedup.
        """
        self._inner = inner
        self.use_cache = use_cache
        self._cache: Optional[Any] = None
        self._cached_ids: List[int] = []
        self.forward_tokens = 0
        self.forward_calls = 0

    def __getattr__(self, name: str) -> Any:
        """Forward everything else to the wrapped SDK model."""
        return getattr(self._inner, name)

    def reset(self) -> None:
        """Drop the cache (the next call is a full forward pass)."""
        self._cache = None
        self._cached_ids = []

    def _common_prefix(self, ids: List[int]) -> int:
        cached = self._cached_ids
        n = min(len(cached), len(ids))
        if ids[:n] == cached[:n]:
            return n
        i = 0
        while i < n and ids[i] == cached[i]:
            i += 1
        return i

    def get_logits_from_input_ids(
        self, input_ids: List[int]
    ) -> npt.NDArray[np.float32]:
        """Return next-token logits for *input_ids* (as a NumPy array).

        Unlike the SDK method this returns an ``ndarray`` rather than a
        Python list: converting 150k floats to a list every step costs more
        than the masked argmax that consumes them.
        """
        if not input_ids:
            raise ValueError("input_ids must not be empty")

        if not self.use_cache:
            self.forward_calls += 1
            self.forward_tokens += len(input_ids)
            return np.asarray(
                self._inner.get_logits_from_input_ids(input_ids),
                dtype=np.float32,
            )

        model = self._inner._model
        device = self._inner._device

        keep = self._common_prefix(input_ids)
        # We always need at least one fresh token to read logits from.
        keep = min(keep, len(input_ids) - 1)

        if self._cache is None or keep == 0:
            from transformers import DynamicCache
            self._cache = DynamicCache()
            keep = 0
        else:
            try:
                self._cache.crop(keep)
            except Exception:
                from transformers import DynamicCache
                self._cache = DynamicCache()
                keep = 0

        fresh = input_ids[keep:]
        tensor = torch.tensor([fresh], device=device, dtype=torch.long)
        with torch.no_grad():
            out = model(
                input_ids=tensor,
                past_key_values=self._cache,
                use_cache=True,
            )
        self._cache = out.past_key_values
        self._cached_ids = list(input_ids)
        self.forward_calls += 1
        self.forward_tokens += len(fresh)
        return out.logits[0, -1].float().cpu().numpy()
