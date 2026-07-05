"""Week 1 engine: single-sequence prefill + decode loop.

Week 2 replaces this with a step()-based engine driven by a continuous-batching
scheduler; the prefill/decode split introduced here carries over unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .kv_cache import ContiguousKVCache
from .model import Transformer
from .sampler import greedy


@dataclass
class GenerationResult:
    prompt_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str = "length"


class Engine:
    def __init__(self, model: Transformer, max_seq_len: int = 4096) -> None:
        self.model = model
        self.cfg = model.cfg
        self.max_seq_len = max_seq_len
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 128,
        sampler: Callable[[torch.Tensor], int] = greedy,
        stop_token_ids: tuple[int, ...] = (),
    ) -> GenerationResult:
        assert len(prompt_ids) > 0, "empty prompt"
        assert len(prompt_ids) + max_new_tokens <= self.max_seq_len, "exceeds max_seq_len"

        cache = ContiguousKVCache.for_model(
            self.cfg, self.max_seq_len, self.device, self.dtype
        )
        result = GenerationResult(prompt_ids=list(prompt_ids))

        # ---- Prefill: process the whole prompt in one forward pass ----
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(len(prompt_ids), device=self.device)
        logits = self.model(ids, positions, cache)
        cache.advance(len(prompt_ids))
        next_id = sampler(logits[-1])

        # ---- Decode: one token per forward pass ----
        for _ in range(max_new_tokens):
            result.output_ids.append(next_id)
            if next_id in stop_token_ids:
                result.finish_reason = "stop"
                break
            if len(result.output_ids) == max_new_tokens:
                break
            pos = cache.seq_len
            ids = torch.tensor([next_id], dtype=torch.long, device=self.device)
            positions = torch.tensor([pos], device=self.device)
            logits = self.model(ids, positions, cache)
            cache.advance(1)
            next_id = sampler(logits[-1])

        return result
