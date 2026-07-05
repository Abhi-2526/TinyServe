"""Sampling strategies. All take logits (vocab,) and return an int token id."""

from __future__ import annotations

import torch


def greedy(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1))


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    if temperature <= 0:
        return greedy(logits)
    probs = torch.softmax(logits.to(torch.float32) / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest set of tokens whose cumulative prob >= top_p
        cutoff = cumsum - sorted_probs >= top_p
        sorted_probs[cutoff] = 0.0
        sorted_probs /= sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1, generator=generator)
        return int(sorted_idx[choice])
    return int(torch.multinomial(probs, 1, generator=generator))
