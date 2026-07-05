"""KV cache implementations.

Week 1: ContiguousKVCache — one pre-allocated slab per sequence.
Week 3 will add PagedKVCache (block tables + free-list allocator) behind
the same interface, so the model code never changes.
"""

from __future__ import annotations

import torch


class KVCache:
    """Interface: append(layer, k, v) -> (k_all, v_all); advance(n); property seq_len."""

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        raise NotImplementedError

    def advance(self, num_tokens: int) -> None:
        raise NotImplementedError


class ContiguousKVCache(KVCache):
    """Pre-allocated contiguous KV storage for a single sequence.

    Shapes: k/v are (num_layers, max_seq_len, num_kv_heads, head_dim).
    ``append`` writes this step's K/V at the current offset and returns a view
    of the full history for that layer. ``advance`` moves the offset once per
    engine step (after all layers have appended).
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (num_layers, max_seq_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: (T, KVH, D). Returns full-history views (seq_len+T, KVH, D)."""
        t = k.shape[0]
        start = self._seq_len
        end = start + t
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: {end} > max_seq_len={self.max_seq_len}"
            )
        self.k[layer_idx, start:end] = k
        self.v[layer_idx, start:end] = v
        return self.k[layer_idx, :end], self.v[layer_idx, :end]

    def advance(self, num_tokens: int) -> None:
        self._seq_len += num_tokens

    @classmethod
    def for_model(cls, cfg, max_seq_len: int, device, dtype) -> "ContiguousKVCache":
        return cls(
            num_layers=cfg.num_hidden_layers,
            max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=device,
            dtype=dtype,
        )
