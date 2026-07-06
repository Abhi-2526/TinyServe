"""KV cache implementations.

Week 2: SlotKVCache — a fixed number of pre-allocated contiguous "slots",
one per concurrent sequence. Enables batched decode across sequences with
different history lengths (the storage layer under continuous batching).

Week 3 will add PagedKVCache (block tables + free-list allocator) behind the
same interface, so the model code never changes.
"""

from __future__ import annotations

import torch


class SlotKVCache:
    """Pre-allocated KV storage for up to `num_slots` concurrent sequences.

    Layout: k/v are (num_layers, num_slots, max_seq_len, num_kv_heads, head_dim).
    Each running sequence owns one slot (a contiguous slab). `seq_lens[slot]`
    tracks how many tokens that slot currently holds.

    The model calls ``append(layer, k, v, slot_ids)`` once per layer per step:
    it writes this step's K/V at each sequence's current offset and returns the
    gathered full history for the batch, padded to the longest sequence in it
    (callers mask out the padding). ``advance`` moves the offsets once per
    engine step, after all layers have appended.
    """

    def __init__(
        self,
        num_layers: int,
        num_slots: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (num_layers, num_slots, max_seq_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.seq_lens = torch.zeros(num_slots, dtype=torch.long, device=device)
        self.max_seq_len = max_seq_len
        self.num_slots = num_slots
        self.device = torch.device(device)
        self._free = list(range(num_slots))

    # ---- slot management ----

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError("no free KV slots")
        return self._free.pop()

    def free(self, slot: int) -> None:
        self.seq_lens[slot] = 0
        self._free.append(slot)

    # ---- model-facing interface ----

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_ids: torch.Tensor):
        """k, v: (B, T, KVH, D); slot_ids: (B,) int64 on the cache's device.

        Two supported shapes:
          - prefill: B == 1, T == prompt chunk length
          - batched decode: T == 1, any B

        Returns (k_all, v_all, kv_lens):
          k_all/v_all: (B, S, KVH, D) where S = max(kv_lens) — padded gather
          kv_lens: (B,) valid history length per sequence (incl. this step)
        """
        B, T = k.shape[0], k.shape[1]
        lens = self.seq_lens[slot_ids]  # (B,)
        if int(lens.max()) + T > self.max_seq_len:
            raise RuntimeError(f"KV cache overflow: slot exceeds max_seq_len={self.max_seq_len}")

        if T == 1:
            # Scatter one token per sequence at its current offset.
            self.k[layer_idx, slot_ids, lens] = k[:, 0]
            self.v[layer_idx, slot_ids, lens] = v[:, 0]
        else:
            assert B == 1, "multi-token append only supported for a single sequence"
            s, start = int(slot_ids[0]), int(lens[0])
            self.k[layer_idx, s, start : start + T] = k[0]
            self.v[layer_idx, s, start : start + T] = v[0]

        kv_lens = lens + T
        S = int(kv_lens.max())
        k_all = self.k[layer_idx, slot_ids, :S]  # (B, S, KVH, D) — gather copy
        v_all = self.v[layer_idx, slot_ids, :S]
        return k_all, v_all, kv_lens

    def advance(self, slot_ids: torch.Tensor, num_tokens: int) -> None:
        self.seq_lens[slot_ids] += num_tokens

    @classmethod
    def for_model(cls, cfg, max_seq_len: int, device, dtype, num_slots: int = 1) -> "SlotKVCache":
        return cls(
            num_layers=cfg.num_hidden_layers,
            num_slots=num_slots,
            max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=device,
            dtype=dtype,
        )


# Week 1 name, kept as an alias: a "contiguous cache" is a 1-slot SlotKVCache.
ContiguousKVCache = SlotKVCache
