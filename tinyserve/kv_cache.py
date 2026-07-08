"""KV cache implementations.

SlotKVCache  — Week 2: one pre-allocated contiguous slab per sequence.
               Simple, but reserves max_seq_len tokens of memory per slot even
               for short sequences (internal fragmentation).
PagedKVCache — Week 3: memory granted in fixed-size blocks (like OS virtual
               memory pages). A sequence holds a *block table* mapping logical
               positions to physical blocks, so memory is claimed as tokens
               are actually generated. Same interface, so model.py is unchanged.

Common interface used by the model:
    append(layer_idx, k, v, slot_ids) -> (k_all, v_all, kv_lens)
    advance(slot_ids, num_tokens)
Used by the engine:
    alloc() / free(slot) / can_admit(n) / reserve(slot_ids, n)
    max_tokens_single_seq
"""

from __future__ import annotations

import math

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

    # ---- slot management (engine-facing) ----

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def max_tokens_single_seq(self) -> int:
        return self.max_seq_len

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError("no free KV slots")
        return self._free.pop()

    def free(self, slot: int) -> None:
        self.seq_lens[slot] = 0
        self._free.append(slot)

    def can_admit(self, num_tokens: int) -> bool:
        return bool(self._free)

    def reserve(self, slot_ids: torch.Tensor, num_tokens: int) -> bool:
        return True  # slabs are pre-reserved at alloc time; nothing to claim

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


class PagedKVCache:
    """Paged KV storage: fixed-size blocks + per-sequence block tables.

    Physical layout: k/v are (num_layers, num_blocks, block_size, KVH, D).
    A sequence's token at logical position p lives in physical block
    ``block_tables[slot, p // block_size]`` at offset ``p % block_size``.

    Memory is claimed block-by-block via ``reserve`` as sequences grow, so a
    50-token conversation holds ceil(50/bs) blocks instead of a full
    max_seq_len slab. When blocks run out, the engine preempts a sequence
    (frees its blocks) rather than crashing — total *virtual* capacity can
    exceed physical capacity, exactly like OS paging.

    ``append`` keeps the SlotKVCache contract: same arguments, same padded
    (B, S, KVH, D) gather return, so the model code is identical for both.
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
        block_size: int = 16,
        num_blocks: int | None = None,
    ) -> None:
        self.block_size = block_size
        self.max_blocks_per_seq = math.ceil(max_seq_len / block_size)
        if num_blocks is None:
            # Same physical capacity as the slot cache (no preemption unless capped).
            num_blocks = num_slots * self.max_blocks_per_seq
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.seq_lens = torch.zeros(num_slots, dtype=torch.long, device=device)
        # Padding entries stay 0 — a valid physical block, so padded gathers
        # read garbage-but-finite values that the attention mask ignores.
        self.block_tables = torch.zeros(
            (num_slots, self.max_blocks_per_seq), dtype=torch.long, device=device
        )
        self.max_seq_len = max_seq_len
        self.num_slots = num_slots
        self.num_blocks = num_blocks
        self.device = torch.device(device)
        self._free_slots = list(range(num_slots))
        self._free_blocks = list(range(num_blocks))
        self._blocks_held = [0] * num_slots

    # ---- slot & block management (engine-facing) ----

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def max_tokens_single_seq(self) -> int:
        return min(self.max_seq_len, self.num_blocks * self.block_size)

    def blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def alloc(self) -> int:
        if not self._free_slots:
            raise RuntimeError("no free KV slots")
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        held = self._blocks_held[slot]
        self._free_blocks.extend(self.block_tables[slot, :held].tolist())
        self._blocks_held[slot] = 0
        self.seq_lens[slot] = 0
        self._free_slots.append(slot)

    def can_admit(self, num_tokens: int) -> bool:
        """Room for a new sequence's prefill? Requires +1 block of headroom so
        a fresh admission can't instantly trigger a preemption storm. Capped at
        num_blocks: a max-size request must still be admittable when the cache
        is completely empty, or the engine would spin forever."""
        need = min(self.blocks_needed(num_tokens) + 1, self.num_blocks)
        return bool(self._free_slots) and len(self._free_blocks) >= need

    def reserve(self, slot_ids: torch.Tensor, num_tokens: int) -> bool:
        """Ensure each sequence has blocks for `num_tokens` more tokens.

        All-or-nothing: returns False (allocating nothing) if the free list
        can't cover the whole batch — the engine then preempts and retries.
        """
        slots = slot_ids.tolist()
        needs = []
        for s in slots:
            target = self.blocks_needed(int(self.seq_lens[s]) + num_tokens)
            needs.append(max(0, target - self._blocks_held[s]))
        if sum(needs) > len(self._free_blocks):
            return False
        for s, need in zip(slots, needs):
            for _ in range(need):
                b = self._free_blocks.pop()
                self.block_tables[s, self._blocks_held[s]] = b
                self._blocks_held[s] += 1
        return True

    # ---- model-facing interface (same contract as SlotKVCache) ----

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_ids: torch.Tensor):
        B, T = k.shape[0], k.shape[1]
        bs = self.block_size
        lens = self.seq_lens[slot_ids]  # (B,)
        kv_lens = lens + T
        if int(kv_lens.max()) > self.max_seq_len:
            raise RuntimeError(f"KV cache overflow: exceeds max_seq_len={self.max_seq_len}")

        # Flat views: physical token index = block_id * block_size + offset.
        kf = self.k[layer_idx].reshape(-1, *self.k.shape[-2:])  # (NB*bs, KVH, D)
        vf = self.v[layer_idx].reshape(-1, *self.v.shape[-2:])

        if T == 1:
            flat = self.block_tables[slot_ids, lens // bs] * bs + lens % bs  # (B,)
            kf[flat] = k[:, 0]
            vf[flat] = v[:, 0]
        else:
            assert B == 1, "multi-token append only supported for a single sequence"
            s, start = int(slot_ids[0]), int(lens[0])
            pos = torch.arange(start, start + T, device=self.device)
            flat = self.block_tables[s, pos // bs] * bs + pos % bs  # (T,)
            kf[flat] = k[0]
            vf[flat] = v[0]

        # Padded gather of full history via the block tables.
        S = int(kv_lens.max())
        pos = torch.arange(S, device=self.device)  # (S,)
        blocks = self.block_tables[slot_ids][:, pos // bs]  # (B, S)
        flat = blocks * bs + pos % bs  # (B, S)
        return kf[flat], vf[flat], kv_lens

    def advance(self, slot_ids: torch.Tensor, num_tokens: int) -> None:
        self.seq_lens[slot_ids] += num_tokens

    @classmethod
    def for_model(
        cls,
        cfg,
        max_seq_len: int,
        device,
        dtype,
        num_slots: int = 1,
        block_size: int = 16,
        num_blocks: int | None = None,
        memory_gb: float | None = None,
    ) -> "PagedKVCache":
        if memory_gb is not None:
            bytes_per_block = (
                2  # K and V
                * cfg.num_hidden_layers
                * block_size
                * cfg.num_key_value_heads
                * cfg.head_dim
                * torch.empty(0, dtype=dtype).element_size()
            )
            num_blocks = max(1, int(memory_gb * 2**30) // bytes_per_block)
        return cls(
            num_layers=cfg.num_hidden_layers,
            num_slots=num_slots,
            max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=device,
            dtype=dtype,
            block_size=block_size,
            num_blocks=num_blocks,
        )


# Week 1 name, kept as an alias: a "contiguous cache" is a 1-slot SlotKVCache.
ContiguousKVCache = SlotKVCache
