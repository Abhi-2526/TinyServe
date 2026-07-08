"""Engines.

Engine       — Week 1: single-sequence prefill + decode loop (kept as the
               correctness reference; test_batching compares against it).
BatchEngine  — Week 2: step()-based engine with iteration-level (continuous)
               batching. Each step() admits waiting requests into free KV
               slots, prefills them, then runs ONE batched decode forward for
               every running sequence. mode="static" disables mid-flight
               admission so you can measure what continuous batching buys.
               Week 3: kv="paged" swaps in the PagedKVCache and adds
               preemption — when physical blocks run out mid-decode, the
               latest-admitted request is evicted (blocks freed, requeued)
               and later resumed by re-prefilling prompt + generated-so-far
               (vLLM's "recompute" preemption).
"""

from __future__ import annotations

import time
from collections import deque

import torch

from .kv_cache import PagedKVCache, SlotKVCache
from .model import Transformer
from .request import Request
from .sampler import greedy, sample


class Engine:
    """Single-sequence engine on the batched (B=1) model API."""

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
        stop_token_ids: tuple[int, ...] = (),
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Request:
        assert len(prompt_ids) > 0, "empty prompt"
        assert len(prompt_ids) + max_new_tokens <= self.max_seq_len, "exceeds max_seq_len"

        cache = SlotKVCache.for_model(
            self.cfg, self.max_seq_len, self.device, self.dtype, num_slots=1
        )
        req = Request(
            prompt_ids=list(prompt_ids),
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            top_p=top_p,
        )
        slot = torch.tensor([0], dtype=torch.long, device=self.device)

        # ---- Prefill ----
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(prompt_ids), device=self.device).unsqueeze(0)
        logits = self.model(ids, positions, cache, slot)
        cache.advance(slot, len(prompt_ids))
        next_id = self._sample(req, logits[0, -1])

        # ---- Decode ----
        while True:
            req.output_ids.append(next_id)
            if next_id in stop_token_ids:
                req.finish_reason = "stop"
                break
            if len(req.output_ids) >= max_new_tokens:
                req.finish_reason = "length"
                break
            pos = int(cache.seq_lens[0])
            ids = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            positions = torch.tensor([[pos]], device=self.device)
            logits = self.model(ids, positions, cache, slot)
            cache.advance(slot, 1)
            next_id = self._sample(req, logits[0, -1])

        req.state = "finished"
        return req

    @staticmethod
    def _sample(req: Request, logits: torch.Tensor) -> int:
        if req.temperature == 0.0:
            return greedy(logits)
        return sample(logits, req.temperature, req.top_p)


class BatchEngine:
    """Continuous-batching engine.

    Usage:
        eng = BatchEngine(model, max_batch_size=8, max_seq_len=2048)
        eng.submit(req); ...
        while eng.has_work():
            eng.step()
        # eng.finished holds completed Requests with timing populated.

    step() = admit new requests into free slots (prefill, one at a time)
             + one batched decode forward for all running sequences.
    """

    def __init__(
        self,
        model: Transformer,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        mode: str = "continuous",
        kv: str = "slot",
        block_size: int = 16,
        num_blocks: int | None = None,
        kv_memory_gb: float | None = None,
    ) -> None:
        assert mode in ("continuous", "static")
        assert kv in ("slot", "paged")
        self.model = model
        self.cfg = model.cfg
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.mode = mode
        self.kv = kv
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        if kv == "paged":
            self.cache = PagedKVCache.for_model(
                self.cfg, max_seq_len, self.device, self.dtype,
                num_slots=max_batch_size, block_size=block_size,
                num_blocks=num_blocks, memory_gb=kv_memory_gb,
            )
        else:
            self.cache = SlotKVCache.for_model(
                self.cfg, max_seq_len, self.device, self.dtype, num_slots=max_batch_size
            )
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.num_preemptions = 0

    # ---- public API ----

    def submit(self, req: Request) -> None:
        assert len(req.prompt_ids) > 0, "empty prompt"
        total = len(req.prompt_ids) + req.max_new_tokens
        # Guarantees a lone request always fits — preemption can then never livelock.
        assert total <= self.cache.max_tokens_single_seq, "request exceeds KV capacity"
        req.t_submitted = time.perf_counter()
        req.state = "waiting"
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.inference_mode()
    def step(self) -> None:
        """One scheduler iteration: admit, then decode one token for everyone."""
        # -- admission (FCFS: stop at the first request that doesn't fit) --
        if self.mode == "continuous" or not self.running:
            while self.waiting and self.cache.can_admit(self._num_tokens(self.waiting[0])):
                self._prefill(self.waiting.popleft())

        # -- batched decode --
        if self.running:
            self._decode_step()

    @staticmethod
    def _num_tokens(req: Request) -> int:
        # A resumed (preempted) request re-prefills prompt + everything generated.
        return len(req.prompt_ids) + len(req.output_ids)

    def run_to_completion(self) -> list[Request]:
        while self.has_work():
            self.step()
        return self.finished

    # ---- internals ----

    def _prefill(self, req: Request) -> None:
        req.slot = self.cache.alloc()
        req.state = "running"
        slot = torch.tensor([req.slot], dtype=torch.long, device=self.device)
        # Resumed request: recompute KV for prompt + tokens generated before
        # preemption (its KV was discarded when it was evicted).
        tokens = req.prompt_ids + req.output_ids
        n = len(tokens)
        if not self.cache.reserve(slot, n):
            raise RuntimeError("prefill reserve failed after can_admit — scheduler bug")
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(n, device=self.device).unsqueeze(0)
        logits = self.model(ids, positions, self.cache, slot)
        self.cache.advance(slot, n)
        token = Engine._sample(req, logits[0, -1])
        self._commit(req, token, time.perf_counter())
        if req.state == "running":
            self.running.append(req)

    def _decode_step(self) -> None:
        # Claim one more token of KV memory for every running sequence; if the
        # paged cache is out of blocks, evict the latest-admitted request and
        # retry (it re-enters at the FRONT of the waiting queue).
        while True:
            slot_ids = torch.tensor(
                [r.slot for r in self.running], dtype=torch.long, device=self.device
            )
            if self.cache.reserve(slot_ids, 1):
                break
            if len(self.running) == 1:
                raise RuntimeError("single request cannot fit — submit() guard violated")
            self._preempt(self.running[-1])

        batch = self.running
        ids = torch.tensor([[r.next_token] for r in batch], dtype=torch.long, device=self.device)
        positions = self.cache.seq_lens[slot_ids].unsqueeze(1)  # (B, 1)
        logits = self.model(ids, positions, self.cache, slot_ids)  # (B, 1, V)
        self.cache.advance(slot_ids, 1)
        now = time.perf_counter()
        still_running: list[Request] = []
        for i, req in enumerate(batch):
            token = Engine._sample(req, logits[i, -1])
            self._commit(req, token, now)
            if req.state == "running":
                still_running.append(req)
        self.running = still_running

    def _commit(self, req: Request, token: int, now: float) -> None:
        """Record a sampled token; finish or schedule it for the next forward."""
        req.output_ids.append(token)
        req.token_times.append(now)
        if len(req.output_ids) == 1:
            req.t_first_token = now
        if not req.ignore_eos and token in req.stop_token_ids:
            self._finish(req, "stop")
        elif len(req.output_ids) >= req.max_new_tokens:
            self._finish(req, "length")
        else:
            req.next_token = token

    def _preempt(self, req: Request) -> None:
        """Evict a running request: discard its KV, requeue it at the front.

        Its generated tokens are kept on the Request; on re-admission,
        _prefill recomputes the KV from prompt + output (recompute-style
        preemption — trades extra compute for freed memory)."""
        self.running.remove(req)
        self.cache.free(req.slot)
        req.slot = -1
        req.state = "waiting"
        req.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(req)

    def _finish(self, req: Request, reason: str) -> None:
        req.state = "finished"
        req.finish_reason = reason
        self.cache.free(req.slot)
        req.slot = -1
        self.finished.append(req)
