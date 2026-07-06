"""Engines.

Engine       — Week 1: single-sequence prefill + decode loop (kept as the
               correctness reference; test_batching compares against it).
BatchEngine  — Week 2: step()-based engine with iteration-level (continuous)
               batching. Each step() admits waiting requests into free KV
               slots, prefills them, then runs ONE batched decode forward for
               every running sequence. mode="static" disables mid-flight
               admission so you can measure what continuous batching buys.
"""

from __future__ import annotations

import time
from collections import deque

import torch

from .kv_cache import SlotKVCache
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
    ) -> None:
        assert mode in ("continuous", "static")
        self.model = model
        self.cfg = model.cfg
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.mode = mode
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.cache = SlotKVCache.for_model(
            self.cfg, max_seq_len, self.device, self.dtype, num_slots=max_batch_size
        )
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []

    # ---- public API ----

    def submit(self, req: Request) -> None:
        assert len(req.prompt_ids) > 0, "empty prompt"
        assert len(req.prompt_ids) + req.max_new_tokens <= self.max_seq_len, "exceeds max_seq_len"
        req.t_submitted = time.perf_counter()
        req.state = "waiting"
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.inference_mode()
    def step(self) -> None:
        """One scheduler iteration: admit, then decode one token for everyone."""
        # -- admission --
        if self.mode == "continuous":
            while self.waiting and self.cache.num_free > 0:
                self._prefill(self.waiting.popleft())
        else:  # static: only refill when the whole batch has drained
            if not self.running:
                while self.waiting and self.cache.num_free > 0:
                    self._prefill(self.waiting.popleft())

        # -- batched decode --
        if self.running:
            self._decode_step()

    def run_to_completion(self) -> list[Request]:
        while self.has_work():
            self.step()
        return self.finished

    # ---- internals ----

    def _prefill(self, req: Request) -> None:
        req.slot = self.cache.alloc()
        req.state = "running"
        slot = torch.tensor([req.slot], dtype=torch.long, device=self.device)
        n = len(req.prompt_ids)
        ids = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(n, device=self.device).unsqueeze(0)
        logits = self.model(ids, positions, self.cache, slot)
        self.cache.advance(slot, n)
        token = Engine._sample(req, logits[0, -1])
        self._commit(req, token, time.perf_counter())
        if req.state == "running":
            self.running.append(req)

    def _decode_step(self) -> None:
        batch = self.running
        slot_ids = torch.tensor([r.slot for r in batch], dtype=torch.long, device=self.device)
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

    def _finish(self, req: Request, reason: str) -> None:
        req.state = "finished"
        req.finish_reason = reason
        self.cache.free(req.slot)
        req.slot = -1
        self.finished.append(req)
