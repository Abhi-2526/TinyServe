"""Request: one generation job moving through the serving system.

Lifecycle: waiting -> running -> finished. Timing fields are recorded by the
engine so the benchmark can compute TTFT (time to first token) and ITL
(inter-token latency) per request.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    prompt_ids: list[int]
    max_new_tokens: int = 128
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False           # benchmarks fix output length for fairness
    temperature: float = 0.0           # 0 = greedy
    top_p: float = 1.0
    arrival_time: float = 0.0          # seconds relative to benchmark start
    request_id: int = 0

    # --- runtime state (owned by the engine) ---
    output_ids: list[int] = field(default_factory=list)
    state: str = "waiting"             # waiting | running | finished
    slot: int = -1
    next_token: int = -1               # sampled but not yet fed back through the model
    finish_reason: str = ""

    # --- timing (absolute perf_counter timestamps) ---
    t_submitted: float = 0.0
    t_first_token: float = 0.0
    token_times: list[float] = field(default_factory=list)

    @property
    def num_generated(self) -> int:
        return len(self.output_ids)

    def ttft(self) -> float:
        """Time to first token, measured from submission (includes queueing)."""
        return self.t_first_token - self.t_submitted

    def itls(self) -> list[float]:
        """Inter-token latencies (gaps between consecutive generated tokens)."""
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]
