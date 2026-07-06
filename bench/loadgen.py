"""Synthetic load generator: Poisson arrivals, lognormal lengths.

Real serving traffic is bursty — requests arrive independently at some average
rate, which is exactly a Poisson process (exponential gaps between arrivals).
Prompt/output lengths follow a lognormal (many short, a long tail), matching
what production traces (e.g. ShareGPT) look like.

Token ids are random ints in a safe range (avoids special tokens); benchmarks
run with ignore_eos=True so every request generates exactly its target length,
which keeps runs comparable across systems.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from tinyserve.request import Request


@dataclass
class Workload:
    requests: list[Request]  # sorted by arrival_time
    rate: float              # requests/second (0 = all arrive at t=0)


def _lognormal_int(rng: random.Random, mean: float, sigma: float, lo: int, hi: int) -> int:
    """Sample a lognormal with the given *linear-space* mean, clipped to [lo, hi]."""
    mu = math.log(mean) - 0.5 * sigma * sigma
    return max(lo, min(hi, round(rng.lognormvariate(mu, sigma))))


def make_workload(
    num_requests: int,
    rate: float = 2.0,
    prompt_len_mean: int = 128,
    output_len_mean: int = 128,
    sigma: float = 0.5,
    max_prompt_len: int = 1024,
    max_output_len: int = 512,
    vocab_size: int = 32000,
    seed: int = 0,
) -> Workload:
    rng = random.Random(seed)
    t = 0.0
    reqs = []
    for i in range(num_requests):
        if rate > 0:
            t += rng.expovariate(rate)  # Poisson process: exponential inter-arrival gaps
        n_prompt = _lognormal_int(rng, prompt_len_mean, sigma, 4, max_prompt_len)
        n_out = _lognormal_int(rng, output_len_mean, sigma, 4, max_output_len)
        # Random ids in [1000, vocab) — skips the low range where special tokens live.
        prompt_ids = [rng.randrange(1000, vocab_size) for _ in range(n_prompt)]
        reqs.append(
            Request(
                prompt_ids=prompt_ids,
                max_new_tokens=n_out,
                ignore_eos=True,
                arrival_time=t,
                request_id=i,
            )
        )
    return Workload(requests=reqs, rate=rate)
