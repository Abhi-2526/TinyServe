"""Compute serving metrics from a list of finished Requests.

Reported:
  throughput_tok_s — total generated tokens / wall-clock
  ttft p50/p99     — time to first token (includes queueing delay)
  itl  p50/p99     — inter-token latency, pooled across all requests
"""

from __future__ import annotations

from tinyserve.request import Request


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. p in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def summarize(requests: list[Request], elapsed_s: float) -> dict:
    total_tokens = sum(r.num_generated for r in requests)
    ttfts = [r.ttft() for r in requests if r.t_first_token > 0]
    itls = [g for r in requests for g in r.itls()]
    return {
        "num_requests": len(requests),
        "total_new_tokens": total_tokens,
        "elapsed_s": round(elapsed_s, 2),
        "throughput_tok_s": round(total_tokens / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        "ttft_p50_ms": round(percentile(ttfts, 50) * 1000, 1),
        "ttft_p99_ms": round(percentile(ttfts, 99) * 1000, 1),
        "itl_p50_ms": round(percentile(itls, 50) * 1000, 1),
        "itl_p99_ms": round(percentile(itls, 99) * 1000, 1),
    }
