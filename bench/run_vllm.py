"""Benchmark vLLM on the IDENTICAL Poisson workload as run_tinyserve.py.

    pip install vllm
    python bench/run_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --num-requests 50 --rate 2.0 --seed 0

Same seed => byte-identical prompts, lengths, and arrival times as the
TinyServe run, so the two JSONs are directly comparable.

Uses vLLM's AsyncLLMEngine so requests are truly submitted at their arrival
times (the offline LLM.generate() API would hand vLLM the whole batch up
front — an unfair advantage). NOTE: vLLM's Python API moves fast; if imports
fail, check your version's docs for AsyncEngineArgs/AsyncLLMEngine/TokensPrompt.
"""

import argparse
import asyncio
import json
import time

from loadgen import make_workload
from metrics import percentile


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--num-requests", type=int, default=50)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--prompt-len-mean", type=int, default=128)
    ap.add_argument("--output-len-mean", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", default="bench/results_vllm.json")
    return ap.parse_args()


async def run_one(engine, sampling_cls, tokens_prompt_cls, req, start):
    """Sleep until this request's arrival time, then stream it and timestamp
    every token — mirrors BatchEngine's _commit bookkeeping."""
    await asyncio.sleep(max(0.0, start + req.arrival_time - time.perf_counter()))
    t_submitted = time.perf_counter()
    params = sampling_cls(
        max_tokens=req.max_new_tokens, ignore_eos=True, temperature=0.0
    )
    token_times = []
    n_seen = 0
    async for out in engine.generate(
        tokens_prompt_cls(prompt_token_ids=req.prompt_ids),
        params,
        request_id=str(req.request_id),
    ):
        n = len(out.outputs[0].token_ids)
        now = time.perf_counter()
        token_times.extend([now] * (n - n_seen))
        n_seen = n
    return t_submitted, token_times, n_seen


async def main():
    args = parse_args()

    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            max_model_len=args.max_seq_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    )

    # vocab_size must match run_tinyserve (it uses the loaded model's config);
    # Qwen2.5 vocab is 151936 but loadgen only needs ids < vocab, and BOTH
    # benches must draw identical ids — so read it from the HF config.
    from transformers import AutoConfig
    vocab_size = AutoConfig.from_pretrained(args.model).vocab_size

    wl = make_workload(
        args.num_requests,
        rate=args.rate,
        prompt_len_mean=args.prompt_len_mean,
        output_len_mean=args.output_len_mean,
        max_prompt_len=args.max_seq_len // 2,
        max_output_len=args.max_seq_len // 4,
        vocab_size=vocab_size,
        seed=args.seed,
    )

    start = time.perf_counter()
    results = await asyncio.gather(
        *[run_one(engine, SamplingParams, TokensPrompt, r, start) for r in wl.requests]
    )
    elapsed = time.perf_counter() - start

    total_tokens = sum(n for _, _, n in results)
    ttfts = [tt[0] - t_sub for t_sub, tt, _ in results if tt]
    itls = [b - a for _, tt, _ in results for a, b in zip(tt, tt[1:])]
    report = {
        "system": "vllm",
        "model": args.model,
        "rate_req_s": args.rate,
        "num_requests": args.num_requests,
        "total_new_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(total_tokens / elapsed, 2),
        "ttft_p50_ms": round(percentile(ttfts, 50) * 1000, 1),
        "ttft_p99_ms": round(percentile(ttfts, 99) * 1000, 1),
        "itl_p50_ms": round(percentile(itls, 50) * 1000, 1),
        "itl_p99_ms": round(percentile(itls, 99) * 1000, 1),
    }
    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
