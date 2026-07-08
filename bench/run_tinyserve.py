"""Benchmark TinyServe's BatchEngine under Poisson load.

Requests are released in real time according to their arrival timestamps; the
engine steps whenever it has work. Compare against bench/baseline_hf.py and
against --mode static to see what continuous batching buys.

Usage (on your GPU machine):
    python bench/run_tinyserve.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --num-requests 50 --rate 2.0 --max-batch-size 8

    # offline / max-throughput mode (everything arrives at t=0):
    python bench/run_tinyserve.py --rate 0 --num-requests 50
"""

import argparse
import json
import time

import torch

from tinyserve import BatchEngine, Transformer

from loadgen import make_workload
from metrics import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--num-requests", type=int, default=50)
    ap.add_argument("--rate", type=float, default=2.0, help="req/s; 0 = all at t=0")
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--mode", choices=["continuous", "static"], default="continuous")
    ap.add_argument("--kv", choices=["paged", "slot"], default="paged")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-memory-gb", type=float, default=None,
                    help="cap physical KV memory; forces preemption under pressure")
    ap.add_argument("--prompt-len-mean", type=int, default=128)
    ap.add_argument("--output-len-mean", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench/results_tinyserve.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Transformer.from_pretrained(args.model, device=device, dtype=dtype)
    engine = BatchEngine(
        model,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        mode=args.mode,
        kv=args.kv,
        block_size=args.block_size,
        kv_memory_gb=args.kv_memory_gb,
    )

    wl = make_workload(
        args.num_requests,
        rate=args.rate,
        prompt_len_mean=args.prompt_len_mean,
        output_len_mean=args.output_len_mean,
        max_prompt_len=args.max_seq_len // 2,
        max_output_len=args.max_seq_len // 4,
        vocab_size=model.cfg.vocab_size,
        seed=args.seed,
    )
    pending = list(wl.requests)  # sorted by arrival_time

    # Warmup (excluded from timing)
    from tinyserve import Engine
    Engine(model, max_seq_len=args.max_seq_len).generate([1000, 1001, 1002], max_new_tokens=4)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    while pending or engine.has_work():
        now = time.perf_counter() - start
        while pending and pending[0].arrival_time <= now:
            engine.submit(pending.pop(0))
        if engine.has_work():
            engine.step()
        elif pending:
            time.sleep(min(0.001, pending[0].arrival_time - now))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    report = {
        "system": f"tinyserve_{args.mode}_{args.kv}",
        "model": args.model,
        "rate_req_s": args.rate,
        "max_batch_size": args.max_batch_size,
        "kv": args.kv,
        **summarize(engine.finished, elapsed),
        "num_preemptions": engine.num_preemptions,
    }
    if args.kv == "paged":
        report["block_size"] = args.block_size
        report["num_blocks"] = engine.cache.num_blocks
        gb = engine.cache.k.element_size() * (engine.cache.k.numel() + engine.cache.v.numel()) / 2**30
        report["kv_memory_gb"] = round(gb, 2)
    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
