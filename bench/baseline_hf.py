"""Naive baseline benchmark: sequential HuggingFace generate() calls.

Record these numbers in Week 1 — every later chart is relative to this.

Usage (on your GPU machine):
    python bench/baseline_hf.py --model Qwen/Qwen2.5-1.5B-Instruct --num-prompts 20
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain the difference between a process and a thread.",
    "Write a haiku about GPUs.",
    "What causes inflation? Answer in three sentences.",
    "Summarize the plot of Hamlet in one paragraph.",
    "Give me a Python one-liner to reverse a string.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_prompts)]

    # Warmup
    ids = tok(prompts[0], return_tensors="pt").input_ids.to(device)
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize() if device == "cuda" else None

    total_new_tokens = 0
    ttfts = []
    start = time.perf_counter()
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        t0 = time.perf_counter()
        # TTFT proxy: time to generate exactly 1 token
        model.generate(ids, max_new_tokens=1, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        ttfts.append(time.perf_counter() - t0)

        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        total_new_tokens += out.shape[1] - ids.shape[1]
    elapsed = time.perf_counter() - start

    report = {
        "system": "naive_hf_sequential",
        "model": args.model,
        "num_prompts": len(prompts),
        "total_new_tokens": total_new_tokens,
        "elapsed_s": round(elapsed, 2),
        "throughput_tok_s": round(total_new_tokens / elapsed, 2),
        "ttft_p50_ms": round(sorted(ttfts)[len(ttfts) // 2] * 1000, 1),
    }
    print(json.dumps(report, indent=2))
    with open("bench/results_baseline.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
