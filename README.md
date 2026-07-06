# TinyServe

A mini LLM inference engine built from scratch in PyTorch — continuous batching, paged KV cache, chunked prefill — benchmarked against vLLM.

**Status: Week 2** — continuous (iteration-level) batching over a slot-based KV cache, with a Poisson load generator and TTFT/ITL metrics.

## Why

Everyone calls LLM APIs. Few can explain *why* continuous batching beats static batching or how PagedAttention avoids KV fragmentation — with code they wrote. TinyServe implements the scheduling and memory layer of a modern inference server on top of stock PyTorch attention (no custom kernels, by design).

## Quickstart

```bash
pip install -e ".[dev]"

# Correctness gate: token-exact match vs HuggingFace (CPU, tiny models)
pytest

# Generate with real weights (GPU)
python examples/generate.py --model Qwen/Qwen2.5-1.5B-Instruct

# Record the naive baseline (every later chart is relative to this)
python bench/baseline_hf.py --model Qwen/Qwen2.5-1.5B-Instruct

# Continuous batching under Poisson load (GPU)
python bench/run_tinyserve.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-requests 50 --rate 2.0 --max-batch-size 8

# Same load, static batching — the comparison that motivates Week 2
python bench/run_tinyserve.py --mode static --num-requests 50 --rate 2.0
```

## Architecture

```
tinyserve/
├── model.py      # Llama-arch forward pass, explicit KV-cache plumbing
│                 #   (state-dict compatible with HF Llama 3.x / Qwen2.x)
├── kv_cache.py   # Week 2: slot-based cache (one slab per sequence). Week 3: paged
├── sampler.py    # greedy / temperature / top-p
├── request.py    # Request lifecycle + per-token timing (TTFT/ITL)
├── engine.py     # Engine (single-seq reference) + BatchEngine (continuous batching)
tests/            # correctness gate (vs HF) + batching gate (vs Engine)
bench/            # baselines, Poisson load generator, metrics
```

## The correctness gate

`tests/test_correctness.py` asserts TinyServe's greedy output matches `model.generate()` **token-for-token** on tiny random-weight models (plain, GQA, tied embeddings, llama3 RoPE scaling). It runs on CPU in seconds and stays green for the life of the project — every optimization gets validated against it.

## Roadmap

- [x] Week 1 — correct engine: explicit KV cache, prefill/decode split, correctness gate, naive baseline numbers
- [x] Week 2 — static → continuous (iteration-level) batching; Poisson load generator
- [ ] Week 3 — paged KV cache: block allocator, block tables, preemption
- [ ] Week 4 — chunked prefill; OpenAI-compatible streaming server; benchmark vs vLLM (TTFT/ITL p50/p99, goodput @ SLO)
- [ ] Week 5 — ablation study, writeup
