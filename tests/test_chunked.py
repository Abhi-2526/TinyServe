"""Week 4 gate: chunked prefill must not change a single token.

Splitting a prompt's prefill into chunks changes WHEN KV is computed, never
WHAT is computed. Verified at the model level (chunked forward == full
forward) and the engine level (chunked BatchEngine == Engine reference),
including chunking + paging + preemption stacked together.

float64 on CPU, same reasoning as test_batching.
"""

import pytest
import torch

from tinyserve import BatchEngine, Engine, ModelConfig, Request, SlotKVCache, Transformer

SEED = 23


def make_model():
    torch.manual_seed(SEED)
    cfg = ModelConfig(
        vocab_size=211,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    return Transformer(cfg).to(torch.float64).eval()


@pytest.fixture(scope="module")
def model():
    return make_model()


def random_prompt(vocab_size, length, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (length,), generator=g).tolist()


@torch.inference_mode()
def test_chunked_prefill_matches_full(model):
    """Model level: prefilling 23 tokens in chunks of 6 == one 23-token pass."""
    tokens = random_prompt(model.cfg.vocab_size, 23, seed=800)
    slot = torch.tensor([0])

    cache_full = SlotKVCache.for_model(model.cfg, 256, "cpu", torch.float64)
    full = model(torch.tensor([tokens]), torch.arange(23).unsqueeze(0), cache_full, slot)

    cache_chunk = SlotKVCache.for_model(model.cfg, 256, "cpu", torch.float64)
    last = None
    for start in range(0, 23, 6):
        end = min(start + 6, 23)
        last = model(
            torch.tensor([tokens[start:end]]),
            torch.arange(start, end).unsqueeze(0),
            cache_chunk,
            slot,
        )
        cache_chunk.advance(slot, end - start)

    torch.testing.assert_close(last[0, -1], full[0, -1], atol=1e-12, rtol=1e-12)
    # And the caches themselves must hold identical KV.
    torch.testing.assert_close(cache_chunk.k[:, 0, :23], cache_full.k[:, 0, :23], atol=0, rtol=0)


def make_requests(model, n, seed_base=0, prompt_base=8):
    prompts = [
        random_prompt(model.cfg.vocab_size, prompt_base + 5 * i, seed=seed_base + i)
        for i in range(n)
    ]
    lens = [10 + 2 * i for i in range(n)]
    eng = Engine(model, max_seq_len=256)
    expected = [eng.generate(p, max_new_tokens=m).output_ids for p, m in zip(prompts, lens)]
    reqs = [
        Request(prompt_ids=p, max_new_tokens=m, ignore_eos=True, request_id=i)
        for i, (p, m) in enumerate(zip(prompts, lens))
    ]
    return reqs, expected


@pytest.mark.parametrize("kv", ["slot", "paged"])
def test_chunked_engine_matches_reference(model, kv):
    """Chunks of 5 over prompts up to 28 tokens, arrivals staggered so chunks
    interleave with other requests' decode steps."""
    reqs, expected = make_requests(model, 4, seed_base=900, prompt_base=13)
    eng = BatchEngine(
        model, max_batch_size=4, max_seq_len=256,
        kv=kv, block_size=4, prefill_chunk_size=5,
    )
    eng.submit(reqs[0])
    eng.step()  # r0 mid-prefill when the others arrive
    for r in reqs[1:]:
        eng.submit(r)
        eng.step()
    done = {r.request_id: r for r in eng.run_to_completion()}
    for i, exp in enumerate(expected):
        assert done[i].output_ids == exp, f"request {i} diverged (kv={kv})"


def test_chunked_with_preemption_exact(model):
    """Chunking + paging + memory pressure together: still token-exact."""
    reqs, expected = make_requests(model, 4, seed_base=1000, prompt_base=13)
    # Largest request: 28 prompt + 16 new = 44 tokens = 11 blocks of 4; give 14.
    eng = BatchEngine(
        model, max_batch_size=4, max_seq_len=256,
        kv="paged", block_size=4, num_blocks=14, prefill_chunk_size=5,
    )
    for r in reqs:
        eng.submit(r)
    done = {r.request_id: r for r in eng.run_to_completion()}
    assert len(done) == 4
    for i, exp in enumerate(expected):
        assert done[i].output_ids == exp, f"request {i} diverged under pressure"
    assert eng.num_preemptions > 0
    assert eng.cache.num_free_blocks == 14
