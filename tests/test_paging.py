"""Week 3 gate: PagedKVCache must be invisible.

Three levels:
1. Cache level — Paged and Slot caches given the same append/advance sequence
   must return bit-identical gathers.
2. Engine level — BatchEngine(kv="paged") must match Engine token-for-token
   (Engine itself is HF-verified).
3. Under memory pressure — with few physical blocks, preemption must fire and
   outputs must STILL be exact: eviction + recompute is invisible to the user.

float64 on CPU, same reasoning as test_batching.
"""

import pytest
import torch

from tinyserve import BatchEngine, Engine, ModelConfig, PagedKVCache, Request, SlotKVCache, Transformer

SEED = 11


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


# ---------- level 1: raw cache equivalence ----------

def test_paged_cache_matches_slot_cache():
    """Same op sequence on both caches -> identical gathers, every call."""
    L, SLOTS, MSL, KVH, D, BS = 2, 3, 64, 2, 8, 4
    torch.manual_seed(0)
    slot_c = SlotKVCache(L, SLOTS, MSL, KVH, D, "cpu", torch.float64)
    paged_c = PagedKVCache(L, SLOTS, MSL, KVH, D, "cpu", torch.float64, block_size=BS)

    def rand(B, T):
        return torch.randn(B, T, KVH, D, dtype=torch.float64)

    def check(layer, k, v, slots_t):
        out_s = slot_c.append(layer, k, v, slots_t)
        out_p = paged_c.append(layer, k, v, slots_t)
        for a, b in zip(out_s, out_p):
            torch.testing.assert_close(a, b, atol=0, rtol=0)  # bit-identical

    # Both caches share the same free-list order, so paired allocs agree.
    a = slot_c.alloc()
    assert paged_c.alloc() == a
    b = slot_c.alloc()
    assert paged_c.alloc() == b
    s_a, s_b = torch.tensor([a]), torch.tensor([b])

    # prefill: 7 tokens crosses a block boundary (7 > BS=4)
    for slot_ids, n in [(s_a, 7), (s_b, 5)]:
        paged_c.reserve(slot_ids, n)
        for layer in range(L):
            k, v = rand(1, n), rand(1, n)
            check(layer, k, v, slot_ids)
        slot_c.advance(slot_ids, n), paged_c.advance(slot_ids, n)

    # 10 batched decode steps over both sequences (mixed history lengths)
    both = torch.tensor([a, b])
    for _ in range(10):
        paged_c.reserve(both, 1)
        for layer in range(L):
            k, v = rand(2, 1), rand(2, 1)
            check(layer, k, v, both)
        slot_c.advance(both, 1), paged_c.advance(both, 1)

    # free + reuse: seq A's slot and blocks recycled by a new sequence
    slot_c.free(a), paged_c.free(a)
    c_slot = slot_c.alloc()
    assert paged_c.alloc() == c_slot
    s_c = torch.tensor([c_slot])
    paged_c.reserve(s_c, 6)
    for layer in range(L):
        k, v = rand(1, 6), rand(1, 6)
        check(layer, k, v, s_c)


def test_block_accounting():
    c = PagedKVCache(1, 2, 32, 2, 4, "cpu", torch.float64, block_size=4, num_blocks=6)
    assert c.num_free_blocks == 6
    s = c.alloc()
    c.reserve(torch.tensor([s]), 7)          # ceil(7/4) = 2 blocks
    assert c.num_free_blocks == 4
    c.advance(torch.tensor([s]), 7)
    c.reserve(torch.tensor([s]), 1)          # 8th token -> still block 2
    assert c.num_free_blocks == 4
    c.advance(torch.tensor([s]), 1)
    c.reserve(torch.tensor([s]), 1)          # 9th token -> new block
    assert c.num_free_blocks == 3
    c.free(s)
    assert c.num_free_blocks == 6 and c.num_free == 2


def test_reserve_all_or_nothing():
    c = PagedKVCache(1, 2, 32, 2, 4, "cpu", torch.float64, block_size=4, num_blocks=2)
    a, b = c.alloc(), c.alloc()
    assert c.reserve(torch.tensor([a]), 4)   # takes 1 block
    assert not c.reserve(torch.tensor([b]), 8)  # needs 2, only 1 free -> refuse
    assert c.num_free_blocks == 1            # and allocated NOTHING


# ---------- level 2: engine equivalence ----------

def make_requests(model, n, seed_base=0):
    prompts = [random_prompt(model.cfg.vocab_size, 8 + 3 * i, seed=seed_base + i) for i in range(n)]
    lens = [10 + 2 * i for i in range(n)]
    eng = Engine(model, max_seq_len=256)
    expected = [eng.generate(p, max_new_tokens=m).output_ids for p, m in zip(prompts, lens)]
    reqs = [
        Request(prompt_ids=p, max_new_tokens=m, ignore_eos=True, request_id=i)
        for i, (p, m) in enumerate(zip(prompts, lens))
    ]
    return reqs, expected


def test_paged_engine_matches_reference(model):
    reqs, expected = make_requests(model, 5, seed_base=500)
    eng = BatchEngine(model, max_batch_size=5, max_seq_len=256, kv="paged", block_size=4)
    for r in reqs:
        eng.submit(r)
    done = {r.request_id: r for r in eng.run_to_completion()}
    for i, exp in enumerate(expected):
        assert done[i].output_ids == exp, f"request {i} diverged"
    assert eng.num_preemptions == 0  # full capacity: paging alone, no eviction


# ---------- level 3: preemption under memory pressure ----------

def test_max_size_request_admits(model):
    """Regression: a request needing ALL physical blocks must still be admitted
    when the cache is empty (the +1 watermark must not deadlock admission)."""
    prompt = random_prompt(model.cfg.vocab_size, 9, seed=700)
    # total = 9 + 11 = 20 tokens = exactly 5 blocks of 4 = num_blocks
    ref = Engine(model, max_seq_len=256).generate(prompt, max_new_tokens=11)
    eng = BatchEngine(
        model, max_batch_size=2, max_seq_len=256, kv="paged", block_size=4, num_blocks=5
    )
    eng.submit(Request(prompt_ids=prompt, max_new_tokens=11, ignore_eos=True))
    done = eng.run_to_completion()
    assert done[0].output_ids == ref.output_ids


def test_preemption_exact_outputs(model):
    """Physical blocks << what the batch wants: requests must get evicted,
    recomputed, and still produce exactly the reference tokens."""
    reqs, expected = make_requests(model, 4, seed_base=600)
    # Each request needs <= ceil((17+16)/4) = 9 blocks alone; give 12 total so
    # the batch of 4 can't coexist but any single request always fits.
    eng = BatchEngine(
        model, max_batch_size=4, max_seq_len=256, kv="paged", block_size=4, num_blocks=12
    )
    for r in reqs:
        eng.submit(r)
    done = {r.request_id: r for r in eng.run_to_completion()}
    assert len(done) == 4
    for i, exp in enumerate(expected):
        assert done[i].output_ids == exp, f"request {i} diverged after preemption"
    assert eng.num_preemptions > 0, "expected memory pressure to force evictions"
    # All memory returned.
    assert eng.cache.num_free_blocks == 12
    assert eng.cache.num_free == 4
