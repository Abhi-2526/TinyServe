"""Week 2 gate: BatchEngine must produce byte-identical outputs to Engine.

Engine is the HF-verified reference (test_correctness.py). If batched decode
with padded KV gather + masking is correct, batching a request with strangers
must not change a single token of its output.

Runs in float64 on CPU: the padded-batch and single-sequence paths reduce in
different orders, so float32 argmax near-ties could flip tokens spuriously.
"""

import pytest
import torch

from tinyserve import BatchEngine, Engine, ModelConfig, Request, Transformer

SEED = 7


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


def reference_outputs(model, prompts, lens):
    eng = Engine(model, max_seq_len=256)
    return [eng.generate(p, max_new_tokens=n).output_ids for p, n in zip(prompts, lens)]


def make_requests(model, n, seed_base=0):
    prompts = [random_prompt(model.cfg.vocab_size, 8 + 3 * i, seed=seed_base + i) for i in range(n)]
    lens = [10 + 2 * i for i in range(n)]
    reqs = [
        Request(prompt_ids=p, max_new_tokens=m, ignore_eos=True, request_id=i)
        for i, (p, m) in enumerate(zip(prompts, lens))
    ]
    return reqs, reference_outputs(model, prompts, lens)


def by_id(finished):
    return {r.request_id: r for r in finished}


def test_continuous_batching_matches_reference(model):
    """All submitted upfront; different lengths force mid-flight departures."""
    reqs, expected = make_requests(model, 5)
    eng = BatchEngine(model, max_batch_size=5, max_seq_len=256)
    for r in reqs:
        eng.submit(r)
    done = by_id(eng.run_to_completion())
    assert len(done) == len(reqs)
    for i, exp in enumerate(expected):
        assert done[i].output_ids == exp, f"request {i} diverged"
        assert done[i].finish_reason == "length"


def test_continuous_batching_staggered_arrivals(model):
    """Requests join a batch already mid-decode (the point of continuous batching)."""
    reqs, expected = make_requests(model, 4, seed_base=100)
    eng = BatchEngine(model, max_batch_size=4, max_seq_len=256)
    eng.submit(reqs[0])
    eng.submit(reqs[1])
    for _ in range(3):
        eng.step()
    eng.submit(reqs[2])  # joins while 0 and 1 have 3 tokens of history
    eng.step()
    eng.submit(reqs[3])
    eng.run_to_completion()
    done = by_id(eng.finished)
    for i, exp in enumerate(expected):
        assert done[i].output_ids == exp, f"request {i} diverged"


def test_slot_reuse_more_requests_than_slots(model):
    """9 requests through 3 slots: waiting queue + free/alloc recycling."""
    reqs, expected = make_requests(model, 9, seed_base=200)
    eng = BatchEngine(model, max_batch_size=3, max_seq_len=256)
    for r in reqs:
        eng.submit(r)
    done = by_id(eng.run_to_completion())
    assert len(done) == 9
    for i, exp in enumerate(expected):
        assert done[i].output_ids == exp, f"request {i} diverged"


def test_static_mode_matches_reference(model):
    reqs, expected = make_requests(model, 6, seed_base=300)
    eng = BatchEngine(model, max_batch_size=3, max_seq_len=256, mode="static")
    for r in reqs:
        eng.submit(r)
    done = by_id(eng.run_to_completion())
    for i, exp in enumerate(expected):
        assert done[i].output_ids == exp, f"request {i} diverged"


def test_stop_token_frees_slot(model):
    """A request that hits a stop token finishes early and releases its slot."""
    prompt = random_prompt(model.cfg.vocab_size, 8, seed=400)
    ref = Engine(model, max_seq_len=256).generate(prompt, max_new_tokens=20)
    stop = ref.output_ids[4]
    cut = ref.output_ids.index(stop) + 1  # first occurrence might be earlier

    eng = BatchEngine(model, max_batch_size=2, max_seq_len=256)
    eng.submit(Request(prompt_ids=prompt, max_new_tokens=20, stop_token_ids=(stop,)))
    done = eng.run_to_completion()
    assert done[0].finish_reason == "stop"
    assert done[0].output_ids == ref.output_ids[:cut]
    assert eng.cache.num_free == 2
