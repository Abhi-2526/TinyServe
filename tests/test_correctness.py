"""The correctness gate: TinyServe must match HuggingFace token-for-token.

Runs on CPU with tiny random-weight models, so it works in CI without a GPU.
This test must stay green for the life of the project — every optimization
(batching, paging, chunked prefill) gets validated against it.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tinyserve import Engine, ModelConfig, Transformer

SEED = 1234


def make_tiny_hf(rope_scaling=None, tie=False):
    torch.manual_seed(SEED)
    hf_cfg = LlamaConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # exercises GQA
        max_position_embeddings=512,
        rope_theta=10000.0,
        rope_scaling=rope_scaling,
        tie_word_embeddings=tie,
    )
    # Pin the attention backend so both sides use SDPA (same numerics path).
    hf_cfg._attn_implementation = "sdpa"
    model = LlamaForCausalLM(hf_cfg).eval()
    # Random-weight models can emit the default EOS token by chance, which
    # makes HF generate() stop early and the token-exact comparison flaky.
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model, hf_cfg


def to_tinyserve(hf_model, hf_cfg):
    from tinyserve.model import extract_rope_config

    rope_theta, rope_scaling = extract_rope_config(hf_cfg)
    cfg = ModelConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        tie_word_embeddings=hf_cfg.tie_word_embeddings,
    )
    model = Transformer(cfg)
    model.load_hf_state_dict(hf_model.state_dict())
    return model.eval()


CASES = {
    "plain": dict(rope_scaling=None, tie=False),
    "tied_embeddings": dict(rope_scaling=None, tie=True),
    "llama3_rope_scaling": dict(
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 128,
        },
        tie=False,
    ),
}


@pytest.fixture(params=CASES.keys())
def models(request):
    hf_model, hf_cfg = make_tiny_hf(**CASES[request.param])
    ts_model = to_tinyserve(hf_model, hf_cfg)
    return hf_model, ts_model


def random_prompt(vocab_size, length, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (length,), generator=g).tolist()


@torch.inference_mode()
def test_prefill_logits_match(models):
    hf_model, ts_model = models
    from tinyserve.kv_cache import ContiguousKVCache

    prompt = random_prompt(ts_model.cfg.vocab_size, 33)
    cache = ContiguousKVCache.for_model(ts_model.cfg, 512, "cpu", torch.float32)
    ids = torch.tensor(prompt)
    ts_logits = ts_model(ids, torch.arange(len(prompt)), cache)

    hf_logits = hf_model(ids.unsqueeze(0)).logits.squeeze(0)
    torch.testing.assert_close(ts_logits, hf_logits, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_greedy_generation_token_exact(models):
    hf_model, ts_model = models
    engine = Engine(ts_model, max_seq_len=512)

    for seed in range(5):
        prompt = random_prompt(ts_model.cfg.vocab_size, 16 + seed * 7, seed=seed)
        ours = engine.generate(prompt, max_new_tokens=32).output_ids

        ids = torch.tensor(prompt).unsqueeze(0)
        hf_out = hf_model.generate(
            ids,
            # Explicit all-ones mask: otherwise HF infers one from pad_token_id
            # and masks out any position whose token id happens to equal it.
            attention_mask=torch.ones_like(ids),
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
        )
        theirs = hf_out.squeeze(0)[len(prompt):].tolist()
        assert ours == theirs, f"seed={seed}: {ours} != {theirs}"


@torch.inference_mode()
def test_decode_matches_prefill(models):
    """Incremental decode over cached history must equal a fresh full forward pass."""
    _, ts_model = models
    from tinyserve.kv_cache import ContiguousKVCache

    tokens = random_prompt(ts_model.cfg.vocab_size, 24, seed=42)

    # Incremental: prefill 20, then decode 4 one at a time
    cache = ContiguousKVCache.for_model(ts_model.cfg, 512, "cpu", torch.float32)
    ids = torch.tensor(tokens[:20])
    ts_model(ids, torch.arange(20), cache)
    cache.advance(20)
    last = None
    for i in range(20, 24):
        last = ts_model(torch.tensor([tokens[i]]), torch.tensor([i]), cache)
        cache.advance(1)

    # Fresh full pass
    cache2 = ContiguousKVCache.for_model(ts_model.cfg, 512, "cpu", torch.float32)
    full = ts_model(torch.tensor(tokens), torch.arange(24), cache2)

    torch.testing.assert_close(last[-1], full[-1], atol=1e-4, rtol=1e-4)
