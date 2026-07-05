"""Llama-architecture transformer with explicit, hand-managed KV cache.

Weight names mirror HuggingFace's LlamaForCausalLM exactly, so
``load_state_dict(hf_model.state_dict())`` works with strict=True.
Supports Llama 3.x (GQA, llama3 RoPE scaling, tied embeddings) and
Qwen2.x-style QKV biases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCache


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 8192
    qkv_bias: bool = False
    tie_word_embeddings: bool = False
    head_dim: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, model_name_or_path: str) -> "ModelConfig":
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_name_or_path)
        qkv_bias = bool(getattr(hf, "attention_bias", False)) or hf.model_type == "qwen2"
        rope_theta, rope_scaling = extract_rope_config(hf)
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=hf.max_position_embeddings,
            qkv_bias=qkv_bias,
            tie_word_embeddings=bool(getattr(hf, "tie_word_embeddings", False)),
            head_dim=getattr(hf, "head_dim", None) or 0,
        )


def extract_rope_config(hf_cfg) -> tuple[float, dict | None]:
    """Read RoPE settings from an HF config across transformers versions.

    v4.x: separate ``rope_theta`` (float) and ``rope_scaling`` (dict | None).
    v5.x: merged into a single ``rope_parameters`` dict, e.g.
          {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0, ...}.

    Returns (rope_theta, rope_scaling) in the v4 shape used internally;
    rope_scaling is None when rope_type is "default".
    """
    rp = getattr(hf_cfg, "rope_parameters", None)
    if isinstance(rp, dict):  # transformers >= 5
        rp = dict(rp)
        theta = float(rp.get("rope_theta", 10000.0))
        rope_type = rp.get("rope_type", rp.get("type", "default"))
        return theta, (rp if rope_type != "default" else None)
    theta = float(getattr(hf_cfg, "rope_theta", 10000.0))
    scaling = getattr(hf_cfg, "rope_scaling", None)
    if scaling is not None:
        scaling = dict(scaling)
        if scaling.get("rope_type", scaling.get("type", "default")) == "default":
            scaling = None
    return theta, scaling


def _compute_inv_freq(cfg: ModelConfig) -> torch.Tensor:
    """Base RoPE inverse frequencies, with optional llama3-style scaling."""
    inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim)
    )
    rs = cfg.rope_scaling
    rope_type = (rs or {}).get("rope_type", (rs or {}).get("type"))
    if rope_type in (None, "default"):
        return inv_freq
    if rope_type != "llama3":
        raise NotImplementedError(f"rope_type={rope_type!r} not supported")

    factor = rs["factor"]
    low = rs["low_freq_factor"]
    high = rs["high_freq_factor"]
    old_ctx = rs["original_max_position_embeddings"]

    low_freq_wavelen = old_ctx / low
    high_freq_wavelen = old_ctx / high
    wavelen = 2 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low) / (high - low)
    smoothed = (1 - smooth) / factor * inv_freq + smooth * inv_freq
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.register_buffer("inv_freq", _compute_inv_freq(cfg), persistent=False)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: (T,) int64 -> cos, sin each (T, head_dim), float32."""
        freqs = torch.outer(positions.to(torch.float32), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    """q: (T, H, D), k: (T, KVH, D); cos/sin: (T, D)."""
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, cache: KVCache) -> torch.Tensor:
        """x: (T, C). Appends this step's K/V to `cache` and attends over the full history."""
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(T, self.n_kv_heads, self.head_dim)
        q, k = _apply_rope(q, k, cos, sin)

        k_all, v_all = cache.append(self.layer_idx, k, v)  # (S, KVH, D)
        S = k_all.shape[0]

        # (1, heads, seq, dim) for SDPA
        q = q.permute(1, 0, 2).unsqueeze(0)
        k_all = k_all.permute(1, 0, 2).unsqueeze(0)
        v_all = v_all.permute(1, 0, 2).unsqueeze(0)
        if self.n_rep > 1:
            k_all = k_all.repeat_interleave(self.n_rep, dim=1)
            v_all = v_all.repeat_interleave(self.n_rep, dim=1)

        if T == S:
            out = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True)
        elif T == 1:
            out = F.scaled_dot_product_attention(q, k_all, v_all)
        else:
            # General case (chunked prefill): query i may attend to kv j iff j <= (S - T) + i
            j = torch.arange(S, device=x.device)
            i = torch.arange(T, device=x.device)
            mask = j[None, :] <= (i[:, None] + (S - T))
            out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)

        out = out.squeeze(0).permute(1, 0, 2).reshape(T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache: KVCache) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg)

    def forward(self, token_ids, positions, cache: KVCache) -> torch.Tensor:
        x = self.embed_tokens(token_ids)
        cos, sin = self.rotary(positions)
        for layer in self.layers:
            x = layer(x, cos, sin, cache)
        return self.norm(x)


class Transformer(nn.Module):
    """Causal LM. State-dict-compatible with HF LlamaForCausalLM / Qwen2ForCausalLM."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = LlamaModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, token_ids, positions, cache: KVCache) -> torch.Tensor:
        """token_ids, positions: (T,). Returns logits (T, vocab)."""
        hidden = self.model(token_ids, positions, cache)
        return self.lm_head(hidden)

    def load_hf_state_dict(self, state_dict: dict) -> None:
        sd = {k: v for k, v in state_dict.items() if not k.endswith("inv_freq")}
        if self.cfg.tie_word_embeddings:
            sd.pop("lm_head.weight", None)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            missing = [m for m in missing if m != "lm_head.weight"]
            assert not missing and not unexpected, (missing, unexpected)
        else:
            self.load_state_dict(sd, strict=True)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Transformer":
        from transformers import AutoModelForCausalLM

        cfg = ModelConfig.from_hf(model_name_or_path)
        model = cls(cfg).to(dtype)
        hf = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
        model.load_hf_state_dict(hf.state_dict())
        del hf
        return model.to(device).eval()
