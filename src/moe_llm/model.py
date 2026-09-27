"""Decoder-only Transformer: RMSNorm, RoPE, GQA, SwiGLU and sparse top-k MoE."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .lora import project


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16384
    hidden_size: int = 768
    num_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    expert_intermediate_size: int = 1536
    num_experts: int = 4  # 1 selects a dense SwiGLU FFN, without a router.
    experts_per_token: int = 2
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "num_layers", "num_attention_heads",
                     "num_key_value_heads", "expert_intermediate_size", "num_experts", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide into attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be a multiple of KV heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 1 <= self.experts_per_token <= self.num_experts:
            raise ValueError("experts_per_token must be within [1, num_experts]")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("RoPE theta and norm epsilon must be positive")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def parameter_counts(self):
        """Count full embeddings in both totals; active count is not FLOPs."""
        d = self.hidden_size
        kv = self.num_key_value_heads * (d // self.num_attention_heads)
        attention = 2 * d * d + 2 * d * kv
        expert = 3 * d * self.expert_intermediate_size
        router = d * self.num_experts if self.num_experts > 1 else 0
        common = attention + router + 2 * d
        embedding = self.vocab_size * d * (1 if self.tie_word_embeddings else 2)
        total = embedding + self.num_layers * (common + self.num_experts * expert) + d
        active = embedding + self.num_layers * (common + self.experts_per_token * expert) + d
        return {"total": total, "active_per_token": active}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        d = config.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        inv_freq = 1.0 / config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate(self, x, offset):
        positions = torch.arange(offset, offset + x.shape[2], device=x.device).float()
        angles = torch.outer(positions, self.inv_freq.float())
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x, attention_mask=None, past=None, use_cache=False, adapter_enabled=False):
        batch, length, _ = x.shape
        def split(proj, heads):
            return project(proj, x, adapter_enabled).view(batch, length, heads, self.head_dim).transpose(1, 2)
        offset = 0 if past is None else past[0].shape[2]
        q = self.rotate(split(self.q_proj, self.heads), offset)
        k = self.rotate(split(self.k_proj, self.kv_heads), offset)
        v = split(self.v_proj, self.kv_heads)
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        cache = (k, v) if use_cache else None
        key_length = k.shape[2]
        # Explicit offset mask is essential for cached multi-token decoding.
        mask = None
        causal = offset == 0 and attention_mask is None
        if not causal:
            query_positions = torch.arange(offset, offset + length, device=x.device)
            mask = torch.arange(key_length, device=x.device)[None, :] <= query_positions[:, None]
            mask = mask[None, None, :, :]
            if attention_mask is not None:
                if attention_mask.shape != (batch, key_length):
                    raise ValueError("attention_mask must cover the entire cached + current sequence")
                mask = mask & attention_mask[:, None, None, :].bool()
        repeats = self.heads // self.kv_heads
        output = F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1),
            attn_mask=mask, dropout_p=0.0, is_causal=causal,
        )
        return project(self.o_proj, output.transpose(1, 2).reshape(batch, length, -1), adapter_enabled), cache


class SwiGLU(nn.Module):
    def __init__(self, d, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(d, intermediate, bias=False)
        self.up_proj = nn.Linear(d, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.num_experts, self.top_k = c.num_experts, c.experts_per_token
        self.router = nn.Linear(c.hidden_size, c.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(c.hidden_size, c.expert_intermediate_size) for _ in range(c.num_experts)
        ])

    def forward(self, x, token_mask):
        flat = x.reshape(-1, x.shape[-1])
        valid = token_mask.reshape(-1).bool()
        # Padding is excluded from dispatch AND load-balancing statistics.
        tokens = flat[valid]
        probabilities = F.softmax(self.router(tokens).float(), dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        result = torch.zeros_like(tokens)
        for number, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == number)
            if rows.numel():
                contribution = expert(tokens[rows]) * weights[rows, slots, None].to(tokens.dtype)
                result = result.index_add(0, rows, contribution)
        output = torch.zeros_like(flat).index_copy(0, valid.nonzero().flatten(), result)
        counts = torch.bincount(indices.flatten(), minlength=self.num_experts).float()
        fraction = counts / max(1, tokens.shape[0] * self.top_k)
        if tokens.shape[0]:
            balance = self.num_experts * (fraction * probabilities.mean(0)).sum()
            entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
        else:
            balance = probabilities.sum() * 0
            entropy = balance.detach()
        return output.view_as(x), balance, fraction.detach(), entropy.detach()


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.attention = Attention(c)
        self.ffn_norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.is_moe = c.num_experts > 1
        self.ffn = SparseMoE(c) if self.is_moe else SwiGLU(c.hidden_size, c.expert_intermediate_size)

    def forward(self, x, token_mask, attention_mask, past=None, use_cache=False, adapter_enabled=False):
        attended, cache = self.attention(self.attn_norm(x), attention_mask, past, use_cache, adapter_enabled)
        x = x + attended
        if self.is_moe:
            feedforward, aux, usage, entropy = self.ffn(self.ffn_norm(x), token_mask)
        else:
            feedforward = self.ffn(self.ffn_norm(x))
            aux, usage, entropy = x.new_zeros(()), x.new_ones(1), x.new_zeros(())
        return x + feedforward, aux, usage, entropy, cache


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                use_cache=False, return_hidden=False, inputs_embeds=None, adapter_enabled=False):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids and inputs_embeds")
        if input_ids is not None:
            if input_ids.ndim != 2 or input_ids.shape[1] == 0:
                raise ValueError("input_ids must have shape [batch, positive_length]")
            x = self.embedding(input_ids)
        else:
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size or inputs_embeds.shape[1] == 0:
                raise ValueError("invalid inputs_embeds shape")
            x = inputs_embeds
        length = x.shape[1]
        offset = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if offset + length > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("cache must contain one entry per layer")
        if self.training and self.config.gradient_checkpointing and (use_cache or past_key_values is not None):
            raise ValueError("training checkpointing cannot be combined with KV caching")
        token_mask = (torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if attention_mask is None
                      else attention_mask[:, -length:].bool())
        aux, usage, entropies, caches = [], [], [], []
        for number, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[number]
            if self.training and self.config.gradient_checkpointing:
                x, a, u, e, cache = checkpoint(
                    layer, x, token_mask, attention_mask, None, False, adapter_enabled, use_reentrant=False,
                )
            else:
                x, a, u, e, cache = layer(x, token_mask, attention_mask, past, use_cache, adapter_enabled)
            aux.append(a)
            usage.append(u)
            entropies.append(e)
            caches.append(cache)
        hidden = self.norm(x)
        result = {"logits": self.lm_head(hidden), "aux_loss": torch.stack(aux).mean(),
                  "expert_usage": torch.stack(usage), "router_entropy": torch.stack(entropies).mean()}
        if use_cache:
            result["past_key_values"] = tuple(caches)
        if return_hidden:  # Future PPO critic/reward heads can consume this tensor.
            result["hidden_states"] = hidden
        return result


def token_log_probs(logits, labels):
    """Aligned (already shifted) labels. Shared building block for SFT/DPO/PPO."""
    valid = labels != -100
    safe = labels.masked_fill(~valid, 0)
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return logp.masked_fill(~valid, 0), valid


def causal_loss_sum(logits, labels):
    logp, valid = token_log_probs(logits, labels)
    return -logp.sum(), valid.sum()


@torch.inference_mode()
def generate(model, prompt_ids, max_new_tokens=64, temperature=0.8, top_k=40, eos_id=2):
    """Single unpadded prompt; returns prompt + continuation, using the KV cache."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("generation currently supports one unpadded prompt")
    if max_new_tokens < 0 or temperature < 0 or top_k < 0:
        raise ValueError("invalid generation options")
    model.eval()
    ids, current, cache = prompt_ids, prompt_ids, None
    for _ in range(min(max_new_tokens, model.config.max_seq_len - ids.shape[1])):
        output = model(current, past_key_values=cache, use_cache=True)
        logits = output["logits"][:, -1].float()
        if temperature == 0:
            next_id = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, -float("inf"))
            next_id = torch.multinomial(logits.softmax(-1), 1)
        ids = torch.cat((ids, next_id), dim=1)
        if next_id.item() == eos_id:
            break
        current, cache = next_id, output["past_key_values"]
    return ids
