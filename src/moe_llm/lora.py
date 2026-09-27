"""Unmerged low-rank visual adapters. Enablement is an explicit forward argument.

An explicit flag also survives activation-checkpoint recomputation; a temporary
module-global flag that is reset after forward would silently break backward.
"""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or alpha <= 0:
            raise ValueError("LoRA requires a Linear, positive rank and alpha")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x, enabled=False):
        out = self.base(x)
        if enabled:
            out = out + nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B) * self.scale
        return out


def project(module, x, enabled=False):
    return module(x, enabled=enabled) if isinstance(module, LoRALinear) else module(x)


def inject_visual_lora(language_model, rank, alpha, targets):
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("LoRA targets must be distinct attention projection names")
    for layer in language_model.layers:
        for name in targets:
            original = getattr(layer.attention, name)
            if isinstance(original, LoRALinear):
                raise ValueError("adapters already installed")
            setattr(layer.attention, name, LoRALinear(original, rank, alpha))
