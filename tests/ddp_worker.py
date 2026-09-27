"""Two-rank gradient oracle: compare variable-length accumulated DDP to one batch."""
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from moe_llm.data import collate
from moe_llm.model import LanguageModel, ModelConfig, causal_loss_sum


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(777)
    config = ModelConfig(vocab_size=32, hidden_size=16, num_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, expert_intermediate_size=24, num_experts=4,
                         experts_per_token=2, max_seq_len=16, gradient_checkpointing=True)
    model = LanguageModel(config)
    oracle = LanguageModel(config)
    oracle.load_state_dict(model.state_dict())
    wrapped = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
    # Distinct lengths AND distinct counts of supervised tokens on each rank.
    examples = []
    for i, length in enumerate((3, 7, 4, 6)):
        ids = torch.tensor([1] + [6 + i] * (length - 1))
        labels = torch.tensor([6 + i] * (length - 1) + [2])
        labels[:i] = -100
        examples.append({"input_ids": ids, "labels": labels})
    denominator = sum(int((x["labels"] != -100).sum()) for x in examples)
    # Repeat to exercise dynamic unused-parameter state across DDP iterations.
    for iteration in range(2):
        wrapped.zero_grad(set_to_none=True)
        oracle.zero_grad(set_to_none=True)
        for micro in range(2):
            batch = collate([examples[rank + world * micro]])
            out = wrapped(batch["input_ids"], attention_mask=batch["attention_mask"])
            total, _ = causal_loss_sum(out["logits"], batch["labels"])
            (total * world / denominator + out["aux_loss"] * 0).backward()
        global_batch = collate(examples)
        out = oracle(global_batch["input_ids"], attention_mask=global_batch["attention_mask"])
        total, _ = causal_loss_sum(out["logits"], global_batch["labels"])
        (total / denominator + out["aux_loss"] * 0).backward()
        for (name, actual), expected in zip(model.named_parameters(), oracle.parameters()):
            a = torch.zeros_like(actual) if actual.grad is None else actual.grad
            b = torch.zeros_like(expected) if expected.grad is None else expected.grad
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5, msg=f"rank {rank}: {name}")
        examples.reverse()
    if rank == 0:
        Path(sys.argv[1]).write_text("Two-rank accumulated gradients match global-token oracle.\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
