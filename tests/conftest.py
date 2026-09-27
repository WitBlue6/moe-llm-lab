import torch
import pytest
from moe_llm.model import ModelConfig


@pytest.fixture(autouse=True)
def reproducible():
    torch.set_num_threads(1)
    torch.manual_seed(123)


@pytest.fixture
def tiny():
    return ModelConfig(vocab_size=262, hidden_size=32, num_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, expert_intermediate_size=48, num_experts=4,
                       experts_per_token=2, max_seq_len=128)
