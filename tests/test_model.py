from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from moe_llm.model import LanguageModel, SparseMoE, causal_loss_sum, token_log_probs, generate


@pytest.mark.parametrize("dense", [False, True])
def test_parameter_counts_and_causality(tiny, dense):
    c = replace(tiny, num_experts=1, experts_per_token=1) if dense else tiny
    model = LanguageModel(c).eval()
    assert sum(p.numel() for p in model.parameters()) == c.parameter_counts()["total"]
    ids = torch.randint(6, c.vocab_size, (2, 8))
    changed = ids.clone()
    changed[:, 4:] = torch.randint(6, c.vocab_size, (2, 4))
    torch.testing.assert_close(model(ids)["logits"][:, :4], model(changed)["logits"][:, :4])


def test_kv_cache_matches_full_prefix_and_chunks(tiny):
    model = LanguageModel(tiny).eval()
    ids = torch.randint(6, 262, (1, 9))
    full = model(ids)["logits"]
    first = model(ids[:, :4], use_cache=True)
    second = model(ids[:, 4:7], past_key_values=first["past_key_values"], use_cache=True)
    third = model(ids[:, 7:], past_key_values=second["past_key_values"], use_cache=True)
    actual = torch.cat((first["logits"], second["logits"], third["logits"]), 1)
    torch.testing.assert_close(actual, full, atol=2e-6, rtol=1e-5)
    assert third["past_key_values"][0][0].shape == (1, 2, 9, 8)


def test_padding_does_not_change_logits_or_router_statistics(tiny):
    model = LanguageModel(tiny).eval()
    ids = torch.randint(6, 262, (2, 4))
    padded = F.pad(ids, (0, 3))
    mask = padded != 0
    clean, padded_out = model(ids), model(padded, attention_mask=mask)
    torch.testing.assert_close(clean["logits"], padded_out["logits"][:, :4])
    torch.testing.assert_close(clean["aux_loss"], padded_out["aux_loss"])
    torch.testing.assert_close(clean["expert_usage"], padded_out["expert_usage"])


def test_sparse_dispatch_matches_explicit_topk_and_has_router_gradients(tiny):
    moe = SparseMoE(tiny)
    x = torch.randn(2, 3, tiny.hidden_size, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    output, aux, usage, entropy = moe(x, mask)
    tokens = x[mask]
    probabilities = moe.router(tokens).float().softmax(-1)
    weights, indices = probabilities.topk(2, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    expected = torch.stack([sum(moe.experts[int(indices[i, j])](tokens[i]) * weights[i, j]
                                for j in range(2)) for i in range(tokens.shape[0])])
    torch.testing.assert_close(output[mask], expected)
    assert output[~mask].count_nonzero() == 0
    torch.testing.assert_close(usage.sum(), torch.tensor(1.0))
    (output.square().sum() + aux * .01).backward()
    assert moe.router.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all() and torch.isfinite(entropy)


def test_empty_experts_are_allowed(tiny):
    moe = SparseMoE(tiny)
    with torch.no_grad():
        moe.router.weight.zero_()  # all tokens select the same two experts
    out, aux, usage, _ = moe(torch.randn(1, 2, 32), torch.ones(1, 2, dtype=torch.bool))
    (out.square().sum() + aux).backward()
    assert (usage == 0).sum() == 2
    assert sum(e.up_proj.weight.grad is None for e in moe.experts) == 2


def test_activation_checkpointing_matches_gradients(tiny):
    plain = LanguageModel(tiny).train()
    checkpointed = LanguageModel(replace(tiny, gradient_checkpointing=True)).train()
    checkpointed.load_state_dict(plain.state_dict())
    ids = torch.randint(6, 262, (2, 5))
    for model in (plain, checkpointed):
        out = model(ids)
        (out["logits"].square().mean() + out["aux_loss"] * .01).backward()
    for a, b in zip(plain.parameters(), checkpointed.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad)


def test_logprob_mask_matches_cross_entropy():
    logits = torch.randn(2, 4, 11, requires_grad=True)
    labels = torch.tensor([[1, -100, 3, 4], [-100, -100, 5, 2]])
    total, count = causal_loss_sum(logits, labels)
    reference = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
    torch.testing.assert_close(total, reference)
    assert count == 5
    total.backward()
    assert logits.grad[labels == -100].count_nonzero() == 0
    logp, mask = token_log_probs(logits, torch.full_like(labels, -100))
    assert logp.sum() == 0 and mask.sum() == 0


def test_tiny_model_overfits_one_batch(tiny):
    model = LanguageModel(tiny)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    ids = torch.tensor([[1, 10, 20, 30, 40], [1, 50, 60, 70, 80]])
    labels = torch.tensor([[10, 20, 30, 40, 2], [50, 60, 70, 80, 2]])
    first = None
    for _ in range(70):
        optimizer.zero_grad()
        out = model(ids)
        ce, count = causal_loss_sum(out["logits"], labels)
        loss = ce / count + .01 * out["aux_loss"]
        first = loss.item() if first is None else first
        loss.backward()
        optimizer.step()
    assert loss.item() < first * .15


def test_generation_matches_uncached_greedy(tiny):
    model = LanguageModel(tiny).eval()
    prompt = torch.tensor([[1, 20, 30]])
    result = generate(model, prompt, 5, temperature=0, eos_id=-1)
    expected = prompt
    for _ in range(5):
        token = model(expected)["logits"][:, -1].argmax(-1, keepdim=True)
        expected = torch.cat((expected, token), 1)
    assert torch.equal(result, expected)


def test_invalid_config_and_context(tiny):
    with pytest.raises(ValueError):
        replace(tiny, experts_per_token=5)
    with pytest.raises(ValueError):
        LanguageModel(tiny)(torch.ones(1, 129, dtype=torch.long))


def test_base_parameter_count_against_meta_model():
    from pathlib import Path
    from moe_llm.model import ModelConfig
    config = ModelConfig.load(Path(__file__).resolve().parents[1] / "configs/moe-base.json")
    with torch.device("meta"):
        model = LanguageModel(config)
    assert sum(p.numel() for p in model.parameters()) == 264315648
    assert config.parameter_counts()["active_per_token"] == 151069440
