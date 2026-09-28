from dataclasses import replace
from pathlib import Path
import pytest
import torch
from moe_llm.data import prepare
from moe_llm.training import TrainConfig, train, load_checkpoint, learning_rate_at

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def config(**kwargs):
    base = TrainConfig(max_steps=4, warmup_steps=1, batch_size=2, grad_accum_steps=2,
                       learning_rate=.002, eval_every=2, save_every=2, device="cpu", precision="fp32")
    return replace(base, **kwargs)


def test_resume_matches_uninterrupted_and_sft_initialization(tmp_path, tiny):
    data = tmp_path / "pretrain-data"
    prepare([FIXTURES / "pretrain.jsonl"], data, "byte", "pretrain", 64, .25)
    c = config()
    full, partial, resumed = [tmp_path / n for n in ("full", "partial", "resumed")]
    train(tiny, c, data, "byte", full)
    train(tiny, c, data, "byte", partial, stop_after=2)
    train(tiny, c, data, "byte", resumed, resume=partial / "step-0000002.pt")
    a, b = load_checkpoint(full / "step-0000004.pt"), load_checkpoint(resumed / "step-0000004.pt")
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert (a["epoch"], a["cursor"], a["trained_tokens"]) == (b["epoch"], b["cursor"], b["trained_tokens"])
    sft_data = tmp_path / "sft-data"
    prepare([FIXTURES / "sft.jsonl"], sft_data, "byte", "sft", 64, .25)
    metrics = train(tiny, config(stage="sft"), sft_data, "byte", tmp_path / "sft", init_from=full / "step-0000004.pt")
    assert metrics["val_tokens"] > 0 and metrics["step"] == 4
    with pytest.raises(ValueError, match="requires"):
        train(tiny, config(stage="sft"), sft_data, "byte", tmp_path / "random-sft")
    with pytest.raises(FileExistsError):
        train(tiny, c, data, "byte", full)
    with pytest.raises(ValueError, match="unchanged"):
        train(tiny, replace(c, learning_rate=.1), data, "byte", tmp_path / "bad-resume", resume=partial / "step-0000002.pt")


def test_scheduler_and_validation():
    c = config(max_steps=10, warmup_steps=2)
    assert learning_rate_at(0, c) == c.learning_rate / 2
    assert learning_rate_at(1, c) == c.learning_rate
    assert learning_rate_at(9, c) == pytest.approx(c.learning_rate * c.min_lr_ratio)
    with pytest.raises(ValueError):
        config(grad_accum_steps=0)


def test_epoch_budget_and_partial_accumulation():
    from moe_llm.training import resolve_training_budget, BatchStream
    c = config(epochs=2, batch_size=2, grad_accum_steps=3, warmup_steps=0)
    resolved, steps = resolve_training_budget(c, 7, 1)
    assert steps == 2 and resolved.max_steps == 4
    stream = BatchStream(list(range(7)), c, 0, 1, collate_fn=lambda rows: rows)
    for _ in range(2):
        first = stream.next_group(3, finish_epoch=True)
        last = stream.next_group(3, finish_epoch=True)
        assert len(first) == 3 and len(last) == 1
        assert sorted(sum(first + last, [])) == list(range(7))
    # DDP sampler pads 7 records to 8, four records per rank.
    resolved, steps = resolve_training_budget(c, 7, 2)
    assert steps == 1 and resolved.max_steps == 2
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="epochs"):
            config(epochs=invalid)
    with pytest.raises(ValueError, match="warmup"):
        resolve_training_budget(config(epochs=1, warmup_steps=2), 1, 1)


def test_epoch_training_tokens_and_resume(tmp_path, tiny):
    from moe_llm.data import TokenDataset
    from moe_llm.training import resolve_training_budget
    data = tmp_path / "data"
    prepare([FIXTURES / "pretrain.jsonl"], data, "byte", "pretrain", 64, .25)
    dataset = TokenDataset(data, "train")
    c = config(epochs=2, batch_size=2, grad_accum_steps=3, warmup_steps=0)
    resolved, steps = resolve_training_budget(c, len(dataset), 1)
    full = train(tiny, c, data, "byte", tmp_path / "full", eval_max_batches=1)
    train(tiny, c, data, "byte", tmp_path / "part", stop_after=steps, eval_max_batches=1)
    train(tiny, c, data, "byte", tmp_path / "resumed",
          resume=tmp_path / f"part/step-{steps:07d}.pt", eval_max_batches=1)
    end = resolved.max_steps
    a = load_checkpoint(tmp_path / f"full/step-{end:07d}.pt")
    b = load_checkpoint(tmp_path / f"resumed/step-{end:07d}.pt")
    assert full["epochs_completed"] == 2 and full["step"] == end
    expected_tokens = 2 * sum(int((dataset[i]['labels'] != -100).sum()) for i in range(len(dataset)))
    assert a['trained_tokens'] == b['trained_tokens'] == expected_tokens
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)


def test_pilot_limits_validation_and_saves_checkpoint(tmp_path, tiny):
    from moe_llm.data import TokenDataset
    from moe_llm.training import evaluate
    from moe_llm.model import LanguageModel
    data = tmp_path / "data"
    prepare([FIXTURES / "pretrain.jsonl"], data, "byte", "pretrain", 64, .5)
    dataset = TokenDataset(data, "val")
    assert len(dataset) > 1
    c = config(batch_size=1)
    result = train(tiny, c, data, "byte", tmp_path / "pilot", stop_after=1, eval_max_batches=1)
    assert result["step"] == 1 and result["val_records"] == 1
    assert result["val_full"] is False
    checkpoint = tmp_path / "pilot" / "step-0000001.pt"
    state = load_checkpoint(checkpoint)
    assert state["provenance"]["eval_max_batches_per_rank"] == 1
    model = LanguageModel(tiny)
    model.load_state_dict(state["model"])
    expected = evaluate(model, torch.utils.data.Subset(dataset, [0]), 1, torch.device("cpu"), "fp32")
    assert result["val_loss"] == pytest.approx(expected["val_loss"])
    assert result["val_tokens"] == expected["val_tokens"]
    full = evaluate(model, dataset, 1, torch.device("cpu"), "fp32")
    assert full["val_full"] and full["val_records"] == len(dataset)
    assert full["val_tokens"] > result["val_tokens"]
    with pytest.raises(ValueError, match="positive"):
        train(tiny, c, data, "byte", tmp_path / "invalid", eval_max_batches=0)
    assert not (tmp_path / "invalid").exists()
