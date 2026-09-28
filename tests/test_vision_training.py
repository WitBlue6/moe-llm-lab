from dataclasses import asdict, replace
from pathlib import Path
import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("transformers")
from moe_llm.model import LanguageModel
from moe_llm.tokenizer import TextTokenizer
from moe_llm.vision import VisionConfig
from moe_llm.vision_data import create_visual_fixture, prepare_visual
from moe_llm.vision_training import VisualTrainConfig, train_visual, load_visual_checkpoint, load_visual_model
from moe_llm.vision_cli import check_text_retention
from moe_llm.data import prepare


def make_base(path, config):
    # Explicit synthetic checkpoint for engineering tests, not a trained SFT model.
    torch.save({"format": "moe-lab-checkpoint-v1", "model": LanguageModel(config).state_dict(),
                "model_config": asdict(config), "train_config": {"stage": "sft"},
                "provenance": {"tokenizer_sha256": TextTokenizer().fingerprint,
                               "artifact_kind": "synthetic_fixture_not_trained"}}, path)


def test_visual_alignment_sft_resume_and_text_retention(tmp_path, tiny):
    base = tmp_path / "synthetic-text-sft.pt"
    tiny = replace(tiny, gradient_checkpointing=True)
    make_base(base, tiny)
    vc = VisionConfig(encoder_type="fixture", image_token_grid=2, lora_rank=4, lora_alpha=8.)
    raw = create_visual_fixture(tmp_path / "images")
    data = tmp_path / "data"
    prepare_visual([raw["jsonl"]], raw["image_root"], data, "byte", vc, 128, .25)
    c = VisualTrainConfig(max_steps=4, warmup_steps=0, batch_size=2, grad_accum_steps=2,
                          learning_rate=.003, device="cpu", precision="fp32", eval_every=2, save_every=2)
    align = tmp_path / "align"
    train_visual(base, vc, c, data, "byte", align)
    parent = align / "step-0000004.pt"
    c = replace(c, stage="sft", adapter_lr_ratio=1.)
    full, partial, resumed = [tmp_path / name for name in ("full", "partial", "resumed")]
    train_visual(base, vc, c, data, "byte", full, init_from=parent)
    train_visual(base, vc, c, data, "byte", partial, init_from=parent, stop_after=2)
    train_visual(base, vc, c, data, "byte", resumed, resume=partial / "step-0000002.pt")
    a = load_visual_checkpoint(full / "step-0000004.pt")
    b = load_visual_checkpoint(resumed / "step-0000004.pt")
    for name in a["adapters"]:
        torch.testing.assert_close(a["adapters"][name], b["adapters"][name], atol=0, rtol=0)
    assert any(v.count_nonzero() > 0 for n, v in a["adapters"].items() if n.endswith("lora_B"))
    assert not any("experts." in key or "base.weight" in key for key in a["adapters"])
    model, tok = load_visual_model(base, full / "step-0000004.pt", "byte", torch.device("cpu"))
    text_data = tmp_path / "text-data"
    fixtures = Path(__file__).resolve().parents[1] / "fixtures/sft.jsonl"
    prepare([fixtures], text_data, "byte", "sft", 64, .25)
    report = check_text_retention(model, base, text_data, tok, torch.device("cpu"))
    assert report["text_logits_identical"] and report["text_max_logit_error"] == 0.
    with pytest.raises(ValueError, match="unchanged"):
        train_visual(base, vc, replace(c, learning_rate=.1), data, "byte", tmp_path / "bad-resume", resume=partial / "step-0000002.pt")
    # The same architecture with different base weights must not load this adapter.
    other = tmp_path / "different-base.pt"
    make_base(other, tiny)
    with pytest.raises(ValueError, match="base checkpoint"):
        load_visual_model(other, full / "step-0000004.pt", "byte", torch.device("cpu"))


def test_visual_eval_limit_and_full_default(tmp_path, tiny):
    from moe_llm.vision_training import evaluate_visual
    from moe_llm.vision_data import VisualDataset
    base = tmp_path / "base.pt"
    make_base(base, tiny)
    vc = VisionConfig(encoder_type="fixture", image_token_grid=2)
    raw = create_visual_fixture(tmp_path / "raw")
    data = tmp_path / "data"
    prepare_visual([raw["jsonl"]], raw["image_root"], data, "byte", vc, 128, .25)
    c = VisualTrainConfig(max_steps=2, warmup_steps=0, batch_size=1, grad_accum_steps=1,
                          device="cpu", precision="fp32", eval_every=1, save_every=2)
    output = tmp_path / "limited"
    result = train_visual(base, vc, c, data, "byte", output, stop_after=1, eval_max_batches=1)
    assert result["val_records"] == 1 and result["val_full"] is False
    checkpoint = output / "step-0000001.pt"
    assert load_visual_checkpoint(checkpoint)["provenance"]["eval_max_batches_per_rank"] == 1
    model, _ = load_visual_model(base, checkpoint, "byte", torch.device("cpu"))
    dataset = VisualDataset(data, "val", vc)
    expected = evaluate_visual(model, torch.utils.data.Subset(dataset, [0]), 1, torch.device("cpu"), "fp32")
    assert result["val_loss"] == pytest.approx(expected["val_loss"])
    assert result["val_tokens"] == expected["val_tokens"]
    full = evaluate_visual(model, dataset, 1, torch.device("cpu"), "fp32")
    assert full["val_full"] and full["val_records"] == len(dataset)
    assert full["val_tokens"] > result["val_tokens"]
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            train_visual(base, vc, c, data, "byte", tmp_path / "invalid", eval_max_batches=invalid)
    assert not (tmp_path / "invalid").exists()
    # One epoch smaller than the accumulation window must stop at its boundary.
    c = replace(c, epochs=1, grad_accum_steps=100)
    result = train_visual(base, vc, c, data, "byte", tmp_path / "epoch", eval_max_batches=1)
    assert result['step'] == 1 and result['epochs_completed'] == 1
    trained = load_visual_checkpoint(tmp_path / 'epoch/step-0000001.pt')
    assert trained['train_config']['max_steps'] == trained['train_config']['epochs'] == 1
