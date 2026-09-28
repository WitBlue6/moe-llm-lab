from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("transformers")
from moe_llm.model import LanguageModel, causal_loss_sum
from moe_llm.tokenizer import TextTokenizer
from moe_llm.vision import VisionConfig, VisionLanguageModel, generate_visual, FrozenVisionEncoder, ImageProcessor
from moe_llm.vision_data import visual_chat, create_visual_fixture, prepare_visual, VisualDataset, collate_visual


def fixture_config():
    return VisionConfig(encoder_type="fixture", image_token_grid=2, lora_rank=4, lora_alpha=8.)


def batch(tok=None):
    tok = tok or TextTokenizer()
    ids, labels, position = visual_chat(tok, [{"role": "user", "content": "color?"},
                                            {"role": "assistant", "content": "red"}], 4)
    return {"input_ids": torch.tensor([ids[:-1]]), "labels": torch.tensor([labels[1:]]),
            "attention_mask": torch.ones(1, len(ids) - 1, dtype=torch.bool),
            "pixel_values": torch.randn(1, 3, 16, 16), "image_positions": torch.tensor([position])}


def test_inputs_embeds_equivalence_and_gradient(tiny):
    model = LanguageModel(tiny).eval()
    ids = torch.tensor([[1, 10, 11, 12]])
    embeds = model.embedding(ids).detach().requires_grad_()
    torch.testing.assert_close(model(ids)["logits"], model(inputs_embeds=embeds)["logits"], rtol=0, atol=0)
    model(inputs_embeds=embeds)["logits"].square().sum().backward()
    assert embeds.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        model(ids, inputs_embeds=embeds)


@pytest.mark.parametrize("stage", ["align", "sft"])
def test_frozen_base_and_exact_text_retention_after_updates(tiny, stage):
    original = LanguageModel(tiny).eval()
    model = VisionLanguageModel(deepcopy(original), fixture_config())
    model.set_stage(stage)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, weight_decay=.1)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    b = batch()
    labels = b.pop("labels")
    for _ in range(3):
        optimizer.zero_grad()
        out = model(**b)
        loss, count = causal_loss_sum(out["logits"], labels)
        (loss / count + .01 * out["aux_loss"]).backward()
        assert model.projector[1].weight.grad.abs().sum() > 0
        assert all(p.grad is None for n, p in model.named_parameters() if n in frozen)
        optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    model.eval()
    ids = torch.randint(6, 262, (2, 10))
    torch.testing.assert_close(model(ids)["logits"], original(ids)["logits"], rtol=0, atol=0)
    assert torch.equal(generate_visual(model, ids[:1], max_new_tokens=4, temperature=0, eos_id=-1),
                       __import__('moe_llm.model', fromlist=['generate']).generate(original, ids[:1], 4, 0, eos_id=-1))
    model.train()
    assert not model.vision_encoder.training and not model.vision_encoder.encoder.training


def test_lora_flag_survives_checkpoint_recomputation(tiny):
    plain = VisionLanguageModel(LanguageModel(tiny), fixture_config())
    checkpointed = VisionLanguageModel(LanguageModel(replace(tiny, gradient_checkpointing=True)), fixture_config())
    with torch.no_grad():
        for name, p in plain.named_parameters():
            if name.endswith("lora_B"):
                p.normal_(0, .02)
    checkpointed.load_state_dict(plain.state_dict())
    b = batch()
    labels = b.pop("labels")
    for model in (plain, checkpointed):
        model.set_stage("sft")
        model.train()
        out = model(**b)
        ce, count = causal_loss_sum(out["logits"], labels)
        (ce / count + .01 * out["aux_loss"]).backward()
    for (name, p), q in zip(plain.named_parameters(), checkpointed.parameters()):
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)


def test_visual_prefill_and_cached_decode_match_full_context(tiny, monkeypatch):
    model = VisionLanguageModel(LanguageModel(tiny), fixture_config()).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("lora_B"):
                p.normal_(0, .02)
    b = batch()
    b.pop("labels")
    full = model(**b)["logits"]
    first = model(b["input_ids"][:, :-2], pixel_values=b["pixel_values"], image_positions=b["image_positions"], use_cache=True)
    last = model(b["input_ids"][:, -2:], past_key_values=first["past_key_values"], visual_context=True, use_cache=True)
    torch.testing.assert_close(last["logits"], full[:, -2:], atol=2e-6, rtol=1e-5)
    calls = []
    original = model.vision_encoder.forward
    def record(x):
        calls.append(1)
        return original(x)
    monkeypatch.setattr(model.vision_encoder, "forward", record)
    result = generate_visual(model, b["input_ids"], b["pixel_values"], b["image_positions"],
                             max_new_tokens=3, temperature=0, eos_id=-1)
    assert len(calls) == 1 and result.shape[1] == b["input_ids"].shape[1] + 3
    with pytest.raises(ValueError):
        model(b["input_ids"][:, -1:], pixel_values=b["pixel_values"], image_positions=b["image_positions"],
              past_key_values=first["past_key_values"])


def test_image_positions_and_masks_are_validated(tiny):
    model = VisionLanguageModel(LanguageModel(tiny), fixture_config())
    b = batch()
    b.pop("labels")
    b["attention_mask"][0, int(b["image_positions"][0])] = False
    with pytest.raises(ValueError, match="visible"):
        model(**b)
    with pytest.raises(ValueError, match="existing visual cache"):
        model(b["input_ids"], visual_context=True)


def test_image_group_split_mask_and_integrity(tmp_path):
    raw = tmp_path / "raw"
    result = create_visual_fixture(raw)
    rows = [json.loads(line) for line in Path(result["jsonl"]).read_text().splitlines()]
    more = deepcopy(rows[0])
    more["messages"][-1]["content"] = "Another description of the same image."
    with Path(result["jsonl"]).open("a") as handle:
        handle.write(json.dumps(more) + '\n' + json.dumps(rows[0]) + '\n')
    prepared = tmp_path / "prepared"
    config = fixture_config()
    report = prepare_visual([result["jsonl"]], result["image_root"], prepared, "byte", config, 128, .25)
    assert report["stats"]["duplicates"] == 1
    split_images = []
    for split in ("train", "val"):
        split_images.append({json.loads(line)["image"] for line in (prepared / f"{split}.jsonl").read_text().splitlines()})
    assert not split_images[0] & split_images[1]
    dataset = VisualDataset(prepared, "train", config)
    row = dataset[0]
    position = row["image_position"]
    assert row["labels"][position - 1:position + 4].eq(-100).all()
    assert row["input_ids"][position:position + 4].eq(0).all()
    b = collate_visual([row, dataset[-1]])
    assert b["attention_mask"][0, position:position + 4].all()
    assert b["labels"][~b["attention_mask"]].eq(-100).all()
    image = Path(result["image_root"]) / rows[0]["image"]
    image.write_bytes(image.read_bytes() + b'changed')
    with pytest.raises(ValueError, match="fingerprint"):
        VisualDataset(prepared, "train", config)


def test_local_siglip_loader_processor_and_patch_pooling(tmp_path):
    from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor
    from PIL import Image
    root = tmp_path / "encoder"
    encoder = SiglipVisionModel(SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, image_size=16, patch_size=4))
    encoder.save_pretrained(root, safe_serialization=True)
    SiglipImageProcessor(size={"height": 16, "width": 16}).save_pretrained(root)
    config = VisionConfig(encoder_path=str(root), image_token_grid=2)
    image = tmp_path / "test.png"
    Image.new("RGB", (23, 19), "red").save(image)
    pixels = ImageProcessor(config)(image)
    model = FrozenVisionEncoder(config)
    features = model(pixels.unsqueeze(0))
    assert features.shape == (1, 4, 16) and not features.requires_grad
    reference = encoder.vision_model.state_dict()
    for name, value in model.encoder.state_dict().items():
        torch.testing.assert_close(reference[name], value, atol=0, rtol=0)
    expected = encoder(pixels.unsqueeze(0)).last_hidden_state.transpose(1, 2).reshape(1, 16, 4, 4)
    expected = torch.nn.functional.adaptive_avg_pool2d(expected, 2).flatten(2).transpose(1, 2)
    torch.testing.assert_close(features, expected, atol=2e-6, rtol=2e-5)
