from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from moe_llm.siglip import SiglipConfig, NativeSiglipVision, NativeImageProcessor, ContrastiveSiglip, sigmoid_pair_loss


def small_config():
    return SiglipConfig(image_size=16, patch_size=4, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=4)


def test_native_features_and_processing_match_reference(tmp_path):
    pytest.importorskip('transformers')
    from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor
    from PIL import Image
    import numpy as np
    c = small_config()
    hf = SiglipVisionModel(SiglipVisionConfig(**asdict(c))).eval()
    hf.save_pretrained(tmp_path)
    ours = NativeSiglipVision.from_local(tmp_path).eval()
    pixels = torch.randn(2, 3, 16, 16)
    torch.testing.assert_close(ours(pixels), hf(pixels).last_hidden_state, atol=2e-6, rtol=2e-5)
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (27, 31, 3), dtype=np.uint8))
    processor = SiglipImageProcessor(size={'height': 16, 'width': 16})
    torch.testing.assert_close(NativeImageProcessor(16)(image), processor(image, return_tensors='pt')['pixel_values'][0], atol=1e-6, rtol=1e-6)
    # Real official configs omit all the default dimension fields.
    p = tmp_path / 'sparse.json'
    p.write_text(json.dumps({'model_type': 'siglip', 'vision_config': {'patch_size': 16}}))
    actual = SiglipConfig.load(p)
    assert (actual.hidden_size, actual.num_hidden_layers, actual.num_attention_heads) == (768, 12, 12)


def test_loss_matches_pairwise_definition_and_has_gradients():
    image = torch.randn(3, 5, requires_grad=True)
    text = torch.randn(3, 5, requires_grad=True)
    scale, bias = torch.tensor(2., requires_grad=True), torch.tensor(-1., requires_grad=True)
    loss = sigmoid_pair_loss(image, text, scale, bias)
    expected = sum(F.softplus((-1 if i == j else 1) * (scale * image[i].dot(text[j]) + bias))
                   for i in range(3) for j in range(3)) / 3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for x in (image, text, scale, bias):
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    aligned = torch.eye(4)
    assert sigmoid_pair_loss(aligned, aligned, torch.tensor(10.), torch.tensor(-5.)) < sigmoid_pair_loss(aligned, aligned.roll(1, 0), torch.tensor(10.), torch.tensor(-5.))


def test_text_padding_does_not_change_embedding():
    model = ContrastiveSiglip(small_config(), 262, text_length=12, projection_size=8, text_layers=1).eval()
    pixels = torch.randn(1, 3, 16, 16)
    a = model(pixels, torch.tensor([[1, 7, 2]]), torch.ones(1, 3, dtype=torch.bool))[1]
    b = model(pixels, torch.tensor([[1, 7, 2, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool))[1]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_scratch_training_resume_export_and_frozen_vlm(tmp_path, tiny):
    from moe_llm.siglip_training import prepare_pairs, train_pairs, export_encoder
    from moe_llm.vision_data import create_visual_fixture, prepare_visual
    from moe_llm.vision import VisionConfig, VisionLanguageModel
    from moe_llm.model import LanguageModel
    fixture = create_visual_fixture(tmp_path / 'fixture')
    rows = [json.loads(x) for x in Path(fixture['jsonl']).read_text().splitlines()]
    captions = tmp_path / 'captions.jsonl'
    captions.write_text(''.join(json.dumps({'image': x['image'], 'caption': x['messages'][-1]['content']}) + '\n' for x in rows))
    data = tmp_path / 'pairs'
    prepare_pairs(captions, fixture['image_root'], data, 'byte', 32, 42, .3)
    c = {'vision': asdict(small_config()), 'text_layers': 1, 'projection_size': 8,
         'max_steps': 2, 'batch_size': 2, 'learning_rate': .001, 'warmup_steps': 0,
         'eval_every': 2, 'save_every': 2, 'eval_samples': 4, 'precision': 'fp32', 'device': 'cpu', 'seed': 42}
    train_pairs(c, data, tmp_path / 'full', 'byte')
    train_pairs(c, data, tmp_path / 'part', 'byte', stop_after=1)
    train_pairs(c, data, tmp_path / 'resume', 'byte', resume=tmp_path / 'part/step-0000001.pt')
    full = torch.load(tmp_path / 'full/step-0000002.pt', weights_only=True)
    resumed = torch.load(tmp_path / 'resume/step-0000002.pt', weights_only=True)
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
    exported = tmp_path / 'export'
    export_encoder(tmp_path / 'full/step-0000002.pt', exported)
    encoder = NativeSiglipVision.from_local(exported)
    for key, v in encoder.state_dict().items():
        torch.testing.assert_close(v, full['model']['vision.' + key], atol=0, rtol=0)
    vc = VisionConfig(encoder_path=str(exported), image_token_grid=2)
    model = LanguageModel(tiny).eval()
    ids = torch.tensor([[1, 7, 2]])
    before = model(ids)['logits'].detach()
    vlm = VisionLanguageModel(model, vc).eval()
    torch.testing.assert_close(vlm(ids)['logits'], before, atol=0, rtol=0)
    assert all(not p.requires_grad for p in vlm.vision_encoder.parameters())
    prepared = tmp_path / 'vlm-data'
    prepare_visual([fixture['jsonl']], fixture['image_root'], prepared, 'byte', vc, 128, .3, 42)
    visual = {k: {json.loads(x)['image'] for x in (prepared / f'{k}.jsonl').read_text().splitlines()} for k in ('train', 'val')}
    pairs = {k: {json.loads(x)['image'] for x in (data / f'{k}.jsonl').read_text().splitlines()} for k in ('train', 'val')}
    assert pairs == visual
