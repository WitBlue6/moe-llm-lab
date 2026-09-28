from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("transformers")
from moe_llm.model import LanguageModel
from moe_llm.tokenizer import TextTokenizer
from moe_llm.vision import VisionConfig
from moe_llm.vision_data import create_visual_fixture, prepare_visual
from moe_llm.vision_training import VisualTrainConfig, load_visual_checkpoint


def test_two_rank_visual_alignment_sft_and_resume(tmp_path, tiny):
    tiny = replace(tiny, gradient_checkpointing=True)
    base = tmp_path / "fixture-text-sft.pt"
    torch.save({"format": "moe-lab-checkpoint-v1", "model": LanguageModel(tiny).state_dict(),
                "model_config": asdict(tiny), "train_config": {"stage": "sft"},
                "provenance": {"tokenizer_sha256": TextTokenizer().fingerprint,
                               "artifact_kind": "synthetic_fixture_not_trained"}}, base)
    vc = VisionConfig(encoder_type="fixture", image_token_grid=2, lora_rank=4, lora_alpha=8.)
    vc_path = tmp_path / "vision.json"
    vc_path.write_text(json.dumps(asdict(vc)))
    raw = create_visual_fixture(tmp_path / "raw")
    data = tmp_path / "data"
    manifest = prepare_visual([raw["jsonl"]], raw["image_root"], data, "byte", vc, 128, .25)
    c = VisualTrainConfig(max_steps=4, batch_size=2, grad_accum_steps=2, warmup_steps=0,
                          eval_every=2, save_every=2, device="cpu", precision="fp32", adapter_lr_ratio=1., epochs=1)
    align_config, sft_config = tmp_path / "align.json", tmp_path / "sft.json"
    align_config.write_text(json.dumps(asdict(replace(c, max_steps=2, epochs=None))))
    sft_config.write_text(json.dumps(asdict(replace(c, stage="sft"))))
    from moe_llm.training import resolve_training_budget
    resolved, _ = resolve_training_budget(c, manifest['split_records']['train'], 2)
    end = resolved.max_steps
    command = [sys.executable, "-m", "torch.distributed.run", "--rdzv-backend=c10d",
               "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1", "--rdzv-conf=is_host=true",
               "--nnodes=1", "--nproc-per-node=2", "-m", "moe_llm.cli", "vision-train",
               "--base-checkpoint", str(base), "--vision-config", str(vc_path), "--data", str(data),
               "--eval-max-batches", "1"]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    parent = str(tmp_path / "align/step-0000002.pt")
    runs = [("align", align_config, []), ("full", sft_config, ["--init-from", parent]),
            ("partial", sft_config, ["--init-from", parent, "--stop-after", "1"]),
            ("resumed", sft_config, ["--resume", str(tmp_path / "partial/step-0000001.pt")])]
    for name, config, extra in runs:
        result = subprocess.run([*command, "--train-config", str(config), "--output", str(tmp_path / name), *extra],
                                capture_output=True, text=True, timeout=60, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((tmp_path / name / "summary.json").read_text())
        assert summary["val_records"] == 4 and summary["val_full"] is False
    full = load_visual_checkpoint(tmp_path / f"full/step-{end:07d}.pt")
    resumed = load_visual_checkpoint(tmp_path / f"resumed/step-{end:07d}.pt")
    assert len(full["rng_states"]) == full["world_size"] == 2
    for key in full["adapters"]:
        torch.testing.assert_close(full["adapters"][key], resumed["adapters"][key], rtol=0, atol=0)
    assert any(v.count_nonzero() > 0 for n, v in full["adapters"].items() if n.endswith("lora_B"))
