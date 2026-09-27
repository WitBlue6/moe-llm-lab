import os
from pathlib import Path
import subprocess
import sys


def test_two_rank_token_normalization_and_dynamic_moe(tmp_path):
    worker = Path(__file__).with_name("ddp_worker.py")
    report = tmp_path / "ddp-result.txt"
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0",
                             "--local-addr=127.0.0.1", "--rdzv-conf=is_host=true",
                             "--nnodes=1", "--nproc-per-node=2", str(worker), str(report)],
                            capture_output=True, text=True, timeout=60,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo0"}
                            if sys.platform == "darwin" else {**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "match" in report.read_text()


def test_ddp_training_checkpoint_and_resume(tmp_path):
    import json
    import torch
    from moe_llm.data import prepare
    from moe_llm.training import load_checkpoint
    project = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    prepare([project / "fixtures/pretrain.jsonl"], data, "byte", "pretrain", 64, .25)
    config = json.loads((project / "configs/train-smoke-pretrain.json").read_text())
    config.update(max_steps=4, save_every=2, eval_every=2)
    train_config = tmp_path / "train.json"
    train_config.write_text(json.dumps(config))
    command = [sys.executable, "-m", "torch.distributed.run", "--rdzv-backend=c10d",
               "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1", "--rdzv-conf=is_host=true",
               "--nnodes=1", "--nproc-per-node=2", "-m", "moe_llm.cli", "train",
               "--model-config", str(project / "configs/moe-tiny.json"),
               "--train-config", str(train_config), "--data", str(data)]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    for name, extra in [("full", []), ("partial", ["--stop-after", "2"]),
                        ("resumed", ["--resume", str(tmp_path / "partial/step-0000002.pt")])]:
        result = subprocess.run([*command, "--output", str(tmp_path / name), *extra],
                                capture_output=True, text=True, timeout=60, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
    full = load_checkpoint(tmp_path / "full/step-0000004.pt")
    resumed = load_checkpoint(tmp_path / "resumed/step-0000004.pt")
    assert full["world_size"] == len(full["rng_states"]) == 2
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)
    assert full["trained_tokens"] == resumed["trained_tokens"]
