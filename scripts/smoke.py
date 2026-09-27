"""Reproducible CLI integration check; all data/models are synthetic and tiny."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="new directory for test runs/logs")
    parser.add_argument("--vision", action="store_true", help="requires uv --extra vision")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    def run(name, *arguments):
        path = root / f"{name}.log"
        with path.open("x") as log:
            result = subprocess.run([sys.executable, "-m", "moe_llm.cli", *map(str, arguments)],
                                    cwd=project, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed: see {path}")
        print(f"PASS {name}", flush=True)
    run("prepare-text", "prepare", "--input", project / "fixtures/pretrain.jsonl", "--output", root / "text-data",
        "--stage", "pretrain", "--max-seq-len", 64, "--val-ratio", .25)
    run("pretrain", "train", "--model-config", project / "configs/moe-tiny.json",
        "--train-config", project / "configs/train-smoke-pretrain.json", "--data", root / "text-data", "--output", root / "pretrain")
    run("prepare-sft", "prepare", "--input", project / "fixtures/sft.jsonl", "--output", root / "sft-data",
        "--stage", "sft", "--max-seq-len", 64, "--val-ratio", .25)
    run("text-sft", "train", "--model-config", project / "configs/moe-tiny.json",
        "--train-config", project / "configs/train-smoke-sft.json", "--data", root / "sft-data",
        "--init-from", root / "pretrain/step-0000008.pt", "--output", root / "text-sft")
    base = root / "text-sft/step-0000004.pt"
    run("text-generation", "generate", "--checkpoint", base, "--chat", "--prompt", "2+1=?", "--temperature", 0, "--max-new-tokens", 8)
    if args.vision:
        run("make-images", "vision-fixture", "--output", root / "visual-raw")
        run("prepare-vision", "vision-prepare", "--input", root / "visual-raw/conversations.jsonl",
            "--image-root", root / "visual-raw/images", "--output", root / "visual-data",
            "--vision-config", project / "configs/vision-fixture.json", "--max-seq-len", 128, "--val-ratio", .25)
        run("visual-align", "vision-train", "--base-checkpoint", base,
            "--vision-config", project / "configs/vision-fixture.json", "--train-config", project / "configs/train-vision-smoke-align.json",
            "--data", root / "visual-data", "--output", root / "visual-align")
        run("visual-sft", "vision-train", "--base-checkpoint", base,
            "--vision-config", project / "configs/vision-fixture.json", "--train-config", project / "configs/train-vision-smoke-sft.json",
            "--data", root / "visual-data", "--init-from", root / "visual-align/step-0000004.pt", "--output", root / "visual-sft")
        adapter = root / "visual-sft/step-0000004.pt"
        run("visual-evaluation", "vision-evaluate", "--base-checkpoint", base, "--checkpoint", adapter,
            "--data", root / "visual-data", "--text-data", root / "sft-data", "--zero-images", "--output", root / "visual-evaluation.json")
        result = json.loads((root / "visual-evaluation.json").read_text())
        if not result["text_logits_identical"]:
            raise AssertionError("pure-text logits changed")
        run("visual-generation", "vision-generate", "--base-checkpoint", base, "--checkpoint", adapter,
            "--image", root / "visual-raw/images/color-00.png", "--prompt", "Describe the image.", "--temperature", 0, "--max-new-tokens", 8)
    (root / "SMOKE_ONLY.txt").write_text("Synthetic engineering verification only. Not a capable pretrained model or benchmark.\n")
    print(f"All smoke checks passed. Artifacts: {root}")


if __name__ == "__main__":
    main()
