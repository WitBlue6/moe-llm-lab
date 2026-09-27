"""Compatibility wrapper; prefer `moe-lab inspect --model-config ...`."""
from pathlib import Path
from moe_llm.cli import main

if __name__ == "__main__":
    main(["inspect", "--model-config", str(Path(__file__).resolve().parents[1] / "configs/moe-base.json")])
