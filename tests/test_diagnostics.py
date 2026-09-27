import json
from moe_llm.cli import main


def test_doctor_cli_success_and_configure(tmp_path):
    report = tmp_path / "doctor.json"
    assert main(["doctor", "--output", str(report)]) is None
    assert "cuda_available" in json.loads(report.read_text())
    from pathlib import Path
    config = Path(__file__).resolve().parents[1] / "configs/moe-base.json"
    output = tmp_path / "model.json"
    assert main(["configure", "--model-config", str(config), "--tokenizer", "byte", "--output", str(output)]) is None
    assert json.loads(output.read_text())["vocab_size"] == 262
