"""Read-only environment diagnostics and vocabulary-compatible model config creation."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import torch


def doctor(output=None):
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "torch_cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "cuda_devices": []}
    for number in range(torch.cuda.device_count()):
        with torch.cuda.device(number):
            properties = torch.cuda.get_device_properties(number)
            report["cuda_devices"].append({"index": number, "name": properties.name,
                "memory_bytes": properties.total_memory, "compute_capability": list(torch.cuda.get_device_capability(number)),
                "bf16_supported": torch.cuda.is_bf16_supported()})
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15)
        report["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    if output:
        with Path(output).open("x") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def configure(model_config, tokenizer, output):
    from .model import ModelConfig
    from .tokenizer import TextTokenizer
    configuration = asdict(ModelConfig.load(model_config))
    configuration["vocab_size"] = TextTokenizer(tokenizer).vocab_size
    validated = ModelConfig(**configuration)
    with Path(output).open("x") as handle:
        json.dump(asdict(validated), handle, indent=2)
    print(json.dumps({"model_config": str(output), "parameters": validated.parameter_counts()}, indent=2))
