"""Token-weighted pretraining/SFT, single-device or torchrun DDP.

The first implementation synchronizes every microbatch deliberately: correctness
with dynamically unused experts takes priority over communication optimization.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .data import TokenDataset, collate
from .model import LanguageModel, ModelConfig, causal_loss_sum
from .tokenizer import TextTokenizer, sha256_file


@dataclass(frozen=True)
class TrainConfig:
    stage: str = "pretrain"
    max_steps: int = 1000
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    aux_loss_coef: float = 0.01
    eval_every: int = 100
    save_every: int = 100
    seed: int = 42
    precision: str = "bf16"
    device: str = "auto"
    cpu_threads: int = 1

    def __post_init__(self):
        if self.stage not in ("pretrain", "sft"):
            raise ValueError("only pretrain and sft are implemented")
        for name in ("max_steps", "batch_size", "grad_accum_steps", "eval_every", "save_every", "cpu_threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0 or self.aux_loss_coef < 0:
            raise ValueError("invalid optimizer configuration")
        if not 0 <= self.warmup_steps < self.max_steps or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("invalid warmup or min_lr_ratio")
        if self.precision not in ("fp32", "bf16", "fp16") or self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("unsupported device or precision")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def learning_rate_at(step, c):
    # Step is zero-based; resume uses the original max_steps scheduling horizon.
    if step < c.warmup_steps:
        return c.learning_rate * (step + 1) / c.warmup_steps
    progress = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps - 1)
    return c.learning_rate * (c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def select_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the server driver and locked PyTorch wheel")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
        return device
    if request != "cpu":
        raise ValueError("training supports CPU or CUDA; MPS has not been validated")
    return torch.device(request)


def amp_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed precision is currently supported only on CUDA; use fp32 for CPU/MPS")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bf16")
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def sum_across_ranks(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


class BatchStream:
    def __init__(self, dataset, config, rank, world, epoch=0, cursor=0, collate_fn=collate):
        self.epoch, self.cursor = epoch, 0
        self.sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                          shuffle=True, seed=config.seed, drop_last=False)
        self.loader = DataLoader(dataset, batch_size=config.batch_size, sampler=self.sampler,
                                 collate_fn=collate_fn, num_workers=0,
                                 generator=torch.Generator().manual_seed(config.seed + rank))
        self._reset()
        for _ in range(cursor):
            next(self.iterator)
            self.cursor += 1

    def _reset(self):
        self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.loader)

    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.cursor = 0
            self._reset()
            batch = next(self.iterator)
        self.cursor += 1
        return batch


def rng_state(device):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def load_checkpoint(path):
    # Only tensors and primitive containers are saved; never allow arbitrary pickle objects.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "moe-lab-checkpoint-v1":
        raise ValueError("unsupported checkpoint format")
    return state


def code_fingerprint():
    root = Path(__file__).parent
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_checkpoint(root, model, optimizer, scaler, step, stream, config, provenance,
                     device, rank, world, trained_tokens):
    local_rng = rng_state(device)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    if rank == 0:
        path = root / f"step-{step:07d}.pt"
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix(".pt.tmp")
        with temporary.open("xb") as handle:
            torch.save({"format": "moe-lab-checkpoint-v1", "model": model.state_dict(),
                        "model_config": asdict(model.config), "train_config": asdict(config),
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "epoch": stream.epoch, "cursor": stream.cursor,
                        "rng_states": states, "world_size": world, "provenance": provenance,
                        "trained_tokens": trained_tokens}, handle)
        temporary.rename(path)
    if world > 1:
        dist.barrier()


@torch.inference_mode()
def evaluate(model, dataset, batch_size, device, precision, rank=0, world=1, max_batches=None):
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    model.eval()
    record_count = len(dataset) if max_batches is None else min(len(dataset), max_batches * batch_size * world)
    subset = Subset(dataset, range(rank, record_count, world))
    if rank == 0:
        print(json.dumps({"event": "validation_start", "val_records": record_count,
                          "val_total_records": len(dataset), "eval_max_batches_per_rank": max_batches}), flush=True)
    loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate,
                        generator=torch.Generator().manual_seed(0))
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_context(device, precision):
            output = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, count = causal_loss_sum(output["logits"], batch["labels"])
        totals += torch.stack((loss.double(), count.double()))
    sum_across_ranks(totals)
    if totals[1].item() == 0:
        raise ValueError("validation has no supervised tokens")
    ce = (totals[0] / totals[1]).item()
    model.train()
    return {"val_loss": ce, "val_perplexity": math.exp(min(ce, 80)), "val_tokens": int(totals[1].item()),
            "val_records": record_count, "val_total_records": len(dataset), "val_full": record_count == len(dataset)}


def train(model_config, config, data_path, tokenizer_path, output, init_from=None, resume=None, stop_after=None, eval_max_batches=None):
    if init_from and resume:
        raise ValueError("init_from loads weights; resume restores training. Choose exactly one.")
    if config.stage == "sft" and not (init_from or resume):
        raise ValueError("SFT requires a pretrained --init-from checkpoint (or --resume)")
    if stop_after is not None and not 1 <= stop_after <= config.max_steps:
        raise ValueError("stop_after must lie within the configured schedule")
    if eval_max_batches is not None and eval_max_batches < 1:
        raise ValueError("eval_max_batches must be positive")
    torch.set_num_threads(config.cpu_threads)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    device = select_device(config.device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        return _train(model_config, config, data_path, tokenizer_path, output, init_from, resume,
                      stop_after, device, rank, world, eval_max_batches)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(model_config, config, data_path, tokenizer_path, output, init_from, resume,
           stop_after, device, rank, world, eval_max_batches):
    # Validate precision before allocating a model.
    with amp_context(device, config.precision):
        pass
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    tok = TextTokenizer(tokenizer_path)
    train_data, val_data = TokenDataset(data_path, "train"), TokenDataset(data_path, "val")
    manifest = train_data.manifest
    if manifest["stage"] != config.stage or manifest["tokenizer_sha256"] != tok.fingerprint:
        raise ValueError("dataset stage/tokenizer differs from training configuration")
    if model_config.vocab_size != tok.vocab_size:
        raise ValueError(f"config vocab_size={model_config.vocab_size}, actual tokenizer size={tok.vocab_size}; update config before training")
    if manifest["max_seq_len"] > model_config.max_seq_len:
        raise ValueError("prepared sequences exceed model context length")
    provenance = {"data_sha256": sha256_file(Path(data_path) / "manifest.json"),
                  "tokenizer_sha256": tok.fingerprint, "code_sha256": code_fingerprint(),
                  "git_revision": git_revision(), "torch": str(torch.__version__),
                  "python": platform.python_version(), "device_type": device.type,
                  "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                  "cuda_runtime": torch.version.cuda, "world_size": world,
                  "eval_max_batches_per_rank": eval_max_batches,
                  "init_from": str(init_from) if init_from else None, "resume": str(resume) if resume else None,
                  "parent_sha256": sha256_file(init_from or resume) if (init_from or resume) else None}
    state = load_checkpoint(resume or init_from) if (resume or init_from) else None
    if state:
        source_config = dict(state["model_config"])
        target_config = asdict(model_config)
        source_config.pop("gradient_checkpointing")
        target_config.pop("gradient_checkpointing")
        if source_config != target_config:
            raise ValueError("checkpoint model architecture mismatch")
        if state["provenance"]["tokenizer_sha256"] != tok.fingerprint:
            raise ValueError("checkpoint tokenizer mismatch")
    if resume:
        if state["train_config"] != asdict(config) or state["model_config"] != asdict(model_config):
            raise ValueError("exact resume requires unchanged configs (use --stop-after for interrupted runs)")
        for key in ("data_sha256", "tokenizer_sha256", "code_sha256", "torch", "device_type", "world_size"):
            if state["provenance"][key] != provenance[key]:
                raise ValueError(f"resume provenance mismatch: {key}")
    root = Path(output)
    # All ranks see the same existing-path error before anyone waits at a barrier.
    exists = torch.tensor(int(root.exists()), device=device)
    sum_across_ranks(exists)
    if exists.item():
        raise FileExistsError(f"use a new output directory, including for resume: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        (root / "run.json").write_text(json.dumps({"model": asdict(model_config), "training": asdict(config),
            "provenance": provenance, "parameters": model_config.parameter_counts(), "data": manifest}, indent=2))
        if tok.path != "byte":
            (root / "tokenizer.json").write_bytes(Path(tok.path).read_bytes())
    if world > 1:
        dist.barrier()
    model = LanguageModel(model_config).to(device)
    if state:
        model.load_state_dict(state["model"])
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.precision == "fp16")
    start, epoch, cursor, trained_tokens = 0, 0, 0, 0
    if resume:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, epoch, cursor = state["step"], state["epoch"], state["cursor"]
        trained_tokens = state["trained_tokens"]
    end = stop_after or config.max_steps
    if end <= start:
        raise ValueError("checkpoint has already reached the requested end step")
    stream = BatchStream(train_data, config, rank, world, epoch, cursor)
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                  find_unused_parameters=True, broadcast_buffers=False) if world > 1 else model
    if resume:
        restore_rng(state["rng_states"][rank], device)
    del state
    model.train()
    tokens_at_start = trained_tokens
    wall_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_metrics = {}
    for step in range(start, end):
        batches = [stream.next() for _ in range(config.grad_accum_steps)]
        denominators = torch.tensor([
            sum(int((b["labels"] != -100).sum()) for b in batches),
            sum(int(b["attention_mask"].sum()) for b in batches),
        ], device=device, dtype=torch.float64)
        sum_across_ranks(denominators)
        if denominators.min().item() <= 0:
            raise ValueError("empty token group")
        lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        # CE sum, weighted auxiliary loss, weighted entropy, expert selection fractions.
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        usage = torch.zeros(model_config.num_layers, model_config.num_experts, device=device)
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            count = batch["attention_mask"].sum()
            with amp_context(device, config.precision):
                out = wrapped(batch["input_ids"], attention_mask=batch["attention_mask"])
                ce_sum, _ = causal_loss_sum(out["logits"], batch["labels"])
                # DDP averages gradients, hence the compensating world-size factor.
                loss = ce_sum * (world / denominators[0]).float()
                loss = loss + config.aux_loss_coef * out["aux_loss"] * (world * count / denominators[1]).float()
            scaler.scale(loss).backward()
            totals += torch.stack((ce_sum.detach().double(), out["aux_loss"].detach().double() * count,
                                   out["router_entropy"].double() * count))
            usage += out["expert_usage"] * count
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        finite = torch.tensor(int(torch.isfinite(grad_norm)), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("non-finite gradients; checkpoint not advanced; reduce LR/use bf16 or fp32")
        scaler.step(optimizer)
        scaler.update()
        sum_across_ranks(totals)
        sum_across_ranks(usage)
        trained_tokens += int(denominators[0].item())
        elapsed = time.perf_counter() - wall_start
        last_metrics = {"step": step + 1, "train_loss": (totals[0] / denominators[0]).item(),
                       "aux_loss": (totals[1] / denominators[1]).item(),
                       "router_entropy": (totals[2] / denominators[1]).item(),
                       "expert_usage": (usage / denominators[1]).tolist(), "learning_rate": lr,
                       "grad_norm": grad_norm.item(), "trained_tokens": trained_tokens,
                       "supervised_tokens_this_step": int(denominators[0].item()),
                       "input_tokens_this_step": int(denominators[1].item()),
                       "supervised_tokens_per_second_this_run": (trained_tokens - tokens_at_start) / max(elapsed, 1e-9),
                       "elapsed_seconds_this_run": elapsed, "allocated_gpu_seconds_this_run": elapsed * world if device.type == "cuda" else 0}
        if (step + 1) % config.eval_every == 0 or step + 1 == end:
            last_metrics.update(evaluate(model, val_data, config.batch_size, device, config.precision, rank, world, eval_max_batches))
        if device.type == "cuda":
            memory = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
            if world > 1:
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            last_metrics["max_rank_peak_memory_bytes"] = memory.item()
        if rank == 0:
            with (root / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(last_metrics) + "\n")
            print(json.dumps({k: v for k, v in last_metrics.items() if k != "expert_usage"}), flush=True)
        if (step + 1) % config.save_every == 0 or step + 1 == end:
            write_checkpoint(root, model, optimizer, scaler, step + 1, stream, config, provenance,
                             device, rank, world, trained_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - wall_start
    last_metrics.update({"elapsed_seconds_this_run": elapsed,
                         "allocated_gpu_seconds_this_run": elapsed * world if device.type == "cuda" else 0,
                         "supervised_tokens_per_second_this_run": (trained_tokens - tokens_at_start) / max(elapsed, 1e-9)})
    if rank == 0:
        (root / "summary.json").write_text(json.dumps(last_metrics, indent=2))
    return last_metrics
