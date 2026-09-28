"""Projector alignment and visual-LoRA SFT with immutable text/vision backbones."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from .model import LanguageModel, ModelConfig, causal_loss_sum
from .tokenizer import TextTokenizer, sha256_file
from .training import (TrainConfig, BatchStream, select_device, amp_context, learning_rate_at,
                       sum_across_ranks, rng_state, restore_rng, load_checkpoint, code_fingerprint, git_revision,
                       resolve_training_budget)
from .vision import VisionConfig, VisionLanguageModel, encoder_fingerprint
from .vision_data import VisualDataset, collate_visual


@dataclass(frozen=True)
class VisualTrainConfig(TrainConfig):
    stage: str = "align"
    learning_rate: float = 1e-3  # projector LR; LoRA uses adapter_lr_ratio times this.
    adapter_lr_ratio: float = .1

    def __post_init__(self):
        common = asdict(self)
        common.pop("adapter_lr_ratio")
        common["stage"] = "sft"
        TrainConfig(**common)
        if self.stage not in ("align", "sft") or not 0 < self.adapter_lr_ratio <= 1:
            raise ValueError("visual stage must be align/sft and adapter_lr_ratio within (0,1]")


def load_visual_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "moe-lab-vision-v1":
        raise ValueError("not a visual adapter checkpoint")
    return state


def build_visual_model(base_checkpoint, vision_config, tokenizer, state=None):
    base = load_checkpoint(base_checkpoint)
    if base["train_config"]["stage"] != "sft":
        raise ValueError("visual training requires a text SFT checkpoint")
    if base["provenance"]["tokenizer_sha256"] != tokenizer.fingerprint:
        raise ValueError("base checkpoint tokenizer mismatch")
    config = ModelConfig(**base["model_config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("base vocabulary differs from tokenizer")
    base_hash = sha256_file(base_checkpoint)
    vision_hash = encoder_fingerprint(vision_config)
    if state:
        if state["base_sha256"] != base_hash or state["tokenizer_sha256"] != tokenizer.fingerprint:
            raise ValueError("visual adapter base checkpoint/tokenizer mismatch")
        if state["encoder_fingerprint"] != vision_hash:
            raise ValueError("visual encoder fingerprint mismatch")
        old_config, new_config = dict(state["vision_config"]), asdict(vision_config)
        # The same frozen encoder may be relocated without changing its content.
        old_config.pop("encoder_path")
        new_config.pop("encoder_path")
        old_config["lora_targets"] = tuple(old_config["lora_targets"])
        if old_config != new_config or state["model_config"] != asdict(config):
            raise ValueError("visual adapter architecture mismatch")
    llm = LanguageModel(config)
    llm.load_state_dict(base["model"])
    del base
    model = VisionLanguageModel(llm, vision_config)
    if state:
        model.load_adapter_state_dict(state["adapters"])
    identity = {"base_sha256": base_hash, "encoder_fingerprint": vision_hash,
                "tokenizer_sha256": tokenizer.fingerprint}
    return model, identity


def load_visual_model(base_checkpoint, checkpoint, tokenizer_path, device, encoder_path=None):
    state = load_visual_checkpoint(checkpoint)
    configuration = dict(state["vision_config"])
    if encoder_path is not None:
        configuration["encoder_path"] = str(encoder_path)
    tokenizer = TextTokenizer(tokenizer_path)
    model, _ = build_visual_model(base_checkpoint, VisionConfig(**configuration), tokenizer, state)
    model.set_stage(state["train_config"]["stage"])
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def evaluate_visual(model, dataset, batch_size, device, precision, rank=0, world=1, zero_images=False, max_batches=None):
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    record_count = len(dataset) if max_batches is None else min(len(dataset), max_batches * batch_size * world)
    if rank == 0:
        print(json.dumps({"event": "validation_start", "val_records": record_count,
                          "val_total_records": len(dataset), "eval_max_batches_per_rank": max_batches}), flush=True)
    model.eval()
    loader = DataLoader(Subset(dataset, range(rank, record_count, world)), batch_size=batch_size,
                        collate_fn=collate_visual, generator=torch.Generator().manual_seed(0))
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        if zero_images:
            batch["pixel_values"] = torch.zeros_like(batch["pixel_values"])
        with amp_context(device, precision):
            out = model(**batch)
            loss, count = causal_loss_sum(out["logits"], labels)
        totals += torch.stack((loss.double(), count.double()))
    sum_across_ranks(totals)
    if totals[1] <= 0:
        raise ValueError("visual validation has no supervised tokens")
    ce = (totals[0] / totals[1]).item()
    model.train()
    prefix = "zero_image" if zero_images else "val"
    return {f"{prefix}_loss": ce, f"{prefix}_perplexity": math.exp(min(ce, 80)),
            f"{prefix}_tokens": int(totals[1].item()), f"{prefix}_records": record_count,
            f"{prefix}_total_records": len(dataset), f"{prefix}_full": record_count == len(dataset)}


def save_visual(root, model, config, identity, provenance, optimizer, scaler, step, stream,
                trained_tokens, device, rank, world):
    states = [None] * world
    local = rng_state(device)
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        target = root / f"step-{step:07d}.pt"
        if target.exists():
            raise FileExistsError(target)
        temporary = target.with_suffix(".pt.tmp")
        with temporary.open("xb") as handle:
            torch.save({"format": "moe-lab-vision-v1", "adapters": model.adapter_state_dict(),
                        "model_config": asdict(model.config), "vision_config": asdict(model.vision_config),
                        "train_config": asdict(config), **identity, "provenance": provenance,
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "epoch": stream.epoch, "cursor": stream.cursor,
                        "trained_tokens": trained_tokens, "rng_states": states, "world_size": world}, handle)
        temporary.rename(target)
    if world > 1:
        dist.barrier()


def train_visual(base_checkpoint, vision_config, config, data_path, tokenizer_path, output,
                 init_from=None, resume=None, stop_after=None, image_root=None, eval_max_batches=None):
    if init_from and resume:
        raise ValueError("choose init_from or resume")
    if config.stage == "sft" and not (init_from or resume):
        raise ValueError("visual SFT requires an alignment adapter via --init-from")
    if stop_after is not None and stop_after < 1:
        raise ValueError("invalid stop_after")
    if eval_max_batches is not None and eval_max_batches < 1:
        raise ValueError("eval_max_batches must be positive")
    torch.set_num_threads(config.cpu_threads)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    device = select_device(config.device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        return _train_visual(base_checkpoint, vision_config, config, data_path, tokenizer_path, output,
                             init_from, resume, stop_after, image_root, device, rank, world, eval_max_batches)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train_visual(base_checkpoint, vc, c, data_path, tokenizer_path, output, init_from, resume,
                  stop_after, image_root, device, rank, world, eval_max_batches):
    with amp_context(device, c.precision):
        pass
    random.seed(c.seed)
    torch.manual_seed(c.seed)
    tok = TextTokenizer(tokenizer_path)
    state = load_visual_checkpoint(resume or init_from) if (resume or init_from) else None
    model, identity = build_visual_model(base_checkpoint, vc, tok, state)
    model.set_stage(c.stage)
    model.to(device)
    train_data = VisualDataset(data_path, "train", vc, image_root)
    val_data = VisualDataset(data_path, "val", vc, image_root)
    manifest = train_data.manifest
    c, steps_per_epoch = resolve_training_budget(c, len(train_data), world)
    if stop_after is not None and stop_after > c.max_steps:
        raise ValueError("stop_after must lie within the configured schedule")
    if rank == 0:
        print(json.dumps({"event": "training_budget", "epochs": c.epochs,
                          "max_steps": c.max_steps, "steps_per_epoch": steps_per_epoch}), flush=True)
    if manifest["tokenizer_sha256"] != tok.fingerprint or manifest["max_seq_len"] > model.config.max_seq_len:
        raise ValueError("visual data tokenizer or context limit mismatch")
    import transformers
    import PIL
    provenance = {"data_sha256": sha256_file(Path(data_path) / "manifest.json"),
                  "code_sha256": code_fingerprint(), "git_revision": git_revision(),
                  "torch": str(torch.__version__), "transformers": transformers.__version__,
                  "pillow": PIL.__version__, "python": platform.python_version(),
                  "device_type": device.type, "world_size": world, "cuda_runtime": torch.version.cuda,
                  "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                  "base_checkpoint": str(base_checkpoint), "parent": str(resume or init_from) if state else None,
                  "parent_sha256": sha256_file(resume or init_from) if state else None,
                  "fixture_vision": vc.encoder_type == "fixture", "eval_max_batches_per_rank": eval_max_batches}
    if resume:
        if state["train_config"] != asdict(c):
            raise ValueError("exact visual resume requires unchanged train config")
        for key in ("data_sha256", "code_sha256", "torch", "transformers", "pillow", "device_type", "world_size"):
            if state["provenance"][key] != provenance[key]:
                raise ValueError(f"visual resume provenance mismatch: {key}")
    root = Path(output)
    existing = torch.tensor(int(root.exists()), device=device)
    sum_across_ranks(existing)
    if existing.item():
        raise FileExistsError(f"use a new output directory: {root}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        (root / "run.json").write_text(json.dumps({"model": asdict(model.config), "vision": asdict(vc),
            "training": asdict(c), "identity": identity, "provenance": provenance, "data": manifest,
            "trainable_parameters": trainable, "total_parameters": sum(p.numel() for p in model.parameters())}, indent=2))
        if tok.path != "byte":
            (root / "tokenizer.json").write_bytes(Path(tok.path).read_bytes())
    if world > 1:
        dist.barrier()
    # Frozen parameters are excluded entirely, including weight decay and optimizer state.
    groups = []
    for is_lora in (False, True):
        for decay in (False, True):
            parameters = [p for name, p in model.named_parameters() if p.requires_grad
                          and name.endswith(("lora_A", "lora_B")) == is_lora and (p.ndim >= 2) == decay]
            if parameters:
                groups.append({"params": parameters, "weight_decay": c.weight_decay if decay else 0.,
                               "lr_scale": c.adapter_lr_ratio if is_lora else 1.})
    optimizer = torch.optim.AdamW(groups, lr=c.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and c.precision == "fp16")
    start, epoch, cursor, trained_tokens = 0, 0, 0, 0
    if resume:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, epoch, cursor = state["step"], state["epoch"], state["cursor"]
        trained_tokens = state["trained_tokens"]
    end = stop_after or c.max_steps
    if end <= start:
        raise ValueError("checkpoint already reached the requested step")
    stream = BatchStream(train_data, c, rank, world, epoch, cursor, collate_visual)
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                  find_unused_parameters=True, broadcast_buffers=False) if world > 1 else model
    if resume:
        restore_rng(state["rng_states"][rank], device)
    del state
    model.train()
    wall_start, tokens_at_start = time.perf_counter(), trained_tokens
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = {}
    for step in range(start, end):
        batches = stream.next_group(c.grad_accum_steps, finish_epoch=c.epochs is not None)
        denominators = torch.tensor([sum(int((b["labels"] != -100).sum()) for b in batches),
                                     sum(int(b["attention_mask"].sum()) for b in batches)],
                                    dtype=torch.float64, device=device)
        sum_across_ranks(denominators)
        if denominators.min() <= 0:
            raise ValueError("empty visual supervision")
        lr = learning_rate_at(step, c)
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        usage = torch.zeros(model.config.num_layers, model.config.num_experts, device=device)
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            count = batch["attention_mask"].sum()
            with amp_context(device, c.precision):
                out = wrapped(**batch)
                ce, _ = causal_loss_sum(out["logits"], labels)
                loss = ce * (world / denominators[0]).float()
                loss += c.aux_loss_coef * out["aux_loss"] * (world * count / denominators[1]).float()
            scaler.scale(loss).backward()
            totals += torch.stack((ce.detach().double(), out["aux_loss"].detach().double() * count,
                                   out["router_entropy"].double() * count))
            usage += out["expert_usage"] * count
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], c.grad_clip)
        finite = torch.tensor(int(torch.isfinite(grad_norm)), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("nonfinite visual gradients; checkpoint not advanced")
        scaler.step(optimizer)
        scaler.update()
        sum_across_ranks(totals)
        sum_across_ranks(usage)
        trained_tokens += int(denominators[0].item())
        metrics = {"step": step + 1, "epochs_completed": stream.epoch + stream.cursor / len(stream.loader),
                   "train_loss": (totals[0] / denominators[0]).item(),
                   "aux_loss": (totals[1] / denominators[1]).item(), "router_entropy": (totals[2] / denominators[1]).item(),
                   "expert_usage": (usage / denominators[1]).tolist(), "projector_learning_rate": lr,
                   "lora_learning_rate": lr * c.adapter_lr_ratio if c.stage == "sft" else 0.,
                   "grad_norm": grad_norm.item(), "trained_tokens": trained_tokens,
                   "supervised_tokens_this_step": int(denominators[0].item()), "input_tokens_this_step": int(denominators[1].item())}
        if (step + 1) % c.eval_every == 0 or step + 1 == end:
            metrics.update(evaluate_visual(model, val_data, c.batch_size, device, c.precision, rank, world,
                                           max_batches=eval_max_batches))
        if device.type == "cuda":
            memory = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
            if world > 1:
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            metrics["max_rank_peak_memory_bytes"] = memory.item()
        elapsed = time.perf_counter() - wall_start
        metrics.update(elapsed_seconds_this_run=elapsed, allocated_gpu_seconds_this_run=elapsed * world if device.type == "cuda" else 0.,
                       supervised_tokens_per_second_this_run=(trained_tokens - tokens_at_start) / max(elapsed, 1e-9))
        if rank == 0:
            with (root / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(json.dumps({k: v for k, v in metrics.items() if k != "expert_usage"}), flush=True)
        if (step + 1) % c.save_every == 0 or step + 1 == end:
            save_visual(root, model, c, identity, provenance, optimizer, scaler, step + 1,
                        stream, trained_tokens, device, rank, world)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - wall_start
    metrics.update(elapsed_seconds_this_run=elapsed, allocated_gpu_seconds_this_run=elapsed * world if device.type == "cuda" else 0.,
                   supervised_tokens_per_second_this_run=(trained_tokens - tokens_at_start) / max(elapsed, 1e-9))
    if rank == 0:
        (root / "summary.json").write_text(json.dumps(metrics, indent=2))
    return metrics
