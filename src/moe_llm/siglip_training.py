"""Scratch image/text sigmoid pretraining, deterministic image split and export.

Usage: uv run --locked --extra vision python -m moe_llm.siglip_training --help
DDP uses differentiable all-gather of text features; each local image sees all
ranks' captions as negatives. No gradient accumulation: it would not enlarge
the contrastive negative pool. Eval uses a fixed, explicitly sized candidate set.
"""
from .progress import TrainingProgress, log_json

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.nn.functional import all_gather

from .siglip import SiglipConfig, NativeImageProcessor, ContrastiveSiglip, sigmoid_pair_loss
from .tokenizer import TextTokenizer, sha256_file
from .data import split_for_fingerprint, read_jsonl
from .vision_data import safe_image_path
from .training import select_device, amp_context, code_fingerprint, rng_state, restore_rng


def prepare_pairs(source, image_root, output, tokenizer_path, text_length=128, seed=42, val_ratio=.05):
    from PIL import Image
    if text_length < 3 or not 0 < val_ratio < 1:
        raise ValueError('invalid text_length or val_ratio')
    tok = TextTokenizer(tokenizer_path)
    if not Path(source).is_file():
        raise FileNotFoundError(source)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    seen, offsets, stats = set(), {'train': [], 'val': []}, {'duplicates_by_image': 0, 'truncated_captions': 0}
    from contextlib import ExitStack
    with ExitStack() as stack:
        handles = {k: stack.enter_context((root / f'{k}.jsonl').open('xb')) for k in offsets}
        for row in read_jsonl(source):
            path = safe_image_path(image_root, row['image'])
            digest = sha256_file(path)
            if digest in seen:
                stats['duplicates_by_image'] += 1; continue
            with Image.open(path) as image:
                image.verify()
            ids = tok.encode(row['caption'].strip())
            if not ids:
                raise ValueError('empty caption')
            stats['truncated_captions'] += len(ids) > text_length - 2
            ids = [tok.bos_id, *ids[:text_length - 2], tok.eos_id]
            seen.add(digest)
            split = split_for_fingerprint(digest, val_ratio, seed)
            offsets[split].append(handles[split].tell())
            payload = {'image': row['image'], 'image_sha256': digest, 'tokens': ids}
            handles[split].write((json.dumps(payload) + '\n').encode())
    if not all(offsets.values()):
        raise ValueError('empty train/val split: use more images and a new directory')
    for split in offsets:
        (root / f'{split}.index.json').write_text(json.dumps(offsets[split]))
    manifest = {'format': 'moe-lab-siglip-pairs-v1', 'image_root': str(Path(image_root).resolve()),
                'tokenizer_sha256': tok.fingerprint, 'vocab_size': tok.vocab_size,
                'source_sha256': sha256_file(source), 'text_length': text_length, 'seed': seed,
                'val_ratio': val_ratio, 'split_records': {k: len(v) for k, v in offsets.items()}, 'stats': stats,
                'artifacts': {p.name: sha256_file(p) for p in root.iterdir()}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


class PairDataset(Dataset):
    def __init__(self, root, split, image_size, image_root=None):
        root = Path(root)
        self.manifest = json.loads((root / 'manifest.json').read_text())
        if self.manifest['format'] != 'moe-lab-siglip-pairs-v1':
            raise ValueError('not a contrastive pairs dataset')
        self.path = root / f'{split}.jsonl'
        index = root / f'{split}.index.json'
        for p in (self.path, index):
            if sha256_file(p) != self.manifest['artifacts'][p.name]:
                raise ValueError('pairs artifact hash mismatch')
        self.offsets = json.loads(index.read_text())
        self.image_root = image_root or self.manifest['image_root']
        self.processor = NativeImageProcessor(image_size)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        from PIL import Image, ImageOps
        with self.path.open('rb') as f:
            f.seek(self.offsets[i]); row = json.loads(f.readline())
        path = safe_image_path(self.image_root, row['image'])
        if sha256_file(path) != row['image_sha256']:
            raise ValueError(f'image changed: {path}')
        with Image.open(path) as image:
            pixels = self.processor(ImageOps.exif_transpose(image))
        return pixels, row['tokens']


def collate_pairs(rows):
    n = max(len(ids) for _, ids in rows)
    tokens = torch.zeros(len(rows), n, dtype=torch.long)
    mask = torch.zeros(len(rows), n, dtype=torch.bool)
    for i, (_, ids) in enumerate(rows):
        tokens[i, :len(ids)] = torch.tensor(ids)
        mask[i, :len(ids)] = True
    return torch.stack([p for p, _ in rows]), tokens, mask


def distributed_loss(outputs, rank, world):
    image, text, scale, bias = outputs
    global_text = torch.cat(all_gather(text.contiguous()), dim=0) if world > 1 else text
    columns = torch.arange(len(image), device=image.device) + rank * len(image)
    return sigmoid_pair_loss(image, global_text, scale, bias, columns)


@torch.inference_mode()
def evaluate_pairs(model, dataset, device, batch_size, max_samples=256):
    n = min(len(dataset), max_samples)
    if n < 2:
        raise ValueError('retrieval evaluation requires at least two images')
    was_training = model.training
    model.eval()
    images, texts = [], []
    loader = DataLoader(Subset(dataset, range(n)), batch_size=batch_size, collate_fn=collate_pairs,
                        generator=torch.Generator().manual_seed(0))
    for batch in loader:
        a, b, scale, bias = model(*(x.to(device) for x in batch))
        images.append(a); texts.append(b)
    images, texts = torch.cat(images), torch.cat(texts)
    logits = images @ texts.T
    labels = torch.arange(n, device=device)
    result = {'val_loss': sigmoid_pair_loss(images, texts, scale, bias).item(),
              'image_to_text_r1': (logits.argmax(1) == labels).float().mean().item(),
              'text_to_image_r1': (logits.argmax(0) == labels).float().mean().item(),
              'val_samples': n, 'val_total_samples': len(dataset), 'val_full': n == len(dataset),
              'random_r1': 1 / n}
    model.train(was_training)
    return result


def build_model(config, manifest):
    return ContrastiveSiglip(SiglipConfig(**config['vision']), manifest['vocab_size'],
        manifest['text_length'], config.get('projection_size', 256), config.get('text_layers', 4))


def train_pairs(config, data, output, tokenizer_path, stop_after=None, resume=None, image_root=None, epochs=None):
    c = dict(config)
    if epochs is not None:
        c['epochs'] = epochs
    requested_epochs = c.get('epochs')
    if requested_epochs is not None and (type(requested_epochs) is not int or requested_epochs < 1):
        raise ValueError('epochs must be a positive integer')
    world = int(os.environ.get('WORLD_SIZE', 1))
    if requested_epochs is not None:
        if c['batch_size'] < 1:
            raise ValueError('batch_size must be positive')
        manifest = json.loads((Path(data) / 'manifest.json').read_text())
        steps_per_epoch = manifest['split_records']['train'] // (world * c['batch_size'])
        if not steps_per_epoch:
            raise ValueError('epoch training requires at least one complete global batch')
        c['max_steps'] = requested_epochs * steps_per_epoch
    for key in ('max_steps', 'batch_size', 'eval_every', 'save_every', 'eval_samples'):
        if c[key] < 1:
            raise ValueError(f'{key} must be positive')
    if c['eval_samples'] < 2 or not 0 <= c['warmup_steps'] < c['max_steps'] or c['learning_rate'] <= 0:
        raise ValueError('invalid evaluation/schedule settings')
    if c.get('precision', 'bf16') not in ('fp32', 'bf16'):
        raise ValueError('scratch SigLIP supports fp32/bf16')
    if stop_after is not None and not 1 <= stop_after <= c['max_steps']:
        raise ValueError('stop_after must lie within schedule')
    torch.set_num_threads(c.get('cpu_threads', 1))
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = select_device(c.get('device', 'cuda'))
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        return _train(c, data, output, tokenizer_path, stop_after, resume, image_root, rank, world, device)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(c, data, output, tokenizer_path, stop_after, resume, image_root, rank, world, device):
    seed = c.get('seed', 42)
    random.seed(seed); torch.manual_seed(seed)
    train_data = PairDataset(data, 'train', c['vision']['image_size'], image_root)
    val_data = PairDataset(data, 'val', c['vision']['image_size'], image_root)
    manifest = train_data.manifest
    tok = TextTokenizer(tokenizer_path)
    if tok.fingerprint != manifest['tokenizer_sha256']:
        raise ValueError('tokenizer mismatch')
    if len(train_data) < world * c['batch_size'] or world * c['batch_size'] < 2:
        raise ValueError('need at least one global batch and two negative-pool candidates')
    provenance = {'data_sha256': sha256_file(Path(data) / 'manifest.json'), 'code_sha256': code_fingerprint(),
                  'tokenizer_sha256': tok.fingerprint, 'world_size': world, 'device_type': device.type,
                  'torch': str(torch.__version__), 'initialization': 'random',
                  'architecture': 'small SigLIP-style; mean image pooling; EOS text pooling',
                  'external_model_calls': 0}
    raw = build_model(c, manifest).to(device)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=c['learning_rate'], weight_decay=c.get('weight_decay', .1))
    step, epoch, cursor, trained_images = 0, 0, 0, 0
    state = None
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state['format'] != 'moe-lab-siglip-training-v1' or state['config'] != c or state['provenance'] != provenance:
            raise ValueError('exact resume requires same configuration, code, data, tokenizer, device and world size')
        raw.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        step, epoch, cursor, trained_images = (state[k] for k in ('step', 'epoch', 'cursor', 'trained_images'))
    end = stop_after or c['max_steps']
    if end <= step:
        raise ValueError('checkpoint already reached requested stopping step')
    root = Path(output)
    exists = torch.tensor(int(root.exists()), device=device)
    if world > 1:
        dist.all_reduce(exists)
    if exists.item():
        raise FileExistsError(f'use a NEW output directory: {root}')
    if rank == 0:
        root.mkdir(parents=True)
        log_json({'event': 'training_budget', 'epochs': c.get('epochs'),
                          'max_steps': c['max_steps'],
                          'steps_per_epoch': len(train_data) // (world * c['batch_size'])})
        (root / 'run.json').write_text(json.dumps({'config': c, 'provenance': provenance,
            'parameters': sum(p.numel() for p in raw.parameters()), 'global_batch': world * c['batch_size']}, indent=2))
    if world > 1:
        dist.barrier()
    model = DDP(raw, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else raw
    if state:
        restore_rng(state['rng'][rank], device)
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, seed=seed, shuffle=True, drop_last=True)
    started, initial_images = time.perf_counter(), trained_images
    with TrainingProgress(rank=rank, total=c['max_steps'], start=step,
                          steps_per_epoch=len(train_data) // (world * c['batch_size']),
                          epochs=c.get('epochs')) as progress:
        while step < end:
            sampler.set_epoch(epoch)
            loader = DataLoader(train_data, batch_size=c['batch_size'], sampler=sampler, drop_last=True,
                                collate_fn=collate_pairs, num_workers=0, generator=torch.Generator().manual_seed(seed + epoch))
            for index, batch in enumerate(loader):
                if index < cursor:
                    continue
                progress.phase("train")
                t = step
                warmup = c['warmup_steps']
                factor = (t + 1) / warmup if t < warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (t - warmup) / max(1, c['max_steps'] - warmup - 1)))
                lr = c['learning_rate'] * factor
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, c.get('precision', 'bf16')):
                    outputs = model(*(x.to(device) for x in batch))
                loss = distributed_loss(outputs, rank, world)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite contrastive loss')
                loss.backward()
                grad = nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                step += 1; cursor = index + 1; trained_images += c['batch_size'] * world
                loss_value = loss.detach()
                if world > 1:
                    dist.all_reduce(loss_value); loss_value /= world
                metrics = {'step': step, 'epochs_completed': epoch + cursor / len(loader),
                           'train_loss': loss_value.item(), 'grad_norm': grad.item(),
                           'learning_rate': lr, 'trained_images': trained_images, 'global_batch': c['batch_size'] * world}
                progress.update(metrics)
                if step % c['eval_every'] == 0 or step == end:
                    if rank == 0:
                        log_json({'event': 'validation_start', 'max_samples': c['eval_samples']})
                        metrics.update(evaluate_pairs(raw, val_data, device, c['batch_size'], c['eval_samples']))
                    if world > 1:
                        dist.barrier()
                elapsed = time.perf_counter() - started
                metrics.update(elapsed_seconds_this_run=elapsed, allocated_gpu_seconds_this_run=elapsed * world if device.type == 'cuda' else 0,
                               images_per_second_this_run=(trained_images - initial_images) / max(elapsed, 1e-9))
                peak = torch.tensor(torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0, device=device)
                if world > 1:
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                metrics['max_rank_peak_memory_bytes'] = peak.item()
                if rank == 0:
                    with (root / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(metrics) + '\n')
                    log_json(metrics)
                if step % c['save_every'] == 0 or step == end:
                    progress.phase("save")
                    local_rng = rng_state(device)
                    states = [None] * world
                    if world > 1:
                        dist.all_gather_object(states, local_rng)
                    else:
                        states = [local_rng]
                    if rank == 0:
                        target = root / f'step-{step:07d}.pt'
                        payload = {'format': 'moe-lab-siglip-training-v1', 'config': c, 'manifest': manifest,
                            'provenance': provenance, 'model': raw.state_dict(), 'optimizer': optimizer.state_dict(),
                            'step': step, 'epoch': epoch, 'cursor': cursor, 'trained_images': trained_images, 'rng': states}
                        temporary = target.with_suffix('.tmp')
                        torch.save(payload, temporary); temporary.rename(target)
                    if world > 1:
                        dist.barrier()
                if step == end:
                    break
            if step < end:
                epoch += 1; cursor = 0
    if rank == 0:
        (root / 'summary.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def export_encoder(checkpoint, output):
    from safetensors.torch import save_file
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('format') != 'moe-lab-siglip-training-v1' or state['step'] < 1:
        raise ValueError('need a trained scratch SigLIP checkpoint')
    model = build_model(state['config'], state['manifest'])
    model.load_state_dict(state['model'], strict=True)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in model.vision.state_dict().items()}, str(root / 'model.safetensors'))
    config = asdict(model.vision.config)
    config['model_type'] = 'moe-lab-siglip-vision'
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    n = model.vision.config.image_size
    (root / 'preprocessor_config.json').write_text(json.dumps({'size': {'height': n, 'width': n},
        'do_resize': True, 'do_rescale': True, 'do_normalize': True, 'resample': 3,
        'rescale_factor': 1 / 255, 'image_mean': [.5]*3, 'image_std': [.5]*3}, indent=2))
    (root / 'SOURCE.json').write_text(json.dumps({'kind': 'locally_trained_siglip_style',
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint), 'step': state['step'],
        'trained_images': state['trained_images'], 'provenance': state['provenance'],
        'note': 'Export does not certify visual quality; evaluate before VLM training.'}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    for key in ('input', 'image-root', 'output', 'tokenizer'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--text-length', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-ratio', type=float, default=.05)
    p = commands.add_parser('train')
    for key in ('config', 'data', 'output', 'tokenizer'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--epochs', type=int, help='positive whole epochs; overrides config epochs and max_steps')
    p.add_argument('--resume')
    p.add_argument('--image-root')
    p = commands.add_parser('export')
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p = commands.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True); p.add_argument('--data', required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-samples', type=int, default=256)
    p.add_argument('--image-root')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        print(json.dumps(prepare_pairs(args.input, args.image_root, args.output, args.tokenizer,
              args.text_length, args.seed, args.val_ratio), indent=2))
    elif args.command == 'train':
        train_pairs(json.loads(Path(args.config).read_text()), args.data, args.output, args.tokenizer,
                    args.stop_after, args.resume, args.image_root, args.epochs)
    elif args.command == 'export':
        export_encoder(args.checkpoint, args.output)
    else:
        torch.set_num_threads(1)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if state.get('format') != 'moe-lab-siglip-training-v1':
            raise ValueError('not a scratch SigLIP checkpoint')
        device = select_device(args.device)
        if sha256_file(Path(args.data) / 'manifest.json') != state['provenance']['data_sha256']:
            raise ValueError('evaluation dataset differs from checkpoint')
        model = build_model(state['config'], state['manifest']).to(device)
        model.load_state_dict(state['model'])
        dataset = PairDataset(args.data, 'val', state['config']['vision']['image_size'], args.image_root)
        print(json.dumps(evaluate_pairs(model, dataset, device, args.batch_size, args.max_samples), indent=2))


if __name__ == '__main__':
    main()
