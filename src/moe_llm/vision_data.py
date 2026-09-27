"""Single-image, multi-turn conversations; split by image bytes before tokenization."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

from .data import read_jsonl, collate, content_fingerprint, split_for_fingerprint
from .tokenizer import TextTokenizer, sha256_file
from .vision import ImageProcessor


def visual_chat(tokenizer, messages, image_tokens, generation=False):
    ids, labels = tokenizer.chat(messages, add_generation_prompt=generation)
    # Insert after the first USER role, without modifying vocabulary or tokenizer.
    position = ids.index(4) + 1
    ids[position:position] = [tokenizer.pad_id] * image_tokens
    labels[position:position] = [-100] * image_tokens
    return ids, labels, position


def safe_image_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("image paths must be relative to image-root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing image or path outside image-root: {relative}")
    return path


def prepare_visual(input_paths, image_root, output, tokenizer_path, vision_config,
                   max_seq_len=512, val_ratio=.05, seed=42):
    from PIL import Image
    if not 0 < val_ratio < 1 or max_seq_len <= vision_config.image_tokens + 4:
        raise ValueError("invalid val_ratio or sequence too short for visual tokens")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    tok = TextTokenizer(tokenizer_path)
    index = {"train": [], "val": []}
    image_hashes, seen = {}, set()
    stats = {"records": 0, "duplicates": 0, "overlong": 0}
    handles = {split: (root / f"{split}.jsonl").open("xb") for split in index}
    try:
        for source in input_paths:
            for record in read_jsonl(source):
                relative = record["image"]
                path = safe_image_path(image_root, relative)
                if relative not in image_hashes:
                    with Image.open(path) as image:
                        image.verify()
                    image_hashes[relative] = sha256_file(path)
                image_hash = image_hashes[relative]
                key = content_fingerprint([image_hash, record["messages"]])
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                split = split_for_fingerprint(image_hash, val_ratio, seed)
                ids, labels, position = visual_chat(tok, record["messages"], vision_config.image_tokens)
                stats["records"] += 1
                if len(ids) - 1 > max_seq_len:
                    stats["overlong"] += 1
                    continue
                row = {"image": relative, "input_ids": ids[:-1], "labels": labels[1:],
                       "image_position": position}
                index[split].append(handles[split].tell())
                handles[split].write((json.dumps(row, separators=(",", ":")) + "\n").encode())
    finally:
        for handle in handles.values():
            handle.close()
    if not all(index.values()):
        raise ValueError("train/val is empty after image-level split; use more images and a new output directory")
    for split, offsets in index.items():
        (root / f"{split}.index.json").write_text(json.dumps(offsets))
    manifest = {"format": "moe-lab-visual-v1", "tokenizer_sha256": tok.fingerprint,
                "vocab_size": tok.vocab_size, "max_seq_len": max_seq_len,
                "image_tokens": vision_config.image_tokens, "seed": seed, "val_ratio": val_ratio,
                "image_root": str(Path(image_root).resolve()), "images": image_hashes,
                "sources": {str(p): sha256_file(p) for p in input_paths}, "stats": stats,
                "split_records": {k: len(v) for k, v in index.items()},
                "artifacts": {p.name: sha256_file(p) for p in sorted(root.iterdir())}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class VisualDataset(Dataset):
    def __init__(self, root, split, vision_config, image_root=None):
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest["format"] != "moe-lab-visual-v1":
            raise ValueError("unsupported visual dataset format")
        if self.manifest["image_tokens"] != vision_config.image_tokens:
            raise ValueError("prepared visual token count differs from vision config")
        self.path = root / f"{split}.jsonl"
        index_path = root / f"{split}.index.json"
        for path in (self.path, index_path):
            if sha256_file(path) != self.manifest["artifacts"][path.name]:
                raise ValueError(f"visual dataset fingerprint mismatch: {path}")
        self.offsets = json.loads(index_path.read_text())
        self.image_root = Path(image_root or self.manifest["image_root"])
        self.processor = ImageProcessor(vision_config)
        # Verify referenced image content once on load; never silently ignore missing/corrupt images.
        for relative, expected in self.manifest["images"].items():
            if sha256_file(safe_image_path(self.image_root, relative)) != expected:
                raise ValueError(f"image fingerprint mismatch: {relative}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            row = json.loads(handle.readline())
        return {"input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
                "labels": torch.tensor(row["labels"], dtype=torch.long),
                "image_position": row["image_position"],
                "pixel_values": self.processor(safe_image_path(self.image_root, row["image"]))}


def collate_visual(examples):
    batch = collate(examples)
    batch["pixel_values"] = torch.stack([row["pixel_values"] for row in examples])
    batch["image_positions"] = torch.tensor([row["image_position"] for row in examples], dtype=torch.long)
    return batch


def create_visual_fixture(output):
    """Generate tiny colored PNGs and conversations; never pretend these are real data."""
    from PIL import Image
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    images = root / "images"
    images.mkdir()
    records = []
    for i in range(32):
        color = (i * 7 % 256, i * 13 % 256, i * 19 % 256)
        name = f"color-{i:02d}.png"
        Image.new("RGB", (16, 16), color).save(images / name)
        records.append({"image": name, "messages": [{"role": "user", "content": "Describe the image."},
            {"role": "assistant", "content": f"A solid color sample {i}."}]})
    (root / "conversations.jsonl").write_text(''.join(json.dumps(row) + "\n" for row in records))
    (root / "README.txt").write_text("Synthetic fixture only. Not a benchmark or pretrained vision model.\n")
    return {"jsonl": str(root / "conversations.jsonl"), "image_root": str(images)}
