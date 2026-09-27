"""JSONL -> deduplicated document split -> mmap token/label records.

Files use little-endian signed int32 pairs [input token, shifted target].
No packing across document boundaries. SFT examples longer than the limit are
skipped intact, avoiding invented EOS targets or partially supervised answers.
"""
import hashlib
import json
import mmap
from pathlib import Path
import struct
import sys

import torch
from torch.utils.data import Dataset

from .tokenizer import TextTokenizer, sha256_file


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected JSON object")
                yield record
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def content_fingerprint(content):
    canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def split_for_fingerprint(fingerprint, val_ratio, seed):
    assignment = hashlib.sha256(f"{seed}:{fingerprint}".encode()).digest()
    return "val" if int.from_bytes(assignment[:8], "big") / 2**64 < val_ratio else "train"


def prepare(input_paths, output, tokenizer, stage, max_seq_len=512, val_ratio=0.05, seed=42):
    if stage not in ("pretrain", "sft") or max_seq_len < 2 or not 0 < val_ratio < 1:
        raise ValueError("invalid stage, sequence length or validation ratio")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    tok = TextTokenizer(tokenizer)
    indexes, handles = {}, {}
    token_counts = {split: {"input": 0, "supervised": 0} for split in ("train", "val")}
    stats = {"documents": 0, "duplicates": 0, "overlong_sft": 0, "empty": 0}
    seen = set()
    for split in ("train", "val"):
        indexes[split] = []
        handles[split] = (root / f"{split}.bin").open("xb")
    def write(split, ids, labels):
        pairs = list(zip(ids[:-1], labels[1:]))
        if not pairs or all(target == -100 for _, target in pairs):
            return
        token_counts[split]["input"] += len(pairs)
        token_counts[split]["supervised"] += sum(target != -100 for _, target in pairs)
        offset = handles[split].tell()
        handles[split].write(struct.pack("<" + "ii" * len(pairs), *(v for p in pairs for v in p)))
        indexes[split].append([offset, len(pairs)])
    try:
        for path in input_paths:
            for record in read_jsonl(path):
                if stage == "pretrain":
                    content = record["text"]
                    if not isinstance(content, str):
                        raise ValueError("pretraining text must be a string")
                    content = content.strip()
                else:
                    content = record["messages"]
                fingerprint = content_fingerprint(content)
                if fingerprint in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(fingerprint)
                split = split_for_fingerprint(fingerprint, val_ratio, seed)
                stats["documents"] += 1
                if stage == "pretrain":
                    encoded = tok.encode(content)
                    if not encoded:
                        stats["empty"] += 1
                        continue
                    ids = [tok.bos_id, *encoded, tok.eos_id]
                    # Overlap one boundary token so every next-token target occurs once.
                    for start in range(0, len(ids) - 1, max_seq_len):
                        window = ids[start:start + max_seq_len + 1]
                        write(split, window, window)
                else:
                    ids, labels = tok.chat(content)
                    if len(ids) - 1 > max_seq_len:
                        stats["overlong_sft"] += 1
                        continue
                    write(split, ids, labels)
    finally:
        for handle in handles.values():
            handle.close()
    if not all(indexes.values()):
        raise ValueError("train or val split is empty; use more documents or another val_ratio/seed and a NEW output directory")
    for split in indexes:
        (root / f"{split}.index.json").write_text(json.dumps(indexes[split]))
    manifest = {"format": "moe-lab-pairs-i32le-v1", "stage": stage, "seed": seed,
                "max_seq_len": max_seq_len, "val_ratio": val_ratio,
                "tokenizer_sha256": tok.fingerprint, "vocab_size": tok.vocab_size,
                "sources": {str(p): sha256_file(p) for p in input_paths}, "stats": stats,
                "split_records": {k: len(v) for k, v in indexes.items()}, "tokens": token_counts,
                "artifacts": {p.name: sha256_file(p) for p in sorted(root.iterdir())}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class TokenDataset(Dataset):
    def __init__(self, root, split):
        if sys.byteorder != "little":
            raise RuntimeError("dataset reader currently requires a little-endian machine")
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest["format"] != "moe-lab-pairs-i32le-v1":
            raise ValueError("unsupported dataset format")
        self.path = root / f"{split}.bin"
        index_path = root / f"{split}.index.json"
        for path in (self.path, index_path):
            if sha256_file(path) != self.manifest["artifacts"][path.name]:
                raise ValueError(f"dataset fingerprint mismatch: {path}")
        self.index = json.loads(index_path.read_text())
        self._file = self._map = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, number):
        if self._map is None:
            self._file = self.path.open("rb")
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        offset, length = self.index[number]
        pair = torch.frombuffer(self._map, dtype=torch.int32, count=length * 2, offset=offset).reshape(-1, 2).long()
        return {"input_ids": pair[:, 0], "labels": pair[:, 1]}

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_map": None}


def collate(examples):
    length = max(len(x["input_ids"]) for x in examples)
    shape = (len(examples), length)
    ids = torch.zeros(shape, dtype=torch.long)
    labels = torch.full(shape, -100, dtype=torch.long)
    mask = torch.zeros(shape, dtype=torch.bool)
    for row, example in enumerate(examples):
        n = len(example["input_ids"])
        ids[row, :n], labels[row, :n], mask[row, :n] = example["input_ids"], example["labels"], True
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}
