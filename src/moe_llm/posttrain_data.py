"""Preference pairs and online prompts, split by the entire prompt content."""
import json
from pathlib import Path

from .data import read_jsonl, content_fingerprint, split_for_fingerprint
from .tokenizer import TextTokenizer, sha256_file


def prepare_posttrain(inputs, output, tokenizer, kind, max_seq_len=512, val_ratio=.05, seed=42):
    if kind not in ('preferences', 'prompts') or max_seq_len < 3 or not 0 < val_ratio < 1:
        raise ValueError('invalid kind, context or validation ratio')
    tok = TextTokenizer(tokenizer)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    offsets = {'train': [], 'val': []}
    stats = {'read': 0, 'duplicates': 0, 'overlong': 0, 'written': 0}
    seen = set()
    from contextlib import ExitStack
    with ExitStack() as stack:
        handles = {k: stack.enter_context((root / f'{k}.jsonl').open('xb')) for k in offsets}
        for source in inputs:
            for row in read_jsonl(source):
                stats['read'] += 1
                messages = row.get('messages')
                if messages is None:
                    messages = [{'role': 'user', 'content': row['prompt']}]
                prompt = tok.chat(messages, add_generation_prompt=True)[0]
                record = {'messages': messages, 'prompt_ids': prompt}
                if kind == 'preferences':
                    for key in ('chosen', 'rejected'):
                        text = row[key]
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(f'{key} must be a nonempty completion string')
                        record[key] = [*tok.encode(text.strip()), tok.eos_id]
                    if record['chosen'] == record['rejected']:
                        raise ValueError('chosen and rejected must differ')
                    length = len(prompt) + max(len(record['chosen']), len(record['rejected']))
                else:
                    if 'answer' in row:
                        if not isinstance(row['answer'], str) or not row['answer'].strip():
                            raise ValueError('answer must be a nonempty string')
                        record['answer'] = row['answer'].strip()
                    # Labels are stored for reward/evaluation, never appended to prompt_ids.
                    length = len(prompt) + 1
                if length > max_seq_len:
                    stats['overlong'] += 1
                    continue
                digest = content_fingerprint(record)
                if digest in seen:
                    stats['duplicates'] += 1
                    continue
                seen.add(digest)
                split = split_for_fingerprint(content_fingerprint(messages), val_ratio, seed)
                offsets[split].append(handles[split].tell())
                handles[split].write((json.dumps(record, ensure_ascii=False) + '\n').encode())
                stats['written'] += 1
    if not all(offsets.values()):
        raise ValueError('empty train/val split: use more prompts and a new output directory')
    for split, index in offsets.items():
        (root / f'{split}.index.json').write_text(json.dumps(index))
    manifest = {'format': 'moe-lab-posttrain-data-v1', 'kind': kind, 'max_seq_len': max_seq_len,
                'seed': seed, 'val_ratio': val_ratio, 'tokenizer_sha256': tok.fingerprint,
                'sources': {str(p): sha256_file(p) for p in inputs}, 'stats': stats,
                'split_records': {k: len(v) for k, v in offsets.items()},
                'artifacts': {p.name: sha256_file(p) for p in root.iterdir()}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


class PosttrainDataset:
    def __init__(self, root, split):
        root = Path(root)
        self.manifest = json.loads((root / 'manifest.json').read_text())
        if self.manifest['format'] != 'moe-lab-posttrain-data-v1':
            raise ValueError('unsupported post-training data')
        for name, digest in self.manifest['artifacts'].items():
            if sha256_file(root / name) != digest:
                raise ValueError(f'post-training artifact changed: {name}')
        self.index = json.loads((root / f'{split}.index.json').read_text())
        self.path = root / f'{split}.jsonl'

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        with self.path.open('rb') as handle:
            handle.seek(self.index[index])
            return json.loads(handle.readline())
