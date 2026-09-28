"""Stream MiniMind-V Parquet into local image files, VLM JSONL, and captions.

Images keep original bytes, named by SHA256 so both stages share split identity.
No-image and documented 8x8 text placeholders are excluded from visual training.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path

from moe_llm.tokenizer import TextTokenizer, sha256_file
from convert_minimind import normalize_sft


def convert(sources, output, limit=None):
    import pyarrow.parquet as pq
    from PIL import Image, UnidentifiedImageError
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    for source in sources.values():
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    images = root / 'images'
    images.mkdir()
    report = {'sources': {k: {'path': str(v), 'sha256': sha256_file(v)} for k, v in sources.items()},
              'converter_sha256': sha256_file(__file__), 'limit_input_rows_per_stage': limit,
              'policy': 'Original image bytes; whole tool/invalid conversations removed; text placeholders removed; first valid caption per unique image for contrastive learning.', 'stats': {}}
    all_images, caption_images = set(), set()
    with ExitStack() as stack:
        captions = stack.enter_context((root / 'captions.jsonl').open('x')) if 'align' in sources else None
        for stage, source in sources.items():
            stats = Counter()
            writer = stack.enter_context((root / f'vision-{stage}.jsonl').open('x'))
            parquet = pq.ParquetFile(source)
            if not {'conversations', 'image_bytes'} <= set(parquet.schema_arrow.names):
                raise ValueError('expected conversations and image_bytes columns')
            for batch in parquet.iter_batches(batch_size=64, columns=['conversations', 'image_bytes']):
                for row in batch.to_pylist():
                    if limit and stats['read'] >= limit:
                        break
                    stats['read'] += 1
                    messages = row['conversations']
                    if isinstance(messages, str):
                        messages = json.loads(messages)
                    if not isinstance(messages, list):
                        stats['skipped_invalid_messages'] += 1; continue
                    markers = sum(m.get('content', '').count('<image>') for m in messages if isinstance(m, dict) and isinstance(m.get('content'), str))
                    if markers != 1:
                        stats['skipped_no_or_multiple_image_markers'] += 1; continue
                    clean = []
                    marker_valid = False
                    for m in messages:
                        if not isinstance(m, dict):
                            clean.append(m); continue
                        m = dict(m)
                        if isinstance(m.get('content'), str) and '<image>' in m['content']:
                            marker_valid = m.get('role') == 'user' and not any(t.get('role') == 'user' for t in clean if isinstance(t, dict))
                            m['content'] = m['content'].replace('<image>', '').strip()
                        clean.append(m)
                    if not marker_valid:
                        stats['skipped_late_or_nonuser_image'] += 1; continue
                    normalized, reason, edits = normalize_sft({'messages': clean})
                    if reason:
                        stats['skipped_' + reason] += 1; continue
                    blob = row['image_bytes']
                    if isinstance(blob, list):
                        if len(blob) != 1:
                            stats['skipped_multiple_images'] += 1; continue
                        blob = blob[0]
                    if not isinstance(blob, bytes) or not blob:
                        stats['skipped_missing_image'] += 1; continue
                    try:
                        with Image.open(io.BytesIO(blob)) as image:
                            size = image.size
                            image.verify()
                        if min(size) <= 8:
                            stats['skipped_tiny_placeholder'] += 1; continue
                    except (UnidentifiedImageError, OSError, ValueError):
                        stats['skipped_bad_image'] += 1; continue
                    digest = hashlib.sha256(blob).hexdigest()
                    relative = f'{digest[:2]}/{digest}.image'
                    if digest not in all_images:
                        target = images / relative
                        target.parent.mkdir(exist_ok=True)
                        with target.open('xb') as f:
                            f.write(blob)
                        all_images.add(digest)
                    normalized['image'] = relative
                    writer.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                    stats['written'] += 1
                    stats.update(edits)
                    if stage == 'align' and digest not in caption_images:
                        caption = normalized['messages'][next(i for i, m in enumerate(normalized['messages']) if m['role'] == 'assistant')]['content']
                        TextTokenizer().encode(caption)
                        captions.write(json.dumps({'image': relative, 'caption': caption}, ensure_ascii=False) + '\n')
                        caption_images.add(digest)
                        stats['contrastive_unique_images'] += 1
                    if stats['read'] % 10000 == 0:
                        print(stage, dict(stats), flush=True)
                if limit and stats['read'] >= limit:
                    break
            if not stats['written']:
                raise ValueError(f'no usable rows for {stage}; retry with a NEW output directory')
            report['stats'][stage] = dict(stats)
    report['unique_images'] = len(all_images)
    report['outputs'] = {p.name: sha256_file(p) for p in root.glob('*.jsonl')}
    (root / 'import-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain')
    parser.add_argument('--sft')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, help='engineering pilot only: cap input rows per stage')
    a = parser.parse_args()
    sources = {k: v for k, v in [('align', a.pretrain), ('sft', a.sft)] if v}
    if not sources:
        parser.error('provide --pretrain and/or --sft')
    convert(sources, a.output, a.limit)


if __name__ == '__main__':
    main()
