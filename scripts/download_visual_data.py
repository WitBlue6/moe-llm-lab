"""Download the pinned MiniMind-V caption/SFT Parquet, including image bytes."""
import argparse
import json
from pathlib import Path
from moe_llm.tokenizer import sha256_file

REPO = 'jingyaogong/minimind-v_dataset'
REVISION = '1e279a8b665cb10383451a6af6fd62b9f35bdd79'
FILES = {
    'pretrain_i2t.parquet': (4326415097, '65761f37d1947d54a1d85457ff70938275e4ef58ba5cedcd02463a3a247c93fd'),
    'sft_i2t.parquet': (4934887104, '712f4026cd0e21b369feddca7334b1e465cb8182b5f298006f3f4f877f926643'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['caption', 'sft', 'both'], default='both')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    names = list(FILES) if args.stage == 'both' else [list(FILES)[args.stage == 'sft']]
    plan = {'repo_id': REPO, 'revision': REVISION, 'files': {k: {'bytes': FILES[k][0], 'sha256': FILES[k][1]} for k in names},
            'dataset_card': f'https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md',
            'note': 'ALLaVA-derived mixed-source data; check upstream image and annotation terms separately.'}
    if args.list:
        print(json.dumps(plan, indent=2)); return
    from huggingface_hub import hf_hub_download
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    record = root / 'DOWNLOAD_PLAN.json'
    if record.exists():
        if json.loads(record.read_text()) != plan:
            raise ValueError('existing directory has a different download plan; use a new one')
    else:
        with record.open('x') as f:
            json.dump(plan, f, indent=2)
    for name in names:
        path = root / name
        if not path.exists():
            hf_hub_download(REPO, name, repo_type='dataset', revision=REVISION, local_dir=str(root), token=False)
        size, digest = FILES[name]
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f'checksum/size mismatch: {path}; not overwritten')
        print(f'Verified {path}', flush=True)
    if not (root / 'SOURCES.json').exists():
        with (root / 'SOURCES.json').open('x') as f:
            json.dump(plan, f, indent=2)


if __name__ == '__main__':
    main()
