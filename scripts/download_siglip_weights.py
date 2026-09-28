"""Optional weight-only route. No external model code is downloaded or run."""
import argparse
import json
from pathlib import Path
from moe_llm.tokenizer import sha256_file

REPO = 'google/siglip-base-patch16-224'
REVISION = '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'
WEIGHT_SHA = '2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8'


def main():
    from huggingface_hub import hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='models/siglip-base-patch16-224')
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    plan = {'repo_id': REPO, 'revision': REVISION, 'kind': 'external_pretrained_weights_native_network'}
    marker = root / 'DOWNLOAD_PLAN.json'
    if marker.exists():
        if json.loads(marker.read_text()) != plan:
            raise ValueError('different download plan; use a new directory')
    elif any(root.iterdir()):
        raise FileExistsError('nonempty directory without a matching plan; use a new directory')
    else:
        marker.write_text(json.dumps(plan, indent=2))
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
        hf_hub_download(REPO, name, revision=REVISION, local_dir=str(root), token=False)
    if (root / 'model.safetensors').stat().st_size != 812672320 or sha256_file(root / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('weight checksum mismatch')
    record = {**plan, 'files': {k: sha256_file(root / k) for k in ('config.json', 'preprocessor_config.json', 'model.safetensors')}}
    target = root / 'SOURCE.json'
    if target.exists():
        if json.loads(target.read_text()) != record:
            raise ValueError('existing SOURCE differs')
    else:
        target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
