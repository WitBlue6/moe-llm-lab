"""Download selected public MiniMind files at a fixed commit and verify SHA256.

Run through the project's locked uv environment. No weights or full repository
snapshot are downloaded. Re-running the same command resumes/reuses HF downloads.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = "jingyaogong/minimind_dataset"
REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"
FILES = {
    "pretrain_t2t_mini.jsonl": (1241043656, "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c"),
    "sft_t2t_mini.jsonl": (1739201170, "abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518"),
    "pretrain_t2t.jsonl": (8275074893, "31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d"),
    "sft_t2t.jsonl": (14096018369, "b5dab3d590b3b4ae3bfeec891ba0c76b0de5acbec4f4b078dcd3912eb1735c79"),
}
PRESETS = {
    "mini": ("pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl"),
    "full-pretrain": ("pretrain_t2t.jsonl", "sft_t2t_mini.jsonl"),
    "full": ("pretrain_t2t.jsonl", "sft_t2t.jsonl"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, size, checksum):
    path = Path(path)
    if path.stat().st_size != size or sha256(path) != checksum:
        raise ValueError(f"size/SHA256 mismatch; existing file was not overwritten: {path}")


def plan(preset):
    return {"repo_id": REPO, "revision": REVISION, "verified_metadata_date": "2026-09-27",
            "preset": preset, "files": {name: {"bytes": FILES[name][0], "sha256": FILES[name][1]}
                                         for name in PRESETS[preset]},
            "dataset_card": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
            "license_labels": ["apache-2.0", "cc-by-nc-2.0"],
            "license_note": "Mixed-source corpus; these labels are not a blanket commercial-use grant."}


def download(preset, output):
    from huggingface_hub import hf_hub_download
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    info = plan(preset)
    record = root / "DOWNLOAD_PLAN.json"
    if record.exists():
        if json.loads(record.read_text()) != info:
            raise ValueError("directory belongs to another download plan; use a new directory")
    else:
        with record.open("x") as handle:
            json.dump(info, handle, indent=2)
    for name, metadata in info["files"].items():
        target = root / name
        if not target.exists():
            print(f"Downloading {name} ({metadata['bytes'] / 1e9:.2f} GB)", flush=True)
            hf_hub_download(repo_id=REPO, repo_type="dataset", filename=name,
                            revision=REVISION, local_dir=str(root), token=False)
        verify(target, metadata["bytes"], metadata["sha256"])
        print(f"Verified {target}", flush=True)
    complete = root / "SOURCES.json"
    if complete.exists():
        if json.loads(complete.read_text()) != info:
            raise ValueError("existing SOURCES.json differs from the verified plan")
    else:
        with complete.open("x") as handle:
            json.dump(info, handle, indent=2)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="mini")
    parser.add_argument("--output", default="data/downloads/minimind-mini-v1")
    parser.add_argument("--list", action="store_true", help="show selected files without downloading")
    args = parser.parse_args()
    if args.list:
        info = plan(args.preset)
        print(json.dumps(info, indent=2))
        print(f"Total download: {sum(f['bytes'] for f in info['files'].values()) / 1e9:.2f} GB")
    else:
        download(args.preset, args.output)


if __name__ == "__main__":
    main()
