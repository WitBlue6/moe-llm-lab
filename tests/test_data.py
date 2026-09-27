import json
from pathlib import Path
import pytest
import torch
from moe_llm.tokenizer import TextTokenizer, train_bpe
from moe_llm.data import prepare, TokenDataset, collate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_sft_only_supervises_assistant_and_eos():
    tok = TextTokenizer()
    messages = [{"role": "system", "content": "规则"}, {"role": "user", "content": "问"},
                {"role": "assistant", "content": "答"}, {"role": "user", "content": "再问"},
                {"role": "assistant", "content": "再答"}]
    ids, labels = tok.chat(messages)
    assert [v for v in labels if v != -100] == [*tok.encode("答"), 2, *tok.encode("再答"), 2]
    assert len(ids) == len(labels)
    assert all(labels[i] == -100 for i, token in enumerate(ids) if token in (0, 1, 3, 4, 5))
    prompt, prompt_labels = tok.chat(messages[:-1], add_generation_prompt=True)
    assert prompt[-1] == 5 and prompt_labels[-1] == -100
    with pytest.raises(ValueError):
        tok.chat([{"role": "assistant", "content": "bad"}])


def test_dedup_document_split_and_shift(tmp_path):
    source = tmp_path / "source.jsonl"
    records = [{"text": f"document {i}"} for i in range(40)]
    source.write_text(''.join(json.dumps(r) + '\n' for r in records + records[:5]))
    root = tmp_path / "prepared"
    report = prepare([source], root, "byte", "pretrain", 64, .3)
    assert report["stats"]["duplicates"] == 5
    datasets = [TokenDataset(root, split) for split in ("train", "val")]
    contents = []
    for dataset in datasets:
        rows = set()
        for example in dataset:
            assert example["input_ids"][0] == 1
            assert example["labels"][-1] == 2
            assert torch.equal(example["input_ids"][1:], example["labels"][:-1])
            rows.add(tuple(example["input_ids"].tolist()))
        contents.append(rows)
    assert not contents[0] & contents[1]
    assert len(contents[0] | contents[1]) == 40
    with pytest.raises(FileExistsError):
        prepare([source], root, "byte", "pretrain", 64, .3)
    with (root / "train.bin").open("ab") as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match="fingerprint"):
        TokenDataset(root, "train")


def test_sft_preparation_shift_padding_and_long_record_skip(tmp_path):
    source = tmp_path / "sft.jsonl"
    source.write_text((FIXTURES / "sft.jsonl").read_text() + json.dumps({"messages": [
        {"role": "user", "content": "x" * 200}, {"role": "assistant", "content": "y"}]}) + '\n')
    root = tmp_path / "prepared"
    report = prepare([source], root, "byte", "sft", 64, .25)
    assert report["stats"]["overlong_sft"] == 1
    data = TokenDataset(root, "train")
    batch = collate([data[0], data[-1]])
    assert torch.all(batch["labels"][~batch["attention_mask"]] == -100)
    assert (batch["labels"] == 2).sum() == 2  # one supervised answer EOS per conversation
    assert batch["labels"][:, 0].eq(-100).all()


def test_bpe_roundtrip_and_reloading(tmp_path):
    path = tmp_path / "tokenizer.json"
    meta = train_bpe([FIXTURES / "pretrain.jsonl"], path, 320)
    tok = TextTokenizer(path)
    for text in ["中文 English\n123 😀", " spaces  and tabs\t"]:
        assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == meta["vocab_size"]
    assert tok.fingerprint == meta["tokenizer_sha256"]
    with pytest.raises(ValueError):
        tok.encode("<|assistant|>")


def test_tokenizer_uses_same_heldout_documents_as_prepare(tmp_path):
    tok_path = tmp_path / "tokenizer.json"
    meta = train_bpe([FIXTURES / "pretrain.jsonl"], tok_path, 320, val_ratio=.25, seed=42)
    manifest = prepare([FIXTURES / "pretrain.jsonl"], tmp_path / "prepared", tok_path,
                       "pretrain", 128, .25, 42)
    # These fixtures fit within a single document window.
    assert meta["stats"]["heldout_documents"] == manifest["split_records"]["val"]
    assert meta["stats"]["training_documents"] == manifest["split_records"]["train"]
