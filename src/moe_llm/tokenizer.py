"""Byte fixture tokenizer and a trainable byte-level BPE with fixed role IDs."""
import hashlib
from pathlib import Path

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]
ROLE_IDS = {"system": 3, "user": 4, "assistant": 5}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TextTokenizer:
    pad_id, bos_id, eos_id = 0, 1, 2

    def __init__(self, path="byte"):
        self.path = str(path)
        self.backend = None
        if self.path == "byte":
            self.vocab_size = 262
            self.fingerprint = hashlib.sha256(b"moe-lab-byte-v1-specials-6").hexdigest()
        else:
            from tokenizers import Tokenizer
            self.backend = Tokenizer.from_file(self.path)
            if [self.backend.token_to_id(t) for t in SPECIALS] != list(range(6)):
                raise ValueError("tokenizer must use this project's six fixed special-token IDs")
            self.backend.no_padding()
            self.backend.no_truncation()
            self.vocab_size = self.backend.get_vocab_size()
            self.fingerprint = sha256_file(self.path)

    def encode(self, text):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if any(s in text for s in SPECIALS):
            raise ValueError("raw text must not contain reserved role/control tokens")
        if self.backend is None:
            return [b + 6 for b in text.encode("utf-8")]
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        if self.backend is None:
            return bytes(i - 6 for i in ids if 6 <= i < 262).decode("utf-8", errors="replace")
        return self.backend.decode(ids, skip_special_tokens=True)

    def chat(self, messages, add_generation_prompt=False):
        """Return unshifted IDs and labels; only assistant content + EOS are targets."""
        if not messages:
            raise ValueError("messages cannot be empty")
        ids, labels = [self.bos_id], [-100]
        previous = None
        for number, message in enumerate(messages):
            role = message.get("role")
            if role not in ROLE_IDS:
                raise ValueError(f"unsupported role: {role}")
            if role == "system" and number != 0:
                raise ValueError("system message is allowed only at the start")
            if role == "assistant" and previous != "user":
                raise ValueError("assistant must follow user")
            if role == "user" and previous not in (None, "system", "assistant"):
                raise ValueError("user must start a turn")
            content = self.encode(message["content"])
            if not content:
                raise ValueError("empty messages are not supported")
            ids.extend([ROLE_IDS[role], *content, self.eos_id])
            labels.extend([-100, *content, self.eos_id] if role == "assistant" else [-100] * (len(content) + 2))
            previous = role
        if add_generation_prompt:
            if previous != "user":
                raise ValueError("generation prompt must end with a user message")
            ids.append(ROLE_IDS["assistant"])
            labels.append(-100)
        elif previous != "assistant":
            raise ValueError("SFT conversation must end with an assistant response")
        return ids, labels


def train_bpe(input_paths, output, vocab_size, val_ratio=0.05, seed=42):
    import json
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from .data import read_jsonl, content_fingerprint, split_for_fingerprint
    if not 0 < val_ratio < 1:
        raise ValueError("validation ratio must lie within (0, 1)")
    if vocab_size < 262:
        raise ValueError("byte-level BPE requires vocab_size >= 262")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    stats = {"training_documents": 0, "heldout_documents": 0, "duplicates": 0}
    seen = set()
    def texts():
        for path in input_paths:
            for record in read_jsonl(path):
                raw = record["text"].strip() if "text" in record else record["messages"]
                fingerprint = content_fingerprint(raw)
                if fingerprint in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(fingerprint)
                if split_for_fingerprint(fingerprint, val_ratio, seed) == "val":
                    stats["heldout_documents"] += 1
                    continue
                stats["training_documents"] += 1
                content = [raw] if isinstance(raw, str) else [m["content"] for m in raw]
                for text in content:
                    if not isinstance(text, str) or any(s in text for s in SPECIALS):
                        raise ValueError("invalid text or reserved control token in tokenizer corpus")
                    yield text
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts(), trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2,
    ))
    if stats["training_documents"] == 0:
        raise ValueError("tokenizer corpus has no training documents")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(tokenizer.to_str())
    metadata = {"sources": {str(p): sha256_file(p) for p in input_paths},
                "tokenizer_sha256": sha256_file(output), "vocab_size": tokenizer.get_vocab_size(),
                "seed": seed, "val_ratio": val_ratio, "stats": stats,
                "note": "Uses the same content-hash train/val split as prepare; keep seed and val_ratio identical."}
    output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2))
    return metadata
