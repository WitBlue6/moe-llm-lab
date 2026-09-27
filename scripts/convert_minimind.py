"""Stream MiniMind JSONL into this project's text and plain-answer SFT schemas.

Whole tool-use conversations are excluded; their context must not be partially
removed. Separate reasoning_content is omitted; a leading closed <think> block
is removed from assistant content. Original files are never modified.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

RESERVED = ("<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>")
TOOL_FIELDS = ("tools", "tool_calls", "function_call", "functions")
TOOL_MARKERS = ("<tool_call", "</tool_call", "<tool_response", "</tool_response")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sft(row):
    stats = Counter()
    if not isinstance(row, dict):
        return None, "invalid_record", stats
    messages = row.get("conversations", row.get("messages"))
    if any(row.get(key) for key in TOOL_FIELDS):
        return None, "tool_conversation", stats
    if not isinstance(messages, list) or not messages:
        return None, "invalid_messages", stats
    result, previous = [], None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None, "invalid_message", stats
        role, content = message.get("role"), message.get("content")
        if role in ("tool", "function") or any(message.get(key) for key in TOOL_FIELDS):
            return None, "tool_conversation", stats
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None, "unsupported_role_or_content", stats
        if any(marker in content.lower() for marker in TOOL_MARKERS):
            return None, "tool_conversation", stats
        if message.get("reasoning_content"):
            stats["reasoning_fields_omitted"] += 1
        if role == "assistant":
            if content.lstrip().startswith("<think>"):
                match = re.match(r"^\s*<think>.*?</think>\s*", content, re.DOTALL)
                if not match:
                    return None, "unclosed_thinking", stats
                content = content[match.end():]
                stats["thinking_prefixes_removed"] += 1
            if "<think>" in content or "</think>" in content:
                return None, "unsupported_thinking_markup", stats
        content = content.strip()
        if not content:
            return None, "empty_content", stats
        if any(marker in content for marker in RESERVED):
            return None, "reserved_control_token", stats
        if role == "system" and index != 0:
            return None, "invalid_turn_order", stats
        if role == "user" and previous not in (None, "system", "assistant"):
            return None, "invalid_turn_order", stats
        if role == "assistant" and previous != "user":
            return None, "invalid_turn_order", stats
        result.append({"role": role, "content": content})
        previous = role
    if previous != "assistant":
        return None, "missing_final_answer", stats
    return {"messages": result}, None, stats


def convert_file(source, target, stage):
    stats = Counter()
    with Path(source).open(encoding="utf-8") as reader, Path(target).open("x", encoding="utf-8") as writer:
        for number, line in enumerate(reader, 1):
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            stats["read"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {source}:{number}") from error
            if stage == "pretrain":
                text = row.get("text") if isinstance(row, dict) else None
                reason = None
                if not isinstance(text, str) or not text.strip():
                    reason = "invalid_text"
                elif any(marker in text for marker in RESERVED):
                    reason = "reserved_control_token"
                normalized = {"text": text.strip()} if reason is None else None
                edits = Counter()
            else:
                normalized, reason, edits = normalize_sft(row)
            if reason:
                stats[f"skipped_{reason}"] += 1
                continue
            stats.update(edits)
            writer.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["written"] += 1
    if not stats["written"]:
        raise ValueError(f"no usable {stage} rows; inspect inputs; retry in a NEW output directory")
    return dict(stats)


def convert(pretrain, sft, output):
    # Validate both inputs before creating any outputs or scanning a large file.
    for source in (pretrain, sft):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"Input file not found: {path.resolve()}. "
                "Run scripts/download_minimind.py first, or pass the actual downloaded path. "
                "No output directory has been created."
            )
        with path.open("rb"):
            pass
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    stats = {"pretrain": convert_file(pretrain, root / "pretrain.jsonl", "pretrain"),
             "sft": convert_file(sft, root / "sft.jsonl", "sft")}
    report = {"format": "moe-lab-minimind-import-v1", "stats": stats,
              "inputs": {str(path): sha256(path) for path in (pretrain, sft)},
              "outputs": {name: sha256(root / name) for name in ("pretrain.jsonl", "sft.jsonl")},
              "converter_sha256": sha256(__file__),
              "policy": "Plain-answer SFT; exclude whole tool conversations; omit separate reasoning; remove closed leading think blocks. No length truncation, splitting, sampling or dedup here."}
    (root / "import-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--output", required=True, help="NEW directory for pretrain.jsonl and sft.jsonl")
    args = parser.parse_args()
    convert(args.pretrain, args.sft, args.output)


if __name__ == "__main__":
    main()
