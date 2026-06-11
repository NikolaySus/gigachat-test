#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def trim_text(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary >= int(max_chars * 0.65):
        return cut[: boundary + 1].strip()
    return cut.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()

    total = 0
    changed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            old_text = str(record["text"])
            new_text = trim_text(old_text, args.max_chars)
            if new_text != old_text:
                changed += 1
            record["text"] = new_text
            metadata = dict(record.get("metadata", {}))
            metadata["trimmed_from_chars"] = len(old_text)
            metadata["trimmed_to_chars"] = len(new_text)
            metadata["trim_policy"] = f"normalized lead text capped at {args.max_chars} chars"
            record["metadata"] = metadata
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "max_chars": args.max_chars,
        "records": total,
        "changed": changed,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
