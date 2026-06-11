#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIXES = (
    "Instruct: Given a text, retrieve semantically similar text\nQuery: ",
    "Instruct: Given a text, retrieve semantically similar text\r\nQuery: ",
)


def strip_prefix(text: str) -> tuple[str, bool]:
    for prefix in PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :].strip(), True
    return text.strip(), False


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip retrieval prefixes from pair-score JSONL records.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stripped_sentence1 = 0
    stripped_sentence2 = 0
    written = 0

    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("objective") != "pair_score":
                continue
            sentence1, changed1 = strip_prefix(str(record["sentence1"]))
            sentence2, changed2 = strip_prefix(str(record["sentence2"]))
            record["sentence1"] = sentence1
            record["sentence2"] = sentence2
            metadata = dict(record.get("metadata") or {})
            metadata["prefix_stripped"] = bool(changed1 or changed2)
            record["metadata"] = metadata
            stripped_sentence1 += int(changed1)
            stripped_sentence2 += int(changed2)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "records": written,
        "stripped_sentence1": stripped_sentence1,
        "stripped_sentence2": stripped_sentence2,
        "note": "Only pair_score records are retained. Retrieval-style prefixes are removed to better match STS evaluation framing.",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
