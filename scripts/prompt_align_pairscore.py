from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RUSTS_PREFIXES = [
    "семантически похожий текст: ",
    "семантически похожий текст \nтекст: ",
]
KNOWN_PREFIX_RE = re.compile(r"^Instruct:.*?\nQuery:\s*", re.S)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_known_prefixes(text: Any) -> str:
    return KNOWN_PREFIX_RE.sub("", str(text or "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Duplicate pair_score rows with RuSTS legacy evaluation prefixes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--strip-known-prefixes", action="store_true")
    args = parser.parse_args()

    source_records = read_jsonl(args.input)
    output_records: list[dict[str, Any]] = []
    skipped = 0
    for record in source_records:
        if record.get("objective") != "pair_score":
            skipped += 1
            continue
        sentence1 = record["sentence1"]
        sentence2 = record["sentence2"]
        if args.strip_known_prefixes:
            sentence1 = strip_known_prefixes(sentence1)
            sentence2 = strip_known_prefixes(sentence2)
        for prefix_index, prefix in enumerate(RUSTS_PREFIXES):
            aligned = dict(record)
            aligned["sentence1"] = prefix + str(sentence1)
            aligned["sentence2"] = prefix + str(sentence2)
            metadata = dict(record.get("metadata") or {})
            metadata.update(
                {
                    "prompt_alignment": "RuSTSBenchmarkSTS legacy_ru ensemble prefix",
                    "prefix_index": prefix_index,
                    "source_unprompted_path": str(args.input),
                    "stripped_known_prefixes": bool(args.strip_known_prefixes),
                }
            )
            aligned["metadata"] = metadata
            output_records.append(aligned)

    write_jsonl(args.output, output_records)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "source_records": len(source_records),
                "records": len(output_records),
                "skipped_non_pair_score": skipped,
                "strip_known_prefixes": bool(args.strip_known_prefixes),
                "prefixes": RUSTS_PREFIXES,
                "fairness": (
                    "Prompt alignment only. Source records keep their existing contamination policy; "
                    "no benchmark rows or released-model scores are introduced."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
