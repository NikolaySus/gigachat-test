from __future__ import annotations

import argparse
import json
from pathlib import Path


RUSTS_PREFIXES = [
    "семантически похожий текст: ",
    "семантически похожий текст \nтекст: ",
]


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def strip_known_prefix(text: str) -> str:
    text = str(text)
    prefixes = [
        "Instruct: Given a text, retrieve semantically similar text\nQuery: ",
        *RUSTS_PREFIXES,
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                changed = True
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Duplicate pair_score records with the legacy RuSTS prefixes used by the frozen wrapper."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=6400)
    args = parser.parse_args()

    records = [record for record in read_jsonl(args.input) if record.get("objective") == "pair_score"]
    output: list[dict] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        if len(batch) < args.batch_size:
            break
        for prefix_index, prefix in enumerate(RUSTS_PREFIXES):
            for record in batch:
                left = strip_known_prefix(record["sentence1"])
                right = strip_known_prefix(record["sentence2"])
                new_record = dict(record)
                new_record["sentence1"] = prefix + left
                new_record["sentence2"] = prefix + right
                new_record["source"] = str(record.get("source", "")) + f":rusts_prefix{prefix_index + 1}"
                metadata = dict(record.get("metadata") or {})
                metadata["prompt_alignment"] = "RuSTSBenchmarkSTS legacy_ru ensemble prefix"
                metadata["prefix_index"] = prefix_index
                new_record["metadata"] = metadata
                output.append(new_record)
                if len(output) >= args.max_records:
                    break
            if len(output) >= args.max_records:
                break
        if len(output) >= args.max_records:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for record in output:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.out),
        "records": len(output),
        "source_records": len(records),
        "batch_size": args.batch_size,
        "prefixes": RUSTS_PREFIXES,
        "construction": (
            "Each ordered source batch is repeated once per RuSTS legacy prefix, "
            "with both pair sides prefixed the same way. This preserves the "
            "source score-bucket batch structure while matching evaluation text framing."
        ),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
