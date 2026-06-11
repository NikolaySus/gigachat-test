from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"

LABEL_STATEMENTS = {
    "neutral": "в тексте нет явной эмоции из списка.",
    "no_emotion": "в тексте нет явной эмоции из списка.",
    "joy": "в тексте выражена радость, одобрение или позитивная эмоция.",
    "sadness": "в тексте выражена грусть, печаль, тоска или усталость.",
    "anger": "в тексте выражена злость, раздражение, гнев или агрессия.",
    "surprise": "в тексте выражено удивление, шок или неожиданность.",
    "fear": "в тексте выражен страх, тревога, опасение или ужас.",
}

LABEL_ORDER = ["neutral", "joy", "sadness", "anger", "surprise", "fear"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CEDR label-statement records from an existing clean component.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()

    source_records = read_jsonl(args.input)
    records = []
    skipped = 0
    for source_index, row in enumerate(source_records):
        group = str(row.get("metadata", {}).get("group", ""))
        if group not in LABEL_STATEMENTS:
            skipped += 1
            continue
        query = str(row["query"])
        positive = f"{query}\nВерная разметка: {LABEL_STATEMENTS[group]}"
        negatives = [
            f"{query}\nВерная разметка: {LABEL_STATEMENTS[label]}"
            for label in LABEL_ORDER
            if label != group
        ]
        records.append(
            {
                "source": f"cedr_label_statement:{args.name}",
                "objective": "contrastive",
                "query": query,
                "positive": positive,
                "negatives": negatives,
                "metadata": {
                    "group": group,
                    "source_index": source_index,
                    "construction": "query_specific_label_statement",
                },
            }
        )

    random.Random(args.seed).shuffle(records)
    path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(path, records)
    summary = {
        "input": str(args.input),
        "output": str(path.relative_to(ROOT)),
        "records": len(records),
        "skipped": skipped,
        "label_order": LABEL_ORDER,
        "construction": "query text vs correct/incorrect Russian emotion label statements",
    }
    path.with_name(path.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
