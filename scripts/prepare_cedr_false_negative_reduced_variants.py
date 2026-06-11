from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    source = DATA_DIR / "open_ru_1r_nc_cedr_rusentitweet_full_local_sentiment_component_full.jsonl"
    rows = read_jsonl(source)

    cross_only = []
    emotion_only = []
    for row in rows:
        item = dict(row)
        item["source"] = "cedr_rusentitweet_full_local_sentiment:cross_label_negatives_only"
        item["negatives"] = list(row.get("negatives", []))[:4]
        item["metadata"] = dict(row.get("metadata", {}))
        item["metadata"]["removed_same_label_far_negatives"] = True
        cross_only.append(item)
        if item["metadata"].get("label") != "neutral":
            emotion_only.append(item)

    outputs = [
        ("open_ru_1r_nc_cedr_rusentitweet_full_crossonly_component_full", cross_only),
        ("open_ru_1r_nc_cedr_rusentitweet_full_crossonly_emotiononly_component_full", emotion_only),
    ]
    for name, records in outputs:
        path = DATA_DIR / f"{name}.jsonl"
        write_jsonl(path, records)
        write_json(
            path.with_name(path.stem + "_summary.json"),
            {
                "source": str(source.relative_to(ROOT)),
                "output": str(path.relative_to(ROOT)),
                "construction": "derived from RuSentiTweet local sentiment full component; keep only first four cross-label hard negatives, dropping same-label topic-far negatives",
                "records": len(records),
                "label_counts": dict(Counter(row.get("metadata", {}).get("label") for row in records)),
                "negative_count_counts": dict(Counter(len(row.get("negatives", [])) for row in records)),
            },
        )
        print(f"prepared {name}: {len(records)} rows")


if __name__ == "__main__":
    main()
