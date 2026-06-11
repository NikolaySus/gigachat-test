from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    CACHE_DIR,
    DATA_DIR,
    build_habr_style_records,
    is_contaminated,
    normalize_text,
    load_cedr_index,
)


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    cedr_index = load_cedr_index()
    dataset = load_dataset("psytechlab/RuSentiTweet", cache_dir=str(CACHE_DIR))
    label_map = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "speech": "neutral",
        "skip": "neutral",
    }
    rows = []
    skipped = Counter()
    seen = set()
    for split_name, split in dataset.items():
        for index, row in enumerate(split):
            raw_label = str(row["label"])
            label = label_map.get(raw_label)
            text = str(row["text"]).strip()
            normalized = normalize_text(text)
            if label is None:
                skipped[f"label_{raw_label}"] += 1
                continue
            if len(normalized) < 12 or len(normalized) > 360:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            rows.append(
                {
                    "source": "psytechlab/RuSentiTweet",
                    "split": split_name,
                    "index": index,
                    "label": label,
                    "raw_label": raw_label,
                    "text": text,
                    "normalized": normalized,
                }
            )

    records, summary = build_habr_style_records(
        rows,
        name="cedr_rusentitweet_skip_speech_as_neutral",
        count=None,
        seed=701,
    )
    summary["loader"] = {
        "skipped": dict(skipped),
        "kept": len(rows),
        "raw_label_counts": dict(Counter(row["raw_label"] for row in rows)),
        "mapped_label_counts": dict(Counter(row["label"] for row in rows)),
        "label_map": label_map,
    }
    path = DATA_DIR / "open_ru_1r_nc_cedr_rusentitweet_skip_speech_as_neutral_component_full.jsonl"
    write_jsonl(path, records)
    write_json(path.with_name(path.stem + "_summary.json"), summary)
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
