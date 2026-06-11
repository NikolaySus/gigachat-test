from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, SENT_PREFIX, read_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]

LABEL_PHRASES = {
    "positive": [
        "положительная эмоциональная окраска, радость, одобрение или симпатия",
        "позитивный комментарий с хорошей или одобрительной эмоцией",
        "сообщение выражает приятное отношение, поддержку или радость",
    ],
    "negative": [
        "отрицательная эмоциональная окраска, недовольство, злость или грусть",
        "негативный комментарий с плохой или критической эмоцией",
        "сообщение выражает неприятное отношение, раздражение или печаль",
    ],
    "neutral": [
        "нейтральный комментарий без выраженной положительной или отрицательной эмоции",
        "информационное или неопределенное сообщение без сильной эмоциональной оценки",
        "сообщение не выражает явной радости, злости, грусти или одобрения",
    ],
}


def label_from_record(record: dict) -> str:
    metadata = record.get("metadata") or {}
    label = metadata.get("label")
    if label not in LABEL_PHRASES:
        raise ValueError(f"Unsupported label in record metadata: {label!r}")
    return label


def strip_prefix(text: str) -> str:
    if text.startswith(SENT_PREFIX):
        return text[len(SENT_PREFIX) :]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RuSentiTweet label-description pair-score calibration data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_rusentitweet_skip_as_neutral_component_full.jsonl",
    )
    parser.add_argument("--name", default="cedr_rusentitweet_skip_label_pairscore")
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--positive-score", type=float, default=0.95)
    parser.add_argument("--negative-score", type=float, default=0.05)
    parser.add_argument("--negative-labels", type=int, default=2)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source_records = read_jsonl(args.input)
    records = []
    label_counts = Counter()
    for record in source_records:
        label = label_from_record(record)
        text = SENT_PREFIX + strip_prefix(record["query"])
        positive_phrase = rng.choice(LABEL_PHRASES[label])
        records.append(
            {
                "source": f"{args.name}:label_positive",
                "objective": "pair_score",
                "sentence1": text,
                "sentence2": positive_phrase,
                "score": args.positive_score,
                "metadata": {"label": label},
            }
        )
        other_labels = [candidate for candidate in LABEL_PHRASES if candidate != label]
        rng.shuffle(other_labels)
        for other_label in other_labels[: args.negative_labels]:
            records.append(
                {
                    "source": f"{args.name}:label_negative",
                    "objective": "pair_score",
                    "sentence1": text,
                    "sentence2": rng.choice(LABEL_PHRASES[other_label]),
                    "score": args.negative_score,
                    "metadata": {"label": label, "negative_label": other_label},
                }
            )
        label_counts[label] += 1
    rng.shuffle(records)

    path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(path, records)
    write_json(
        path.with_name(path.stem + "_summary.json"),
        {
            "source": args.name,
            "input": str(args.input.relative_to(ROOT)),
            "records": len(records),
            "input_label_counts": dict(label_counts),
            "positive_score": args.positive_score,
            "negative_score": args.negative_score,
            "negative_labels": args.negative_labels,
            "construction": "pair-score calibration between tweets and sentiment label descriptions",
        },
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
