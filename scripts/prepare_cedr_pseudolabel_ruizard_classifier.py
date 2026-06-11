from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_cedr_neutral_lexical_distractors import read_jsonl
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_json,
    write_jsonl,
)


TARGET_LABELS = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
MODEL_TO_TARGET = {
    "neutral": "neutral",
    "joy": "joy",
    "enthusiasm": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    "fear": "fear",
    "anger": "anger",
}
PREFIX_RE = re.compile(r"^Определи эмоции в комментарии:.*?\nкомментарий:\s*", re.S)


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = PREFIX_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_ok(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 10 or len(normalized) > 360:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 0.35 * max(1, len(text))


def iter_candidate_texts(paths: list[Path]) -> list[tuple[str, str]]:
    items = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for field in ("query", "positive", "sentence1", "sentence2", "text"):
                if field not in row:
                    continue
                text = clean_text(row[field])
                normalized = normalize_text(text)
                if not text_ok(text) or normalized in seen:
                    continue
                seen.add(normalized)
                items.append((text, str(path.name)))
            for field in ("negatives", "positives"):
                for value in row.get(field, []) or []:
                    text = clean_text(value)
                    normalized = normalize_text(text)
                    if not text_ok(text) or normalized in seen:
                        continue
                    seen.add(normalized)
                    items.append((text, str(path.name)))
    return items


def classify_texts(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
) -> list[tuple[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    predictions = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float()
            probs = torch.softmax(logits, dim=1)
            scores, indices = probs.max(dim=1)
            for score, index in zip(scores.tolist(), indices.tolist(), strict=True):
                predictions.append((id2label[int(index)], float(score)))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label clean CEDR-like text pools with RuIzard classifier.")
    parser.add_argument("--name", default="cedr_ruizard_classifier_pseudolabel_7200")
    parser.add_argument("--model-name", default="Djacon/rubert-tiny2-russian-emotion-detection")
    parser.add_argument("--count", type=int, default=7200)
    parser.add_argument("--neutral-threshold", type=float, default=0.88)
    parser.add_argument("--emotion-threshold", type=float, default=0.72)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4011)
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    paths = args.path or [
        DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_scan500k_8000.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_lenta_negative_topic_neutral_reported_2400.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_blog_neutral_broad_1600.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_djacon_rugoemotions_prior9000.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_seara_rugoemotions_strict_prior9000.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_skywater_ruemotions_contrastive_strict_7200.jsonl",
    ]
    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    candidates = iter_candidate_texts(paths)
    texts = [text for text, _source in candidates]
    predictions = classify_texts(texts, model_name=args.model_name, batch_size=args.batch_size)

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for (text, source), (raw_label, confidence) in zip(candidates, predictions, strict=True):
        label = MODEL_TO_TARGET.get(raw_label)
        if label is None:
            skipped["unmapped_label"] += 1
            continue
        threshold = args.neutral_threshold if label == "neutral" else args.emotion_threshold
        if confidence < threshold:
            skipped["low_confidence"] += 1
            continue
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        pools[label].append({"text": text, "source_file": source, "raw_label": raw_label, "confidence": confidence})

    targets = {"neutral": int(args.count * 0.45)}
    remaining = args.count - targets["neutral"]
    per_emotion = remaining // (len(TARGET_LABELS) - 1)
    for label in TARGET_LABELS:
        if label == "neutral":
            continue
        targets[label] = per_emotion
    for label in TARGET_LABELS[1 : 1 + (remaining - per_emotion * (len(TARGET_LABELS) - 1))]:
        targets[label] += 1

    selected: dict[str, list[dict[str, Any]]] = {}
    for label in TARGET_LABELS:
        rows = sorted(pools[label], key=lambda row: row["confidence"], reverse=True)
        top_window = rows[: max(targets[label] * 3, targets[label])]
        rng.shuffle(top_window)
        selected[label] = top_window[: min(targets[label], len(top_window))]

    records = []
    for label, rows in selected.items():
        for index, item in enumerate(rows):
            positives = [candidate for candidate in rows if candidate is not item]
            if not positives:
                continue
            negatives = []
            for negative_label in TARGET_LABELS:
                if negative_label == label or not selected.get(negative_label):
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(selected[negative_label])["text"])
            records.append(
                {
                    "source": "cedr_ruizard_classifier_pseudolabel",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": label,
                        "raw_label": item["raw_label"],
                        "confidence": item["confidence"],
                        "source_file": item["source_file"],
                        "index": index,
                    },
                }
            )
    rng.shuffle(records)

    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "records": len(records),
            "requested": args.count,
            "model_name": args.model_name,
            "neutral_threshold": args.neutral_threshold,
            "emotion_threshold": args.emotion_threshold,
            "candidate_count": len(candidates),
            "available_by_label": {label: len(rows) for label, rows in pools.items()},
            "target_by_label": targets,
            "selected_by_label": {label: len(rows) for label, rows in selected.items()},
            "record_counts": dict(Counter(row["metadata"]["group"] for row in records)),
            "skipped": dict(skipped),
            "paths": [str(path) for path in paths],
            "construction": "high-confidence pseudo-labels from a public RuIzard emotion classifier over clean non-CEDR pools",
            "contamination_policy": "source pools are clean and exact/near CEDR overlap is checked again",
        },
    )
    print(f"prepared {out}: {len(records)} rows")


if __name__ == "__main__":
    main()
