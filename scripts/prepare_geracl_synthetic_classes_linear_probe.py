#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARROW = Path(
    "/home/nop/.cache/huggingface/datasets/deepvk___ge_ra_cl_synthethic_dataset/"
    "synthetic_classes_train/0.0.0/2f6c644f75d55df8043bc3e875c2aa3073cbde8d/"
    "ge_ra_cl_synthethic_dataset-train.arrow"
)


def clean_text(text: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text.replace("\ufeff", " ")).strip()
    return text[:max_chars].strip()


def usable_text(text: str, *, min_chars: int, max_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    sample = text[:max_chars]
    letters = sum(ch.isalpha() for ch in sample)
    cyrillic = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in sample)
    if letters < 20:
        return False
    return cyrillic / max(letters, 1) >= 0.45


def scenario_rows(
    *,
    arrow_path: Path,
    max_source_rows: int,
    min_chars: int,
    max_chars: int,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    dataset = Dataset.from_file(str(arrow_path))
    by_scenario_label: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    skipped = Counter()
    seen: set[tuple[str, str, str]] = set()

    for index, row in enumerate(dataset):
        if max_source_rows > 0 and index >= max_source_rows:
            break
        text = clean_text(str(row.get("text") or ""), max_chars=max_chars)
        if not usable_text(text, min_chars=min_chars, max_chars=max_chars):
            skipped["low_quality_text"] += 1
            continue
        scenarios = row.get("scenarios") or []
        for scenario_index, scenario in enumerate(scenarios[:5]):
            classes = row.get(f"classes_{scenario_index}") or []
            if not classes:
                skipped["missing_classes"] += 1
                continue
            # In GeRaCl synthetic_classes_train the first class is the generated
            # correct class; remaining classes are candidate distractors.
            label = str(classes[0]).strip()
            scenario = str(scenario).strip()
            if not scenario or not label:
                skipped["missing_label"] += 1
                continue
            key = (scenario, label, text[:200])
            if key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(key)
            by_scenario_label[scenario][label].append(text)

    return by_scenario_label, {
        "arrow_path": str(arrow_path),
        "source_rows_seen": min(len(dataset), max_source_rows) if max_source_rows > 0 else len(dataset),
        "skipped": dict(skipped),
        "scenarios": len(by_scenario_label),
        "raw_label_counts_by_scenario": {
            scenario: len(labels) for scenario, labels in by_scenario_label.items()
        },
    }


def build_batches(
    by_scenario_label: dict[str, dict[str, list[str]]],
    *,
    batch_count: int,
    labels_per_batch: int,
    supports_per_label: int,
    queries_per_label: int,
    min_records_per_label: int,
    seed: int,
    ridge_lambda: float,
    encode_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    usable_scenarios: dict[str, list[str]] = {}
    for scenario, by_label in by_scenario_label.items():
        labels = [label for label, texts in by_label.items() if len(texts) >= min_records_per_label]
        if len(labels) >= labels_per_batch:
            usable_scenarios[scenario] = labels
            for label in labels:
                rng.shuffle(by_label[label])
    if not usable_scenarios:
        raise ValueError("No usable scenarios after filtering")

    scenarios = sorted(usable_scenarios, key=lambda item: len(usable_scenarios[item]), reverse=True)
    cursors: dict[tuple[str, str], int] = defaultdict(int)
    sampled = Counter()
    records: list[dict[str, Any]] = []
    per_label = supports_per_label + queries_per_label

    for batch_index in range(batch_count):
        scenario = scenarios[batch_index % len(scenarios)]
        if batch_index and batch_index % len(scenarios) == 0:
            rng.shuffle(scenarios)
        labels = rng.sample(usable_scenarios[scenario], labels_per_batch)
        for label in labels:
            texts = by_scenario_label[scenario][label]
            for item_index in range(per_label):
                cursor_key = (scenario, label)
                cursor = cursors[cursor_key] % len(texts)
                text = texts[cursor]
                cursors[cursor_key] += 1
                if cursors[cursor_key] % len(texts) == 0:
                    rng.shuffle(texts)
                role = "support" if item_index < supports_per_label else "query"
                full_label = f"geracl_synth::{scenario}::{label}"
                records.append(
                    {
                        "source": "deepvk/GeRaCl_synthethic_dataset:synthetic_classes_train_linear_probe_clean",
                        "objective": "linear_probe_labeled_text",
                        "text": text,
                        "label": full_label,
                        "role": role,
                        "ridge_lambda": ridge_lambda,
                        "use_bias": True,
                        "encode_batch_size": encode_batch_size,
                        "metadata": {
                            "scenario": scenario,
                            "positive_class": label,
                            "batch_index": batch_index,
                            "role": role,
                            "contamination_policy": (
                                "Derived only from deepvk/GeRaCl_synthethic_dataset "
                                "synthetic_classes_train. The ru_mteb_classes GeRaCl config is "
                                "explicitly excluded because it contains benchmark-like label schemas. "
                                "No ruMTEB rows, released outputs, released teacher, or released latent "
                                "weights are used."
                            ),
                        },
                    }
                )
                sampled[(scenario, label)] += 1

    return records, {
        "records": len(records),
        "batch_count": batch_count,
        "batch_size": labels_per_batch * per_label,
        "labels_per_batch": labels_per_batch,
        "supports_per_label": supports_per_label,
        "queries_per_label": queries_per_label,
        "usable_scenarios": len(usable_scenarios),
        "usable_labels_total": sum(len(labels) for labels in usable_scenarios.values()),
        "top_scenarios": {scenario: len(usable_scenarios[scenario]) for scenario in scenarios[:30]},
        "sampled_pairs": len(sampled),
        "sampled_pair_examples": {
            f"{scenario}::{label}": count
            for (scenario, label), count in sampled.most_common(30)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-path", type=Path, default=DEFAULT_ARROW)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/fair_geracl_synth_classes_linear_probe_b12_60_seed3091.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=60)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--supports-per-label", type=int, default=2)
    parser.add_argument("--queries-per-label", type=int, default=1)
    parser.add_argument("--min-records-per-label", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--max-source-rows", type=int, default=60000)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3091)
    args = parser.parse_args()

    by_scenario_label, source_summary = scenario_rows(
        arrow_path=args.arrow_path,
        max_source_rows=args.max_source_rows,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    records, batch_summary = build_batches(
        by_scenario_label,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        supports_per_label=args.supports_per_label,
        queries_per_label=args.queries_per_label,
        min_records_per_label=args.min_records_per_label,
        seed=args.seed,
        ridge_lambda=args.ridge_lambda,
        encode_batch_size=args.encode_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "seed": args.seed,
        "source_summary": source_summary,
        "batch_summary": batch_summary,
        "fairness": (
            "Clean synthetic_classes_train only. ru_mteb_classes and ru_mteb_extended_classes "
            "are not used."
        ),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
