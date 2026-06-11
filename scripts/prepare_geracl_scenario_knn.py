#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def label_id(scenario: str, class_name: str) -> str:
    return f"{scenario} :: {class_name}"


def build_records(
    source_rows: list[dict[str, Any]],
    *,
    seed: int,
    max_records: int,
    classes_per_episode: int,
    min_classes_per_scenario: int,
    batch_size: int,
    supports_per_class: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_scenario_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    skipped = Counter()

    for row in source_rows:
        metadata = row.get("metadata") or {}
        scenario = str(metadata.get("scenario") or "").strip()
        class_name = str(metadata.get("positive_class") or "").strip()
        if not scenario or not class_name:
            skipped["missing_metadata"] += 1
            continue
        if not row.get("query") or not row.get("positive"):
            skipped["missing_text"] += 1
            continue
        by_scenario_class[scenario][class_name].append(row)

    scenario_order = [
        scenario
        for scenario, classes in by_scenario_class.items()
        if len(classes) >= min_classes_per_scenario
    ]
    rng.shuffle(scenario_order)

    scenario_records: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenario_order:
        classes = by_scenario_class[scenario]
        class_names = list(classes)
        for class_rows in classes.values():
            rng.shuffle(class_rows)

        records: list[dict[str, Any]] = []
        query_pool = []
        for class_name, class_rows in classes.items():
            for row in class_rows:
                query_pool.append((class_name, row))
        rng.shuffle(query_pool)

        for class_name, row in query_pool:
            negatives = [name for name in class_names if name != class_name]
            if len(negatives) < classes_per_episode - 1:
                continue
            neg_classes = rng.sample(negatives, classes_per_episode - 1)
            episode_classes = [class_name, *neg_classes]
            supports: dict[str, list[str]] = {}
            for episode_class in episode_classes:
                candidates = classes[episode_class]
                support_rows = rng.sample(candidates, k=min(supports_per_class, len(candidates)))
                supports[label_id(scenario, episode_class)] = [
                    str(support_row["positive"]) for support_row in support_rows
                ]
            records.append(
                {
                    "source": "deepvk/GeRaCl_synthethic_dataset:scenario_knn",
                    "objective": "knn_classification",
                    "query": str(row["query"]),
                    "label": label_id(scenario, class_name),
                    "supports": supports,
                    "metadata": {
                        "scenario": scenario,
                        "positive_class": class_name,
                        "classes_per_episode": classes_per_episode,
                        "supports_per_class": supports_per_class,
                    },
                }
            )

        if len(records) >= batch_size:
            scenario_records[scenario] = records

    # Keep adjacent batches inside one scenario. The trainer's KNN loss builds
    # the class set across the batch, so grouped batches avoid many absent
    # labels and better approximate episodic classification.
    ordered: list[dict[str, Any]] = []
    scenario_counts: Counter[str] = Counter()
    while len(ordered) + batch_size <= max_records:
        progressed = False
        rng.shuffle(scenario_order)
        for scenario in scenario_order:
            records = scenario_records.get(scenario) or []
            if len(records) < batch_size or len(ordered) + batch_size > max_records:
                continue
            batch = records[:batch_size]
            del records[:batch_size]
            ordered.extend(batch)
            scenario_counts[scenario] += len(batch)
            progressed = True
        if not progressed:
            break

    summary = {
        "source_records": len(source_rows),
        "output_records": len(ordered),
        "seed": seed,
        "objective": "knn_classification",
        "classes_per_episode": classes_per_episode,
        "supports_per_class": supports_per_class,
        "batch_size_grouping": batch_size,
        "eligible_scenarios": len(scenario_records),
        "top_scenarios": scenario_counts.most_common(20),
        "skipped": dict(skipped),
        "contamination_policy": "Derived only from the previously audited clean GeRaCl source; no ruMTEB benchmark rows are introduced.",
    }
    return ordered, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_geracl.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_geracl_scenario_knn_c6_p1_b4_2400_seed2521.jsonl",
    )
    parser.add_argument("--seed", type=int, default=2521)
    parser.add_argument("--max-records", type=int, default=2400)
    parser.add_argument("--classes-per-episode", type=int, default=6)
    parser.add_argument("--min-classes-per-scenario", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--supports-per-class", type=int, default=1)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    records, summary = build_records(
        rows,
        seed=args.seed,
        max_records=args.max_records,
        classes_per_episode=args.classes_per_episode,
        min_classes_per_scenario=args.min_classes_per_scenario,
        batch_size=args.batch_size,
        supports_per_class=args.supports_per_class,
    )
    write_jsonl(args.output, records)
    write_json(args.output.with_name(args.output.stem + "_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
