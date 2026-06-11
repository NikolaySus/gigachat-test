#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

FIELD_TO_OECD_LABEL = {
    "Agricultural and Biological Sciences": "Сельское хозяйство, лесное хозяйство, рыбное хозяйство",
    "Biochemistry, Genetics and Molecular Biology": "Биологические науки",
    "Business, Management and Accounting": "Экономика и бизнес",
    "Chemical Engineering": "Химические технологии",
    "Chemistry": "Химические науки",
    "Computer Science": "Электротехника, электронная техника, информационные технологии",
    "Earth and Planetary Sciences": "Науки о Земле и смежные экологические науки",
    "Economics, Econometrics and Finance": "Экономика и бизнес",
    "Energy": "Энергетика и рациональное природопользование",
    "Engineering": "Механика и машиностроение",
    "Environmental Science": "Науки о Земле и смежные экологические науки",
    "Health Professions": "Науки о здоровье",
    "Materials Science": "Технологии материалов",
    "Mathematics": "Математика",
    "Medicine": "Клиническая медицина",
    "Neuroscience": "Биологические науки",
    "Physics and Astronomy": "Физика и астрономия",
    "Psychology": "Психологические науки",
    "Social Sciences": "Социологические науки",
    "Veterinary": "Ветеринарные науки",
}


SUBFIELD_TO_OECD_LABEL = {
    "Law": "Право",
    "Education": "Науки об образовании",
    "Political Science and International Relations": "Политологические науки",
    "Sociology and Political Science": "Социологические науки",
    "Communication": "СМИ и массовые коммуникации",
    "Cultural Studies": "Искусствоведение",
    "Language and Linguistics": "Языкознание и литература",
    "Archeology": "История и археология",
    "History": "История и археология",
    "Philosophy": "Философия, этика, религиоведение",
    "Architecture": "Строительство и архитектура",
    "Building and Construction": "Строительство и архитектура",
    "Geography, Planning and Development": "Социальная и экономическая география",
    "Development": "Социальная и экономическая география",
    "Public Health, Environmental and Occupational Health": "Науки о здоровье",
    "Civil and Structural Engineering": "Строительство и архитектура",
    "Electrical and Electronic Engineering": "Электротехника, электронная техника, информационные технологии",
    "Industrial and Manufacturing Engineering": "Механика и машиностроение",
    "Mechanical Engineering": "Механика и машиностроение",
    "Aerospace Engineering": "Механика и машиностроение",
    "Materials Chemistry": "Технологии материалов",
    "Food Science": "Сельское хозяйство, лесное хозяйство, рыбное хозяйство",
    "Animal Science and Zoology": "Животноводство и молочное дело",
}


def mapped_label(metadata: dict[str, Any]) -> str | None:
    subfield = str(metadata.get("openalex_subfield") or "")
    field = str(metadata.get("openalex_field") or "")
    return SUBFIELD_TO_OECD_LABEL.get(subfield) or FIELD_TO_OECD_LABEL.get(field)


def make_batches(
    rows: list[dict[str, Any]],
    *,
    batch_count: int,
    labels_per_batch: int,
    positives_per_label: int,
    min_docs_per_label: int,
    seed: int,
    loss: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for row in rows:
        metadata = row.get("metadata", {})
        label = mapped_label(metadata)
        if not label:
            skipped["unmapped_field_or_subfield"] += 1
            continue
        item = dict(row)
        item["mapped_oecd_label"] = label
        by_label[label].append(item)
    labels = [label for label, values in by_label.items() if len(values) >= min_docs_per_label]
    labels = sorted(labels, key=lambda label: len(by_label[label]), reverse=True)
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need {labels_per_batch} labels, got {len(labels)}")
    for label in labels:
        rng.shuffle(by_label[label])
    label_pool = labels[: max(labels_per_batch, min(len(labels), labels_per_batch * 8))]
    cursors = Counter()
    sampled = Counter()
    records: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        selected = rng.sample(label_pool, labels_per_batch)
        for label in selected:
            values = by_label[label]
            for text_index in range(positives_per_label):
                cursor = cursors[label] % len(values)
                source = values[cursor]
                cursors[label] += 1
                if cursors[label] % len(values) == 0:
                    rng.shuffle(values)
                metadata = dict(source.get("metadata", {}))
                metadata.update(
                    {
                        "mapped_oecd_label": label,
                        "batch_index": batch_index,
                        "text_index": text_index,
                        "contamination_policy": (
                            "Built from the RuSciBench-overlap-filtered OpenAlex dataset. "
                            "OpenAlex field/subfield names are mapped to public OECD-style "
                            "taxonomy labels; no RuSciBench rows or released model used."
                        ),
                    }
                )
                records.append(
                    {
                        "source": "openalex:ru_primary_topic_oecd_mapped",
                        "objective": "labeled_text",
                        "text": source["text"],
                        "label": f"oecd_mapped::{label}",
                        "loss": loss,
                        "metadata": metadata,
                    }
                )
                sampled[label] += 1
    return records, {
        "records": len(records),
        "usable_mapped_labels": labels,
        "label_pool": label_pool,
        "source_mapped_label_counts": {label: len(by_label[label]) for label in labels},
        "sampled_label_counts": dict(sampled),
        "skipped": dict(skipped),
        "batch_size": labels_per_batch * positives_per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/contrastive/openalex_ru_fos_subfield_circle_b8_3200_seed2641.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/openalex_ru_oecd_mapped_circle_b4_1600_seed2651.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=400)
    parser.add_argument("--labels-per-batch", type=int, default=2)
    parser.add_argument("--positives-per-label", type=int, default=2)
    parser.add_argument("--min-docs-per-label", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2651)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines()]
    records, summary = make_batches(
        rows,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        positives_per_label=args.positives_per_label,
        min_docs_per_label=args.min_docs_per_label,
        seed=args.seed,
        loss=args.loss,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary.update(
        {
            "source": str(args.source),
            "output": str(args.output),
            "field_mapping": FIELD_TO_OECD_LABEL,
            "subfield_mapping": SUBFIELD_TO_OECD_LABEL,
            "loss": args.loss,
            "seed": args.seed,
            "contamination_policy": (
                "Inherits exact text/title/prefix RuSciBench filtering from the source OpenAlex dataset."
            ),
        }
    )
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
