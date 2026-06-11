#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# Clean taxonomy mapping from GRNTI top-level labels to the public RuSciBench
# OECD-style label inventory. This uses label names only, not benchmark rows.
GRNTI_NUMBER_TO_OECD29 = {
    1: "Электротехника, электронная техника, информационные технологии",
    2: "Физика и астрономия",
    3: "Биологические науки",
    4: "Биологические науки",
    5: "Экономика и бизнес",
    6: "Экономика и бизнес",
    7: "Энергетика и рациональное природопользование",
    8: "Прочие технологии",
    9: "Науки о Земле и смежные экологические науки",
    10: "Науки о Земле и смежные экологические науки",
    11: "Науки о Земле и смежные экологические науки",
    12: "Науки о Земле и смежные экологические науки",
    13: "Энергетика и рациональное природопользование",
    14: "Право",
    15: "Социологические науки",
    16: "Строительство и архитектура",
    17: "Электротехника, электронная техника, информационные технологии",
    18: "Искусствоведение",
    19: "История и археология",
    20: "Электротехника, электронная техника, информационные технологии",
    21: "Социологические науки",
    22: "Физика и астрономия",
    23: "Искусствоведение",
    24: "Прочие технологии",
    25: "Сельское хозяйство, лесное хозяйство, рыбное хозяйство",
    26: "Языкознание и литература",
    27: "СМИ и массовые коммуникации",
    28: "Математика",
    29: "Механика и машиностроение",
    30: "Клиническая медицина",
    31: "Технологии материалов",
    32: "Прочие технологии",
    33: "Механика и машиностроение",
    34: "Науки об образовании",
    35: "Прочие технологии",
    36: "Прочие технологии",
    37: "Экономика и бизнес",
    38: "Науки о Земле и смежные экологические науки",
    39: "Прочие технологии",
    40: "Сельское хозяйство, лесное хозяйство, рыбное хозяйство",
    41: "Политологические науки",
    42: "Электротехника, электронная техника, информационные технологии",
    43: "Психологические науки",
    44: "Философия, этика, религиоведение",
    45: "Животноводство и молочное дело",
    46: "Электротехника, электронная техника, информационные технологии",
    47: "Сельское хозяйство, лесное хозяйство, рыбное хозяйство",
    48: "Социологические науки",
    49: "Математика",
    50: "Строительство и архитектура",
    51: "Механика и машиностроение",
    52: "Физика и астрономия",
    53: "Науки о здоровье",
    54: "Философия, этика, религиоведение",
    55: "Химические технологии",
    56: "Химические науки",
    57: "Экономика и бизнес",
    58: "Электротехника, электронная техника, информационные технологии",
    59: "Энергетика и рациональное природопользование",
    60: "Энергетика и рациональное природопользование",
    61: "Языкознание и литература",
    62: "Электротехника, электронная техника, информационные технологии",
}


def grnti_number(label: str) -> int | None:
    match = re.match(r"\s*(\d+)\b", label)
    return int(match.group(1)) if match else None


def read_source(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_oecd: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    source_counts = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            grnti_label = str(record.get("metadata", {}).get("grnti_label", "")).strip()
            number = grnti_number(grnti_label)
            oecd_label = GRNTI_NUMBER_TO_OECD29.get(number or -1)
            if oecd_label is None:
                skipped["unmapped_grnti"] += 1
                continue
            item = {
                "text": str(record["text"]),
                "grnti_label": grnti_label,
                "oecd_label": oecd_label,
                "file": record.get("metadata", {}).get("file"),
                "source_contamination_policy": record.get("metadata", {}).get("contamination_policy"),
            }
            by_oecd[oecd_label].append(item)
            source_counts[grnti_label] += 1
    return by_oecd, {
        "source_grnti_counts": dict(source_counts),
        "source_oecd_counts": {label: len(values) for label, values in by_oecd.items()},
        "skipped": dict(skipped),
    }


def make_batches(
    by_oecd: dict[str, list[dict[str, Any]]],
    *,
    batch_count: int,
    labels_per_batch: int,
    positives_per_label: int,
    min_docs_per_label: int,
    seed: int,
    loss: str,
    encode_batch_size: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    labels = [label for label, rows in by_oecd.items() if len(rows) >= min_docs_per_label]
    labels = sorted(labels, key=lambda label: len(by_oecd[label]), reverse=True)
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need {labels_per_batch} labels, got {len(labels)}")
    for label in labels:
        rng.shuffle(by_oecd[label])
    cursors = Counter()
    sampled = Counter()
    records: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        selected = rng.sample(labels, labels_per_batch)
        for label in selected:
            values = by_oecd[label]
            for text_index in range(positives_per_label):
                cursor = cursors[label] % len(values)
                item = values[cursor]
                cursors[label] += 1
                if cursors[label] % len(values) == 0:
                    rng.shuffle(values)
                records.append(
                    {
                        "source": "kaggle/ergkerg/russian-scientific-articles:grnti_to_oecd29_metric",
                        "objective": "labeled_text",
                        "text": item["text"],
                        "label": f"kaggle_oecd29::{label}",
                        "loss": loss,
                        **({"encode_batch_size": encode_batch_size} if encode_batch_size else {}),
                        "metadata": {
                            "mapped_oecd_label": label,
                            "grnti_label": item["grnti_label"],
                            "file": item["file"],
                            "batch_index": batch_index,
                            "text_index": text_index,
                            "contamination_policy": (
                                "Derived from audited Kaggle GRNTI article JSONL after "
                                "RuSciBench GRNTI/OECD title/prefix overlap removal. "
                                "GRNTI labels are mapped to the public RuSciBench OECD-style "
                                "label inventory using label names only; no benchmark rows or "
                                "released model are used."
                            ),
                        },
                    }
                )
                sampled[label] += 1
    return records, {
        "records": len(records),
        "usable_oecd_labels": labels,
        "sampled_oecd_counts": dict(sampled),
        "batch_size": labels_per_batch * positives_per_label,
        "encode_batch_size": encode_batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_labeled_circle_b4_1600_seed2551.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_to_oecd29_circle_b4_2400_seed2661.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=600)
    parser.add_argument("--labels-per-batch", type=int, default=2)
    parser.add_argument("--positives-per-label", type=int, default=2)
    parser.add_argument("--min-docs-per-label", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2661)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    parser.add_argument("--encode-batch-size", type=int, default=0)
    args = parser.parse_args()

    by_oecd, source_summary = read_source(args.source)
    records, batch_summary = make_batches(
        by_oecd,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        positives_per_label=args.positives_per_label,
        min_docs_per_label=args.min_docs_per_label,
        seed=args.seed,
        loss=args.loss,
        encode_batch_size=args.encode_batch_size or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "mapping": GRNTI_NUMBER_TO_OECD29,
        "loss": args.loss,
        "encode_batch_size": args.encode_batch_size or None,
        "seed": args.seed,
        "batch_count": args.batch_count,
        "labels_per_batch": args.labels_per_batch,
        "positives_per_label": args.positives_per_label,
        **source_summary,
        **batch_summary,
        "contamination_policy": (
            "Inherits RuSciBench GRNTI/OECD title/prefix overlap filtering from source JSONL."
        ),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
