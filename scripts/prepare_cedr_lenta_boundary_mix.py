from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows: list[dict[str, Any]], *, count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= len(rows):
        return rng.sample(rows, count)
    output = rows[:]
    while len(output) < count:
        output.extend(rng.sample(rows, min(len(rows), count - len(output))))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Lenta-centered CEDR boundary mix.")
    parser.add_argument("--seed", type=int, default=903)
    parser.add_argument("--name", default="cedr_lenta_boundary_mix_v1_9600")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    components = [
        (
            "lenta_reported_x2",
            DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl",
            6400,
        ),
        (
            "semeval2025_rus",
            DATA_DIR / "open_ru_1r_nc_cedr_semeval2025_rus_tracka_train_dev_clean.jsonl",
            1600,
        ),
        (
            "resd_dialog",
            DATA_DIR / "open_ru_1r_nc_cedr_resd_clean_emotion_component.jsonl",
            800,
        ),
        (
            "neutral_pairscore_reported",
            DATA_DIR / "open_ru_1r_nc_cedr_neutral_label_pairscore_reported_3000.jsonl",
            800,
        ),
    ]

    mixed = []
    summary_components = {}
    for label, path, count in components:
        rows = read_jsonl(path)
        selected = sample_rows(rows, count=count, rng=rng)
        for row in selected:
            row = dict(row)
            row["source"] = f"cedr_lenta_boundary_mix_v1:{label}:{row.get('source', '')}"
            mixed.append(row)
        summary_components[label] = {
            "path": str(path.relative_to(ROOT)),
            "available": len(rows),
            "selected": len(selected),
            "objectives": dict(Counter(row.get("objective", "contrastive") for row in selected)),
        }

    rng.shuffle(mixed)
    output_path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(output_path, mixed)
    summary = {
        "name": args.name,
        "records": len(mixed),
        "seed": args.seed,
        "components": summary_components,
        "objective_counts": dict(Counter(row.get("objective", "contrastive") for row in mixed)),
        "contamination_policy": "inherits exact/near CEDR filtering from component datasets; no CEDR rows used",
        "motivation": "Keep the only useful Lenta neutral-boundary signal dominant, while adding small clean Russian emotion/boundary support.",
    }
    output_path.with_name(output_path.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
