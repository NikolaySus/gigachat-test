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
    rows: list[dict[str, Any]] = []
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


def parse_component(value: str) -> tuple[str, Path, int, str | None]:
    parts = value.split(":", 3)
    if len(parts) not in {3, 4}:
        raise ValueError("Component must be label:path:count[:source_contains]")
    label, path, count = parts[:3]
    contains = parts[3] if len(parts) == 4 else None
    return label, Path(path), int(count), contains


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic weighted JSONL mix.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--component", action="append", required=True, help="label:path:count[:source_contains]")
    parser.add_argument("--seed", type=int, default=1237)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"name": args.name, "seed": args.seed, "components": []}
    for label, path, count, contains in [parse_component(item) for item in args.component]:
        rows = read_jsonl(path)
        if contains:
            rows = [row for row in rows if contains in str(row.get("source", ""))]
        rng.shuffle(rows)
        selected = rows[: min(count, len(rows))]
        for index, row in enumerate(selected):
            mixed = dict(row)
            mixed["source"] = f"weighted_mix:{args.name}:{label}:{row.get('source', '')}"
            metadata = dict(mixed.get("metadata") or {})
            metadata["mix_component"] = label
            metadata["mix_source_index"] = index
            mixed["metadata"] = metadata
            records.append(mixed)
        summary["components"].append(
            {
                "label": label,
                "path": str(path),
                "source_contains": contains,
                "available": len(rows),
                "requested": count,
                "selected": len(selected),
                "trigger_counts": dict(Counter((row.get("metadata") or {}).get("trigger_group", "none") for row in selected)),
            }
        )

    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    summary["records"] = len(records)
    summary["objective_counts"] = dict(Counter(row.get("objective", "unknown") for row in records))
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
