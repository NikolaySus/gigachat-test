from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Need {count} records, got {len(records)}")
    records = records[:]
    random.Random(seed).shuffle(records)
    return records[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mix J: GeRaCl:Habr:DeepVK:GroundedStrict:GrandMaster = 2:1:1:1:1.")
    parser.add_argument("--geracl-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--habr-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"))
    parser.add_argument("--deepvk-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"))
    parser.add_argument("--grounded-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_grounded_rag_v2_q180_doc1200_neg2.jsonl"))
    parser.add_argument("--grandmaster-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_grandmaster_clustered_3200.jsonl"))
    parser.add_argument("--mix-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixj_geracl2_habr1_deepvk1_groundedstrict1_grandmaster1_19200.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()

    geracl = read_jsonl(args.geracl_path)
    habr = read_jsonl(args.habr_path)
    deepvk = read_jsonl(args.deepvk_path)
    grounded = read_jsonl(args.grounded_path)
    grandmaster = read_jsonl(args.grandmaster_path)

    selected = {
        "geracl": sample(geracl, count=6400, seed=args.seed),
        "habr_harder": sample(habr, count=3200, seed=args.seed + 1),
        "deepvk_filtered": sample(deepvk, count=3200, seed=args.seed + 2),
        "grounded_strict": sample(grounded, count=3200, seed=args.seed + 3),
        "grandmaster": sample(grandmaster, count=3200, seed=args.seed + 4),
    }

    mixed: list[dict[str, Any]] = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(args.seed + 5).shuffle(mixed)
    write_jsonl(args.mix_out, mixed)

    summary_path = args.summary_out or args.mix_out.with_name(args.mix_out.stem + "_summary.json")
    write_json(
        summary_path,
        {
            "output": str(args.mix_out),
            "seed": args.seed,
            "ratio": {
                "geracl": 2,
                "habr_harder": 1,
                "deepvk_ru_hnp_filtered": 1,
                "grounded_strict": 1,
                "grandmaster": 1,
            },
            "counts": {
                "geracl_source": len(geracl),
                "geracl_used": len(selected["geracl"]),
                "habr_harder_source": len(habr),
                "habr_harder_used": len(selected["habr_harder"]),
                "deepvk_filtered_source": len(deepvk),
                "deepvk_used": len(selected["deepvk_filtered"]),
                "grounded_strict_source": len(grounded),
                "grounded_strict_used": len(selected["grounded_strict"]),
                "grandmaster_source": len(grandmaster),
                "grandmaster_used": len(selected["grandmaster"]),
                "total": len(mixed),
            },
            "batch_size": 4,
            "max_steps_1x": len(mixed) // 4,
            "source_paths": {
                "geracl": str(args.geracl_path),
                "habr_harder": str(args.habr_path),
                "deepvk_filtered": str(args.deepvk_path),
                "grounded_strict": str(args.grounded_path),
                "grandmaster": str(args.grandmaster_path),
            },
        },
    )
    print(f"Wrote {args.mix_out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
