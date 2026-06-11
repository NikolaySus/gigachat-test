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


def shuffled(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    records = records[:]
    random.Random(seed).shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mix L: TERRa-oriented broad mix with clean sensitive source combo.")
    parser.add_argument("--geracl-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--habr-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"))
    parser.add_argument("--deepvk-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"))
    parser.add_argument("--grounded-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_grounded_rag_v2_q180_doc1200_neg2.jsonl"))
    parser.add_argument("--sensitive-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_sensitive_topic_mvrcii_uc_berkeley_3200.jsonl"))
    parser.add_argument("--mix-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixl_geracl12800_habrall_deepvkall_grounded3200_sensitive_mvrcii_ucb_27020.jsonl"))
    parser.add_argument("--geracl-remaining-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixl_geracl_remaining_seed131_3200.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--geracl-count", type=int, default=12800)
    parser.add_argument("--grounded-count", type=int, default=3200)
    parser.add_argument("--batch-size-stage1", type=int, default=4)
    parser.add_argument("--batch-size-stage2", type=int, default=8)
    args = parser.parse_args()

    geracl_all = read_jsonl(args.geracl_path)
    habr_all = read_jsonl(args.habr_path)
    deepvk_all = read_jsonl(args.deepvk_path)
    grounded_all = read_jsonl(args.grounded_path)
    sensitive_all = read_jsonl(args.sensitive_path)

    if len(geracl_all) < args.geracl_count:
        raise ValueError(f"Need {args.geracl_count} GeRaCl records, got {len(geracl_all)}")
    if len(grounded_all) < args.grounded_count:
        raise ValueError(f"Need {args.grounded_count} Grounded records, got {len(grounded_all)}")

    geracl_shuffled = shuffled(geracl_all, seed=args.seed)
    geracl_stage1 = geracl_shuffled[: args.geracl_count]
    geracl_remaining = geracl_shuffled[args.geracl_count :]
    habr = shuffled(habr_all, seed=args.seed + 1)
    deepvk = shuffled(deepvk_all, seed=args.seed + 2)
    grounded = shuffled(grounded_all, seed=args.seed + 3)[: args.grounded_count]
    sensitive = shuffled(sensitive_all, seed=args.seed + 4)

    selected = {
        "geracl": geracl_stage1,
        "habr_harder": habr,
        "deepvk_filtered": deepvk,
        "grounded_strict": grounded,
        "sensitive_mvrcii_uc_berkeley": sensitive,
    }
    mixed: list[dict[str, Any]] = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(args.seed + 5).shuffle(mixed)

    write_jsonl(args.mix_out, mixed)
    write_jsonl(args.geracl_remaining_out, geracl_remaining)

    summary_path = args.summary_out or args.mix_out.with_name(args.mix_out.stem + "_summary.json")
    write_json(
        summary_path,
        {
            "output": str(args.mix_out),
            "geracl_remaining_output": str(args.geracl_remaining_out),
            "seed": args.seed,
            "ratio_target": {
                "geracl": 4,
                "habr_harder": "all_unique",
                "deepvk_ru_hnp_filtered": "all_unique",
                "grounded_strict": 1,
                "sensitive_topic_discrimination_mvrcii_uc_berkeley": 1,
            },
            "counts": {
                "geracl_source": len(geracl_all),
                "geracl_used": len(geracl_stage1),
                "geracl_remaining": len(geracl_remaining),
                "habr_harder_source": len(habr_all),
                "habr_harder_used": len(habr),
                "deepvk_filtered_source": len(deepvk_all),
                "deepvk_used": len(deepvk),
                "grounded_strict_source": len(grounded_all),
                "grounded_strict_used": len(grounded),
                "sensitive_source": len(sensitive_all),
                "sensitive_used": len(sensitive),
                "total_stage1": len(mixed),
            },
            "batch_size_stage1": args.batch_size_stage1,
            "batch_size_stage2": args.batch_size_stage2,
            "max_steps_stage1_1x": len(mixed) // args.batch_size_stage1,
            "max_steps_stage2_1x": len(geracl_remaining) // args.batch_size_stage2,
            "source_paths": {
                "geracl": str(args.geracl_path),
                "habr_harder": str(args.habr_path),
                "deepvk_filtered": str(args.deepvk_path),
                "grounded_strict": str(args.grounded_path),
                "sensitive_topic_discrimination_mvrcii_uc_berkeley": str(args.sensitive_path),
            },
        },
    )
    print(f"Wrote {args.mix_out}")
    print(f"Wrote {args.geracl_remaining_out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
