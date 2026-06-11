from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from prepare_open_ru_sts_v2 import ru_hnp_records
from prepare_open_ru_sts_v13_recovery import (
    GERACL_CLASSES_SOURCE,
    GERACL_POSITIVES_SOURCE,
    PROMPTRIEVER_SOURCE,
    geracl_synthetic_class_records,
    geracl_synthetic_positive_records,
    promptriever_records,
)
from prepare_open_ru_sts_v14_grounded_rag import SOURCE as GROUNDED_RAG_SOURCE
from prepare_open_ru_sts_v14_grounded_rag import build_records as grounded_rag_records
from prepare_open_ru_sts_v15_habr_qa_sbs import SOURCE as HABR_QA_SBS_SOURCE
from prepare_open_ru_sts_v15_habr_qa_sbs import build_records as habr_qa_sbs_records


RU_HNP_SOURCE = "deepvk/ru-HNP"
GERACL_SOURCE = "deepvk/GeRaCl_synthethic_dataset"


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            record.setdefault("objective", "contrastive")
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_records(records: list[dict], count: int, rng: random.Random) -> list[dict]:
    shuffled = records[:]
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def normalize_source(records: list[dict], source: str) -> list[dict]:
    normalized = []
    for record in records:
        item = dict(record)
        item["source"] = source
        normalized.append(item)
    return normalized


def write_summary(path: Path, *, source_records: dict[str, list[dict]], mixed_records: list[dict], args) -> None:
    summary = {
        "name": "open_ru_1r_nc",
        "seed": args.seed,
        "batch_size_assumption": args.batch_size,
        "mixed_records": len(mixed_records),
        "mixed_steps_per_source": args.mixed_steps_per_source,
        "target_records_per_source": args.target_steps_per_source * args.batch_size,
        "source_counts": {
            source: len(records)
            for source, records in source_records.items()
        },
        "mixed_counts": dict(Counter(record["source"] for record in mixed_records)),
        "outputs": {
            "mixed": str(args.mixed_out),
            "ru_hnp": str(args.ru_hnp_out),
            "geracl": str(args.geracl_out),
            "promptriever": str(args.promptriever_out),
            "grounded_rag": str(args.grounded_rag_out),
            "habr_qa_sbs_hard": str(args.habr_out),
        },
        "contamination_policy": [
            "Only sources previously marked as no known ruMTEB contamination are used.",
            "GeRaCl ru_mteb_classes and ru_mteb_extended_classes configs are not loaded.",
            "Habr QA SBS uses the hard lexical-similarity filter from STS-v16.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare no-contamination open-RU data for fair 1R-NC reinit experiments.")
    parser.add_argument("--mixed-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_train.jsonl"))
    parser.add_argument("--ru-hnp-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_deepvk_ru_hnp.jsonl"))
    parser.add_argument("--geracl-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--promptriever-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_promptriever.jsonl"))
    parser.add_argument("--grounded-rag-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_grounded_rag_v2.jsonl"))
    parser.add_argument("--habr-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_hard.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_summary.json"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mixed-steps-per-source", type=int, default=200)
    parser.add_argument("--target-steps-per-source", type=int, default=2000)
    parser.add_argument("--negatives-per-record", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    rng = random.Random(args.seed)
    target_records = args.target_steps_per_source * args.batch_size
    mixed_records_per_source = args.mixed_steps_per_source * args.batch_size

    ru_hnp = normalize_source(
        ru_hnp_records(
            limit=max(1, target_records // 3 + 512),
            negatives_per_record=args.negatives_per_record,
            positives_per_query=2,
            rng=rng,
        )[:target_records],
        RU_HNP_SOURCE,
    )
    geracl = normalize_source(
        (
            geracl_synthetic_class_records(target_records // 2, 2, rng)
            + geracl_synthetic_positive_records(target_records // 2, 2, rng)
        )[:target_records],
        GERACL_SOURCE,
    )
    promptriever = normalize_source(
        promptriever_records(target_records, args.negatives_per_record),
        PROMPTRIEVER_SOURCE,
    )
    grounded_rag, _grounded_summary = grounded_rag_records(
        limit=target_records,
        negatives_per_record=2,
        random_negatives_per_record=1,
        doc_max_chars=3500,
        seed=args.seed,
    )
    grounded_rag = normalize_source(grounded_rag, GROUNDED_RAG_SOURCE)
    habr, _habr_summary = habr_qa_sbs_records(
        limit=target_records,
        seed=args.seed,
        min_question_chars=25,
        min_best_chars=120,
        min_bad_chars=80,
        min_best_words=12,
        min_bad_words=8,
        max_answer_chars=3000,
        min_best_bad_similarity=0.18,
        max_best_bad_similarity=0.86,
    )
    habr = normalize_source(habr, HABR_QA_SBS_SOURCE + ":hard")

    source_records = {
        RU_HNP_SOURCE: ru_hnp,
        GERACL_SOURCE: geracl,
        PROMPTRIEVER_SOURCE: promptriever,
        GROUNDED_RAG_SOURCE: grounded_rag,
        HABR_QA_SBS_SOURCE + ":hard": habr,
    }
    mixed_records = []
    for records in source_records.values():
        mixed_records.extend(sample_records(records, mixed_records_per_source, rng))
    rng.shuffle(mixed_records)

    write_jsonl(args.ru_hnp_out, ru_hnp)
    write_jsonl(args.geracl_out, geracl)
    write_jsonl(args.promptriever_out, promptriever)
    write_jsonl(args.grounded_rag_out, grounded_rag)
    write_jsonl(args.habr_out, habr)
    write_jsonl(args.mixed_out, mixed_records)
    write_summary(args.summary_out, source_records=source_records, mixed_records=mixed_records, args=args)

    print(f"Wrote {len(mixed_records)} mixed records to {args.mixed_out}")
    for source, records in source_records.items():
        print(f"{source}: {len(records)} records")
    print(f"Wrote summary to {args.summary_out}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
