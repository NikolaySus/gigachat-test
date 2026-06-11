from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_cedr_neutral_lexical_distractors import (
    DEFAULT_SOURCES,
    LEXEMES,
    clean_text,
    detect_groups,
    fields_from_record,
    quality_ok,
    read_jsonl,
)
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]

LABEL_STATEMENTS = {
    "neutral": "В комментарии нет явной эмоции из списка: радость, грусть, удивление, страх или злость.",
    "joy": "В комментарии выражена радость или положительная эмоция.",
    "sadness": "В комментарии выражена грусть, печаль или тоска.",
    "surprise": "В комментарии выражено удивление, шок или неожиданность.",
    "fear": "В комментарии выражен страх, тревога или опасение.",
    "anger": "В комментарии выражена злость, раздражение или гнев.",
}

REPORTING_MARKERS = re.compile(
    r"\b("
    r"сообщил\w*|заявил\w*|рассказал\w*|отметил\w*|указал\w*|пишет|пишут|"
    r"исследован\w*|опрос\w*|статистик\w*|рейтинг\w*|данн\w*|"
    r"причин\w*|связан\w*|из-за|по поводу|в случае|при этом|"
    r"демонстрац\w*|обсужден\w*|анализ\w*|провер\w*"
    r")\b",
    re.IGNORECASE,
)


def statement_text(label: str) -> str:
    return CEDR_PREFIX + LABEL_STATEMENTS[label]


def collect_neutral_candidates(
    paths: list[Path],
    *,
    cedr_index: Any,
    require_reporting_marker: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = Counter()
    for path in paths:
        if not path.exists():
            skipped[f"missing:{path.name}"] += 1
            continue
        for record_index, record in enumerate(read_jsonl(path)):
            source = str(record.get("source") or path.stem)
            for text_index, raw_text in enumerate(fields_from_record(record)):
                text = clean_text(raw_text)
                normalized = normalize_text(text)
                if normalized in seen:
                    skipped["duplicate"] += 1
                    continue
                seen.add(normalized)
                groups = detect_groups(text)
                if not groups:
                    skipped["no_lexeme"] += 1
                    continue
                if len(groups) > 2:
                    skipped["too_many_lexeme_groups"] += 1
                    continue
                if not quality_ok(text):
                    skipped["quality"] += 1
                    continue
                if require_reporting_marker and not REPORTING_MARKERS.search(text):
                    skipped["not_report_like"] += 1
                    continue
                if is_contaminated(text, cedr_index):
                    skipped["cedr_overlap"] += 1
                    continue
                candidates.append(
                    {
                        "text": text,
                        "trigger_groups": groups,
                        "source": source,
                        "path": str(path),
                        "record_index": record_index,
                        "text_index": text_index,
                    }
                )
    return candidates, {"skipped": dict(skipped)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build explicit neutral-label pair-score rows from clean non-CEDR emotion-lexeme texts."
    )
    parser.add_argument("--source", type=Path, action="append", default=None)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--name", default="cedr_neutral_label_pairscore_reported_3000")
    parser.add_argument("--false-labels-per-row", type=int, default=5)
    parser.add_argument("--positive-score", type=float, default=1.0)
    parser.add_argument("--negative-score", type=float, default=0.0)
    parser.add_argument(
        "--require-reporting-marker",
        action="store_true",
        help="Keep only report-like factual contexts. Off by default to retain more neutral lexical boundary cases.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    paths = args.source or [
        *DEFAULT_SOURCES,
        DATA_DIR / "open_ru_1r_nc_geracl.jsonl",
        DATA_DIR / "open_ru_1r_nc_grounded_rag_v2_q180_doc1200_neg2.jsonl",
        DATA_DIR / "open_ru_1r_nc_grandmaster_clustered_3200.jsonl",
    ]
    cedr_index = load_cedr_index()
    candidates, meta = collect_neutral_candidates(
        paths,
        cedr_index=cedr_index,
        require_reporting_marker=args.require_reporting_marker,
    )
    rng.shuffle(candidates)
    selected = candidates[: args.count]

    records: list[dict[str, Any]] = []
    false_labels = ["joy", "sadness", "surprise", "fear", "anger"]
    for item in selected:
        sentence1 = CEDR_PREFIX + item["text"]
        records.append(
            {
                "source": f"neutral_label_pairscore:{item['source']}",
                "objective": "pair_score",
                "sentence1": sentence1,
                "sentence2": statement_text("neutral"),
                "score": args.positive_score,
                "metadata": {
                    "group": "neutral",
                    "kind": "neutral_label_positive",
                    "trigger_groups": item["trigger_groups"],
                    "path": item["path"],
                    "record_index": item["record_index"],
                    "text_index": item["text_index"],
                },
            }
        )
        ordered_false = false_labels[:]
        rng.shuffle(ordered_false)
        for label in ordered_false[: args.false_labels_per_row]:
            records.append(
                {
                    "source": f"neutral_label_pairscore:{item['source']}",
                    "objective": "pair_score",
                    "sentence1": sentence1,
                    "sentence2": statement_text(label),
                    "score": args.negative_score,
                    "metadata": {
                        "group": "neutral",
                        "kind": "emotion_label_negative",
                        "negative_label": label,
                        "trigger_groups": item["trigger_groups"],
                        "path": item["path"],
                        "record_index": item["record_index"],
                        "text_index": item["text_index"],
                    },
                }
            )

    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    trigger_counts = Counter(group for item in selected for group in item["trigger_groups"])
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "output": str(out.relative_to(ROOT)),
            "source_paths": [str(path) for path in paths],
            "candidate_count": len(candidates),
            "selected_texts": len(selected),
            "records": len(records),
            "false_labels_per_row": args.false_labels_per_row,
            "require_reporting_marker": args.require_reporting_marker,
            "trigger_counts": dict(trigger_counts),
            "record_kinds": dict(Counter(record["metadata"]["kind"] for record in records)),
            "skipped": meta["skipped"],
            "lexemes": {key: value.pattern for key, value in LEXEMES.items()},
            "construction": "neutral emotion-lexeme text is paired with explicit neutral statement and low-score emotion label statements",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records or labels used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows from {len(selected)} texts")


if __name__ == "__main__":
    main()
