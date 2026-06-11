from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rapidfuzz import fuzz


TOKEN_RE = re.compile(r"[\w]+", re.U)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def normalize_text(value: Any) -> str:
    text = str(value).lower().replace("ё", "е")
    text = URL_RE.sub(" ", text)
    text = re.sub(r"[^\w\s\[\]#]+", " ", text, flags=re.U)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_text(text)))


def snippet(value: str, limit: int = 220) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def load_cedr(cache_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset = load_dataset("mteb/CEDRClassification", cache_dir=str(cache_dir), trust_remote_code=True)
    for split, ds in dataset.items():
        for idx, row in enumerate(ds):
            text = normalize_text(row["text"])
            if not text:
                continue
            rows.append(
                {
                    "dataset": "mteb/CEDRClassification",
                    "split": split,
                    "index": idx,
                    "text": text,
                    "raw_text": row["text"],
                    "label": row["label"],
                }
            )
    return rows


def load_brighter(cache_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", cache_dir=str(cache_dir))
    for split, ds in dataset.items():
        for idx, row in enumerate(ds):
            text = normalize_text(row["text"])
            if not text:
                continue
            labels = [name for name in EMOTION_LABELS if int(row.get(name, 0)) == 1]
            rows.append(
                {
                    "dataset": "brighter-dataset/BRIGHTER-emotion-categories:rus",
                    "split": split,
                    "index": idx,
                    "id": row.get("id"),
                    "text": text,
                    "raw_text": row["text"],
                    "labels": labels,
                }
            )
    return rows


def match_record(src: dict[str, Any], tgt: dict[str, Any], **scores: Any) -> dict[str, Any]:
    record = {
        "source_split": src["split"],
        "source_index": src["index"],
        "source_id": src.get("id"),
        "source_labels": src.get("labels", []),
        "cedr_split": tgt["split"],
        "cedr_index": tgt["index"],
        "cedr_label": tgt.get("label", []),
        "source_text": snippet(src["raw_text"]),
        "cedr_text": snippet(tgt["raw_text"]),
    }
    record.update(scores)
    return record


def exact_matches(source: list[dict[str, Any]], target: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in target:
        target_by_text[item["text"]].append(item)
    matches: list[dict[str, Any]] = []
    for src in source:
        for tgt in target_by_text.get(src["text"], []):
            matches.append(match_record(src, tgt, exact=1.0))
    return matches


def containment_matches(
    source: list[dict[str, Any]],
    target: list[dict[str, Any]],
    *,
    min_chars: int,
    max_matches: int,
) -> list[dict[str, Any]]:
    long_target = [item for item in target if len(item["text"]) >= min_chars]
    matches: list[dict[str, Any]] = []
    for src in source:
        if len(src["text"]) < min_chars:
            continue
        for tgt in long_target:
            if src["text"] == tgt["text"]:
                continue
            if src["text"] in tgt["text"] or tgt["text"] in src["text"]:
                matches.append(match_record(src, tgt, containment=True))
                break
        if len(matches) >= max_matches:
            break
    return matches


def lexical_near_matches(
    source: list[dict[str, Any]],
    target: list[dict[str, Any]],
    *,
    jaccard_threshold: float,
    fuzz_threshold: int,
    max_matches: int,
) -> list[dict[str, Any]]:
    indexed: list[tuple[dict[str, Any], set[str]]] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    for item in target:
        tokens = token_set(item["text"])
        if len(tokens) < 4:
            continue
        indexed.append((item, tokens))
        index = len(indexed) - 1
        for token in tokens:
            if len(token) >= 4:
                inverted[token].append(index)

    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, int]] = set()
    for src in source:
        src_tokens = token_set(src["text"])
        if len(src_tokens) < 4:
            continue
        candidate_counts: dict[int, int] = defaultdict(int)
        for token in src_tokens:
            if len(token) < 4:
                continue
            for index in inverted.get(token, []):
                candidate_counts[index] += 1

        for index, _ in sorted(candidate_counts.items(), key=lambda item: -item[1])[:300]:
            tgt, tgt_tokens = indexed[index]
            key = (src["split"], src["index"], tgt["split"], tgt["index"])
            if key in seen:
                continue
            intersection = len(src_tokens & tgt_tokens)
            union = len(src_tokens | tgt_tokens)
            jaccard = intersection / union if union else 0.0
            if jaccard < jaccard_threshold:
                continue
            ratio = fuzz.token_set_ratio(src["text"], tgt["text"])
            partial = fuzz.partial_ratio(src["text"], tgt["text"])
            if ratio < fuzz_threshold and partial < fuzz_threshold:
                continue
            seen.add(key)
            matches.append(match_record(src, tgt, jaccard=jaccard, fuzz_ratio=ratio, partial_ratio=partial))

    matches.sort(key=lambda item: (item.get("jaccard", 0.0), item.get("fuzz_ratio", 0.0)), reverse=True)
    return matches[:max_matches]


def write_markdown(path: Path, summary: dict[str, Any], samples: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# BRIGHTER Russian vs CEDR Contamination Audit",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    for section, rows in samples.items():
        lines.extend([f"## {section}", ""])
        if not rows:
            lines.extend(["No matches.", ""])
            continue
        lines.append("| BRIGHTER | CEDR | Scores |")
        lines.append("|---|---|---|")
        for row in rows[:50]:
            score_bits = []
            for key in ("exact", "containment", "jaccard", "fuzz_ratio", "partial_ratio"):
                if key in row:
                    value = row[key]
                    if isinstance(value, float):
                        score_bits.append(f"{key}={value:.4f}")
                    else:
                        score_bits.append(f"{key}={value}")
            lines.append(
                "| "
                + f"{row['source_split']}#{row['source_index']} {row['source_id']} {row['source_labels']}: {row['source_text']} "
                + "| "
                + f"{row['cedr_split']}#{row['cedr_index']} {row['cedr_label']}: {row['cedr_text']} "
                + "| "
                + ", ".join(score_bits)
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BRIGHTER Russian emotion data against CEDR benchmark text.")
    parser.add_argument("--cache-dir", type=Path, default=Path("results/mteb_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/contamination/brighter_cedr"))
    parser.add_argument("--jaccard-threshold", type=float, default=0.78)
    parser.add_argument("--fuzz-threshold", type=int, default=92)
    parser.add_argument("--max-matches", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cedr = load_cedr(args.cache_dir)
    brighter = load_brighter(args.cache_dir)

    exact = exact_matches(brighter, cedr)
    containment = containment_matches(brighter, cedr, min_chars=60, max_matches=args.max_matches)
    lexical = lexical_near_matches(
        brighter,
        cedr,
        jaccard_threshold=args.jaccard_threshold,
        fuzz_threshold=args.fuzz_threshold,
        max_matches=args.max_matches,
    )

    contaminated_keys = {
        (row["source_split"], row["source_index"])
        for rows in (exact, containment, lexical)
        for row in rows
    }
    label_counts = Counter(label for row in brighter for label in row["labels"])
    empty_count = sum(1 for row in brighter if not row["labels"])
    clean_label_counts = Counter(
        label
        for row in brighter
        if (row["split"], row["index"]) not in contaminated_keys
        for label in row["labels"]
    )
    clean_empty_count = sum(
        1 for row in brighter if not row["labels"] and (row["split"], row["index"]) not in contaminated_keys
    )

    summary = {
        "cedr_rows_total": len(cedr),
        "cedr_train_rows": sum(1 for row in cedr if row["split"] == "train"),
        "cedr_test_rows": sum(1 for row in cedr if row["split"] == "test"),
        "brighter_rows_total": len(brighter),
        "brighter_train_rows": sum(1 for row in brighter if row["split"] == "train"),
        "brighter_dev_rows": sum(1 for row in brighter if row["split"] == "dev"),
        "brighter_test_rows": sum(1 for row in brighter if row["split"] == "test"),
        "exact_matches": len(exact),
        "containment_matches": len(containment),
        "lexical_near_matches": len(lexical),
        "unique_brighter_rows_flagged": len(contaminated_keys),
        "brighter_rows_clean_after_filters": len(brighter) - len(contaminated_keys),
        "brighter_empty_label_rows": empty_count,
        "clean_brighter_empty_label_rows": clean_empty_count,
        "brighter_label_counts": dict(label_counts),
        "clean_brighter_label_counts": dict(clean_label_counts),
        "jaccard_threshold": args.jaccard_threshold,
        "fuzz_threshold": args.fuzz_threshold,
    }

    result = {
        "summary": summary,
        "matches": {
            "exact": exact,
            "containment": containment,
            "lexical_near": lexical,
        },
    }
    (args.output_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    flagged_rows = [
        row for row in brighter if (row["split"], row["index"]) in contaminated_keys
    ]
    (args.output_dir / "flagged_brighter_rows.json").write_text(
        json.dumps(flagged_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "audit.md", summary, result["matches"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_dir / 'audit.md'}")


if __name__ == "__main__":
    main()
