from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[\w]+", re.U)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


def normalize_text(value: Any) -> str:
    text = str(value).lower().replace("ё", "е")
    text = URL_RE.sub(" ", text)
    text = re.sub(r"[^\w\s\[\]#]+", " ", text, flags=re.U)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_text(text)))


def snippet(value: str, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "..."


def iter_dataset(dataset: DatasetDict):
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            yield split, index, row


def load_cedr(cache_dir: Path) -> list[dict[str, Any]]:
    dataset = load_dataset("mteb/CEDRClassification", cache_dir=str(cache_dir), trust_remote_code=True)
    rows = []
    for split, index, row in iter_dataset(dataset):
        text = normalize_text(row["text"])
        if text:
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "text": text,
                    "raw_text": row["text"],
                    "label": row["label"],
                }
            )
    return rows


def load_candidate(name: str, cache_dir: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if name == "brighter":
        dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", cache_dir=str(cache_dir))
        label_names = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        for split, index, row in iter_dataset(dataset):
            text = normalize_text(row["text"])
            labels = [label for label in label_names if int(row.get(label, 0)) == 1]
            rows.append({"split": split, "index": index, "text": text, "raw_text": row["text"], "labels": labels})
    elif name == "go_ekman":
        dataset = load_dataset("SkyWater21/ru_go_emotions_ekman", "simplified_ekman", cache_dir=str(cache_dir))
        label_names = dataset["train"].features["labels_ekman"].feature.names
        for split, index, row in iter_dataset(dataset):
            text = normalize_text(row["ru_text"])
            labels = [label_names[label] for label in row["labels_ekman"]]
            rows.append({"split": split, "index": index, "text": text, "raw_text": row["ru_text"], "labels": labels})
    elif name == "go_fine":
        dataset = load_dataset("seara/ru_go_emotions", "simplified", cache_dir=str(cache_dir))
        label_names = dataset["train"].features["labels"].feature.names
        for split, index, row in iter_dataset(dataset):
            text = normalize_text(row["ru_text"])
            labels = [label_names[label] for label in row["labels"]]
            rows.append({"split": split, "index": index, "text": text, "raw_text": row["ru_text"], "labels": labels})
    elif name == "ru_sentiment_social":
        dataset = load_dataset("DmitrySharonov/ru_sentiment_neg_pos_neutral", cache_dir=str(cache_dir))
        for split, index, row in iter_dataset(dataset):
            text = normalize_text(row["text"])
            rows.append({"split": split, "index": index, "text": text, "raw_text": row["text"], "labels": [row["label"]]})
    elif name == "rusentiment_union":
        dataset = load_dataset("Megnis/RuSentimentUnion", cache_dir=str(cache_dir))
        for split, index, row in iter_dataset(dataset):
            text = normalize_text(row["text"])
            rows.append({"split": split, "index": index, "text": text, "raw_text": row["text"], "labels": [row["label"]]})
    elif name == "twitter_emotions_ekman":
        dataset = load_dataset("AiLab-IMCS-UL/twitter_emotions-ru", "simplified_ekman", cache_dir=str(cache_dir))
        for split, index, row in iter_dataset(dataset):
            text = normalize_text(row["ru_text"])
            label = dataset[split].features["labels_ekman"].int2str(row["labels_ekman"])
            rows.append({"split": split, "index": index, "text": text, "raw_text": row["ru_text"], "labels": [label]})
    elif name == "ruemotions":
        dataset = load_dataset("Darkester/RuEmotions", cache_dir=str(cache_dir))
        for split, index, row in iter_dataset(dataset):
            text = normalize_text(row["text"])
            rows.append({"split": split, "index": index, "text": text, "raw_text": row["text"], "labels": [row["emotion"]]})
    elif name == "rusentitweet":
        dataset = load_dataset("psytechlab/RuSentiTweet", cache_dir=str(cache_dir))
        for split, index, row in iter_dataset(dataset):
            text = normalize_text(row["text"])
            rows.append({"split": split, "index": index, "text": text, "raw_text": row["text"], "labels": [row["label"]]})
    else:
        raise ValueError(f"Unknown candidate: {name}")

    rows = [row for row in rows if row["text"]]
    if limit is not None:
        rows = rows[:limit]
    return rows


def exact_matches(source: list[dict[str, Any]], target: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target:
        target_by_text[row["text"]].append(row)
    matches = []
    for src in source:
        for tgt in target_by_text.get(src["text"], []):
            matches.append(format_match(src, tgt, exact=1.0))
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

    matches = []
    seen = set()
    for src in source:
        src_tokens = token_set(src["text"])
        if len(src_tokens) < 4:
            continue
        candidate_counts: dict[int, int] = defaultdict(int)
        for token in src_tokens:
            if len(token) >= 4:
                for index in inverted.get(token, []):
                    candidate_counts[index] += 1
        for index, _ in sorted(candidate_counts.items(), key=lambda item: -item[1])[:250]:
            tgt, tgt_tokens = indexed[index]
            key = (src["split"], src["index"], tgt["split"], tgt["index"])
            if key in seen:
                continue
            jaccard = len(src_tokens & tgt_tokens) / len(src_tokens | tgt_tokens)
            if jaccard < jaccard_threshold:
                continue
            ratio = fuzz.token_set_ratio(src["text"], tgt["text"])
            partial = fuzz.partial_ratio(src["text"], tgt["text"])
            if ratio < fuzz_threshold and partial < fuzz_threshold:
                continue
            seen.add(key)
            matches.append(format_match(src, tgt, jaccard=jaccard, fuzz_ratio=ratio, partial_ratio=partial))

    matches.sort(key=lambda item: (item.get("jaccard", 0.0), item.get("fuzz_ratio", 0.0)), reverse=True)
    return matches[:max_matches]


def format_match(src: dict[str, Any], tgt: dict[str, Any], **scores: Any) -> dict[str, Any]:
    return {
        "source_split": src["split"],
        "source_index": src["index"],
        "source_labels": src.get("labels", []),
        "cedr_split": tgt["split"],
        "cedr_index": tgt["index"],
        "cedr_label": tgt.get("label", []),
        "source_text": snippet(src["raw_text"]),
        "cedr_text": snippet(tgt["raw_text"]),
        **scores,
    }


def write_markdown(path: Path, summary: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    lines = ["# CEDR Candidate Contamination Audit", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Flagged Samples", ""])
    if not samples:
        lines.append("No exact or lexical near matches.")
    else:
        lines.append("| Source | CEDR | Scores |")
        lines.append("|---|---|---|")
        for row in samples[:80]:
            scores = []
            for key in ("exact", "jaccard", "fuzz_ratio", "partial_ratio"):
                if key in row:
                    value = row[key]
                    scores.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
            lines.append(
                "| "
                + f"{row['source_split']}#{row['source_index']} {row['source_labels']}: {row['source_text']}"
                + " | "
                + f"{row['cedr_split']}#{row['cedr_index']} {row['cedr_label']}: {row['cedr_text']}"
                + " | "
                + ", ".join(scores)
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["brighter", "go_ekman", "ru_sentiment_social", "rusentiment_union"])
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "mteb_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "contamination" / "cedr_candidates")
    parser.add_argument("--jaccard-threshold", type=float, default=0.78)
    parser.add_argument("--fuzz-threshold", type=int, default=92)
    parser.add_argument("--max-matches", type=int, default=250)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cedr = load_cedr(args.cache_dir)
    all_summary = {}

    for name in args.datasets:
        source = load_candidate(name, args.cache_dir)
        exact = exact_matches(source, cedr)
        lexical = lexical_near_matches(
            source,
            cedr,
            jaccard_threshold=args.jaccard_threshold,
            fuzz_threshold=args.fuzz_threshold,
            max_matches=args.max_matches,
        )
        flagged = {(row["source_split"], row["source_index"]) for row in exact + lexical}
        label_counts = Counter(label for row in source for label in row.get("labels", []))
        clean_label_counts = Counter(
            label
            for row in source
            if (row["split"], row["index"]) not in flagged
            for label in row.get("labels", [])
        )
        summary = {
            "dataset": name,
            "source_rows": len(source),
            "cedr_rows": len(cedr),
            "exact_matches": len(exact),
            "lexical_near_matches": len(lexical),
            "unique_flagged_source_rows": len(flagged),
            "clean_rows": len(source) - len(flagged),
            "label_counts": dict(label_counts),
            "clean_label_counts": dict(clean_label_counts),
        }
        all_summary[name] = summary
        flagged_rows = sorted(
            [{"split": split, "index": index} for split, index in flagged],
            key=lambda item: (item["split"], item["index"]),
        )
        (args.output_dir / f"{name}_flagged_rows.json").write_text(
            json.dumps(flagged_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / f"{name}_audit.json").write_text(
            json.dumps({"summary": summary, "exact": exact, "lexical": lexical}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_markdown(args.output_dir / f"{name}_audit.md", summary, exact + lexical)

    (args.output_dir / "summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
