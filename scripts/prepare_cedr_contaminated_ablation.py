from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"

CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
LABELS = {
    0: "joy",
    1: "sadness",
    2: "surprise",
    3: "fear",
    4: "anger",
}
LABEL_RU = {
    0: "радость",
    1: "грусть",
    2: "удивление",
    3: "страх",
    4: "злость",
}
LABEL_DETAIL = {
    0: "в комментарии выражена радость, одобрение или позитивная эмоция",
    1: "в комментарии выражена грусть, печаль, тоска или усталость",
    2: "в комментарии выражено удивление, шок или неожиданность",
    3: "в комментарии выражен страх, тревога, опасение или ужас",
    4: "в комментарии выражена злость, раздражение, гнев или агрессия",
}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prefixed(text: str) -> str:
    return CEDR_PREFIX + " ".join(str(text).split())


def load_cedr(cache_dir: Path) -> dict[str, list[dict[str, Any]]]:
    dataset = load_dataset(
        "ai-forever/cedr-classification",
        revision="c0ba03d058e3e1b2f3fd20518875a4563dd12db4",
        cache_dir=str(cache_dir),
    )
    return {
        split: [
            {
                "split": split,
                "index": index,
                "text": str(row["text"]),
                "labels": [int(label) for label in row["label"]],
            }
            for index, row in enumerate(dataset[split])
        ]
        for split in ("train", "test")
    }


def label_pools(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    pools: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for label in row["labels"]:
            pools[int(label)].append(row)
    return dict(pools)


def mteb_style_support_set(
    rows: list[dict[str, Any]],
    *,
    query: dict[str, Any],
    rng: random.Random,
    samples_per_label: int,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if not (row["split"] == query["split"] and row["index"] == query["index"])
    ]
    rng.shuffle(candidates)
    label_counter: dict[int, int] = defaultdict(int)
    selected = []
    for row in candidates:
        if any(label_counter[int(label)] < samples_per_label for label in row["labels"]):
            selected.append(row)
            for label in row["labels"]:
                label_counter[int(label)] += 1
        if all(label_counter[label] >= samples_per_label for label in LABELS):
            break
    if not selected:
        raise RuntimeError("Unable to build non-empty CEDR support set")
    return selected


def support_texts(
    pools: dict[int, list[dict[str, Any]]],
    *,
    query: dict[str, Any],
    label: int,
    rng: random.Random,
    count: int,
) -> list[str]:
    candidates = [
        row for row in pools.get(label, [])
        if not (row["split"] == query["split"] and row["index"] == query["index"])
    ]
    if not candidates:
        return []
    if len(candidates) >= count:
        selected = rng.sample(candidates, count)
    else:
        selected = [rng.choice(candidates) for _ in range(count)]
    return [prefixed(row["text"]) for row in selected]


def make_support_records(
    rows: list[dict[str, Any]],
    *,
    supports_from: list[dict[str, Any]],
    name: str,
    seed: int,
    supports_per_label: int,
    positive_weight: float,
    similarity_threshold: float,
    support_pooling: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pools = label_pools(supports_from)
    records = []
    for row in rows:
        records.append(
            {
                "source": f"CONTAMINATED_CEDR:{name}",
                "objective": "multilabel_support_classification",
                "query": prefixed(row["text"]),
                "labels": [LABELS[label] for label in row["labels"]],
                "supports": {
                    LABELS[label]: support_texts(
                        pools,
                        query=row,
                        label=label,
                        rng=rng,
                        count=supports_per_label,
                    )
                    for label in LABELS
                },
                "positive_weight": positive_weight,
                "similarity_threshold": similarity_threshold,
                "support_pooling": support_pooling,
                "metadata": {
                    "cedr_split": row["split"],
                    "cedr_index": row["index"],
                    "cedr_labels": row["labels"],
                    "contamination": "direct CEDR benchmark rows and labels",
                    "supports_from": sorted({item["split"] for item in supports_from}),
                },
            }
        )
    rng.shuffle(records)
    return records


def make_knn_episode_records(
    rows: list[dict[str, Any]],
    *,
    supports_from: list[dict[str, Any]],
    name: str,
    seed: int,
    samples_per_label: int,
    knn_k: int,
    vote_temperature: float,
    decision_threshold: float,
    exact_set_weight: float,
    margin_weight: float,
    vote_margin: float,
    support_chunk_size: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for row in rows:
        support_rows = mteb_style_support_set(
            supports_from,
            query=row,
            rng=rng,
            samples_per_label=samples_per_label,
        )
        records.append(
            {
                "source": f"CONTAMINATED_CEDR:{name}",
                "objective": "cedr_knn_episode",
                "query": prefixed(row["text"]),
                "labels": [LABELS[label] for label in row["labels"]],
                "supports": [
                    {
                        "text": prefixed(support["text"]),
                        "labels": [LABELS[label] for label in support["labels"]],
                        "metadata": {
                            "cedr_split": support["split"],
                            "cedr_index": support["index"],
                            "cedr_labels": support["labels"],
                        },
                    }
                    for support in support_rows
                ],
                "class_order": [LABELS[index] for index in sorted(LABELS)],
                "knn_k": knn_k,
                "vote_temperature": vote_temperature,
                "decision_threshold": decision_threshold,
                "exact_set_weight": exact_set_weight,
                "margin_weight": margin_weight,
                "vote_margin": vote_margin,
                "support_chunk_size": support_chunk_size,
                "metadata": {
                    "cedr_split": row["split"],
                    "cedr_index": row["index"],
                    "cedr_labels": row["labels"],
                    "contamination": "direct CEDR benchmark rows and labels",
                    "construction": "mteb_style_cedr_knn_episode",
                    "supports_from": sorted({item["split"] for item in supports_from}),
                    "support_count": len(support_rows),
                    "samples_per_label": samples_per_label,
                },
            }
        )
    rng.shuffle(records)
    return records


def label_statement(text: str, label: int) -> str:
    return f"{text}\nВерная разметка CEDR: {LABEL_RU[label]} ({LABEL_DETAIL[label]})."


def make_label_statement_records(rows: list[dict[str, Any]], *, name: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for row in rows:
        positives = [
            prefixed(label_statement(row["text"], label))
            for label in row["labels"]
        ]
        if positives:
            positive = positives[0]
            extra_positives = positives[1:]
        else:
            positive = prefixed(f"{row['text']}\nВерная разметка CEDR: нет явной эмоции из списка.")
            extra_positives = []
        false_labels = [label for label in LABELS if label not in row["labels"]]
        rng.shuffle(false_labels)
        records.append(
            {
                "source": f"CONTAMINATED_CEDR:{name}",
                "objective": "contrastive",
                "query": prefixed(row["text"]),
                "positive": positive,
                "positives": extra_positives,
                "negatives": [prefixed(label_statement(row["text"], label)) for label in false_labels],
                "metadata": {
                    "cedr_split": row["split"],
                    "cedr_index": row["index"],
                    "cedr_labels": row["labels"],
                    "contamination": "direct CEDR benchmark rows and labels",
                    "construction": "same-text label statement",
                },
            }
        )
    rng.shuffle(records)
    return records


def make_labeled_text_records(rows: list[dict[str, Any]], *, name: str, loss: str) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        labels = row["labels"] or ["no_emotion"]
        for label in labels:
            label_name = LABELS[int(label)] if isinstance(label, int) else str(label)
            records.append(
                {
                    "source": f"CONTAMINATED_CEDR:{name}",
                    "objective": "labeled_text",
                    "text": prefixed(row["text"]),
                    "label": label_name,
                    "loss": loss,
                    "metadata": {
                        "cedr_split": row["split"],
                        "cedr_index": row["index"],
                        "cedr_labels": row["labels"],
                        "expanded_label": label_name,
                        "contamination": "direct CEDR benchmark rows and labels",
                        "construction": "labeled_text_metric_learning",
                    },
                }
            )
    return records


def summarize(records: list[dict[str, Any]], *, output: Path, description: str) -> dict[str, Any]:
    split_counts = Counter()
    label_counts = Counter()
    objective_counts = Counter()
    for record in records:
        objective_counts[record["objective"]] += 1
        metadata = record.get("metadata", {})
        split_counts[metadata.get("cedr_split", "?")] += 1
        labels = metadata.get("cedr_labels", [])
        if labels:
            for label in labels:
                label_counts[LABELS[int(label)]] += 1
        else:
            label_counts["no_emotion"] += 1
    return {
        "output": str(output.relative_to(ROOT)),
        "records": len(records),
        "description": description,
        "objective_counts": dict(objective_counts),
        "cedr_split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "contamination": "YES: direct CEDR benchmark rows/labels are intentionally used for diagnostic ablation only",
        "do_not_use_for_fair_results": True,
    }


def write_variant(name: str, records: list[dict[str, Any]], description: str) -> None:
    path = DATA_DIR / f"{name}.jsonl"
    write_jsonl(path, records)
    write_json(path.with_name(path.stem + "_summary.json"), summarize(records, output=path, description=description))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare intentionally contaminated CEDR diagnostic ablations.")
    parser.add_argument("--seed", type=int, default=9041)
    parser.add_argument("--supports-per-label", type=int, default=8)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.45)
    parser.add_argument("--support-pooling", choices=("max", "mean_top2", "logsumexp"), default="max")
    parser.add_argument("--episode-vote-temperature", type=float, default=0.03)
    parser.add_argument("--episode-decision-threshold", type=float, default=0.5)
    parser.add_argument("--episode-exact-set-weight", type=float, default=0.5)
    parser.add_argument("--episode-margin-weight", type=float, default=0.25)
    parser.add_argument("--episode-vote-margin", type=float, default=0.15)
    parser.add_argument("--episode-support-chunk-size", type=int, default=12)
    args = parser.parse_args()

    cedr = load_cedr(ROOT / "data" / "hf_cache")
    train_rows = cedr["train"]
    test_rows = cedr["test"]
    all_rows = train_rows + test_rows

    common = {
        "seed": args.seed,
        "supports_per_label": args.supports_per_label,
        "positive_weight": args.positive_weight,
        "similarity_threshold": args.similarity_threshold,
        "support_pooling": args.support_pooling,
    }
    write_variant(
        "CONTAMINATED_cedr_train_support_knn_proxy",
        make_support_records(
            train_rows,
            supports_from=train_rows,
            name="train_support_knn_proxy",
            **common,
        ),
        "CEDR train split rows with CEDR train split supports; tests benchmark-train contamination and KNN geometry.",
    )
    write_variant(
        "CONTAMINATED_cedr_test_support_knn_proxy",
        make_support_records(
            test_rows,
            supports_from=train_rows,
            name="test_support_knn_proxy",
            **common,
        ),
        "CEDR test split rows with CEDR train split supports; direct eval contamination.",
    )
    write_variant(
        "CONTAMINATED_cedr_all_support_knn_proxy",
        make_support_records(
            all_rows,
            supports_from=train_rows,
            name="all_support_knn_proxy",
            **common,
        ),
        "CEDR train+test rows with CEDR train split supports; direct train and eval contamination.",
    )
    episode_common = {
        "samples_per_label": args.supports_per_label,
        "knn_k": 5,
        "vote_temperature": args.episode_vote_temperature,
        "decision_threshold": args.episode_decision_threshold,
        "exact_set_weight": args.episode_exact_set_weight,
        "margin_weight": args.episode_margin_weight,
        "vote_margin": args.episode_vote_margin,
        "support_chunk_size": args.episode_support_chunk_size,
    }
    write_variant(
        "CONTAMINATED_cedr_train_knn_episode",
        make_knn_episode_records(
            train_rows,
            supports_from=train_rows,
            name="train_knn_episode",
            seed=args.seed + 101,
            **episode_common,
        ),
        "CEDR train rows as MTEB-style 5-NN episodes with direct CEDR labels.",
    )
    write_variant(
        "CONTAMINATED_cedr_test_knn_episode",
        make_knn_episode_records(
            test_rows,
            supports_from=train_rows,
            name="test_knn_episode",
            seed=args.seed + 102,
            **episode_common,
        ),
        "CEDR test rows as MTEB-style 5-NN episodes against CEDR train supports; direct eval contamination.",
    )
    write_variant(
        "CONTAMINATED_cedr_all_knn_episode",
        make_knn_episode_records(
            all_rows,
            supports_from=train_rows,
            name="all_knn_episode",
            seed=args.seed + 103,
            **episode_common,
        ),
        "CEDR train+test rows as MTEB-style 5-NN episodes against CEDR train supports.",
    )
    write_variant(
        "CONTAMINATED_cedr_test_label_statement",
        make_label_statement_records(test_rows, name="test_label_statement", seed=args.seed + 17),
        "CEDR test rows converted to same-text label-statement contrastive rows; isolates label-boundary signal.",
    )
    write_variant(
        "CONTAMINATED_cedr_all_labeled_supcon",
        make_labeled_text_records(all_rows, name="all_labeled_supcon", loss="supcon"),
        "CEDR train+test rows as labeled_text supervised contrastive metric-learning data; directly targets KNN clustering.",
    )
    write_variant(
        "CONTAMINATED_cedr_all_labeled_circle",
        make_labeled_text_records(all_rows, name="all_labeled_circle", loss="circle"),
        "CEDR train+test rows as labeled_text circle-loss metric-learning data; directly targets KNN clustering.",
    )
    print("Wrote contaminated CEDR ablation datasets under", DATA_DIR)


if __name__ == "__main__":
    main()
