from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def strip_query_prefix(text: str) -> str:
    marker = "\nQuery:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    if text.startswith("Query:"):
        return text[len("Query:") :].strip()
    return text.strip()


def load_cluster_texts(path: Path) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    seen_by_cluster: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            cluster = str((record.get("metadata") or {}).get("cluster", ""))
            if not cluster:
                continue
            candidates = [
                strip_query_prefix(record.get("query", "")),
                record.get("positive", "").strip(),
            ]
            candidates.extend(str(text).strip() for text in record.get("positives", []))
            for text in candidates:
                if len(text) < 80:
                    continue
                if text in seen_by_cluster[cluster]:
                    continue
                seen_by_cluster[cluster].add(text)
                clusters[cluster].append(text)
    return {cluster: texts for cluster, texts in clusters.items() if len(texts) >= 2}


def make_batches(
    clusters: dict[str, list[str]],
    *,
    batch_count: int,
    clusters_per_batch: int,
    positives_per_cluster: int,
    seed: int,
    loss_name: str,
) -> list[dict]:
    rng = random.Random(seed)
    cluster_ids = list(clusters)
    if len(cluster_ids) < clusters_per_batch:
        raise ValueError(
            f"Need at least {clusters_per_batch} usable clusters, got {len(cluster_ids)}"
        )
    records = []
    for batch_index in range(batch_count):
        selected_clusters = rng.sample(cluster_ids, clusters_per_batch)
        for cluster in selected_clusters:
            texts = clusters[cluster]
            if len(texts) >= positives_per_cluster:
                selected_texts = rng.sample(texts, positives_per_cluster)
            else:
                selected_texts = [rng.choice(texts) for _ in range(positives_per_cluster)]
            for text_index, text in enumerate(selected_texts):
                records.append(
                    {
                        "source": "Vikhrmodels/GrandMaster-PRO-MAX:clustered:labeled_metric",
                        "objective": "labeled_text",
                        "text": text,
                        "label": f"grandmaster_cluster_{cluster}",
                        "loss": loss_name,
                        "metadata": {
                            "cluster": int(cluster) if cluster.isdigit() else cluster,
                            "batch_index": batch_index,
                            "text_index": text_index,
                            "group": f"grandmaster_cluster_{cluster}",
                            "contamination_policy": "Derived from clean open-data GrandMaster cluster IDs; no ruMTEB rows.",
                        },
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/contrastive/open_ru_1r_nc_grandmaster_clustered_3200.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--batch-count", type=int, default=240)
    parser.add_argument("--clusters-per-batch", type=int, default=4)
    parser.add_argument("--positives-per-cluster", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="supcon")
    args = parser.parse_args()

    clusters = load_cluster_texts(args.input)
    records = make_batches(
        clusters,
        batch_count=args.batch_count,
        clusters_per_batch=args.clusters_per_batch,
        positives_per_cluster=args.positives_per_cluster,
        seed=args.seed,
        loss_name=args.loss,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = args.summary or args.output.with_name(args.output.stem + "_summary.json")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "records": len(records),
        "batch_count": args.batch_count,
        "batch_size": args.clusters_per_batch * args.positives_per_cluster,
        "clusters_per_batch": args.clusters_per_batch,
        "positives_per_cluster": args.positives_per_cluster,
        "usable_clusters": len(clusters),
        "loss": args.loss,
        "seed": args.seed,
        "contamination_policy": "Clean open-data GrandMaster cluster IDs; no ruMTEB rows.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
