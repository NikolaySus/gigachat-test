from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


RESULTS_COMMIT = "ec3c9cbbdffc44b53052775d78196a2fff838b8c"
MODEL_REVISION = "40b27667b9ad586d7812675df76e5062ccc80b0e"
TASKS = [
    "CEDRClassification",
    "GeoreviewClassification",
    "GeoreviewClusteringP2P",
    "HeadlineClassification",
    "InappropriatenessClassification",
    "KinopoiskClassification",
    "MIRACLReranking",
    "MIRACLRetrieval",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "RUParaPhraserSTS",
    "RiaNewsRetrieval",
    "RuBQReranking",
    "RuBQRetrieval",
    "RuReviewsClassification",
    "RuSTSBenchmarkSTS",
    "RuSciBenchGRNTIClassification",
    "RuSciBenchGRNTIClusteringP2P",
    "RuSciBenchOECDClassification",
    "RuSciBenchOECDClusteringP2P",
    "STS22",
    "SensitiveTopicsClassification",
    "TERRa",
]


def main_score(result: dict) -> float:
    for split in ("test", "dev", "validation"):
        scores = result.get("scores", {}).get(split)
        if isinstance(scores, list):
            for row in scores:
                if isinstance(row, dict) and "main_score" in row:
                    return float(row["main_score"])
    for scores in result.get("scores", {}).values():
        if isinstance(scores, list):
            for row in scores:
                if isinstance(row, dict) and "main_score" in row:
                    return float(row["main_score"])
    raise KeyError("main_score")


def fetch_official_scores() -> dict[str, float]:
    base = (
        "https://huggingface.co/datasets/mteb/results/resolve/"
        f"{RESULTS_COMMIT}/results/ai-sage__Giga-Embeddings-instruct/{MODEL_REVISION}"
    )
    scores = {}
    for task in TASKS:
        with urllib.request.urlopen(f"{base}/{task}.json", timeout=60) as response:
            scores[task] = main_score(json.load(response))
    return scores


def local_result_files(results_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in results_dir.rglob("*.json") if path.name != "repro_manifest.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local official-repro MTEB output with public mteb/results.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--write-md", type=Path, default=None)
    args = parser.parse_args()

    official = fetch_official_scores()
    local_files = local_result_files(args.results_dir)
    rows = []
    for task, official_score in official.items():
        path = local_files.get(task)
        if path is None:
            continue
        local_score = main_score(json.loads(path.read_text(encoding="utf-8")))
        rows.append((task, local_score, official_score, local_score - official_score))

    lines = [
        "| Task | Local repro | Official | Delta |",
        "|---|---:|---:|---:|",
    ]
    for task, local_score, official_score, delta in sorted(rows, key=lambda row: abs(row[3]), reverse=True):
        lines.append(f"| {task} | {local_score:.6f} | {official_score:.6f} | {delta:+.6f} |")

    if rows:
        mean_delta = sum(row[3] for row in rows) / len(rows)
        mean_abs = sum(abs(row[3]) for row in rows) / len(rows)
        local_avg = sum(row[1] for row in rows) / len(rows)
        official_avg = sum(row[2] for row in rows) / len(rows)
        header = [
            f"Matched tasks: {len(rows)}",
            f"Local avg: {local_avg:.6f}",
            f"Official avg: {official_avg:.6f}",
            f"Mean delta: {mean_delta:+.6f}",
            f"Mean abs delta: {mean_abs:.6f}",
            "",
        ]
    else:
        header = ["Matched tasks: 0", ""]

    text = "\n".join(header + lines) + "\n"
    print(text)
    if args.write_md is not None:
        args.write_md.parent.mkdir(parents=True, exist_ok=True)
        args.write_md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
