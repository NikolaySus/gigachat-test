from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rumteb_contamination import (
    RUMTEB_TASK_DATASETS,
    contaminated_tasks_from_manifest,
    extract_main_score,
    load_training_manifest,
    task_category,
)


def find_task_results(results_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(results_dir.rglob("*.json")):
        if path.name in {"model_meta.json", "evaluation_manifest.json", "summary.json"}:
            continue
        paths.append(path)
    return paths


def scope_for_task(task_name: str, contaminated_tasks: dict[str, list[str]]) -> str:
    return "contaminated" if task_name in contaminated_tasks else "clean"


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_summary(results_dir: Path, manifest_path: Path | None) -> dict[str, Any]:
    training_manifest = load_training_manifest(manifest_path)
    contaminated_tasks = contaminated_tasks_from_manifest(training_manifest)

    rows = []
    for path in find_task_results(results_dir):
        data = json.loads(path.read_text(encoding="utf-8"))
        task_name = data.get("task_name", path.stem)
        score = extract_main_score(data)
        if score is None:
            continue
        contaminated_by = contaminated_tasks.get(task_name, [])
        rows.append(
            {
                "task_name": task_name,
                "score": score,
                "category": task_category(task_name),
                "scope": scope_for_task(task_name, contaminated_tasks),
                "dataset_id": RUMTEB_TASK_DATASETS.get(task_name),
                "contaminated_by": contaminated_by,
                "result_path": str(path),
            }
        )

    by_scope = defaultdict(list)
    by_scope_category = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scope["all"].append(row["score"])
        by_scope[row["scope"]].append(row["score"])
        by_scope_category["all"][row["category"]].append(row["score"])
        by_scope_category[row["scope"]][row["category"]].append(row["score"])

    return {
        "results_dir": str(results_dir),
        "training_manifest_path": str(manifest_path) if manifest_path else None,
        "training_manifest_name": training_manifest["name"] if training_manifest else None,
        "task_count": len(rows),
        "averages": {
            scope: average(scores)
            for scope, scores in sorted(by_scope.items())
        },
        "category_averages": {
            scope: {
                category: average(scores)
                for category, scores in sorted(categories.items())
            }
            for scope, categories in sorted(by_scope_category.items())
        },
        "contaminated_tasks": contaminated_tasks,
        "tasks": sorted(rows, key=lambda row: row["task_name"]),
    }


def format_score(score: float | None) -> str:
    if score is None:
        return "n/a"
    return f"{score:.6f}"


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# ruMTEB Summary",
        "",
        f"Results directory: `{summary['results_dir']}`",
        f"Training manifest: `{summary['training_manifest_path'] or 'none'}`",
        "",
        "## Averages",
        "",
        "| Scope | Average |",
        "| --- | ---: |",
    ]
    for scope in ("all", "clean", "contaminated"):
        lines.append(f"| {scope} | {format_score(summary['averages'].get(scope))} |")

    lines.extend([
        "",
        "## Category Averages",
        "",
        "| Scope | Category | Average |",
        "| --- | --- | ---: |",
    ])
    for scope in ("all", "clean", "contaminated"):
        for category, score in summary["category_averages"].get(scope, {}).items():
            lines.append(f"| {scope} | {category} | {format_score(score)} |")

    lines.extend([
        "",
        "## Per-Task Scores",
        "",
        "| Task | Category | Score | Scope | Dataset | Contaminated by |",
        "| --- | --- | ---: | --- | --- | --- |",
    ])
    for row in summary["tasks"]:
        contaminated_by = ", ".join(row["contaminated_by"]) if row["contaminated_by"] else ""
        lines.append(
            "| "
            f"{row['task_name']} | "
            f"{row['category']} | "
            f"{format_score(row['score'])} | "
            f"{row['scope']} | "
            f"{row['dataset_id'] or ''} | "
            f"{contaminated_by} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ruMTEB results with contamination-aware averages.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = build_summary(args.results_dir, args.training_manifest)
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, out_dir / "summary.md")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
