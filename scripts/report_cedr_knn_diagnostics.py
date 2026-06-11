from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value:.6f}"


def confusion_map(report: dict[str, Any]) -> dict[str, int]:
    return {
        str(item["key"]): int(item["count"])
        for item in report["aggregate"].get("top_confusions", [])
    }


def per_label_accuracy(report: dict[str, Any]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for experiment in report.get("experiments", []):
        for label, row in experiment.get("per_true", {}).items():
            if row.get("exact_accuracy") is not None:
                values[str(label)].append(float(row["exact_accuracy"]))
    return {label: sum(rows) / len(rows) for label, rows in values.items() if rows}


def hard_indices(report: dict[str, Any]) -> set[int]:
    return {int(item["test_index"]) for item in report["aggregate"].get("hard_examples", [])}


def prediction_stability(report: dict[str, Any]) -> dict[str, Any]:
    stable_wrong = 0
    unstable = 0
    correct_all = 0
    wrong_counts = Counter()
    for row in report["aggregate"].get("per_test_predictions", []):
        wrong = sum(1 for item in row["predictions"] if not item["correct"])
        wrong_counts[wrong] += 1
        if wrong == 0:
            correct_all += 1
        elif wrong == len(row["predictions"]):
            stable_wrong += 1
        else:
            unstable += 1
    return {
        "correct_all": correct_all,
        "stable_wrong": stable_wrong,
        "unstable": unstable,
        "wrong_count_histogram": dict(sorted(wrong_counts.items())),
    }


def neighbor_label_counts(report: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for example in report["aggregate"].get("neighbor_examples", []):
        for experiment in example.get("experiments", []):
            for neighbor in experiment.get("neighbors", []):
                counts[str(neighbor["label"])] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a markdown comparison for CEDR KNN diagnostics.")
    parser.add_argument("--report", action="append", nargs=2, metavar=("NAME", "JSON"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [(name, load_json(Path(path))) for name, path in args.report]
    base_name, base_report = reports[0]
    base_confusions = confusion_map(base_report)
    base_hard = hard_indices(base_report)

    encode_modes = {
        name: report.get("encode_task_name")
        for name, report in reports
    }
    encode_mode_text = ", ".join(
        f"{name}={value or 'generic-MTEB-call'}" for name, value in encode_modes.items()
    )

    lines = [
        "# CEDR KNN Diagnostic",
        "",
        "Evaluation uses the frozen-wrapper model path: legacy Russian prompt, eager attention, seed reset, 10 sampled 5-NN multilabel experiments.",
        f"Encode task-name mode: {encode_mode_text}.",
        "",
        "## Aggregate",
        "",
        "| Model | Mean accuracy | Mean macro F1 | Stable wrong | Unstable | Correct in all 10 | Hard-overlap with first |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports:
        aggregate = report["aggregate"]
        stability = prediction_stability(report)
        hard_overlap = len(hard_indices(report) & base_hard)
        lines.append(
            f"| {name} | {pct(float(aggregate['mean_accuracy']))} | {pct(float(aggregate['mean_f1']))} | "
            f"{stability['stable_wrong']} | {stability['unstable']} | {stability['correct_all']} | {hard_overlap} |"
        )

    labels = sorted({label for _, report in reports for label in per_label_accuracy(report)})
    lines.extend(["", "## Per-Label Exact Accuracy", "", "| Label | " + " | ".join(name for name, _ in reports) + " |"])
    lines.append("|---" + "|---:" * len(reports) + "|")
    for label in labels:
        row = [label]
        for _, report in reports:
            row.append(pct(per_label_accuracy(report).get(label, float("nan"))))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Top Confusions", ""])
    top_keys = Counter()
    for name, report in reports:
        for key, count in confusion_map(report).items():
            top_keys[key] += count
    selected = [key for key, _ in top_keys.most_common(20)]
    lines.append("| Confusion | " + " | ".join(name for name, _ in reports) + " |")
    lines.append("|---" + "|---:" * len(reports) + "|")
    for key in selected:
        row = [key]
        for _, report in reports:
            row.append(str(confusion_map(report).get(key, 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Neighbor Labels Around Hard Errors", ""])
    neighbor_labels = sorted({label for _, report in reports for label in neighbor_label_counts(report)})
    lines.append("| Neighbor label | " + " | ".join(name for name, _ in reports) + " |")
    lines.append("|---" + "|---:" * len(reports) + "|")
    for label in neighbor_labels:
        row = [label]
        for _, report in reports:
            row.append(str(neighbor_label_counts(report).get(label, 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Most Persistent Hard Examples", ""])
    for name, report in reports:
        lines.extend([f"### {name}", ""])
        for item in report["aggregate"].get("hard_examples", [])[:10]:
            text = str(item["text"]).replace("\n", " ")
            if len(text) > 180:
                text = text[:177] + "..."
            lines.append(
                f"- `{item['test_index']}` wrong `{item['wrong_count']}/10`, label `{item['label']}`: {text}"
            )
        lines.append("")

    lines.extend(["", "## Notes", ""])
    lines.append(
        "- `Stable wrong` means the same test item is wrong in all 10 sampled support sets; these are geometry/domain failures, not just sampling noise."
    )
    lines.append(
        "- `Unstable` means the item flips depending on sampled train supports; these can often be improved by better support-like training or calibration."
    )
    lines.append(
        f"- The first report (`{base_name}`) is used only as the hard-example overlap reference."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
