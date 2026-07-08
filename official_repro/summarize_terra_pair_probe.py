from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TERRa pair-wrapper probe results.")
    parser.add_argument("input", type=Path, nargs="?", default=Path("results/official_repro/terra_pair_wrapper_probe_0ad.json"))
    parser.add_argument("--write-md", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    baseline = float(data.get("baseline_frozen_legacy_ru_0ad", 0.675025))
    target = float(data.get("target_score", 0.7))
    rows: list[tuple[float, str, str, dict]] = []
    for group_name in ("variants", "pair_variants"):
        for name, scores in data.get(group_name, {}).items():
            rows.append((float(scores["main_score"]), group_name, name, scores))
    rows.sort(reverse=True, key=lambda item: item[0])

    lines = [
        "# TERRa Pair Wrapper Probe Summary",
        "",
        f"Input: `{args.input}`",
        "",
        f"Baseline frozen `legacy_ru` on `0ad5b29...`: `{baseline:.6f}`",
        f"Target: `{target:.6f}`",
        "",
        "| Rank | Family | Variant | TERRa | Delta vs baseline | Delta vs target | Best metric |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for rank, (score, group_name, name, scores) in enumerate(rows, start=1):
        metric_values = {
            key: value
            for key, value in scores.items()
            if key.endswith("_ap") and isinstance(value, int | float)
        }
        best_metric = max(metric_values, key=metric_values.get) if metric_values else "main_score"
        lines.append(
            f"| {rank} | `{group_name}` | `{name}` | {score:.6f} | "
            f"{score - baseline:+.6f} | {score - target:+.6f} | `{best_metric}` |"
        )

    if rows:
        best_score, best_group, best_name, _ = rows[0]
        lines.extend(
            [
                "",
                "## Conclusion",
                "",
                f"Best variant: `{best_group}/{best_name}` with TERRa `{best_score:.6f}`.",
            ]
        )
        if best_score >= target:
            lines.append("This meets the target.")
        else:
            lines.append("This does not meet the target yet.")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.write_md is not None:
        args.write_md.parent.mkdir(parents=True, exist_ok=True)
        args.write_md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
