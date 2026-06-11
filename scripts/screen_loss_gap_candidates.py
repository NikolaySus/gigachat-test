from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from giga_model_utils import ModelLoadConfig, load_giga_embeddings
from latent_experiment_modules import load_latent_experiment_checkpoint
from train_exp01b_latent_memory import compute_batch_loss, latent_state_clone, load_latent_state_from_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
OUTPUT_DIR = ROOT / "results" / "official_repro" / "loss_gap_screen"

CLEAN_BASE = (
    ROOT
    / "experiments/exp01_reinit_fair/checkpoints/"
    / "open_ru_1r_nc_mixh_habrfull_plus_geracl_remaining_4096_eager_frozenrepro/latest.pt"
)

EXCLUDE_NAME_PARTS = {
    "a050",
    "postmanip",
    "soup",
    "teacher_scored",
    "benchmark",
    "stsb",
    "ru_stsbenchmark",
    "mteb",
}

CONTRASTIVE_TEMPS = [0.02, 0.05, 0.10]
LABELED_TEMPS = [0.05, 0.10]
LABELED_LOSSES = ["supcon", "circle", "multi_similarity"]
DEFAULT_TEMPERATURE = 0.02


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def is_curated_clean(path: Path) -> bool:
    name = path.name.lower()
    if any(part in name for part in EXCLUDE_NAME_PARTS):
        return False
    if name.startswith("open_ru_1r_nc_"):
        return True
    if name.startswith("rusts_external_"):
        summary = ROOT / "results" / "contamination" / "rusts_external" / f"{path.stem}_summary.json"
        return summary.exists()
    return False


def discover_datasets(limit: int | None = None) -> list[Path]:
    paths = sorted(path for path in DATA_DIR.glob("*.jsonl") if is_curated_clean(path))
    return paths[:limit] if limit is not None else paths


def read_records(path: Path, *, max_records: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    objective_counts: Counter[str] = Counter()
    malformed = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            record.setdefault("objective", "contrastive")
            objective_counts[str(record["objective"])] += 1
            records.append(record)
            if len(records) >= max_records:
                break
    metadata = {
        "sampled_records": len(records),
        "objective_counts": dict(objective_counts),
        "malformed_sample_rows": malformed,
    }
    return records, metadata


def record_text_count(record: dict[str, Any]) -> int:
    objective = record.get("objective", "contrastive")
    if objective == "contrastive":
        return 2 + len(record.get("positives", [])) + len(record.get("negatives", []))
    if objective == "pair_score":
        return 2
    if objective == "labeled_text":
        return 1
    if objective == "knn_classification":
        return 1 + sum(len(values) for values in record.get("supports", {}).values())
    if objective in {"prototype_classification", "prototype_none_classification", "prototype_uniform_classification"}:
        return 1 + sum(len(values) for values in record.get("prototypes", {}).values())
    return 2


def adaptive_batch_size(records: list[dict[str, Any]], *, max_texts: int) -> int:
    if not records:
        return 0
    counts = sorted(record_text_count(record) for record in records[:32])
    typical = counts[min(len(counts) - 1, max(0, int(len(counts) * 0.75) - 1))]
    return max(1, max_texts // max(1, typical))


def batches(records: list[dict[str, Any]], *, batch_size: int, max_batches: int) -> list[list[dict[str, Any]]]:
    result = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        if batch:
            result.append(batch)
        if len(result) >= max_batches:
            break
    return result


def labels_have_positive_and_negative(records: list[dict[str, Any]]) -> bool:
    labels = [str(record.get("label")) for record in records]
    if len(set(labels)) < 2:
        return False
    counts = Counter(labels)
    return any(count >= 2 for count in counts.values())


def objective_records(records: list[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("objective", "contrastive") == objective]


def make_views(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    contrastive_records = objective_records(records, "contrastive")
    if contrastive_records:
        for temperature in CONTRASTIVE_TEMPS:
            views.append(
                {
                    "loss": "contrastive",
                    "temperature": temperature,
                    "records": contrastive_records,
                    "pair_score_loss_weight": 1.0,
                }
            )

    pair_records = objective_records(records, "pair_score")
    if pair_records:
        views.append(
            {
                "loss": "pair_score",
                "temperature": DEFAULT_TEMPERATURE,
                "records": pair_records,
                "pair_score_loss_weight": 1.0,
            }
        )

    for objective in [
        "prototype_classification",
        "knn_classification",
        "prototype_none_classification",
        "prototype_uniform_classification",
    ]:
        proto_records = objective_records(records, objective)
        if proto_records:
            views.append(
                {
                    "loss": objective,
                    "temperature": DEFAULT_TEMPERATURE,
                    "records": proto_records,
                    "pair_score_loss_weight": 1.0,
                }
            )

    labeled_records = objective_records(records, "labeled_text")
    if labeled_records:
        for loss_name in LABELED_LOSSES:
            for temperature in LABELED_TEMPS:
                copied = [dict(record, loss=loss_name) for record in labeled_records]
                views.append(
                    {
                        "loss": f"labeled_{loss_name}",
                        "temperature": temperature,
                        "records": copied,
                        "pair_score_loss_weight": 1.0,
                    }
                )
    return views


def load_clean_latent(model, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_latent_experiment_checkpoint(model, checkpoint)
    return latent_state_clone(model)


def set_latent_state(model, state: dict[str, torch.Tensor]) -> None:
    model.latent_attention_model.load_state_dict(
        {name: tensor.to(next(model.parameters()).device) for name, tensor in state.items()},
        strict=True,
    )


@torch.no_grad()
def eval_view(
    view: dict[str, Any],
    *,
    tokenizer,
    model,
    latent_state: dict[str, torch.Tensor],
    max_length: int,
    max_texts_per_batch: int,
    max_batches: int,
) -> dict[str, Any]:
    set_latent_state(model, latent_state)
    model.eval()
    model.latent_attention_model.eval()

    records = view["records"]
    batch_size = adaptive_batch_size(records, max_texts=max_texts_per_batch)
    losses: list[float] = []
    skipped_batches = 0
    for batch in batches(records, batch_size=batch_size, max_batches=max_batches):
        if view["loss"].startswith("labeled_") and not labels_have_positive_and_negative(batch):
            skipped_batches += 1
            continue
        try:
            loss = compute_batch_loss(
                batch,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=float(view["temperature"]),
                pair_score_loss_weight=float(view["pair_score_loss_weight"]),
            )
            value = float(loss.detach().cpu())
        except (RuntimeError, ValueError, KeyError) as error:
            skipped_batches += 1
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
            continue
        if math.isfinite(value):
            losses.append(value)
        else:
            skipped_batches += 1
    if not losses:
        return {
            "ok": False,
            "loss": None,
            "valid_batches": 0,
            "skipped_batches": skipped_batches,
            "batch_size": batch_size,
        }
    return {
        "ok": True,
        "loss": sum(losses) / len(losses),
        "valid_batches": len(losses),
        "skipped_batches": skipped_batches,
        "batch_size": batch_size,
    }


def write_outputs(
    *,
    rows: list[dict[str, Any]],
    skips: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda row: row["loss_gap"], reverse=True)
    payload = {
        "args": vars(args),
        "elapsed_seconds": elapsed_seconds,
        "rows": rows_sorted,
        "skips": skips,
    }
    (output_dir / "loss_gap_screen.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "rank",
        "dataset",
        "loss",
        "temperature",
        "our_loss",
        "released_loss",
        "loss_gap",
        "loss_ratio",
        "valid_batches",
        "batch_size",
        "sampled_records",
        "objective_counts",
    ]
    with (output_dir / "loss_gap_screen.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows_sorted, start=1):
            writer.writerow(
                {
                    **{field: row.get(field) for field in fieldnames if field != "rank"},
                    "rank": rank,
                    "objective_counts": json.dumps(row.get("objective_counts", {}), ensure_ascii=False),
                }
            )

    lines = [
        "# Loss-Gap Candidate Screen",
        "",
        f"Elapsed: {elapsed_seconds / 60:.1f} min",
        f"Rows: {len(rows_sorted)}",
        f"Skipped views/datasets: {len(skips)}",
        "",
        "Released latent weights are used only as an offline diagnostic target.",
        "",
        "| Rank | Dataset | Loss | Temp | Ours | Released | Gap | Ratio | Batches |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows_sorted[: args.top_k], start=1):
        lines.append(
            f"| {rank} | `{row['dataset']}` | {row['loss']} | {row['temperature']:.2f} | "
            f"{row['our_loss']:.6f} | {row['released_loss']:.6f} | {row['loss_gap']:+.6f} | "
            f"{row['loss_ratio']:.3f} | {row['valid_batches']} |"
        )
    lines.extend(
        [
            "",
            "## Top Training Candidates",
            "",
            "| Rank | Data path | Loss | Temperature | Gap |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for rank, row in enumerate(rows_sorted[: args.train_top_k], start=1):
        lines.append(
            f"| {rank} | `data/contrastive/{row['dataset']}` | {row['loss']} | "
            f"{row['temperature']:.2f} | {row['loss_gap']:+.6f} |"
        )
    lines.extend(["", "## Skip Summary", ""])
    for skip in skips[:100]:
        lines.append(f"- `{skip.get('dataset', '-')}` {skip.get('loss', '')}: {skip['reason']}")
    (output_dir / "loss_gap_screen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    reliable = [
        row
        for row in rows_sorted
        if row["loss_gap"] > 0.0 and int(row.get("valid_batches", 0)) >= 8
    ]
    reliable_lines = [
        "# Loss-Gap Candidate Screen: Reliable Top Rows",
        "",
        "Rows here have positive gap and at least 8 valid batches.",
        "",
        "| Rank | Dataset | Loss | Temp | Ours | Released | Gap | Ratio | Batches |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(reliable[: args.top_k], start=1):
        reliable_lines.append(
            f"| {rank} | `{row['dataset']}` | {row['loss']} | {row['temperature']:.2f} | "
            f"{row['our_loss']:.6f} | {row['released_loss']:.6f} | {row['loss_gap']:+.6f} | "
            f"{row['loss_ratio']:.3f} | {row['valid_batches']} |"
        )
    reliable_lines.extend(
        [
            "",
            "## Recommended First Ablations",
            "",
            "| Rank | Data path | Loss | Temperature | Gap |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for rank, row in enumerate(reliable[: min(8, args.train_top_k)], start=1):
        reliable_lines.append(
            f"| {rank} | `data/contrastive/{row['dataset']}` | {row['loss']} | "
            f"{row['temperature']:.2f} | {row['loss_gap']:+.6f} |"
        )
    (output_dir / "loss_gap_screen_reliable.md").write_text(
        "\n".join(reliable_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen clean datasets by released-vs-trained latent loss gap.")
    parser.add_argument("--clean-base", default=str(CLEAN_BASE.relative_to(ROOT)))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(ROOT)))
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--max-texts-per-batch", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--train-top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1831)
    args = parser.parse_args()

    started_at = time.perf_counter()
    random.seed(args.seed)
    output_dir = resolve_path(args.output_dir)
    clean_base = resolve_path(args.clean_base)

    tokenizer, model = load_giga_embeddings(
        ModelLoadConfig(
            max_length=args.max_length,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    )
    released_state = latent_state_clone(model)
    clean_state = load_clean_latent(model, clean_base)

    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    datasets = discover_datasets(args.dataset_limit)
    for dataset_index, path in enumerate(datasets, start=1):
        print(f"[{dataset_index}/{len(datasets)}] {path.name}", flush=True)
        records, metadata = read_records(path, max_records=args.max_records)
        if not records:
            skips.append({"dataset": path.name, "reason": "no sampled records"})
            continue
        views = make_views(records)
        if not views:
            skips.append({"dataset": path.name, "reason": "no supported objectives", **metadata})
            continue
        for view in views:
            our = eval_view(
                view,
                tokenizer=tokenizer,
                model=model,
                latent_state=clean_state,
                max_length=args.max_length,
                max_texts_per_batch=args.max_texts_per_batch,
                max_batches=args.max_batches,
            )
            released = eval_view(
                view,
                tokenizer=tokenizer,
                model=model,
                latent_state=released_state,
                max_length=args.max_length,
                max_texts_per_batch=args.max_texts_per_batch,
                max_batches=args.max_batches,
            )
            if not our["ok"] or not released["ok"]:
                skips.append(
                    {
                        "dataset": path.name,
                        "loss": view["loss"],
                        "temperature": view["temperature"],
                        "reason": "no valid batches for ours or released",
                        "our": our,
                        "released": released,
                        **metadata,
                    }
                )
                continue
            our_loss = float(our["loss"])
            released_loss = float(released["loss"])
            rows.append(
                {
                    "dataset": path.name,
                    "loss": view["loss"],
                    "temperature": float(view["temperature"]),
                    "our_loss": our_loss,
                    "released_loss": released_loss,
                    "loss_gap": our_loss - released_loss,
                    "loss_ratio": our_loss / max(released_loss, 1e-12),
                    "valid_batches": min(int(our["valid_batches"]), int(released["valid_batches"])),
                    "batch_size": int(our["batch_size"]),
                    **metadata,
                }
            )

    write_outputs(
        rows=rows,
        skips=skips,
        output_dir=output_dir,
        args=args,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    print(output_dir / "loss_gap_screen.md")


if __name__ == "__main__":
    main()
