from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def load_latent_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("latent_attention_model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state dict")
    return {
        str(name): value.detach().cpu()
        for name, value in state.items()
        if torch.is_tensor(value) and torch.is_floating_point(value)
    }


def common_keys(*states: dict[str, torch.Tensor]) -> list[str]:
    keys = set(states[0])
    for state in states[1:]:
        keys &= set(state)
    return sorted(
        key
        for key in keys
        if all(tuple(state[key].shape) == tuple(states[0][key].shape) for state in states[1:])
    )


def dot_and_norms(
    base: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    keys: list[str],
) -> dict[str, float]:
    dot = 0.0
    candidate_norm_sq = 0.0
    target_norm_sq = 0.0
    after_dist_sq = 0.0
    for key in keys:
        target_delta = (target[key].float() - base[key].float()).reshape(-1)
        candidate_delta = (candidate[key].float() - base[key].float()).reshape(-1)
        target_minus_candidate = (target[key].float() - candidate[key].float()).reshape(-1)
        dot += float(torch.dot(candidate_delta, target_delta))
        candidate_norm_sq += float(torch.dot(candidate_delta, candidate_delta))
        target_norm_sq += float(torch.dot(target_delta, target_delta))
        after_dist_sq += float(torch.dot(target_minus_candidate, target_minus_candidate))

    candidate_norm = math.sqrt(max(candidate_norm_sq, 0.0))
    target_norm = math.sqrt(max(target_norm_sq, 0.0))
    after_distance = math.sqrt(max(after_dist_sq, 0.0))
    cosine = dot / (candidate_norm * target_norm) if candidate_norm > 0 and target_norm > 0 else 0.0
    projection_fraction = dot / target_norm_sq if target_norm_sq > 0 else 0.0
    orthogonal_sq = candidate_norm_sq - (dot * dot / target_norm_sq) if target_norm_sq > 0 else candidate_norm_sq
    orthogonal_norm = math.sqrt(max(orthogonal_sq, 0.0))
    distance_improvement = 1.0 - after_distance / target_norm if target_norm > 0 else 0.0

    return {
        "common_tensors": float(len(keys)),
        "candidate_delta_norm": candidate_norm,
        "released_delta_norm": target_norm,
        "cosine_to_released_delta": cosine,
        "projection_fraction_of_released_delta": projection_fraction,
        "orthogonal_delta_norm": orthogonal_norm,
        "distance_to_released_before": target_norm,
        "distance_to_released_after": after_distance,
        "relative_distance_improvement": distance_improvement,
    }


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        path_obj = Path(value)
        label = path_obj.parent.name if path_obj.name in {"latest.pt", "latent.pt"} else path_obj.stem
        path = value
    return label, resolve_path(path)


def format_float(value: float) -> str:
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.6f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], base: Path, target: Path) -> None:
    lines = [
        "# Latent Direction Metrics",
        "",
        "Released latent weights are used here only as an offline diagnostic target.",
        "They are not used as labels, distillation outputs, regularization targets, or training losses.",
        "",
        f"Base checkpoint: `{base.relative_to(ROOT) if base.is_relative_to(ROOT) else base}`",
        f"Released latent checkpoint: `{target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}`",
        "",
        "| Checkpoint | Cosine | Projection | Distance improvement | Delta norm | After distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    format_float(row["cosine_to_released_delta"]),
                    format_float(row["projection_fraction_of_released_delta"]),
                    format_float(row["relative_distance_improvement"]),
                    format_float(row["candidate_delta_norm"]),
                    format_float(row["distance_to_released_after"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Positive cosine means the normal training update points partly toward the released latent block.")
    lines.append("- Projection is the fraction of the base-to-released direction covered by the candidate update.")
    lines.append("- Distance improvement is positive only when the trained checkpoint is closer to the released latent block than the base.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    base_path = resolve_path(args.base)
    target_path = resolve_path(args.target)
    base = load_latent_state(base_path)
    target = load_latent_state(target_path)

    rows: list[dict[str, Any]] = []
    for value in args.checkpoint:
        label, path = parse_checkpoint(value)
        candidate = load_latent_state(path)
        keys = common_keys(base, target, candidate)
        if not keys:
            raise RuntimeError(f"No compatible latent tensors for {path}")
        metrics = dot_and_norms(base, target, candidate, keys)
        rows.append(
            {
                "label": label,
                "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                **metrics,
            }
        )

    rows.sort(key=lambda row: row["relative_distance_improvement"], reverse=True)
    if args.output_json:
        output = resolve_path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(resolve_path(args.output_md), rows, base_path, target_path)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
