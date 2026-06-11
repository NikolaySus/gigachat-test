from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def parse_weighted_path(value: str) -> tuple[float, Path]:
    if "=" not in value:
        return 1.0, Path(value)
    weight, path = value.split("=", 1)
    return float(weight), Path(path)


def latent_state_from_checkpoint(checkpoint: dict[str, Any]) -> OrderedDict[str, torch.Tensor]:
    state = checkpoint.get("latent_attention_model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a latent state dict")
    return OrderedDict((key, value) for key, value in state.items() if torch.is_tensor(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Average latent-only checkpoint state dicts.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Path or weight=path.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weighted_paths = [parse_weighted_path(item) for item in args.checkpoint]
    total_weight = sum(weight for weight, _ in weighted_paths)
    if total_weight <= 0:
        raise ValueError("Total checkpoint weight must be positive.")

    averaged = None
    template: dict[str, Any] | None = None
    for weight, path in weighted_paths:
        checkpoint = torch.load(path, map_location="cpu")
        state = latent_state_from_checkpoint(checkpoint)
        factor = weight / total_weight
        if averaged is None:
            if isinstance(checkpoint, dict) and "latent_attention_model" in checkpoint:
                template = dict(checkpoint)
            averaged = OrderedDict((key, value.detach().float().mul(factor)) for key, value in state.items())
            continue
        if state.keys() != averaged.keys():
            missing = sorted(set(averaged) ^ set(state))
            raise ValueError(f"Checkpoint keys differ for {path}: {missing[:10]}")
        for key, value in state.items():
            averaged[key].add_(value.detach().float(), alpha=factor)

    if averaged is None:
        raise RuntimeError("No checkpoints were averaged")

    output_state: dict[str, Any] | OrderedDict[str, torch.Tensor]
    if template is not None:
        template["latent_attention_model"] = averaged
        template["averaged_from"] = [{"weight": weight, "path": str(path)} for weight, path in weighted_paths]
        output_state = template
    else:
        output_state = averaged

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_state, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
