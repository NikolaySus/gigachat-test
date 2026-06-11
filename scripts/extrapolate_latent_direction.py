from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("latent_attention_model", checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrapolate a latent checkpoint along a source-to-target direction.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_state(args.base)
    target = load_state(args.target)
    if base.keys() != target.keys():
        raise ValueError("Checkpoint keys differ")

    extrapolated = {}
    for key in base:
        base_tensor = base[key].float()
        target_tensor = target[key].float()
        value = base_tensor + args.alpha * (target_tensor - base_tensor)
        extrapolated[key] = value.to(dtype=base[key].dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent_attention_model": extrapolated,
            "base": str(args.base),
            "target": str(args.target),
            "alpha": args.alpha,
        },
        args.output,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
