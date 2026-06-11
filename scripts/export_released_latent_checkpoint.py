#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from giga_model_utils import ModelLoadConfig, load_giga_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the released Giga latent-attention state as a small checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="ai-sage/Giga-Embeddings-instruct")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    _, model = load_giga_embeddings(
        ModelLoadConfig(
            model_name=args.model_name,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": 0,
        "config": {
            "source": args.model_name,
            "description": "Released model latent-attention state exported for teacher distillation.",
        },
        "checkpoint_architecture": "original_latent_attention",
        "latent_module_config": (
            model.latent_attention_model.export_config()
            if hasattr(model.latent_attention_model, "export_config")
            else {"architecture": "original_latent_attention"}
        ),
        "latent_attention_model": {
            name: value.detach().cpu()
            for name, value in model.latent_attention_model.state_dict().items()
        },
    }
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "output": str(args.output),
                "keys": len(payload["latent_attention_model"]),
                "latent_module_config": payload["latent_module_config"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
