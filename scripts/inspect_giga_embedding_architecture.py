from __future__ import annotations

import argparse
from pathlib import Path

import torch

from giga_model_utils import ModelLoadConfig, load_giga_embeddings, write_json


def summarize_model(model) -> dict:
    latent = model.latent_attention_model
    cfg = model.config
    latent_cfg = cfg.latent_attention_config
    text_cfg = cfg.text_config

    attention = latent.cross_attend_blocks[0]
    feed_forward = latent.cross_attend_blocks[1]

    return {
        "model_type": type(model).__name__,
        "paper_claim_checked": {
            "figure_1_direction": "LLM output is Q; latent array is K,V",
            "implementation_direction": "hidden_states are Q; learnable latents are K,V",
            "latent_slots_are_output_slots": False,
            "pooling": "mask-aware mean pooling over token positions after latent attention",
        },
        "text_encoder": {
            "hidden_size": text_cfg.hidden_size,
            "layers": text_cfg.num_hidden_layers,
            "attention_heads": text_cfg.num_attention_heads,
            "max_position_embeddings": text_cfg.max_position_embeddings,
            "is_decoder": text_cfg.is_decoder,
        },
        "latent_attention": {
            "num_latents": latent_cfg.num_latents_value,
            "latent_dim": latent_cfg.latent_dim,
            "hidden_dim": latent_cfg.hidden_dim,
            "num_cross_heads": latent_cfg.num_cross_heads,
            "cross_dim_head": latent_cfg.cross_dim_head,
            "mlp_mult": latent_cfg.mult,
            "latents_shape": list(latent.latents.shape),
            "attention_class": type(attention).__name__,
            "feed_forward_class": type(feed_forward).__name__,
            "attention_mask_used_inside_attention": False,
        },
        "parameter_counts": {
            "total": sum(p.numel() for p in model.parameters()),
            "latent_attention_model": sum(p.numel() for p in latent.parameters()),
            "trainable_now": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/architecture_summary.json"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    args = parser.parse_args()

    _, model = load_giga_embeddings(
        ModelLoadConfig(
            batch_size=1,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    )
    summary = summarize_model(model)
    summary["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(args.out, summary)
    print(f"Wrote {args.out}")
    print(
        "Baseline latent block: token hidden states query learned latent K,V; "
        "then FFN residual; final embedding is mean-pooled over token positions."
    )


if __name__ == "__main__":
    main()
