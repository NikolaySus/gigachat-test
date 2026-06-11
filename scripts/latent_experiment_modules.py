from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class HierarchicalLatentAttentionModel(nn.Module):
    """Two-stage latent-memory block compatible with GigarEmbedModel.

    The released model's latent block keeps token positions as the output shape:
    token hidden states attend to learned latent K/V slots, then the embedding
    model mean-pools token positions. This module preserves that contract while
    adding a second, smaller latent-memory stage.
    """

    checkpoint_architecture = "hierarchical_latent_attention"

    def __init__(
        self,
        *,
        stage1_attention: nn.Module,
        stage1_feed_forward: nn.Module,
        stage1_latents: torch.Tensor,
        stage2_attention: nn.Module,
        stage2_feed_forward: nn.Module,
        stage2_latents: torch.Tensor,
    ) -> None:
        super().__init__()
        self.stage1_cross_attend_blocks = nn.ModuleList([stage1_attention, stage1_feed_forward])
        self.stage2_cross_attend_blocks = nn.ModuleList([stage2_attention, stage2_feed_forward])
        self.stage1_latents = nn.Parameter(stage1_latents.detach().clone())
        self.stage2_latents = nn.Parameter(stage2_latents.detach().clone())

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = hidden_states.size(0)

        stage1_attention, stage1_feed_forward = self.stage1_cross_attend_blocks
        stage1_latents = self.stage1_latents.repeat(batch_size, 1, 1)
        stage1_attended = stage1_attention(hidden_states, context=stage1_latents, attention_mask=attention_mask)
        stage1_output = stage1_feed_forward(stage1_attended + hidden_states) + stage1_attended + hidden_states

        stage2_attention, stage2_feed_forward = self.stage2_cross_attend_blocks
        stage2_latents = self.stage2_latents.repeat(batch_size, 1, 1)
        stage2_attended = stage2_attention(stage1_output, context=stage2_latents, attention_mask=attention_mask)
        return stage2_feed_forward(stage2_attended + stage1_output) + stage2_attended + stage1_output

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": self.checkpoint_architecture,
            "stage1_latents": int(self.stage1_latents.shape[0]),
            "stage2_latents": int(self.stage2_latents.shape[0]),
            "latent_dim": int(self.stage1_latents.shape[1]),
        }


def _new_attention_like(original_attention: nn.Module) -> nn.Module:
    num_heads = int(original_attention.num_heads)
    inner_dimension = int(original_attention.to_q.out_features)
    return type(original_attention)(
        query_dimension=original_attention.to_q.in_features,
        context_dimension=original_attention.to_kv.in_features,
        num_heads=num_heads,
        head_dim=inner_dimension // num_heads,
    )


def _new_feed_forward_like(original_feed_forward: nn.Module, latent_dim: int) -> nn.Module:
    return type(original_feed_forward)(latent_dim)


def install_hierarchical_latent_attention(
    model,
    *,
    stage2_latents: int = 128,
    copy_stage1: bool = True,
    initialize_fn=None,
) -> HierarchicalLatentAttentionModel:
    original = model.latent_attention_model
    original_attention, original_feed_forward = original.cross_attend_blocks
    latent_dim = int(original.latents.shape[1])
    device = original.latents.device
    dtype = original.latents.dtype

    if copy_stage1:
        stage1_attention = copy.deepcopy(original_attention)
        stage1_feed_forward = copy.deepcopy(original_feed_forward)
        stage1_latents = original.latents.detach().clone()
    else:
        stage1_attention = _new_attention_like(original_attention).to(device=device, dtype=dtype)
        stage1_feed_forward = _new_feed_forward_like(original_feed_forward, latent_dim).to(device=device, dtype=dtype)
        stage1_latents = torch.empty_like(original.latents)
        if initialize_fn is not None:
            stage1_attention.apply(initialize_fn)
            stage1_feed_forward.apply(initialize_fn)
        torch.nn.init.normal_(stage1_latents)

    stage2_attention = _new_attention_like(original_attention).to(device=device, dtype=dtype)
    stage2_feed_forward = _new_feed_forward_like(original_feed_forward, latent_dim).to(device=device, dtype=dtype)
    stage2_latent_tensor = torch.empty(stage2_latents, latent_dim, device=device, dtype=dtype)
    if initialize_fn is not None:
        stage2_attention.apply(initialize_fn)
        stage2_feed_forward.apply(initialize_fn)
    torch.nn.init.normal_(stage2_latent_tensor)

    hierarchical = HierarchicalLatentAttentionModel(
        stage1_attention=stage1_attention,
        stage1_feed_forward=stage1_feed_forward,
        stage1_latents=stage1_latents.to(device=device, dtype=dtype),
        stage2_attention=stage2_attention,
        stage2_feed_forward=stage2_feed_forward,
        stage2_latents=stage2_latent_tensor,
    ).to(device=device, dtype=dtype)
    model.latent_attention_model = hierarchical
    return hierarchical


def load_latent_experiment_checkpoint(model, checkpoint: dict[str, Any]) -> None:
    architecture = checkpoint.get("checkpoint_architecture", "original_latent_attention")
    if architecture == HierarchicalLatentAttentionModel.checkpoint_architecture:
        module_config = checkpoint.get("latent_module_config", {})
        install_hierarchical_latent_attention(
            model,
            stage2_latents=int(module_config.get("stage2_latents", 128)),
            copy_stage1=True,
            initialize_fn=getattr(model, "_init_weights", None),
        )
    elif architecture != "original_latent_attention":
        raise ValueError(f"Unsupported latent checkpoint architecture: {architecture}")

    state_dict = checkpoint.get("latent_attention_model", checkpoint)
    model.latent_attention_model.load_state_dict(state_dict)
