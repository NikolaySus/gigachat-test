from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from latent_experiment_modules import load_latent_experiment_checkpoint
except ModuleNotFoundError:
    from scripts.latent_experiment_modules import load_latent_experiment_checkpoint


MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"


@dataclass(frozen=True)
class ModelLoadConfig:
    model_name: str = MODEL_NAME
    max_length: int = 512
    batch_size: int = 8
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    local_files_only: bool = False
    latent_checkpoint: Path | None = None


def patch_transformers_for_python_313() -> None:
    """Work around transformers 4.51 remote-code import on Python 3.13."""
    from transformers.configuration_utils import PretrainedConfig
    import transformers.modeling_utils as modeling_utils

    if not hasattr(PretrainedConfig, "torch_dtype"):
        PretrainedConfig.torch_dtype = None
    if not hasattr(modeling_utils, "init_empty_weights"):
        from contextlib import nullcontext

        modeling_utils.init_empty_weights = nullcontext


def resolve_model_source(model_name: str, local_files_only: bool) -> str:
    if not local_files_only or model_name != MODEL_NAME:
        return model_name

    cache_root = Path.home() / ".cache/huggingface/hub/models--ai-sage--Giga-Embeddings-instruct/snapshots"
    snapshots = sorted(cache_root.glob("*/config.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if snapshots:
        return str(snapshots[0].parent)
    return model_name


def load_giga_embeddings(config: ModelLoadConfig = ModelLoadConfig()):
    patch_transformers_for_python_313()
    if config.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_source = resolve_model_source(config.model_name, config.local_files_only)
    attn_implementation = config.attn_implementation
    if attn_implementation is None:
        attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_source,
        torch_dtype=config.torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        device_map="auto" if config.device == "cuda" else None,
        local_files_only=config.local_files_only,
    )
    if config.device != "cuda":
        model = model.to(config.device)
    if config.latent_checkpoint is not None:
        checkpoint = torch.load(config.latent_checkpoint, map_location="cpu")
        load_latent_experiment_checkpoint(model, checkpoint)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def encode_texts(
    texts: Iterable[str],
    tokenizer,
    model,
    *,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
    texts = list(texts)
    vectors: list[torch.Tensor] = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded, return_embeddings=True)
        emb = output["sentence_embeddings"] if isinstance(output, dict) else output
        vectors.append(F.normalize(emb.float(), p=2, dim=-1).cpu())
    return torch.cat(vectors, dim=0).numpy()


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors @ vectors.T


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
