from __future__ import annotations

import argparse
import json
import random
import time
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from giga_model_utils import ModelLoadConfig, load_giga_embeddings, write_json
from latent_experiment_modules import HierarchicalLatentAttentionModel, install_hierarchical_latent_attention


class ContrastiveJsonlDataset(Dataset):
    def __init__(self, path: Path):
        self.records = []
        with path.open(encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                objective = record.get("objective", "contrastive")
                if objective == "contrastive":
                    if "query" not in record or "positive" not in record:
                        raise ValueError(f"{path}:{line_no}: expected `query` and `positive` fields")
                    record.setdefault("negatives", [])
                    record.setdefault("positives", [])
                    if not isinstance(record["positives"], list):
                        raise ValueError(f"{path}:{line_no}: expected `positives` to be a list")
                elif objective == "pair_score":
                    if "sentence1" not in record or "sentence2" not in record or "score" not in record:
                        raise ValueError(f"{path}:{line_no}: expected `sentence1`, `sentence2`, and `score` fields")
                    record["score"] = float(record["score"])
                elif objective == "prototype_classification":
                    if "query" not in record or "label" not in record or "prototypes" not in record:
                        raise ValueError(
                            f"{path}:{line_no}: expected `query`, `label`, and `prototypes` fields"
                        )
                    if not isinstance(record["prototypes"], dict) or not record["prototypes"]:
                        raise ValueError(f"{path}:{line_no}: expected non-empty `prototypes` mapping")
                elif objective in {"knn_classification", "ridge_classification"}:
                    if "query" not in record or "label" not in record or "supports" not in record:
                        raise ValueError(
                            f"{path}:{line_no}: expected `query`, `label`, and `supports` fields"
                        )
                    if not isinstance(record["supports"], dict) or not record["supports"]:
                        raise ValueError(f"{path}:{line_no}: expected non-empty `supports` mapping")
                elif objective == "multilabel_support_classification":
                    if "query" not in record or "labels" not in record or "supports" not in record:
                        raise ValueError(
                            f"{path}:{line_no}: expected `query`, `labels`, and `supports` fields"
                        )
                    if not isinstance(record["labels"], list):
                        raise ValueError(f"{path}:{line_no}: expected `labels` to be a list")
                    if not isinstance(record["supports"], dict) or not record["supports"]:
                        raise ValueError(f"{path}:{line_no}: expected non-empty `supports` mapping")
                elif objective == "cedr_knn_episode":
                    if "query" not in record or "labels" not in record or "supports" not in record:
                        raise ValueError(
                            f"{path}:{line_no}: expected `query`, `labels`, and `supports` fields"
                        )
                    if not isinstance(record["labels"], list):
                        raise ValueError(f"{path}:{line_no}: expected `labels` to be a list")
                    if not isinstance(record["supports"], list) or not record["supports"]:
                        raise ValueError(f"{path}:{line_no}: expected non-empty `supports` list")
                    for support in record["supports"]:
                        if not isinstance(support, dict) or "text" not in support or "labels" not in support:
                            raise ValueError(
                                f"{path}:{line_no}: every support must contain `text` and `labels`"
                            )
                        if not isinstance(support["labels"], list):
                            raise ValueError(f"{path}:{line_no}: support `labels` must be a list")
                elif objective == "labeled_text":
                    if "text" not in record or "label" not in record:
                        raise ValueError(f"{path}:{line_no}: expected `text` and `label` fields")
                elif objective == "hierarchical_labeled_text":
                    if "text" not in record or "labels" not in record:
                        raise ValueError(f"{path}:{line_no}: expected `text` and `labels` fields")
                    if not isinstance(record["labels"], dict) or not record["labels"]:
                        raise ValueError(f"{path}:{line_no}: expected non-empty `labels` mapping")
                elif objective == "linear_probe_labeled_text":
                    if "text" not in record or "label" not in record or "role" not in record:
                        raise ValueError(f"{path}:{line_no}: expected `text`, `label`, and `role` fields")
                    if record["role"] not in {"support", "query"}:
                        raise ValueError(f"{path}:{line_no}: expected role to be `support` or `query`")
                elif objective in {"prototype_none_classification", "prototype_uniform_classification"}:
                    if "query" not in record or "label" not in record or "prototypes" not in record:
                        raise ValueError(
                            f"{path}:{line_no}: expected `query`, `label`, and `prototypes` fields"
                        )
                    if not isinstance(record["prototypes"], dict) or not record["prototypes"]:
                        raise ValueError(f"{path}:{line_no}: expected non-empty `prototypes` mapping")
                else:
                    raise ValueError(f"{path}:{line_no}: unsupported objective `{objective}`")
                record["objective"] = objective
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No records found in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": records,
    }


def filter_records(
    records: list[dict[str, Any]],
    *,
    objectives: list[str] | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    objective_set = set(objectives or [])
    source_set = set(sources or [])
    filtered = []
    for record in records:
        if objective_set and record["objective"] not in objective_set:
            continue
        if source_set and record.get("source") not in source_set:
            continue
        filtered.append(record)
    return filtered


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_for_exp01b(model, *, freeze_llm: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if freeze_llm:
        for parameter in model.latent_attention_model.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True


def reinitialize_latent_block(model) -> None:
    model.latent_attention_model.apply(model._init_weights)
    if hasattr(model.latent_attention_model, "latents"):
        torch.nn.init.normal_(model.latent_attention_model.latents)
    if hasattr(model.latent_attention_model, "stage1_latents"):
        torch.nn.init.normal_(model.latent_attention_model.stage1_latents)
    if hasattr(model.latent_attention_model, "stage2_latents"):
        torch.nn.init.normal_(model.latent_attention_model.stage2_latents)


def configure_latent_module(model, config: dict[str, Any]) -> None:
    architecture = config.get("latent_architecture", "original_latent_attention")
    if architecture == "original_latent_attention":
        return
    if architecture == HierarchicalLatentAttentionModel.checkpoint_architecture:
        install_hierarchical_latent_attention(
            model,
            stage2_latents=int(config.get("stage2_latents", 128)),
            copy_stage1=bool(config.get("copy_stage1_from_original", True)),
            initialize_fn=getattr(model, "_init_weights", None),
        )
        return
    raise ValueError(f"Unsupported latent_architecture: {architecture}")


def encode_train(texts: list[str], tokenizer, model, *, max_length: int) -> torch.Tensor:
    device = next(model.parameters()).device
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded, return_embeddings=True)
    embeddings = output["sentence_embeddings"] if isinstance(output, dict) else output
    return embeddings.float()


def latent_state_clone(model, *, detach: bool = True) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in model.latent_attention_model.state_dict().items():
        tensor = value.detach() if detach else value
        state[name] = tensor.clone()
    return state


def parameter_anchor_loss(model, anchor_state: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    current_state = dict(model.latent_attention_model.named_parameters())
    for name, anchor in anchor_state.items():
        parameter = current_state.get(name)
        if parameter is None:
            continue
        losses.append(F.mse_loss(parameter.float(), anchor.to(parameter.device).float()))
    if not losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def parameter_direction_loss(
    model,
    *,
    anchor_state: dict[str, torch.Tensor],
    direction_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    numerator = torch.zeros((), device=next(model.parameters()).device)
    denominator = torch.zeros((), device=next(model.parameters()).device)
    current_state = dict(model.latent_attention_model.named_parameters())
    for name, anchor in anchor_state.items():
        parameter = current_state.get(name)
        direction_target = direction_state.get(name)
        if parameter is None or direction_target is None or not parameter.is_floating_point():
            continue
        anchor = anchor.to(parameter.device).float()
        direction = direction_target.to(parameter.device).float() - anchor
        current_delta = parameter.float() - anchor
        numerator = numerator + (current_delta * direction).sum()
        denominator = denominator + (direction * direction).sum()
    if float(denominator.detach().cpu()) == 0.0:
        return torch.zeros((), device=next(model.parameters()).device)
    return numerator / denominator.clamp_min(1e-12)


def parameter_direction_target_loss(
    model,
    *,
    anchor_state: dict[str, torch.Tensor],
    direction_state: dict[str, torch.Tensor],
    target_projection: float,
) -> torch.Tensor:
    projection = parameter_direction_loss(
        model,
        anchor_state=anchor_state,
        direction_state=direction_state,
    )
    target = torch.tensor(
        target_projection,
        device=projection.device,
        dtype=projection.dtype,
    )
    return F.mse_loss(projection, target)


def load_latent_state_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("latent_attention_model", checkpoint)
    return {name: value.detach().clone() for name, value in state.items()}


def texts_from_records(records: list[dict[str, Any]], *, max_texts: int) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for record in records:
        candidates = []
        if record["objective"] == "contrastive":
            candidates.extend([record["query"], record["positive"]])
            candidates.extend(record.get("negatives", []))
        elif record["objective"] == "pair_score":
            candidates.extend([record["sentence1"], record["sentence2"]])
        elif record["objective"] == "prototype_classification":
            candidates.append(record["query"])
            for prototype_texts in record["prototypes"].values():
                candidates.extend(prototype_texts)
        elif record["objective"] == "knn_classification":
            candidates.append(record["query"])
            for support_texts in record["supports"].values():
                candidates.extend(support_texts)
        elif record["objective"] == "multilabel_support_classification":
            candidates.append(record["query"])
            for support_texts in record["supports"].values():
                candidates.extend(support_texts)
        elif record["objective"] == "cedr_knn_episode":
            candidates.append(record["query"])
            candidates.extend(support["text"] for support in record["supports"])
        elif record["objective"] == "labeled_text":
            candidates.append(record["text"])
        elif record["objective"] in {"prototype_none_classification", "prototype_uniform_classification"}:
            candidates.append(record["query"])
            for prototype_texts in record["prototypes"].values():
                candidates.extend(prototype_texts)
        for text in candidates:
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
            if len(texts) >= max_texts:
                return texts
    return texts


def teacher_encode_with_latent_state(
    texts: list[str],
    *,
    tokenizer,
    model,
    max_length: int,
    teacher_latent_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    current_latent_state = latent_state_clone(model)
    latent_was_training = model.latent_attention_model.training
    try:
        model.latent_attention_model.load_state_dict(teacher_latent_state)
        model.latent_attention_model.eval()
        with torch.no_grad():
            return encode_train(texts, tokenizer, model, max_length=max_length).detach()
    finally:
        model.latent_attention_model.load_state_dict(current_latent_state)
        if latent_was_training:
            model.latent_attention_model.train()
        else:
            model.latent_attention_model.eval()


def distillation_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    teacher_latent_state: dict[str, torch.Tensor],
    max_texts: int,
) -> torch.Tensor:
    texts = texts_from_records(records, max_texts=max_texts)
    if not texts:
        return torch.zeros((), device=next(model.parameters()).device)
    teacher_embeddings = teacher_encode_with_latent_state(
        texts,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        teacher_latent_state=teacher_latent_state,
    )
    student_embeddings = encode_train(texts, tokenizer, model, max_length=max_length)
    return 1.0 - F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1).mean()


def pairwise_distillation_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    teacher_latent_state: dict[str, torch.Tensor],
    max_texts: int,
) -> torch.Tensor:
    texts = texts_from_records(records, max_texts=max_texts)
    if len(texts) < 2:
        return torch.zeros((), device=next(model.parameters()).device)
    teacher_embeddings = F.normalize(
        teacher_encode_with_latent_state(
            texts,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            teacher_latent_state=teacher_latent_state,
        ).float(),
        p=2,
        dim=1,
    )
    student_embeddings = F.normalize(
        encode_train(texts, tokenizer, model, max_length=max_length).float(),
        p=2,
        dim=1,
    )
    teacher_similarities = teacher_embeddings @ teacher_embeddings.T
    student_similarities = student_embeddings @ student_embeddings.T
    mask = ~torch.eye(
        teacher_similarities.shape[0],
        device=teacher_similarities.device,
        dtype=torch.bool,
    )
    return F.mse_loss(student_similarities[mask], teacher_similarities[mask])


def combined_distillation_losses(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    teacher_latent_state: dict[str, torch.Tensor],
    distillation_max_texts: int,
    pairwise_max_texts: int,
) -> dict[str, torch.Tensor]:
    max_texts = max(distillation_max_texts, pairwise_max_texts)
    texts = texts_from_records(records, max_texts=max_texts)
    device = next(model.parameters()).device
    if not texts:
        zero = torch.zeros((), device=device)
        return {"distill": zero, "pairdist": zero}

    teacher_embeddings = teacher_encode_with_latent_state(
        texts,
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        teacher_latent_state=teacher_latent_state,
    )
    student_embeddings = encode_train(texts, tokenizer, model, max_length=max_length)

    losses: dict[str, torch.Tensor] = {}
    distill_count = min(distillation_max_texts, len(texts))
    if distill_count > 0:
        losses["distill"] = 1.0 - F.cosine_similarity(
            student_embeddings[:distill_count],
            teacher_embeddings[:distill_count],
            dim=1,
        ).mean()
    else:
        losses["distill"] = torch.zeros((), device=device)

    pair_count = min(pairwise_max_texts, len(texts))
    if pair_count >= 2:
        teacher_pair = F.normalize(teacher_embeddings[:pair_count].float(), p=2, dim=1)
        student_pair = F.normalize(student_embeddings[:pair_count].float(), p=2, dim=1)
        teacher_similarities = teacher_pair @ teacher_pair.T
        student_similarities = student_pair @ student_pair.T
        mask = ~torch.eye(
            teacher_similarities.shape[0],
            device=teacher_similarities.device,
            dtype=torch.bool,
        )
        losses["pairdist"] = F.mse_loss(student_similarities[mask], teacher_similarities[mask])
    else:
        losses["pairdist"] = torch.zeros((), device=device)
    return losses


def contrastive_loss(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = query_embeddings @ candidate_embeddings.T
    logits = logits / temperature
    if positive_mask is not None:
        log_denominator = torch.logsumexp(logits, dim=1)
        masked_logits = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
        log_numerator = torch.logsumexp(masked_logits, dim=1)
        return (log_denominator - log_numerator).mean()
    return F.cross_entropy(logits, labels)


def pair_score_loss(
    sentence1_embeddings: torch.Tensor,
    sentence2_embeddings: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    similarities = torch.sum(sentence1_embeddings * sentence2_embeddings, dim=1)
    similarities = (similarities + 1.0) / 2.0
    return F.mse_loss(similarities, scores)


def pair_score_rank_loss(
    sentence1_embeddings: torch.Tensor,
    sentence2_embeddings: torch.Tensor,
    scores: torch.Tensor,
    *,
    scale: float = 20.0,
    min_score_gap: float = 0.05,
) -> torch.Tensor:
    similarities = torch.sum(sentence1_embeddings * sentence2_embeddings, dim=1)
    similarities = (similarities + 1.0) / 2.0
    score_diffs = scores.unsqueeze(1) - scores.unsqueeze(0)
    ordered_mask = score_diffs > min_score_gap
    if not ordered_mask.any():
        return torch.zeros((), device=similarities.device, dtype=similarities.dtype)
    similarity_diffs = similarities.unsqueeze(1) - similarities.unsqueeze(0)
    violations = -scale * similarity_diffs.masked_select(ordered_mask)
    zeros = torch.zeros(1, device=violations.device, dtype=violations.dtype)
    return torch.logsumexp(torch.cat([zeros, violations]), dim=0)


def pair_score_correlation_loss(
    sentence1_embeddings: torch.Tensor,
    sentence2_embeddings: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    similarities = torch.sum(sentence1_embeddings * sentence2_embeddings, dim=1)
    if similarities.numel() < 2:
        return torch.zeros((), device=similarities.device, dtype=similarities.dtype)
    centered_similarities = similarities - similarities.mean()
    centered_scores = scores.to(similarities.dtype) - scores.to(similarities.dtype).mean()
    sim_norm = centered_similarities.norm()
    score_norm = centered_scores.norm()
    if float(sim_norm.detach().cpu()) == 0.0 or float(score_norm.detach().cpu()) == 0.0:
        return torch.zeros((), device=similarities.device, dtype=similarities.dtype)
    correlation = (centered_similarities * centered_scores).sum() / (sim_norm * score_norm).clamp_min(1e-12)
    return 1.0 - correlation


def labeled_metric_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
    loss_name: str,
    margin: float = 0.25,
    gamma: float = 32.0,
    alpha: float = 2.0,
    beta: float = 50.0,
    base: float = 0.5,
) -> torch.Tensor:
    texts = [record["text"] for record in records]
    labels = [str(record["label"]) for record in records]
    encode_batch_size = int(records[0].get("encode_batch_size", 0) or 0)
    if encode_batch_size > 0 and encode_batch_size < len(texts):
        embedding_chunks = [
            encode_train(texts[start : start + encode_batch_size], tokenizer, model, max_length=max_length)
            for start in range(0, len(texts), encode_batch_size)
        ]
        embeddings = torch.cat(embedding_chunks, dim=0)
    else:
        embeddings = encode_train(texts, tokenizer, model, max_length=max_length)
    embeddings = F.normalize(embeddings, dim=1)
    return labeled_metric_loss_from_embeddings(
        embeddings,
        labels,
        temperature=temperature,
        loss_name=loss_name,
        margin=margin,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        base=base,
    )


def labeled_metric_loss_from_embeddings(
    embeddings: torch.Tensor,
    labels: list[str],
    *,
    temperature: float,
    loss_name: str,
    margin: float = 0.25,
    gamma: float = 32.0,
    alpha: float = 2.0,
    beta: float = 50.0,
    base: float = 0.5,
) -> torch.Tensor:
    similarities = embeddings @ embeddings.T
    label_to_id = {label: index for index, label in enumerate(sorted(set(labels)))}
    labels_tensor = torch.tensor(
        [label_to_id[label] for label in labels],
        device=similarities.device,
        dtype=torch.long,
    )
    same_label = labels_tensor.unsqueeze(0).eq(labels_tensor.unsqueeze(1))
    eye = torch.eye(len(labels), device=similarities.device, dtype=torch.bool)
    positive_mask = same_label & ~eye
    negative_mask = ~same_label

    if not positive_mask.any() or not negative_mask.any():
        return similarities.sum() * 0.0

    loss_name = loss_name.lower()
    if loss_name == "supcon":
        logits = similarities / temperature
        logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
        log_denominator = torch.logsumexp(logits, dim=1)
        positive_logits = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
        log_numerator = torch.logsumexp(positive_logits, dim=1)
        valid = positive_mask.any(dim=1)
        valid_rows = valid.nonzero(as_tuple=True)[0]
        return (
            log_denominator.index_select(0, valid_rows)
            - log_numerator.index_select(0, valid_rows)
        ).mean()

    losses = []
    for row in range(similarities.shape[0]):
        positives = similarities[row][positive_mask[row]]
        negatives = similarities[row][negative_mask[row]]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        if loss_name == "circle":
            alpha_p = torch.relu(1.0 + margin - positives.detach())
            alpha_n = torch.relu(negatives.detach() + margin)
            delta_p = 1.0 - margin
            delta_n = margin
            logit_p = -gamma * alpha_p * (positives - delta_p)
            logit_n = gamma * alpha_n * (negatives - delta_n)
            losses.append(F.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0)))
        elif loss_name == "multi_similarity":
            hard_pos = positives[positives < negatives.max() + margin]
            hard_neg = negatives[negatives > positives.min() - margin]
            if hard_pos.numel() == 0 or hard_neg.numel() == 0:
                continue
            pos_term = torch.log1p(torch.exp(-alpha * (hard_pos - base)).sum()) / alpha
            neg_term = torch.log1p(torch.exp(beta * (hard_neg - base)).sum()) / beta
            losses.append(pos_term + neg_term)
        else:
            raise ValueError(f"Unsupported labeled_metric_loss `{loss_name}`")
    if not losses:
        return similarities.sum() * 0.0
    return torch.stack(losses).mean()


def hierarchical_labeled_metric_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
    loss_name: str,
    label_weights: dict[str, float] | None = None,
    margin: float = 0.25,
    gamma: float = 32.0,
    alpha: float = 2.0,
    beta: float = 50.0,
    base: float = 0.5,
) -> torch.Tensor:
    texts = [record["text"] for record in records]
    encode_batch_size = int(records[0].get("encode_batch_size", 0) or 0)
    if encode_batch_size > 0 and encode_batch_size < len(texts):
        embedding_chunks = [
            encode_train(texts[start : start + encode_batch_size], tokenizer, model, max_length=max_length)
            for start in range(0, len(texts), encode_batch_size)
        ]
        embeddings = torch.cat(embedding_chunks, dim=0)
    else:
        embeddings = encode_train(texts, tokenizer, model, max_length=max_length)
    embeddings = F.normalize(embeddings, dim=1)
    keys = sorted({key for record in records for key in record["labels"]})
    weights = label_weights or {}
    losses = []
    for key in keys:
        key_records = [record for record in records if key in record["labels"]]
        if len(key_records) < 3:
            continue
        indices = [index for index, record in enumerate(records) if key in record["labels"]]
        key_embeddings = embeddings.index_select(
            0,
            torch.tensor(indices, device=embeddings.device, dtype=torch.long),
        )
        labels = [f"{key}::{record['labels'][key]}" for record in key_records]
        loss = labeled_metric_loss_from_embeddings(
            key_embeddings,
            labels,
            temperature=temperature,
            loss_name=loss_name,
            margin=margin,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            base=base,
        )
        if not torch.isfinite(loss):
            continue
        losses.append(float(weights.get(key, 1.0)) * loss)
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).sum()


def encode_texts_for_records(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    text_key: str = "text",
) -> torch.Tensor:
    texts = [record[text_key] for record in records]
    encode_batch_size = int(records[0].get("encode_batch_size", 0) or 0)
    if encode_batch_size > 0 and encode_batch_size < len(texts):
        chunks = [
            encode_train(texts[start : start + encode_batch_size], tokenizer, model, max_length=max_length)
            for start in range(0, len(texts), encode_batch_size)
        ]
        return torch.cat(chunks, dim=0)
    return encode_train(texts, tokenizer, model, max_length=max_length)


def linear_probe_labeled_text_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
    ridge_lambda: float,
    use_bias: bool = True,
) -> torch.Tensor:
    embeddings = F.normalize(
        encode_texts_for_records(records, tokenizer=tokenizer, model=model, max_length=max_length),
        dim=1,
    )
    support_indices = [index for index, record in enumerate(records) if record["role"] == "support"]
    query_indices = [index for index, record in enumerate(records) if record["role"] == "query"]
    if not support_indices or not query_indices:
        return torch.zeros((), device=embeddings.device)

    support_labels = [str(records[index]["label"]) for index in support_indices]
    class_order = sorted(set(support_labels))
    if len(class_order) < 2:
        return torch.zeros((), device=embeddings.device)
    class_to_index = {label: index for index, label in enumerate(class_order)}
    valid_query_indices = [
        index for index in query_indices if str(records[index]["label"]) in class_to_index
    ]
    if not valid_query_indices:
        return torch.zeros((), device=embeddings.device)

    support = embeddings.index_select(
        0, torch.tensor(support_indices, device=embeddings.device, dtype=torch.long)
    )
    queries = embeddings.index_select(
        0, torch.tensor(valid_query_indices, device=embeddings.device, dtype=torch.long)
    )
    if use_bias:
        support = torch.cat([support, torch.ones(support.shape[0], 1, device=support.device)], dim=1)
        queries = torch.cat([queries, torch.ones(queries.shape[0], 1, device=queries.device)], dim=1)

    target = torch.zeros(
        support.shape[0],
        len(class_order),
        device=support.device,
        dtype=support.dtype,
    )
    target[
        torch.arange(support.shape[0], device=support.device),
        torch.tensor([class_to_index[label] for label in support_labels], device=support.device),
    ] = 1.0
    identity = torch.eye(support.shape[1], device=support.device, dtype=support.dtype)
    if use_bias:
        identity[-1, -1] = 0.0
    gram = support.T @ support + ridge_lambda * identity
    weights = torch.linalg.solve(gram, support.T @ target)
    logits = (queries @ weights) / temperature
    labels = torch.tensor(
        [class_to_index[str(records[index]["label"])] for index in valid_query_indices],
        device=queries.device,
        dtype=torch.long,
    )
    return F.cross_entropy(logits, labels)


def prototype_classification_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
) -> torch.Tensor:
    class_order = sorted({label for record in records for label in record["prototypes"]})
    class_to_index = {label: index for index, label in enumerate(class_order)}
    query_embeddings = encode_train(
        [record["query"] for record in records],
        tokenizer,
        model,
        max_length=max_length,
    )
    logits_per_record = []
    labels = []
    for query_embedding, record in zip(query_embeddings, records, strict=True):
        prototypes = []
        for label in class_order:
            texts = record["prototypes"].get(label)
            if not texts:
                prototypes.append(torch.zeros_like(query_embedding))
                continue
            prototype_embeddings = encode_train(texts, tokenizer, model, max_length=max_length)
            prototypes.append(prototype_embeddings.mean(dim=0))
        prototype_matrix = torch.stack(prototypes)
        logits_per_record.append(query_embedding @ prototype_matrix.T)
        labels.append(class_to_index[record["label"]])
    logits = torch.stack(logits_per_record) / temperature
    label_tensor = torch.tensor(labels, device=query_embeddings.device, dtype=torch.long)
    return F.cross_entropy(logits, label_tensor)


def knn_classification_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
) -> torch.Tensor:
    class_order = sorted({label for record in records for label in record["supports"]})
    class_to_index = {label: index for index, label in enumerate(class_order)}
    query_embeddings = encode_train(
        [record["query"] for record in records],
        tokenizer,
        model,
        max_length=max_length,
    )
    logits_per_record = []
    labels = []
    for query_embedding, record in zip(query_embeddings, records, strict=True):
        class_logits = []
        for label in class_order:
            texts = record["supports"].get(label)
            if not texts:
                class_logits.append(torch.full((), -100.0, device=query_embedding.device))
                continue
            support_embeddings = encode_train(texts, tokenizer, model, max_length=max_length)
            similarities = query_embedding @ support_embeddings.T
            # CEDR uses 5-NN voting; logsumexp over individual supports keeps
            # nearest support examples visible instead of collapsing to a mean prototype.
            class_logits.append(torch.logsumexp(similarities / temperature, dim=0))
        logits_per_record.append(torch.stack(class_logits))
        labels.append(class_to_index[record["label"]])
    logits = torch.stack(logits_per_record)
    label_tensor = torch.tensor(labels, device=query_embeddings.device, dtype=torch.long)
    return F.cross_entropy(logits, label_tensor)


def ridge_classification_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
) -> torch.Tensor:
    losses = []
    for record in records:
        supports_by_label = record["supports"]
        class_order = sorted(supports_by_label)
        class_to_index = {label: index for index, label in enumerate(class_order)}
        if record["label"] not in class_to_index:
            continue

        support_texts = []
        support_labels = []
        for label in class_order:
            for text in supports_by_label[label]:
                support_texts.append(text)
                support_labels.append(class_to_index[label])
        if len(set(support_labels)) < 2 or len(support_texts) < 2:
            continue

        query_embedding = encode_train(
            [record["query"]],
            tokenizer,
            model,
            max_length=max_length,
        )
        support_embeddings = encode_train(
            support_texts,
            tokenizer,
            model,
            max_length=max_length,
        )
        query_embedding = F.normalize(query_embedding.float(), p=2, dim=1)
        support_embeddings = F.normalize(support_embeddings.float(), p=2, dim=1)

        targets = torch.zeros(
            (len(support_texts), len(class_order)),
            device=support_embeddings.device,
            dtype=support_embeddings.dtype,
        )
        target_indices = torch.tensor(
            support_labels,
            device=support_embeddings.device,
            dtype=torch.long,
        )
        targets.scatter_(1, target_indices.unsqueeze(1), 1.0)

        ridge_lambda = float(record.get("ridge_lambda", 1.0))
        kernel = support_embeddings @ support_embeddings.T
        eye = torch.eye(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
        dual_weights = torch.linalg.solve(kernel + ridge_lambda * eye, targets)
        logits = (query_embedding @ support_embeddings.T @ dual_weights) / temperature
        label_tensor = torch.tensor(
            [class_to_index[record["label"]]],
            device=logits.device,
            dtype=torch.long,
        )
        losses.append(F.cross_entropy(logits, label_tensor))

    if not losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def multilabel_support_classification_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
    positive_weight: float = 2.0,
    similarity_threshold: float = 0.45,
    support_pooling: str = "max",
) -> torch.Tensor:
    class_order = sorted({label for record in records for label in record["supports"]})
    if not class_order:
        return torch.zeros((), device=next(model.parameters()).device)
    query_embeddings = encode_train(
        [record["query"] for record in records],
        tokenizer,
        model,
        max_length=max_length,
    )
    logits_per_record = []
    targets = []
    for query_embedding, record in zip(query_embeddings, records, strict=True):
        label_set = {str(label) for label in record["labels"]}
        row_logits = []
        row_targets = []
        for label in class_order:
            texts = record["supports"].get(label)
            if not texts:
                row_logits.append(torch.full((), -100.0, device=query_embedding.device))
            else:
                support_embeddings = encode_train(texts, tokenizer, model, max_length=max_length)
                similarities = query_embedding @ support_embeddings.T
                if support_pooling == "logsumexp":
                    pooled_similarity = temperature * torch.logsumexp(similarities / temperature, dim=0)
                elif support_pooling == "mean_top2":
                    pooled_similarity = similarities.topk(k=min(2, similarities.numel())).values.mean()
                elif support_pooling == "max":
                    pooled_similarity = similarities.max()
                else:
                    raise ValueError(f"Unsupported support_pooling `{support_pooling}`")
                row_logits.append((pooled_similarity - similarity_threshold) / temperature)
            row_targets.append(1.0 if label in label_set else 0.0)
        logits_per_record.append(torch.stack(row_logits))
        targets.append(row_targets)
    logits = torch.stack(logits_per_record)
    target_tensor = torch.tensor(targets, device=query_embeddings.device, dtype=logits.dtype)
    pos_weight = torch.full((len(class_order),), positive_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target_tensor, pos_weight=pos_weight)


def encode_train_chunked(
    texts: list[str],
    tokenizer,
    model,
    *,
    max_length: int,
    chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), chunk_size):
        chunks.append(
            encode_train(
                texts[start : start + chunk_size],
                tokenizer,
                model,
                max_length=max_length,
            )
        )
    return torch.cat(chunks, dim=0)


def cedr_knn_episode_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
) -> torch.Tensor:
    class_order = list(records[0].get("class_order") or ["joy", "sadness", "surprise", "fear", "anger"])
    class_to_index = {label: index for index, label in enumerate(class_order)}
    knn_k = int(records[0].get("knn_k", 5))
    vote_temperature = float(records[0].get("vote_temperature", temperature))
    decision_threshold = float(records[0].get("decision_threshold", 0.5))
    exact_set_weight = float(records[0].get("exact_set_weight", 0.25))
    margin_weight = float(records[0].get("margin_weight", 0.25))
    vote_margin = float(records[0].get("vote_margin", 0.2))
    support_chunk_size = max(1, int(records[0].get("support_chunk_size", 12)))

    query_embeddings = encode_train(
        [record["query"] for record in records],
        tokenizer,
        model,
        max_length=max_length,
    )
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    probability_rows = []
    target_rows = []
    margin_losses = []
    for query_embedding, record in zip(query_embeddings, records, strict=True):
        supports = record["supports"]
        support_embeddings = encode_train_chunked(
            [support["text"] for support in supports],
            tokenizer,
            model,
            max_length=max_length,
            chunk_size=support_chunk_size,
        )
        support_embeddings = F.normalize(support_embeddings, p=2, dim=1)
        support_targets = torch.zeros(
            (len(supports), len(class_order)),
            device=query_embedding.device,
            dtype=query_embedding.dtype,
        )
        for support_index, support in enumerate(supports):
            for label in support["labels"]:
                label_index = class_to_index.get(str(label))
                if label_index is not None:
                    support_targets[support_index, label_index] = 1.0

        similarities = query_embedding.unsqueeze(0) @ support_embeddings.T
        similarities = similarities.squeeze(0)
        top_k = min(knn_k, similarities.numel())
        top_values, top_indices = similarities.topk(k=top_k)
        vote_weights = torch.softmax(top_values / vote_temperature, dim=0)
        probabilities = vote_weights @ support_targets.index_select(0, top_indices)

        target = torch.zeros(len(class_order), device=query_embedding.device, dtype=query_embedding.dtype)
        for label in record["labels"]:
            label_index = class_to_index.get(str(label))
            if label_index is not None:
                target[label_index] = 1.0
        probability_rows.append(probabilities)
        target_rows.append(target)

        vote_centered = probabilities - decision_threshold
        positive_mask = target.bool()
        negative_mask = ~positive_mask
        if positive_mask.any():
            margin_losses.append(F.relu(vote_margin - vote_centered[positive_mask]).mean())
        if negative_mask.any():
            margin_losses.append(F.relu(vote_margin + vote_centered[negative_mask]).mean())

    probabilities = torch.stack(probability_rows).clamp(1e-5, 1.0 - 1e-5)
    targets = torch.stack(target_rows)
    bce = F.binary_cross_entropy(probabilities, targets)
    prediction_logits = (probabilities - decision_threshold) / max(vote_temperature, 1e-6)
    exact_set = F.binary_cross_entropy_with_logits(prediction_logits, targets)
    if margin_losses:
        margin = torch.stack(margin_losses).mean()
    else:
        margin = torch.zeros((), device=query_embeddings.device)
    return bce + exact_set_weight * exact_set + margin_weight * margin


def prototype_none_classification_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
) -> torch.Tensor:
    class_order = sorted({label for record in records for label in record["prototypes"]})
    class_to_index = {label: index for index, label in enumerate(class_order)}
    query_embeddings = encode_train(
        [record["query"] for record in records],
        tokenizer,
        model,
        max_length=max_length,
    )
    logits_per_record = []
    labels = []
    for query_embedding, record in zip(query_embeddings, records, strict=True):
        prototypes = []
        for label in class_order:
            texts = record["prototypes"].get(label)
            if not texts:
                prototypes.append(torch.zeros_like(query_embedding))
                continue
            prototype_embeddings = encode_train(texts, tokenizer, model, max_length=max_length)
            prototypes.append(prototype_embeddings.mean(dim=0))
        prototype_matrix = torch.stack(prototypes)
        logits_per_record.append(query_embedding @ prototype_matrix.T)
        labels.append(str(record["label"]))

    logits = torch.stack(logits_per_record)
    emotion_rows = [index for index, label in enumerate(labels) if label in class_to_index]
    neutral_rows = [index for index, label in enumerate(labels) if label not in class_to_index]
    losses = []
    if emotion_rows:
        row_tensor = torch.tensor(emotion_rows, device=logits.device, dtype=torch.long)
        label_tensor = torch.tensor(
            [class_to_index[labels[index]] for index in emotion_rows],
            device=logits.device,
            dtype=torch.long,
        )
        losses.append(F.cross_entropy(logits.index_select(0, row_tensor) / temperature, label_tensor))
    if neutral_rows:
        row_tensor = torch.tensor(neutral_rows, device=logits.device, dtype=torch.long)
        neutral_logits = logits.index_select(0, row_tensor)
        margins = torch.tensor(
            [
                float(records[index].get("metadata", {}).get("neutral_margin", 0.35))
                for index in neutral_rows
            ],
            device=logits.device,
            dtype=logits.dtype,
        ).unsqueeze(1)
        losses.append(F.softplus((neutral_logits - margins) / temperature).mean())
    if not losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(losses).sum()


def prototype_uniform_classification_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
) -> torch.Tensor:
    class_order = sorted({label for record in records for label in record["prototypes"]})
    class_to_index = {label: index for index, label in enumerate(class_order)}
    query_embeddings = encode_train(
        [record["query"] for record in records],
        tokenizer,
        model,
        max_length=max_length,
    )
    logits_per_record = []
    labels = []
    for query_embedding, record in zip(query_embeddings, records, strict=True):
        prototypes = []
        for label in class_order:
            texts = record["prototypes"].get(label)
            if not texts:
                prototypes.append(torch.zeros_like(query_embedding))
                continue
            prototype_embeddings = encode_train(texts, tokenizer, model, max_length=max_length)
            prototypes.append(prototype_embeddings.mean(dim=0))
        prototype_matrix = torch.stack(prototypes)
        logits_per_record.append(query_embedding @ prototype_matrix.T)
        labels.append(str(record["label"]))

    logits = torch.stack(logits_per_record) / temperature
    emotion_rows = [index for index, label in enumerate(labels) if label in class_to_index]
    neutral_rows = [index for index, label in enumerate(labels) if label not in class_to_index]
    losses = []
    if emotion_rows:
        row_tensor = torch.tensor(emotion_rows, device=logits.device, dtype=torch.long)
        label_tensor = torch.tensor(
            [class_to_index[labels[index]] for index in emotion_rows],
            device=logits.device,
            dtype=torch.long,
        )
        losses.append(F.cross_entropy(logits.index_select(0, row_tensor), label_tensor))
    if neutral_rows:
        row_tensor = torch.tensor(neutral_rows, device=logits.device, dtype=torch.long)
        neutral_logits = logits.index_select(0, row_tensor)
        target = torch.full_like(neutral_logits, 1.0 / neutral_logits.shape[1])
        log_probs = F.log_softmax(neutral_logits, dim=1)
        losses.append(F.kl_div(log_probs, target, reduction="batchmean"))
    if not losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(losses).sum()


def make_dataloader(
    *,
    data_path: Path,
    batch_size: int,
    objectives: list[str] | None = None,
    sources: list[str] | None = None,
    shuffle: bool = True,
) -> DataLoader:
    dataset = ContrastiveJsonlDataset(data_path)
    if objectives or sources:
        records = filter_records(dataset.records, objectives=objectives, sources=sources)
        if not records:
            raise ValueError(
                f"No records left in {data_path} after filtering "
                f"objectives={objectives} sources={sources}"
            )
        dataset.records = records
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_records,
    )


def compute_batch_loss(
    records: list[dict[str, Any]],
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
    pair_score_loss_weight: float,
    pair_score_rank_loss_weight: float = 0.0,
    pair_score_correlation_loss_weight: float = 0.0,
    pair_score_rank_scale: float = 20.0,
    pair_score_rank_min_gap: float = 0.05,
    multi_positive_metadata_key: str | None = None,
) -> torch.Tensor:
    contrastive_records = [record for record in records if record["objective"] == "contrastive"]
    pair_score_records = [record for record in records if record["objective"] == "pair_score"]
    prototype_records = [
        record for record in records if record["objective"] == "prototype_classification"
    ]
    knn_records = [
        record for record in records if record["objective"] == "knn_classification"
    ]
    ridge_records = [
        record for record in records if record["objective"] == "ridge_classification"
    ]
    multilabel_support_records = [
        record for record in records if record["objective"] == "multilabel_support_classification"
    ]
    cedr_knn_episode_records = [
        record for record in records if record["objective"] == "cedr_knn_episode"
    ]
    prototype_none_records = [
        record for record in records if record["objective"] == "prototype_none_classification"
    ]
    prototype_uniform_records = [
        record for record in records if record["objective"] == "prototype_uniform_classification"
    ]
    labeled_records = [record for record in records if record["objective"] == "labeled_text"]
    hierarchical_labeled_records = [
        record for record in records if record["objective"] == "hierarchical_labeled_text"
    ]
    linear_probe_labeled_records = [
        record for record in records if record["objective"] == "linear_probe_labeled_text"
    ]
    losses_for_step = []
    if contrastive_records:
        queries = [record["query"] for record in contrastive_records]
        positive_groups = [
            [record["positive"], *record.get("positives", [])]
            for record in contrastive_records
        ]
        positives = [positive for group in positive_groups for positive in group]
        positive_spans = []
        offset = 0
        for group in positive_groups:
            positive_spans.append((offset, offset + len(group)))
            offset += len(group)
        flat_negatives = [
            negative
            for record in contrastive_records
            for negative in record.get("negatives", [])
        ]
        candidates = positives + flat_negatives
        labels = torch.tensor(
            [start for start, _end in positive_spans],
            device=next(model.parameters()).device,
            dtype=torch.long,
        )
        query_embeddings = encode_train(queries, tokenizer, model, max_length=max_length)
        candidate_embeddings = encode_train(candidates, tokenizer, model, max_length=max_length)
        positive_mask = None
        if any((end - start) > 1 for start, end in positive_spans) or multi_positive_metadata_key:
            positive_mask = torch.zeros(
                (len(queries), len(candidates)),
                device=query_embeddings.device,
                dtype=torch.bool,
            )
            for query_index, (start, end) in enumerate(positive_spans):
                positive_mask[query_index, start:end] = True
        if multi_positive_metadata_key:
            query_values = [
                str(record.get("metadata", {}).get(multi_positive_metadata_key, ""))
                for record in contrastive_records
            ]
            for query_index, query_value in enumerate(query_values):
                if not query_value:
                    continue
                for candidate_index, candidate_value in enumerate(query_values):
                    if candidate_value == query_value:
                        start, end = positive_spans[candidate_index]
                        positive_mask[query_index, start:end] = True
        losses_for_step.append(
            contrastive_loss(
                query_embeddings,
                candidate_embeddings,
                labels,
                temperature=temperature,
                positive_mask=positive_mask,
            )
        )
    if pair_score_records:
        sentence1 = [record["sentence1"] for record in pair_score_records]
        sentence2 = [record["sentence2"] for record in pair_score_records]
        scores = torch.tensor(
            [record["score"] for record in pair_score_records],
            device=next(model.parameters()).device,
            dtype=torch.float32,
        )
        sentence1_embeddings = encode_train(sentence1, tokenizer, model, max_length=max_length)
        sentence2_embeddings = encode_train(sentence2, tokenizer, model, max_length=max_length)
        losses_for_step.append(
            pair_score_loss_weight * pair_score_loss(sentence1_embeddings, sentence2_embeddings, scores)
        )
        if pair_score_rank_loss_weight > 0.0:
            losses_for_step.append(
                pair_score_rank_loss_weight
                * pair_score_rank_loss(
                    sentence1_embeddings,
                    sentence2_embeddings,
                    scores,
                    scale=pair_score_rank_scale,
                    min_score_gap=pair_score_rank_min_gap,
                )
            )
        if pair_score_correlation_loss_weight > 0.0:
            losses_for_step.append(
                pair_score_correlation_loss_weight
                * pair_score_correlation_loss(sentence1_embeddings, sentence2_embeddings, scores)
            )
    if prototype_records:
        losses_for_step.append(
            prototype_classification_loss(
                prototype_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
            )
        )
    if knn_records:
        losses_for_step.append(
            knn_classification_loss(
                knn_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
            )
        )
    if ridge_records:
        losses_for_step.append(
            ridge_classification_loss(
                ridge_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
            )
        )
    if multilabel_support_records:
        losses_for_step.append(
            multilabel_support_classification_loss(
                multilabel_support_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
                positive_weight=float(multilabel_support_records[0].get("positive_weight", 2.0)),
                similarity_threshold=float(multilabel_support_records[0].get("similarity_threshold", 0.45)),
                support_pooling=str(multilabel_support_records[0].get("support_pooling", "max")),
            )
        )
    if cedr_knn_episode_records:
        losses_for_step.append(
            cedr_knn_episode_loss(
                cedr_knn_episode_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
            )
        )
    if prototype_none_records:
        losses_for_step.append(
            prototype_none_classification_loss(
                prototype_none_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
            )
        )
    if prototype_uniform_records:
        losses_for_step.append(
            prototype_uniform_classification_loss(
                prototype_uniform_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
            )
        )
    if labeled_records:
        losses_for_step.append(
            labeled_metric_loss(
                labeled_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
                loss_name=str(labeled_records[0].get("loss", "supcon")),
                margin=float(labeled_records[0].get("margin", 0.25)),
                gamma=float(labeled_records[0].get("gamma", 32.0)),
                alpha=float(labeled_records[0].get("alpha", 2.0)),
                beta=float(labeled_records[0].get("beta", 50.0)),
                base=float(labeled_records[0].get("base", 0.5)),
            )
        )
    if hierarchical_labeled_records:
        label_weights = hierarchical_labeled_records[0].get("label_weights")
        if label_weights is not None and not isinstance(label_weights, dict):
            raise ValueError("hierarchical_labeled_text `label_weights` must be a mapping")
        losses_for_step.append(
            hierarchical_labeled_metric_loss(
                hierarchical_labeled_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
                loss_name=str(hierarchical_labeled_records[0].get("loss", "supcon")),
                label_weights=label_weights,
                margin=float(hierarchical_labeled_records[0].get("margin", 0.25)),
                gamma=float(hierarchical_labeled_records[0].get("gamma", 32.0)),
                alpha=float(hierarchical_labeled_records[0].get("alpha", 2.0)),
                beta=float(hierarchical_labeled_records[0].get("beta", 50.0)),
                base=float(hierarchical_labeled_records[0].get("base", 0.5)),
            )
        )
    if linear_probe_labeled_records:
        losses_for_step.append(
            linear_probe_labeled_text_loss(
                linear_probe_labeled_records,
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
                ridge_lambda=float(linear_probe_labeled_records[0].get("ridge_lambda", 1.0)),
                use_bias=bool(linear_probe_labeled_records[0].get("use_bias", True)),
            )
        )
    if not losses_for_step:
        raise RuntimeError("Batch produced no loss records")
    return torch.stack(losses_for_step).sum()


def validate(
    dataloader: DataLoader,
    *,
    tokenizer,
    model,
    max_length: int,
    temperature: float,
    pair_score_loss_weight: float,
    pair_score_correlation_loss_weight: float = 0.0,
    multi_positive_metadata_key: str | None = None,
) -> float:
    was_training = model.training
    latent_was_training = model.latent_attention_model.training
    backbone_was_training = model.model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_batch_loss(
                batch["records"],
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=temperature,
                pair_score_loss_weight=pair_score_loss_weight,
                pair_score_correlation_loss_weight=pair_score_correlation_loss_weight,
                multi_positive_metadata_key=multi_positive_metadata_key,
            )
            losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    if latent_was_training:
        model.latent_attention_model.train()
    else:
        model.latent_attention_model.eval()
    if backbone_was_training:
        model.model.train()
    else:
        model.model.eval()
    return sum(losses) / len(losses)


def train(config: dict[str, Any]) -> None:
    started_at = time.perf_counter()
    set_seed(int(config.get("seed", 13)))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_giga_embeddings(
        ModelLoadConfig(
            model_name=config.get("model_name", "ai-sage/Giga-Embeddings-instruct"),
            max_length=int(config["max_length"]),
            local_files_only=bool(config.get("local_files_only", False)),
            attn_implementation=config.get("attn_implementation"),
            latent_checkpoint=Path(config["initial_latent_checkpoint"])
            if config.get("initial_latent_checkpoint")
            else None,
        )
    )
    model.train()
    configure_latent_module(model, config)
    freeze_for_exp01b(model, freeze_llm=bool(config.get("freeze_llm", True)))
    if bool(config.get("reinit_latent", False)):
        reinitialize_latent_block(model)
    if bool(config.get("freeze_llm", True)):
        model.model.eval()
        model.latent_attention_model.train()

    retention_config = config.get("retention") or {}
    teacher_latent_state = None
    anchor_latent_state = None
    negative_direction_state = None
    if retention_config:
        if retention_config.get("teacher_latent_checkpoint"):
            teacher_latent_state = load_latent_state_from_checkpoint(
                Path(retention_config["teacher_latent_checkpoint"])
            )
        else:
            teacher_latent_state = latent_state_clone(model)
        if retention_config.get("parameter_anchor_checkpoint"):
            anchor_latent_state = load_latent_state_from_checkpoint(
                Path(retention_config["parameter_anchor_checkpoint"])
            )
        else:
            anchor_latent_state = teacher_latent_state
        if retention_config.get("negative_direction_checkpoint"):
            negative_direction_state = load_latent_state_from_checkpoint(
                Path(retention_config["negative_direction_checkpoint"])
            )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters selected")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    log_every = int(config.get("log_every", 10))
    default_save_every = int(config.get("save_every", config.get("max_steps", 1)))
    max_length = int(config["max_length"])
    temperature = float(config["temperature"])
    losses: list[float] = []
    validation_config = config.get("validation") or {}
    validation_history: list[dict[str, Any]] = []
    best_validation_loss: float | None = None
    best_validation_step: int | None = None
    best_checkpoint_path = output_dir / "best.pt"
    rehearsal_iterators: list[dict[str, Any]] = []
    if retention_config.get("rehearsal_data_paths"):
        for rehearsal_source in retention_config["rehearsal_data_paths"]:
            rehearsal_dataloader = make_dataloader(
                data_path=Path(rehearsal_source["data_path"]),
                batch_size=int(rehearsal_source.get(
                    "batch_size",
                    retention_config.get("rehearsal_batch_size", config["batch_size"]),
                )),
                objectives=rehearsal_source.get("objectives", retention_config.get("objectives")),
                sources=rehearsal_source.get("sources", retention_config.get("sources")),
            )
            rehearsal_iterators.append({
                "iterator": cycle(rehearsal_dataloader),
                "loss_weight": float(rehearsal_source.get("loss_weight", 1.0)),
                "temperature": rehearsal_source.get("temperature"),
            })
    elif retention_config.get("rehearsal_data_path"):
        rehearsal_dataloader = make_dataloader(
            data_path=Path(retention_config["rehearsal_data_path"]),
            batch_size=int(retention_config.get("rehearsal_batch_size", config["batch_size"])),
            objectives=retention_config.get("objectives"),
            sources=retention_config.get("sources"),
        )
        rehearsal_iterators.append({
            "iterator": cycle(rehearsal_dataloader),
            "loss_weight": 1.0,
            "temperature": retention_config.get("rehearsal_temperature"),
        })
    distillation_iterator = None
    if retention_config.get("distillation_data_path"):
        distillation_dataloader = make_dataloader(
            data_path=Path(retention_config["distillation_data_path"]),
            batch_size=int(retention_config.get(
                "distillation_batch_size",
                retention_config.get("rehearsal_batch_size", config["batch_size"]),
            )),
            objectives=retention_config.get("distillation_objectives"),
            sources=retention_config.get("distillation_sources"),
        )
        distillation_iterator = cycle(distillation_dataloader)
    raw_stages = config.get("stages")
    if raw_stages is None:
        raw_stages = [
            {
                "name": config.get("name", "exp01b"),
                "data_path": config["data_path"],
                "max_steps": int(config["max_steps"]),
                "batch_size": int(config["batch_size"]),
                "pair_score_loss_weight": float(config.get("pair_score_loss_weight", 1.0)),
                "pair_score_rank_loss_weight": float(config.get("pair_score_rank_loss_weight", 0.0)),
                "pair_score_correlation_loss_weight": float(
                    config.get("pair_score_correlation_loss_weight", 0.0)
                ),
                "pair_score_rank_scale": float(config.get("pair_score_rank_scale", 20.0)),
                "pair_score_rank_min_gap": float(config.get("pair_score_rank_min_gap", 0.05)),
                "temperature": temperature,
            }
        ]

    completed_steps = 0
    total_steps = sum(int(stage["max_steps"]) for stage in raw_stages)
    for stage_index, stage in enumerate(raw_stages, start=1):
        best_validation_loss = None
        best_validation_step = None
        stage_name = stage.get("name", f"stage{stage_index}")
        stage_steps = int(stage["max_steps"])
        stage_temperature = float(stage.get("temperature", temperature))
        stage_pair_weight = float(stage.get("pair_score_loss_weight", config.get("pair_score_loss_weight", 1.0)))
        stage_pair_rank_weight = float(
            stage.get("pair_score_rank_loss_weight", config.get("pair_score_rank_loss_weight", 0.0))
        )
        stage_pair_correlation_weight = float(
            stage.get(
                "pair_score_correlation_loss_weight",
                config.get("pair_score_correlation_loss_weight", 0.0),
            )
        )
        stage_pair_rank_scale = float(
            stage.get("pair_score_rank_scale", config.get("pair_score_rank_scale", 20.0))
        )
        stage_pair_rank_min_gap = float(
            stage.get("pair_score_rank_min_gap", config.get("pair_score_rank_min_gap", 0.05))
        )
        stage_task_weight = float(stage.get("task_loss_weight", config.get("task_loss_weight", 1.0)))
        stage_multi_positive_key = stage.get(
            "multi_positive_metadata_key",
            config.get("multi_positive_metadata_key"),
        )
        stage_save_every = int(stage.get("save_every", default_save_every))
        if "learning_rate" in stage:
            for group in optimizer.param_groups:
                group["lr"] = float(stage["learning_rate"])
        dataloader = make_dataloader(
            data_path=Path(stage.get("data_path", config.get("data_path"))),
            batch_size=int(stage.get("batch_size", config["batch_size"])),
            objectives=stage.get("objectives"),
            sources=stage.get("sources"),
            shuffle=bool(stage.get("shuffle", True)),
        )
        validation_dataloader = None
        if validation_config.get("data_path"):
            validation_dataloader = make_dataloader(
                data_path=Path(validation_config["data_path"]),
                batch_size=int(stage.get("validation_batch_size", stage.get("batch_size", config["batch_size"]))),
                objectives=stage.get("objectives"),
                sources=stage.get("sources"),
                shuffle=False,
            )
        eval_every = int(stage.get("eval_every", validation_config.get("eval_every", 0)))
        patience = int(stage.get("patience", validation_config.get("patience", 0)))
        min_delta = float(stage.get("min_delta", validation_config.get("min_delta", 0.0)))
        no_improvement_checks = 0
        iterator = cycle(dataloader)
        progress = tqdm(range(1, stage_steps + 1), desc=f"{config.get('name', 'exp01b')}:{stage_name}")
        actual_stage_steps = 0
        for stage_step in progress:
            actual_stage_steps = stage_step
            step_started_at = time.perf_counter()
            global_step = completed_steps + stage_step
            batch = next(iterator)

            optimizer.zero_grad(set_to_none=True)
            distillation_records = None

            loss = torch.zeros((), device=next(model.parameters()).device)
            loss_components: dict[str, float] = {}

            distillation_weight = float(stage.get(
                "distillation_loss_weight",
                retention_config.get("distillation_loss_weight", 0.0),
            ))
            pairwise_distillation_weight = float(stage.get(
                "pairwise_distillation_loss_weight",
                retention_config.get("pairwise_distillation_loss_weight", 0.0),
            ))
            if distillation_weight > 0.0 and pairwise_distillation_weight > 0.0:
                if teacher_latent_state is None:
                    raise RuntimeError("distillation losses require retention configuration")
                if distillation_iterator is not None:
                    distillation_records = next(distillation_iterator)["records"]
                else:
                    distillation_records = batch["records"]
                distillation_losses = combined_distillation_losses(
                    distillation_records,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=int(retention_config.get("distillation_max_length", max_length)),
                    teacher_latent_state=teacher_latent_state,
                    distillation_max_texts=int(retention_config.get("distillation_max_texts", 8)),
                    pairwise_max_texts=int(retention_config.get("pairwise_distillation_max_texts", 12)),
                )
                distill_loss = distillation_losses["distill"]
                pairwise_distill_loss = distillation_losses["pairdist"]
                loss = loss + distillation_weight * distill_loss
                loss = loss + pairwise_distillation_weight * pairwise_distill_loss
                loss_components["distill"] = float(distill_loss.detach().cpu())
                loss_components["pairdist"] = float(pairwise_distill_loss.detach().cpu())
            elif distillation_weight > 0.0:
                if teacher_latent_state is None:
                    raise RuntimeError("distillation_loss_weight requires retention configuration")
                if distillation_iterator is not None:
                    distillation_records = next(distillation_iterator)["records"]
                else:
                    distillation_records = batch["records"]
                distill_loss = distillation_loss(
                    distillation_records,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=int(retention_config.get("distillation_max_length", max_length)),
                    teacher_latent_state=teacher_latent_state,
                    max_texts=int(retention_config.get("distillation_max_texts", 8)),
                )
                loss = loss + distillation_weight * distill_loss
                loss_components["distill"] = float(distill_loss.detach().cpu())

            elif pairwise_distillation_weight > 0.0:
                if teacher_latent_state is None:
                    raise RuntimeError(
                        "pairwise_distillation_loss_weight requires retention configuration"
                    )
                if distillation_iterator is not None:
                    distillation_records = next(distillation_iterator)["records"]
                else:
                    distillation_records = batch["records"]
                pairwise_distill_loss = pairwise_distillation_loss(
                    distillation_records,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=int(retention_config.get("distillation_max_length", max_length)),
                    teacher_latent_state=teacher_latent_state,
                    max_texts=int(retention_config.get("pairwise_distillation_max_texts", 12)),
                )
                loss = loss + pairwise_distillation_weight * pairwise_distill_loss
                loss_components["pairdist"] = float(pairwise_distill_loss.detach().cpu())

            task_loss = compute_batch_loss(
                batch["records"],
                tokenizer=tokenizer,
                model=model,
                max_length=max_length,
                temperature=stage_temperature,
                pair_score_loss_weight=stage_pair_weight,
                pair_score_rank_loss_weight=stage_pair_rank_weight,
                pair_score_correlation_loss_weight=stage_pair_correlation_weight,
                pair_score_rank_scale=stage_pair_rank_scale,
                pair_score_rank_min_gap=stage_pair_rank_min_gap,
                multi_positive_metadata_key=stage_multi_positive_key,
            )
            loss = loss + stage_task_weight * task_loss
            loss_components["task"] = float(task_loss.detach().cpu())

            rehearsal_weight = float(stage.get(
                "rehearsal_loss_weight",
                retention_config.get("rehearsal_loss_weight", 0.0),
            ))
            if rehearsal_weight > 0.0:
                active_rehearsal_iterators = rehearsal_iterators or [
                    {"iterator": None, "loss_weight": 1.0, "temperature": None}
                ]
                rehearsal_losses = []
                for rehearsal_source in active_rehearsal_iterators:
                    rehearsal_records = (
                        next(rehearsal_source["iterator"])["records"]
                        if rehearsal_source["iterator"] is not None
                        else batch["records"]
                    )
                    rehearsal_loss = compute_batch_loss(
                        rehearsal_records,
                        tokenizer=tokenizer,
                        model=model,
                        max_length=max_length,
                        temperature=float(
                            rehearsal_source.get("temperature")
                            or retention_config.get("rehearsal_temperature", stage_temperature)
                        ),
                        pair_score_loss_weight=float(
                            retention_config.get("rehearsal_pair_score_loss_weight", stage_pair_weight)
                        ),
                        pair_score_rank_loss_weight=float(
                            retention_config.get("rehearsal_pair_score_rank_loss_weight", 0.0)
                        ),
                        pair_score_correlation_loss_weight=float(
                            retention_config.get("rehearsal_pair_score_correlation_loss_weight", 0.0)
                        ),
                        pair_score_rank_scale=float(
                            retention_config.get("rehearsal_pair_score_rank_scale", stage_pair_rank_scale)
                        ),
                        pair_score_rank_min_gap=float(
                            retention_config.get("rehearsal_pair_score_rank_min_gap", stage_pair_rank_min_gap)
                        ),
                        multi_positive_metadata_key=retention_config.get(
                            "multi_positive_metadata_key",
                            stage_multi_positive_key,
                        ),
                    )
                    rehearsal_losses.append(float(rehearsal_source["loss_weight"]) * rehearsal_loss)
                rehearsal_loss = torch.stack(rehearsal_losses).sum()
                loss = loss + rehearsal_weight * rehearsal_loss
                loss_components["rehearsal"] = float(rehearsal_loss.detach().cpu())

            anchor_weight = float(stage.get(
                "parameter_anchor_weight",
                retention_config.get("parameter_anchor_weight", 0.0),
            ))
            if anchor_weight > 0.0:
                if anchor_latent_state is None:
                    raise RuntimeError("parameter_anchor_weight requires retention configuration")
                anchor_loss = parameter_anchor_loss(model, anchor_latent_state)
                loss = loss + anchor_weight * anchor_loss
                loss_components["anchor"] = float(anchor_loss.detach().cpu())

            negative_direction_weight = float(stage.get(
                "negative_direction_weight",
                retention_config.get("negative_direction_weight", 0.0),
            ))
            if negative_direction_weight != 0.0:
                if anchor_latent_state is None or negative_direction_state is None:
                    raise RuntimeError(
                        "negative_direction_weight requires teacher/negative direction checkpoints"
                    )
                if "negative_direction_target_projection" in stage:
                    direction_loss = parameter_direction_target_loss(
                        model,
                        anchor_state=anchor_latent_state,
                        direction_state=negative_direction_state,
                        target_projection=float(stage["negative_direction_target_projection"]),
                    )
                elif "negative_direction_target_projection" in retention_config:
                    direction_loss = parameter_direction_target_loss(
                        model,
                        anchor_state=anchor_latent_state,
                        direction_state=negative_direction_state,
                        target_projection=float(retention_config["negative_direction_target_projection"]),
                    )
                else:
                    direction_loss = parameter_direction_loss(
                        model,
                        anchor_state=anchor_latent_state,
                        direction_state=negative_direction_state,
                    )
                loss = loss + negative_direction_weight * direction_loss
                loss_components["neg_dir"] = float(direction_loss.detach().cpu())

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            elapsed_seconds = time.perf_counter() - started_at
            step_seconds = time.perf_counter() - step_started_at
            steps_per_second = global_step / elapsed_seconds if elapsed_seconds > 0 else 0.0
            remaining_seconds = (total_steps - global_step) / steps_per_second if steps_per_second > 0 else 0.0
            if stage_step % log_every == 0 or stage_step == 1:
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    task=f"{loss_components['task']:.4f}",
                    distill=f"{loss_components.get('distill', 0.0):.4f}",
                    anchor=f"{loss_components.get('anchor', 0.0):.6f}",
                    neg_dir=f"{loss_components.get('neg_dir', 0.0):.4f}",
                    step_s=f"{step_seconds:.2f}",
                    elapsed_m=f"{elapsed_seconds / 60:.1f}",
                    eta_m=f"{remaining_seconds / 60:.1f}",
                    global_step=global_step,
                )
            if global_step % stage_save_every == 0 or global_step == total_steps:
                save_checkpoint(
                    model,
                    output_dir / f"step-{global_step}.pt",
                    config,
                    global_step,
                    losses,
                    elapsed_seconds,
                    validation_history=validation_history,
                    best_validation_loss=best_validation_loss,
                )
                save_checkpoint(
                    model,
                    output_dir / "latest.pt",
                    config,
                    global_step,
                    losses,
                    elapsed_seconds,
                    validation_history=validation_history,
                    best_validation_loss=best_validation_loss,
                )
            if validation_dataloader is not None and eval_every > 0 and (
                stage_step % eval_every == 0 or stage_step == stage_steps
            ):
                validation_loss = validate(
                    validation_dataloader,
                    tokenizer=tokenizer,
                    model=model,
                    max_length=max_length,
                    temperature=stage_temperature,
                    pair_score_loss_weight=stage_pair_weight,
                    pair_score_correlation_loss_weight=stage_pair_correlation_weight,
                    multi_positive_metadata_key=stage_multi_positive_key,
                )
                validation_entry = {
                    "step": global_step,
                    "stage": stage_name,
                    "stage_step": stage_step,
                    "loss": validation_loss,
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
                validation_history.append(validation_entry)
                improved = best_validation_loss is None or validation_loss < best_validation_loss - min_delta
                if improved:
                    best_validation_loss = validation_loss
                    best_validation_step = global_step
                    no_improvement_checks = 0
                    if bool(validation_config.get("save_best", True)):
                        save_checkpoint(
                            model,
                            best_checkpoint_path,
                            config,
                            global_step,
                            losses,
                            validation_entry["elapsed_seconds"],
                            validation_history=validation_history,
                            best_validation_loss=best_validation_loss,
                            best_validation_step=best_validation_step,
                        )
                else:
                    no_improvement_checks += 1
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    val_loss=f"{validation_loss:.4f}",
                    global_step=global_step,
                    best_val=f"{best_validation_loss:.4f}" if best_validation_loss is not None else "n/a",
                )
                if patience > 0 and no_improvement_checks >= patience:
                    print(
                        f"Early stopping stage {stage_name} at step {global_step}: "
                        f"no validation improvement for {patience} checks."
                    )
                    break
        completed_steps += actual_stage_steps
    if bool(validation_config.get("restore_best", True)) and best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=next(model.parameters()).device)
        model.latent_attention_model.load_state_dict(checkpoint["latent_attention_model"])
        elapsed_seconds = time.perf_counter() - started_at
        restored_step = int(checkpoint.get("step", completed_steps))
        save_checkpoint(
            model,
            output_dir / "latest.pt",
            config,
            restored_step,
            losses,
            elapsed_seconds,
            validation_history=validation_history,
            best_validation_loss=best_validation_loss,
            best_validation_step=best_validation_step,
        )


def save_checkpoint(
    model,
    path: Path,
    config: dict[str, Any],
    step: int,
    losses: list[float],
    elapsed_seconds: float,
    *,
    validation_history: list[dict[str, Any]] | None = None,
    best_validation_loss: float | None = None,
    best_validation_step: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "config": config,
            "losses": losses,
            "elapsed_seconds": elapsed_seconds,
            "validation_history": validation_history or [],
            "best_validation_loss": best_validation_loss,
            "best_validation_step": best_validation_step,
            "checkpoint_architecture": config.get("latent_architecture", "original_latent_attention"),
            "latent_module_config": (
                model.latent_attention_model.export_config()
                if hasattr(model.latent_attention_model, "export_config")
                else {"architecture": "original_latent_attention"}
            ),
            "latent_attention_model": model.latent_attention_model.state_dict(),
        },
        path,
    )
    write_json(
        path.with_suffix(".json"),
        {
            "step": step,
            "config": config,
            "last_loss": losses[-1] if losses else None,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60,
            "steps_per_second": step / elapsed_seconds if elapsed_seconds > 0 else None,
            "validation_history": validation_history or [],
            "best_validation_loss": best_validation_loss,
            "best_validation_step": best_validation_step,
            "checkpoint": str(path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Experiment 1b original latent-memory block.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        config = json.load(file)
    train(config)


if __name__ == "__main__":
    main()
