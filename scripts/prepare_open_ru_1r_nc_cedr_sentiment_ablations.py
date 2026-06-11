from __future__ import annotations

import csv
import io
import json
import math
import random
import re
import urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
CONFIG_DIR = ROOT / "configs" / "experiments"
CACHE_DIR = ROOT / "results" / "mteb_cache"
CONTAM_DIR = ROOT / "results" / "contamination" / "cedr_candidates"
RAW_DIR = ROOT / "data" / "raw" / "rusentiment"

SENT_PREFIX = (
    "Определи эмоциональную окраску и смысл комментария: положительный, отрицательный или нейтральный\n"
    "комментарий: "
)
LABELS = ["neutral", "positive", "negative"]
RUSENTIMENT_URLS = {
    "preselected": "https://raw.githubusercontent.com/strawberrypie/rusentiment/master/Dataset/rusentiment_preselected_posts.csv",
    "random": "https://raw.githubusercontent.com/strawberrypie/rusentiment/master/Dataset/rusentiment_random_posts.csv",
    "test": "https://raw.githubusercontent.com/strawberrypie/rusentiment/master/Dataset/rusentiment_test.csv",
}


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.U)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value: str) -> set[str]:
    return set(re.findall(r"[\w]+", normalize_text(value), flags=re.U))


def jaccard(left: str, right: str) -> float:
    a = token_set(left)
    b = token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cedr_index() -> dict[str, Any]:
    dataset = load_dataset("mteb/CEDRClassification", cache_dir=str(CACHE_DIR), trust_remote_code=True)
    texts = []
    for split in dataset.values():
        for row in split:
            text = normalize_text(row["text"])
            if text:
                texts.append(text)
    inverted: dict[str, list[int]] = defaultdict(list)
    tokenized = []
    for index, text in enumerate(texts):
        tokens = token_set(text)
        tokenized.append(tokens)
        for token in tokens:
            if len(token) >= 4:
                inverted[token].append(index)
    return {"texts": texts, "exact": set(texts), "tokenized": tokenized, "inverted": inverted}


def is_contaminated(text: str, cedr_index: dict[str, Any]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    if normalized in cedr_index["exact"]:
        return True
    tokens = token_set(normalized)
    if len(tokens) < 4:
        return False
    candidate_counts: dict[int, int] = defaultdict(int)
    for token in tokens:
        if len(token) >= 4:
            for index in cedr_index["inverted"].get(token, []):
                candidate_counts[index] += 1
    for index, _count in sorted(candidate_counts.items(), key=lambda item: -item[1])[:250]:
        cedr_tokens = cedr_index["tokenized"][index]
        if len(cedr_tokens) < 4:
            continue
        overlap = len(tokens & cedr_tokens) / len(tokens | cedr_tokens)
        if overlap >= 0.78:
            cedr_text = cedr_index["texts"][index]
            ratio = SequenceMatcher(None, normalized, cedr_text).ratio()
            if ratio >= 0.9:
                return True
    return False


def load_rusentitweet(cedr_index: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset("psytechlab/RuSentiTweet", cache_dir=str(CACHE_DIR))
    rows = []
    skipped = Counter()
    seen = set()
    for split_name, split in dataset.items():
        for index, row in enumerate(split):
            label = str(row["label"])
            text = str(row["text"]).strip()
            normalized = normalize_text(text)
            if label not in LABELS:
                skipped[f"label_{label}"] += 1
                continue
            if len(normalized) < 12 or len(normalized) > 360:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            rows.append(
                {
                    "source": "psytechlab/RuSentiTweet",
                    "split": split_name,
                    "index": index,
                    "label": label,
                    "text": text,
                    "normalized": normalized,
                }
            )
    return rows, {"skipped": dict(skipped), "kept": len(rows), "label_counts": dict(Counter(row["label"] for row in rows))}


def download_rusentiment_file(name: str, url: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"rusentiment_{name}.csv"
    if not path.exists():
        data = urllib.request.urlopen(url, timeout=60).read()
        path.write_bytes(data)
    return path


def load_original_rusentiment(cedr_index: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    skipped = Counter()
    seen = set()
    for split_name, url in RUSENTIMENT_URLS.items():
        path = download_rusentiment_file(split_name, url)
        text = path.read_text(encoding="utf-8-sig")
        for index, row in enumerate(csv.DictReader(io.StringIO(text))):
            label = str(row.get("label", "")).strip()
            value = str(row.get("text", "")).strip()
            normalized = normalize_text(value)
            if label not in LABELS:
                skipped[f"label_{label}"] += 1
                continue
            if len(normalized) < 12 or len(normalized) > 360:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(value, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            rows.append(
                {
                    "source": "text-machine-lab/rusentiment",
                    "split": split_name,
                    "index": index,
                    "label": label,
                    "text": value,
                    "normalized": normalized,
                }
            )
    return rows, {"skipped": dict(skipped), "kept": len(rows), "label_counts": dict(Counter(row["label"] for row in rows))}


def select_by_prior(rows: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    # Match CEDR's rough neutral-vs-emotional shape without pretending sentiment labels are CEDR emotions.
    targets = {"neutral": count // 2, "positive": count // 4, "negative": count - count // 2 - count // 4}
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    selected = []
    for label, target in targets.items():
        pool = by_label[label][:]
        rng.shuffle(pool)
        if len(pool) < target:
            raise ValueError(f"Need {target} rows for {label}, got {len(pool)}")
        selected.extend(pool[:target])
    rng.shuffle(selected)
    return selected


def nearest_by_label(
    item: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    same_label: bool,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["normalized"] != item["normalized"] and (row["label"] == item["label"]) == same_label]
    sample = rng.sample(candidates, k=min(limit, len(candidates)))
    return sorted(sample, key=lambda row: jaccard(item["normalized"], row["normalized"]), reverse=True)


def far_same_label(
    item: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["normalized"] != item["normalized"] and row["label"] == item["label"]]
    sample = rng.sample(candidates, k=min(limit, len(candidates)))
    return [row for row in sample if jaccard(item["normalized"], row["normalized"]) <= 0.02]


def build_habr_style_records(
    rows: list[dict[str, Any]],
    *,
    name: str,
    count: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count is None:
        selected = rows[:]
        random.Random(seed).shuffle(selected)
    else:
        selected = select_by_prior(rows, count=count, seed=seed)
    rng = random.Random(seed + 17)
    records = []
    pos_scores = []
    cross_scores = []
    same_far_count = 0
    for item in selected:
        same_near = nearest_by_label(item, rows, same_label=True, limit=96, rng=rng)
        cross_near = nearest_by_label(item, rows, same_label=False, limit=192, rng=rng)
        same_far = far_same_label(item, rows, limit=128, rng=rng)
        if not same_near or not cross_near:
            continue
        positive = same_near[0]
        negatives = []
        for row in cross_near[:4]:
            negatives.append(SENT_PREFIX + row["text"])
            cross_scores.append(jaccard(item["normalized"], row["normalized"]))
        for row in same_far[:2]:
            negatives.append(SENT_PREFIX + row["text"])
            same_far_count += 1
        pos_scores.append(jaccard(item["normalized"], positive["normalized"]))
        records.append(
            {
                "source": f"{name}:sentiment_local_hardneg",
                "objective": "contrastive",
                "query": SENT_PREFIX + item["text"],
                "positive": SENT_PREFIX + positive["text"],
                "negatives": negatives[:6],
                "metadata": {
                    "label": item["label"],
                    "source_dataset": item["source"],
                    "split": item["split"],
                    "index": item["index"],
                    "positive_jaccard": round(pos_scores[-1], 6),
                },
            }
        )
    rng.shuffle(records)
    summary = {
        "source": name,
        "construction": "local sentiment discrimination: same-label lexical-near positive, cross-label lexical-near hard negatives, same-label topic-far negatives",
        "requested": count if count is not None else "all_filtered",
        "kept": len(records),
        "selected_by_label": dict(Counter(row["metadata"]["label"] for row in records)),
        "positive_jaccard_mean": sum(pos_scores) / len(pos_scores) if pos_scores else 0.0,
        "cross_negative_jaccard_mean": sum(cross_scores) / len(cross_scores) if cross_scores else 0.0,
        "same_label_far_negative_count": same_far_count,
        "prefix": SENT_PREFIX,
    }
    return records, summary


def sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Need {count} records, got {len(records)}")
    selected = records[:]
    random.Random(seed).shuffle(selected)
    return selected[:count]


def make_mix(name: str, sentiment_records: list[dict[str, Any]], summary: dict[str, Any], *, seed: int) -> None:
    geracl = read_jsonl(DATA_DIR / "open_ru_1r_nc_geracl.jsonl")
    habr = read_jsonl(DATA_DIR / "open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl")
    deepvk = read_jsonl(DATA_DIR / "open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl")
    grandmaster = read_jsonl(DATA_DIR / "open_ru_1r_nc_grandmaster_clustered_3200.jsonl")
    selected = {
        "geracl": sample(geracl, count=6400, seed=seed),
        "habr_harder_full": habr,
        "deepvk_filtered": sample(deepvk, count=3200, seed=seed + 2),
        "grandmaster": grandmaster,
        "sentiment_local": sentiment_records,
    }
    mixed = []
    for rows in selected.values():
        mixed.extend(rows)
    random.Random(seed + 31).shuffle(mixed)

    data_path = DATA_DIR / f"open_ru_1r_nc_{name}.jsonl"
    write_jsonl(data_path, mixed)
    max_steps = math.ceil(len(mixed) / 3)
    write_json(
        data_path.with_name(data_path.stem + "_summary.json"),
        {
            "output": str(data_path.relative_to(ROOT)),
            "seed": seed,
            "base_run": "Mix H HabrFull stage 1",
            "stage2": "none",
            "component_counts": {key: len(value) for key, value in selected.items()},
            "total_records": len(mixed),
            "batch_size": 3,
            "max_steps_1x_batch3": max_steps,
            "sentiment_component": summary,
        },
    )
    write_json(
        CONFIG_DIR / f"exp01r_nc_{name}_4096_eager_frozenrepro.json",
        {
            "name": f"exp01r_nc_{name}_4096_eager_frozenrepro",
            "description": f"Stage-1-only sentiment ablation for CEDR: {name}",
            "model_name": "ai-sage/Giga-Embeddings-instruct",
            "local_files_only": True,
            "attn_implementation": "eager",
            "latent_architecture": "original_latent_attention",
            "initial_latent_checkpoint": None,
            "freeze_llm": True,
            "reinit_latent": True,
            "data_path": str(data_path.relative_to(ROOT)),
            "output_dir": f"experiments/exp01_reinit_fair/checkpoints/{name}_4096_eager_frozenrepro",
            "max_length": 4096,
            "batch_size": 3,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "temperature": 0.02,
            "max_steps": max_steps,
            "log_every": 50,
            "save_every": max_steps,
            "seed": seed,
        },
    )


def main() -> None:
    seed = 301
    cedr_index = load_cedr_index()
    rusentitweet, rusentitweet_loader = load_rusentitweet(cedr_index)
    rusentiment, rusentiment_loader = load_original_rusentiment(cedr_index)

    write_json(
        CONTAM_DIR / "sentiment_local_sources_summary.json",
        {
            "cedr_rows": len(cedr_index["texts"]),
            "rusentitweet": rusentitweet_loader,
            "original_rusentiment": rusentiment_loader,
        },
    )

    variants = [
        ("cedr_rusentitweet_local_sentiment", rusentitweet, {"loader": rusentitweet_loader}, seed + 1, 3200),
        (
            "cedr_rusentitweet_full_local_sentiment",
            rusentitweet,
            {"loader": rusentitweet_loader},
            seed + 5,
            None,
        ),
        ("cedr_rusentiment_local_sentiment", rusentiment, {"loader": rusentiment_loader}, seed + 2, 3200),
        (
            "cedr_rusentitweet_rusentiment_local_sentiment",
            rusentitweet + rusentiment,
            {"loader": {"rusentitweet": rusentitweet_loader, "original_rusentiment": rusentiment_loader}},
            seed + 3,
            3200,
        ),
    ]
    for name, rows, extra_summary, variant_seed, count in variants:
        records, summary = build_habr_style_records(rows, name=name, count=count, seed=variant_seed)
        summary.update(extra_summary)
        suffix = "full" if count is None else str(count)
        write_jsonl(DATA_DIR / f"open_ru_1r_nc_{name}_component_{suffix}.jsonl", records)
        write_json(DATA_DIR / f"open_ru_1r_nc_{name}_component_{suffix}_summary.json", summary)
        make_mix(name, records, summary, seed=variant_seed)
        print(f"prepared {name}: {len(records)} sentiment rows")


if __name__ == "__main__":
    main()
