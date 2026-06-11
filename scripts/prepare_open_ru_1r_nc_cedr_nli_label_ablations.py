from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
CONFIG_DIR = ROOT / "configs" / "experiments"
REPORT_PATH = ROOT / "results" / "official_repro" / "cedr_stage1_ablation_summary.md"

CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
LABEL_RU = {
    "joy": "радость",
    "sadness": "грусть",
    "anger": "злость",
    "fear": "страх",
    "surprise": "удивление",
    "no_emotion": "без явной эмоции",
}
LABEL_DETAIL = {
    "joy": "в тексте выражена радость, одобрение или позитивная эмоция",
    "sadness": "в тексте выражена грусть, печаль, тоска или усталость",
    "anger": "в тексте выражена злость, раздражение, гнев или агрессия",
    "fear": "в тексте выражен страх, тревога, опасение или ужас",
    "surprise": "в тексте выражено удивление, шок или неожиданность",
    "no_emotion": "в тексте нет явной эмоции из списка",
}


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


def strip_prefix(text: str) -> str:
    if text.startswith(CEDR_PREFIX):
        text = text[len(CEDR_PREFIX) :]
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Need {count} records, got {len(records)}")
    rows = records[:]
    random.Random(seed).shuffle(rows)
    return rows[:count]


def label_statement(text: str, label: str, *, style: str) -> str:
    if style == "short":
        return f"{text}\nЭмоция: {LABEL_RU[label]}."
    if style == "definition":
        return f"{text}\nВерная разметка: {LABEL_DETAIL[label]}."
    if style == "multi_label":
        positives = LABEL_RU[label] if label != "no_emotion" else "нет явной эмоции"
        return f"{text}\nПодходящие метки CEDR: {positives}."
    raise ValueError(f"unsupported style: {style}")


def convert_component_records(
    component_records: list[dict[str, Any]],
    *,
    name: str,
    style: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    skipped = Counter()
    for record in component_records:
        group = record.get("metadata", {}).get("group")
        if group not in LABEL_RU:
            skipped["unknown_group"] += 1
            continue
        text = strip_prefix(record["query"])
        if len(text) < 10:
            skipped["short_text"] += 1
            continue
        false_labels = [label for label in LABEL_RU if label != group]
        rng.shuffle(false_labels)
        negatives = [CEDR_PREFIX + label_statement(text, label, style=style) for label in false_labels]
        records.append(
            {
                "source": f"cedr_label_nli:{name}",
                "objective": "contrastive",
                "query": CEDR_PREFIX + text,
                "positive": CEDR_PREFIX + label_statement(text, str(group), style=style),
                "negatives": negatives,
                "metadata": {
                    "group": group,
                    "source_dataset": record.get("metadata", {}).get("source_dataset"),
                    "source_split": record.get("metadata", {}).get("split"),
                    "source_index": record.get("metadata", {}).get("index"),
                    "style": record.get("metadata", {}).get("style"),
                    "construction": "query_specific_label_statement",
                    "label_statement_style": style,
                },
            }
        )
    rng.shuffle(records)
    return records, {
        "records": len(records),
        "skipped": dict(skipped),
        "groups": dict(Counter(row["metadata"]["group"] for row in records)),
        "source_datasets": dict(Counter(row["metadata"]["source_dataset"] for row in records)),
        "styles": dict(Counter(row["metadata"]["style"] for row in records)),
        "label_statement_style": style,
    }


def make_mix(name: str, addon_records: list[dict[str, Any]], summary: dict[str, Any], *, seed: int) -> None:
    geracl = read_jsonl(DATA_DIR / "open_ru_1r_nc_geracl.jsonl")
    habr = read_jsonl(DATA_DIR / "open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl")
    deepvk = read_jsonl(DATA_DIR / "open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl")
    grandmaster = read_jsonl(DATA_DIR / "open_ru_1r_nc_grandmaster_clustered_3200.jsonl")
    selected = {
        "geracl": sample(geracl, count=6400, seed=seed),
        "habr_harder_full": habr,
        "deepvk_filtered": sample(deepvk, count=3200, seed=seed + 2),
        "grandmaster": grandmaster,
        "cedr_label_nli": addon_records,
    }
    mixed = []
    for rows in selected.values():
        mixed.extend(rows)
    random.Random(seed + 31).shuffle(mixed)

    data_path = DATA_DIR / f"open_ru_1r_nc_{name}.jsonl"
    write_jsonl(data_path, mixed)
    max_steps = math.ceil(len(mixed) / 2)
    write_json(
        data_path.with_name(data_path.stem + "_summary.json"),
        {
            "output": str(data_path.relative_to(ROOT)),
            "seed": seed,
            "base_run": "Mix H HabrFull batch-2 control recipe",
            "stage2": "none",
            "component_counts": {key: len(value) for key, value in selected.items()},
            "total_records": len(mixed),
            "batch_size": 2,
            "max_steps_1x_batch2": max_steps,
            "addon_component": summary,
            "contamination_policy": "uses only previously CEDR-audited mined component rows; CEDR benchmark labels are not used",
        },
    )
    write_json(
        CONFIG_DIR / f"exp01r_nc_{name}_4096_eager_frozenrepro.json",
        {
            "name": f"exp01r_nc_{name}_4096_eager_frozenrepro",
            "description": f"CEDR query-specific label/NLI ablation: {name}",
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
            "batch_size": 2,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "temperature": 0.02,
            "max_steps": max_steps,
            "log_every": 50,
            "save_every": max_steps,
            "seed": seed,
        },
    )


def prepare_variant(*, base_name: str, out_name: str, style: str, seed: int) -> dict[str, Any]:
    component_path = DATA_DIR / f"open_ru_1r_nc_{base_name}_component.jsonl"
    component_records = read_jsonl(component_path)
    records, summary = convert_component_records(component_records, name=out_name, style=style, seed=seed)
    out_component = DATA_DIR / f"open_ru_1r_nc_{out_name}_component.jsonl"
    write_jsonl(out_component, records)
    summary.update(
        {
            "name": out_name,
            "base_component": str(component_path.relative_to(ROOT)),
            "component_path": str(out_component.relative_to(ROOT)),
            "construction": "query-specific positive label statement with same-text false-label negatives",
            "motivation": "avoid same-label comment-to-comment over-clustering and train label boundary geometry closer to CEDR logistic classification",
        }
    )
    write_json(out_component.with_name(out_component.stem + "_summary.json"), summary)
    make_mix(out_name, records, summary, seed=seed)
    return summary


def append_report(summaries: list[dict[str, Any]]) -> None:
    lines = [
        "",
        "## CEDR label/NLI ablations prepared",
        "",
        "These variants convert already CEDR-audited mined sentiment rows into query-specific label-statement pairs. The query and positive share the same comment text; the positive adds the predicted emotion label, while negatives add false labels for the same text. This follows the SimCSE/NLI-style idea of explicit entailment/contradiction pairs and avoids the false-positive pressure from clustering unrelated same-label comments.",
        "",
        "| run | addon rows | total rows | batch | steps | label style | groups |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for summary in summaries:
        total_rows = 17169 + int(summary["records"])
        steps = math.ceil(total_rows / 2)
        groups = ", ".join(f"{key} {value}" for key, value in sorted(summary["groups"].items()))
        lines.append(
            f"| `{summary['name']}` | {summary['records']} | {total_rows} | 2 | {steps} | "
            f"{summary['label_statement_style']} | {groups} |"
        )
    lines.extend(
        [
            "",
            "First gate remains `CEDRClassification` only. Promote a variant only if it beats the batch-2 no-addon control (`0.643996`) by a meaningful margin; the hard target is `0.68`.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    summaries = [
        prepare_variant(
            base_name="cedr_sentiment_mined_v2_informal_1600",
            out_name="cedr_label_nli_v1_informal_1600",
            style="definition",
            seed=521,
        ),
        prepare_variant(
            base_name="cedr_sentiment_mined_v2_mixed_1600",
            out_name="cedr_label_nli_v1_mixed_1600",
            style="definition",
            seed=531,
        ),
        prepare_variant(
            base_name="cedr_sentiment_mined_v2_polarity_1200",
            out_name="cedr_label_nli_v1_polarity_1200",
            style="short",
            seed=541,
        ),
    ]
    append_report(summaries)
    for summary in summaries:
        print(f"prepared {summary['name']}: {summary['records']} addon rows")


if __name__ == "__main__":
    main()
