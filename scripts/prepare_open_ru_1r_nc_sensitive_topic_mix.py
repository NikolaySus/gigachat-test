from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


SENSITIVE_TOPICS = [
    "offline_crime",
    "online_crime",
    "drugs",
    "gambling",
    "pornography",
    "prostitution",
    "slavery",
    "suicide",
    "terrorism",
    "weapons",
    "body_shaming",
    "health_shaming",
    "politics",
    "racism",
    "religion",
    "sexual_minorities",
    "sexism",
    "social_injustice",
]

SENSITIVE_SOURCE_FILTERS = {
    "all": None,
    "mvrcii": {"mvrcii/safety-moderation-benchmark:topic"},
    "mvrcii_uc_berkeley": {
        "mvrcii/safety-moderation-benchmark:topic",
        "ucberkeley-dlab/measuring-hate-speech:target_topic",
    },
    "veles": {"Vikhrmodels/Veles-2.5:lexicon_sensitive_topic"},
    "mnwa": {"Mnwa/russian-toxic:toxicity_style"},
    "textdetox": {"textdetox/multilingual_toxicity_dataset:ru"},
    "uc_berkeley": {"ucberkeley-dlab/measuring-hate-speech:target_topic"},
}

HARD_NEGATIVES = {
    "offline_crime": ["online_crime", "terrorism", "weapons", "social_injustice"],
    "online_crime": ["offline_crime", "gambling", "politics"],
    "drugs": ["suicide", "health_shaming", "offline_crime"],
    "gambling": ["online_crime", "drugs", "social_injustice"],
    "pornography": ["prostitution", "sexual_minorities", "sexism"],
    "prostitution": ["pornography", "slavery", "offline_crime"],
    "slavery": ["social_injustice", "offline_crime", "prostitution"],
    "suicide": ["health_shaming", "drugs", "offline_crime"],
    "terrorism": ["weapons", "politics", "religion"],
    "weapons": ["terrorism", "offline_crime", "politics"],
    "body_shaming": ["health_shaming", "sexism"],
    "health_shaming": ["body_shaming", "suicide", "drugs"],
    "politics": ["religion", "terrorism", "social_injustice"],
    "racism": ["religion", "sexism", "sexual_minorities", "social_injustice"],
    "religion": ["politics", "terrorism", "racism"],
    "sexual_minorities": ["sexism", "racism", "pornography"],
    "sexism": ["sexual_minorities", "body_shaming", "social_injustice"],
    "social_injustice": ["politics", "racism", "slavery", "sexism"],
}

LEXICON = {
    "offline_crime": [
        "убийств",
        "нападени",
        "похищен",
        "тюрьм",
        "заключенн",
        "преступлен",
        "полици",
        "суд",
        "насили",
        "изнасил",
    ],
    "online_crime": [
        "взлом",
        "хакер",
        "парол",
        "фишинг",
        "мошеннич",
        "скам",
        "персональн",
        "данн",
        "пиратск",
        "вирус",
        "докс",
    ],
    "drugs": ["наркот", "алкогол", "табак", "сигарет", "курени", "кокаин", "героин", "марихуан", "спайс"],
    "gambling": ["казино", "ставк", "букмекер", "азарт", "лотере", "рулетк", "покер", "выигрыш"],
    "pornography": ["порно", "секс", "эротик", "интим", "обнажен", "bdsm", "бдсм"],
    "prostitution": ["проститу", "эскорт", "бордел", "секс услуг", "секс-услуг"],
    "slavery": ["рабств", "торговл людьми", "траффик", "эксплуатац"],
    "suicide": ["суицид", "самоуб", "убить себя", "покончить с собой", "смерт"],
    "terrorism": ["террор", "экстрем", "радикал", "взрыв", "бомб", "игил"],
    "weapons": ["оруж", "пистолет", "автомат", "винтовк", "патрон", "нож", "танк", "ракет"],
    "body_shaming": ["толст", "жирн", "урод", "внешност", "сиськ", "лыс", "некрасив"],
    "health_shaming": ["инвалид", "болезн", "псих", "аутиз", "депресс", "диагноз", "здоров"],
    "politics": ["путин", "навальн", "выбор", "митинг", "власть", "либерал", "коммунист", "войн", "армия"],
    "racism": ["русск", "кавказ", "евре", "негр", "хач", "наци", "этнич", "раса", "мигрант"],
    "religion": ["бог", "христ", "ислам", "мусульман", "церков", "религи", "православ", "молитв"],
    "sexual_minorities": ["гей", "лесби", "лгбт", "гомосек", "трансгендер", "квир"],
    "sexism": ["баб", "мужик", "женщин", "фемини", "патриарх", "сексист", "шлюх"],
    "social_injustice": ["бедност", "неравен", "несправедлив", "пенси", "зарплат", "олигарх", "коррупц"],
}

MVR_TOPIC_MAP = {
    "cbrn_threats": ["weapons", "terrorism"],
    "csam_sex_crimes": ["pornography", "offline_crime"],
    "defamation_libel_slander": ["social_injustice", "politics"],
    "espionage_hacking_doxing": ["online_crime"],
    "fraud_scam_phishing": ["online_crime", "gambling"],
    "illegal_violent_crimes": ["offline_crime", "weapons"],
    "privacy_pii_violations": ["online_crime"],
    "self_harm_suicide": ["suicide", "health_shaming"],
}

UC_BERKELEY_MAP = {
    "target_race": "racism",
    "target_religion": "religion",
    "target_gender": "sexism",
    "target_sexuality": "sexual_minorities",
    "target_disability": "health_shaming",
    "target_politics": "politics",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contamination_keys() -> set[str]:
    dataset = load_dataset(
        "ai-forever/sensitive-topics-classification",
        revision="416b34a802308eac30e4192afc0ff99bb8dcc7f2",
    )
    keys = set()
    for split in ("train", "test"):
        keys.update(normalize_text(text) for text in dataset[split]["text"])
    return keys


def extract_conversation_text(conversation: Any) -> str:
    if not isinstance(conversation, list):
        return ""
    parts = []
    for message in conversation:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or message.get("value") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def extract_mvr_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return normalize_space(str(messages or ""))
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}" if role else content)
    return normalize_space("\n".join(parts))


def lexicon_labels(text: str) -> list[str]:
    normalized = normalize_text(text)
    labels = []
    for topic, needles in LEXICON.items():
        if any(needle in normalized for needle in needles):
            labels.append(topic)
    return labels


def add_grouped(groups: dict[str, list[dict[str, Any]]], *, text: str, labels: list[str], source: str) -> None:
    text = normalize_space(text)
    if not (30 <= len(text) <= 3500):
        return
    unique_labels = [label for label in dict.fromkeys(labels) if label in SENSITIVE_TOPICS]
    if not unique_labels:
        return
    item = {"text": text, "labels": unique_labels, "source": source}
    for label in unique_labels:
        groups[label].append(item)


def mvr_groups(limit: int, contaminated: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    dataset = load_dataset("mvrcii/safety-moderation-benchmark", split="train")
    topic_names = dataset.features["topic"].names
    label_names = dataset.features["label"].names
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = kept = contaminated_count = 0
    for row in dataset:
        scanned += 1
        if scanned > limit:
            break
        if label_names[int(row["label"])] == "safe":
            continue
        source_topic = topic_names[int(row["topic"])]
        labels = MVR_TOPIC_MAP.get(source_topic, [])
        if not labels:
            continue
        text = extract_mvr_text(row.get("text"))
        if normalize_text(text) in contaminated:
            contaminated_count += 1
            continue
        add_grouped(groups, text=text, labels=labels, source="mvrcii/safety-moderation-benchmark:topic")
        kept += 1
    return groups, {"scanned": scanned, "kept": kept, "contaminated": contaminated_count}


def veles_groups(limit: int, contaminated: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    dataset = load_dataset("Vikhrmodels/Veles-2.5", split="train", streaming=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = kept = contaminated_count = 0
    for row in dataset:
        scanned += 1
        if scanned > limit:
            break
        text = extract_conversation_text(row.get("conversations"))
        labels = lexicon_labels(text)
        if normalize_text(text) in contaminated:
            contaminated_count += 1
            continue
        before = sum(len(values) for values in groups.values())
        add_grouped(groups, text=text, labels=labels, source="Vikhrmodels/Veles-2.5:lexicon_sensitive_topic")
        after = sum(len(values) for values in groups.values())
        if after > before:
            kept += 1
    return groups, {"scanned": scanned, "kept": kept, "contaminated": contaminated_count}


def toxicity_groups(
    contaminated: set[str],
    *,
    mnwa_limit: int,
    textdetox_limit: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary: dict[str, Any] = {}
    for dataset_id, split, limit, source in [
        ("Mnwa/russian-toxic", "train", mnwa_limit, "Mnwa/russian-toxic:toxicity_style"),
        ("textdetox/multilingual_toxicity_dataset", "ru", textdetox_limit, "textdetox/multilingual_toxicity_dataset:ru"),
    ]:
        try:
            dataset = load_dataset(dataset_id, split=split)
        except Exception as exc:  # noqa: BLE001
            summary[source] = {"error": str(exc)}
            continue
        cols = set(dataset.column_names)
        text_col = "text" if "text" in cols else "comment" if "comment" in cols else None
        label_col = "label" if "label" in cols else "toxic" if "toxic" in cols else "is_toxic" if "is_toxic" in cols else None
        if text_col is None:
            summary[source] = {"error": f"No text column in {sorted(cols)}"}
            continue
        scanned = kept = contaminated_count = 0
        for row in dataset:
            scanned += 1
            if scanned > limit:
                break
            text = normalize_space(str(row.get(text_col) or ""))
            if normalize_text(text) in contaminated:
                contaminated_count += 1
                continue
            label_value = row.get(label_col) if label_col else 1
            label = "social_injustice" if bool(label_value) else "body_shaming"
            add_grouped(groups, text=text, labels=[label], source=source)
            kept += 1
        summary[source] = {"scanned": scanned, "kept": kept, "contaminated": contaminated_count}
    return groups, summary


def uc_berkeley_groups(limit: int, contaminated: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    except Exception as exc:  # noqa: BLE001
        return groups, {"error": str(exc), "kept": 0}
    scanned = kept = contaminated_count = 0
    for row in dataset:
        scanned += 1
        if scanned > limit:
            break
        text = normalize_space(str(row.get("text") or ""))
        if normalize_text(text) in contaminated:
            contaminated_count += 1
            continue
        labels = [topic for col, topic in UC_BERKELEY_MAP.items() if bool(row.get(col))]
        before = sum(len(values) for values in groups.values())
        add_grouped(groups, text=text, labels=labels, source="ucberkeley-dlab/measuring-hate-speech:target_topic")
        after = sum(len(values) for values in groups.values())
        if after > before:
            kept += 1
    return groups, {"scanned": scanned, "kept": kept, "contaminated": contaminated_count}


def merge_groups(*group_sets: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for groups in group_sets:
        for topic, items in groups.items():
            for item in items:
                key = normalize_text(item["text"])
                if key in seen[topic]:
                    continue
                seen[topic].add(key)
                merged[topic].append(item)
    return merged


def filter_groups_by_source(
    groups: dict[str, list[dict[str, Any]]],
    source_filter: set[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    if source_filter is None:
        return groups
    filtered: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for topic, items in groups.items():
        filtered_items = [item for item in items if item.get("source") in source_filter]
        if filtered_items:
            filtered[topic].extend(filtered_items)
    return filtered


def make_records(
    groups: dict[str, list[dict[str, Any]]],
    *,
    count: int,
    seed: int,
    source_name: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    usable = {topic: items for topic, items in groups.items() if len(items) >= 2}
    if len(usable) < 2:
        raise RuntimeError("Not enough sensitive-topic groups with at least two records")
    topics = list(usable)
    records: list[dict[str, Any]] = []
    attempts = 0
    topic_cycle = topics[:]
    while len(records) < count and attempts < count * 100:
        attempts += 1
        if not topic_cycle:
            topic_cycle = topics[:]
            rng.shuffle(topic_cycle)
        topic = topic_cycle.pop()
        query_item, positive_item = rng.sample(usable[topic], 2)
        negative_topics = [candidate for candidate in HARD_NEGATIVES.get(topic, []) if candidate in usable]
        if not negative_topics:
            negative_topics = [candidate for candidate in topics if candidate != topic]
        if not negative_topics:
            continue
        negatives = []
        negative_labels = []
        for negative_topic in rng.sample(negative_topics, k=min(2, len(negative_topics))):
            negatives.append(rng.choice(usable[negative_topic])["text"])
            negative_labels.append(negative_topic)
        records.append(
            {
                "source": source_name,
                "query": "Instruct: Classify sensitive-topic type and retrieve text from the same fine-grained topic\nQuery: "
                + query_item["text"],
                "positive": positive_item["text"],
                "negatives": negatives,
                "metadata": {
                    "topic": topic,
                    "query_labels": query_item["labels"],
                    "positive_labels": positive_item["labels"],
                    "negative_topics": negative_labels,
                    "query_source": query_item["source"],
                    "positive_source": positive_item["source"],
                },
                "objective": "contrastive",
            }
        )
    if len(records) < count:
        raise RuntimeError(f"Could only build {len(records)} records out of requested {count}")
    rng.shuffle(records)
    return records


def sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Need {count} records, got {len(records)}")
    records = records[:]
    random.Random(seed).shuffle(records)
    return records[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean sensitive-topic discrimination Mix K.")
    parser.add_argument("--geracl-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--habr-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"))
    parser.add_argument("--deepvk-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"))
    parser.add_argument("--grounded-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_grounded_rag_v2_q180_doc1200_neg2.jsonl"))
    parser.add_argument("--sensitive-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_sensitive_topic_discrimination_3200.jsonl"))
    parser.add_argument("--mix-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixk_geracl2_habr1_deepvk1_groundedstrict1_sensitive1_19200.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--sensitive-count", type=int, default=3200)
    parser.add_argument("--mvr-limit", type=int, default=120000)
    parser.add_argument("--veles-limit", type=int, default=62881)
    parser.add_argument("--mnwa-limit", type=int, default=80000)
    parser.add_argument("--textdetox-limit", type=int, default=10000)
    parser.add_argument("--uc-berkeley-limit", type=int, default=50000)
    parser.add_argument(
        "--sensitive-source-filter",
        choices=sorted(SENSITIVE_SOURCE_FILTERS),
        default="all",
        help="Restrict SensitiveTopicDiscrimination records to one underlying source before pair construction.",
    )
    args = parser.parse_args()

    contaminated = contamination_keys()
    mvr, mvr_summary = mvr_groups(args.mvr_limit, contaminated)
    veles, veles_summary = veles_groups(args.veles_limit, contaminated)
    toxicity, toxicity_summary = toxicity_groups(
        contaminated,
        mnwa_limit=args.mnwa_limit,
        textdetox_limit=args.textdetox_limit,
    )
    uc_berkeley, uc_summary = uc_berkeley_groups(args.uc_berkeley_limit, contaminated)
    all_groups = merge_groups(mvr, veles, toxicity, uc_berkeley)
    groups = filter_groups_by_source(all_groups, SENSITIVE_SOURCE_FILTERS[args.sensitive_source_filter])
    sensitive = make_records(
        groups,
        count=args.sensitive_count,
        seed=args.seed,
        source_name=f"clean_sensitive_topic_discrimination:{args.sensitive_source_filter}",
    )
    write_jsonl(args.sensitive_out, sensitive)

    geracl = read_jsonl(args.geracl_path)
    habr = read_jsonl(args.habr_path)
    deepvk = read_jsonl(args.deepvk_path)
    grounded = read_jsonl(args.grounded_path)
    selected = {
        "geracl": sample(geracl, count=6400, seed=args.seed),
        "habr_harder": sample(habr, count=3200, seed=args.seed + 1),
        "deepvk_filtered": sample(deepvk, count=3200, seed=args.seed + 2),
        "grounded_strict": sample(grounded, count=3200, seed=args.seed + 3),
        "sensitive_topic_discrimination": sensitive,
    }
    mixed: list[dict[str, Any]] = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(args.seed + 5).shuffle(mixed)
    write_jsonl(args.mix_out, mixed)

    topic_counts = Counter(record["metadata"]["topic"] for record in sensitive)
    source_counts = Counter(record["metadata"]["query_source"] for record in sensitive)
    summary_path = args.summary_out or args.mix_out.with_name(args.mix_out.stem + "_summary.json")
    write_json(
        summary_path,
        {
            "output": str(args.mix_out),
            "sensitive_output": str(args.sensitive_out),
            "seed": args.seed,
            "sensitive_source_filter": args.sensitive_source_filter,
            "ratio": {
                "geracl": 2,
                "habr_harder": 1,
                "deepvk_ru_hnp_filtered": 1,
                "grounded_strict": 1,
                "sensitive_topic_discrimination": 1,
            },
            "counts": {
                "geracl_source": len(geracl),
                "geracl_used": len(selected["geracl"]),
                "habr_harder_source": len(habr),
                "habr_harder_used": len(selected["habr_harder"]),
                "deepvk_filtered_source": len(deepvk),
                "deepvk_used": len(selected["deepvk_filtered"]),
                "grounded_strict_source": len(grounded),
                "grounded_strict_used": len(selected["grounded_strict"]),
                "sensitive_used": len(sensitive),
                "total": len(mixed),
            },
            "contamination_filter": {
                "excluded_exact_normalized_rumteb_sensitive_texts": len(contaminated),
                "excluded_datasets": [
                    "NiGuLa/Russian_Sensitive_Topics",
                    "ai-forever/sensitive-topics-classification",
                    "mteb/SensitiveTopicsClassification",
                ],
            },
            "sensitive_topic_counts": dict(topic_counts),
            "sensitive_query_source_counts": dict(source_counts),
            "usable_group_counts": {topic: len(items) for topic, items in sorted(groups.items())},
            "all_group_counts": {topic: len(items) for topic, items in sorted(all_groups.items())},
            "source_summaries": {
                "mvrcii": mvr_summary,
                "veles": veles_summary,
                "toxicity": toxicity_summary,
                "uc_berkeley": uc_summary,
            },
            "batch_size": 4,
            "max_steps_1x": len(mixed) // 4,
            "source_paths": {
                "geracl": str(args.geracl_path),
                "habr_harder": str(args.habr_path),
                "deepvk_filtered": str(args.deepvk_path),
                "grounded_strict": str(args.grounded_path),
                "sensitive_topic_discrimination": str(args.sensitive_out),
            },
        },
    )
    print(f"Wrote {args.sensitive_out}")
    print(f"Wrote {args.mix_out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
