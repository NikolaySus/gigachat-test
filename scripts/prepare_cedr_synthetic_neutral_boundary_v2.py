from __future__ import annotations

import argparse
import random
from pathlib import Path

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_neutral_lexical_distractors import read_jsonl
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


TRIGGERS = {
    "joy": ["радующий", "веселый", "забавный", "приятный", "хороший"],
    "sadness": ["печальный", "грустный", "тоскливый", "бедственный", "болезненный"],
    "surprise": ["странный", "удивительный", "необычный", "неожиданный", "изумительный"],
    "fear": ["опасный", "тревожный", "страшный", "угрожающий", "пугающий"],
    "anger": ["злой", "яростный", "гневный", "агрессивный", "мстительный"],
}

TEMPLATES = {
    "reported_fact": [
        "В сообщении говорится, что очевидцы назвали эпизод {trigger}, но автор только пересказывает факт.",
        "По словам участников, случай показался им {trigger}, однако фраза остается новостным описанием.",
        "В отчете указано, что реакция была охарактеризована как {trigger}, без личной эмоции рассказчика.",
        "Издание передало оценку свидетелей: событие сочли {trigger}, но текст не выражает переживание.",
    ],
    "object_attribute": [
        "Описание предмета содержит признак «{trigger} вид», это характеристика объекта, а не эмоция.",
        "В тексте сказано, что устройство имеет {trigger} эффект, но речь идет о визуальной особенности.",
        "Автор описывает {trigger} образ героя как литературную деталь, а не собственное чувство.",
        "Фраза про {trigger} внешний вид относится к свойству вещи или персонажа.",
    ],
    "emotion_concept": [
        "Материал объясняет понятие «{trigger} реакция» как термин, не выражая эмоцию автора.",
        "Текст обсуждает, как люди распознают {trigger} поведение, в справочном стиле.",
        "В статье перечислены признаки состояния «{trigger}», но это классификация, а не переживание.",
        "Фраза использует эмоциональное слово {trigger} как тему исследования.",
    ],
    "crime_news": [
        "Новость описывает конфликт и {trigger} поступок участника в фактической форме.",
        "В сводке упомянуто {trigger} происшествие, но предложение остается нейтральным сообщением.",
        "Текст сообщает о нарушении и называет его {trigger}, не выражая отношение автора.",
        "Описание инцидента содержит слово {trigger}, потому что это часть фабулы события.",
    ],
    "quoted_speech": [
        "В цитате собеседник использовал слово {trigger}, а автор лишь передал чужую речь.",
        "Фраза пересказывает чужое мнение: ситуацию назвали {trigger}, без оценки рассказчика.",
        "Сообщение фиксирует реплику о том, что результат был {trigger}, в нейтральном контексте.",
        "Автор цитирует выражение «{trigger} случай», не добавляя собственной эмоциональной реакции.",
    ],
    "meta_literary": [
        "Разбор текста отмечает {trigger} мотив в сюжете как художественный прием.",
        "Критик описывает {trigger} черту персонажа, а не сообщает о своем состоянии.",
        "В пересказе произведения слово {trigger} относится к роли героя в истории.",
        "Литературный комментарий использует {trigger} эпитет как часть анализа образа.",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build richer synthetic neutral-boundary rows for CEDR.")
    parser.add_argument("--name", default="cedr_synthetic_neutral_boundary_v2_3600")
    parser.add_argument("--count", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=1121)
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    go_rows = read_jsonl(args.go_path)
    emotion_pools: dict[str, list[str]] = {group: [] for group in GROUPS if group != "neutral"}
    for row in go_rows:
        group = row.get("metadata", {}).get("group")
        if group in emotion_pools:
            emotion_pools[group].append(row["query"])

    base_items = []
    for trigger_group, triggers in TRIGGERS.items():
        for frame, templates in TEMPLATES.items():
            for trigger in triggers:
                for template in templates:
                    base_items.append(
                        {
                            "text": template.format(trigger=trigger),
                            "trigger_group": trigger_group,
                            "frame": frame,
                        }
                    )
    rng.shuffle(base_items)

    records = []
    while len(records) < args.count:
        item = base_items[len(records) % len(base_items)]
        positives = [
            candidate
            for candidate in base_items
            if candidate["trigger_group"] == item["trigger_group"]
            and candidate["frame"] == item["frame"]
            and candidate["text"] != item["text"]
        ]
        negative_groups = [item["trigger_group"]] + [
            group for group in ["joy", "sadness", "surprise", "fear", "anger"] if group != item["trigger_group"]
        ]
        negatives = []
        for negative_group in negative_groups:
            pool = emotion_pools.get(negative_group) or []
            if pool:
                negatives.append(rng.choice(pool))
        records.append(
            {
                "source": "synthetic:cedr_neutral_boundary_v2",
                "objective": "contrastive",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                "negatives": negatives,
                "metadata": {
                    "group": "neutral",
                    "trigger_group": item["trigger_group"],
                    "frame": item["frame"],
                    "synthetic": True,
                },
            }
        )

    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "records": len(records),
            "seed": args.seed,
            "per_trigger_group": {
                group: sum(1 for row in records if row["metadata"]["trigger_group"] == group)
                for group in TRIGGERS
            },
            "per_frame": {
                frame: sum(1 for row in records if row["metadata"]["frame"] == frame)
                for frame in TEMPLATES
            },
            "go_path": str(args.go_path),
            "construction": "synthetic neutral CEDR-style hard-boundary frames with emotion lexemes and clean GoEmotions-RU hard negatives",
            "contamination_policy": "synthetic templates only; no CEDR records used",
        },
    )
    print(f"prepared {out}: {len(records)} rows")


if __name__ == "__main__":
    main()
