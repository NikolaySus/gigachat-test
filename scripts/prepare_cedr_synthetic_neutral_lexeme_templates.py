from __future__ import annotations

import argparse
import random
from pathlib import Path

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_neutral_lexical_distractors import read_jsonl
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


LEXEMES = {
    "joy": ["радость", "радующий", "веселый", "улыбка", "счастье", "хороший"],
    "sadness": ["грусть", "печаль", "тоска", "скорбь", "беда", "боль"],
    "surprise": ["удивительный", "странный", "изумительный", "неожиданный", "недоумение"],
    "fear": ["страх", "опасение", "угроза", "тревога", "боюсь"],
    "anger": ["злость", "гнев", "ярость", "разозлил", "агрессивный", "мстительный"],
}

NEUTRAL_TEMPLATES = [
    "В материале упоминается слово «{lexeme}», но текст описывает факт и не выражает личную эмоцию.",
    "Автор сообщает о теме «{lexeme}» в нейтральном информационном стиле.",
    "Фраза содержит эмоционально окрашенное слово «{lexeme}», однако является описанием события.",
    "В тексте говорится о реакции людей на тему «{lexeme}», без выраженной эмоции автора.",
    "Это нейтральная справочная фраза, где «{lexeme}» является предметом обсуждения.",
    "Сообщение фиксирует наличие оценки «{lexeme}» как факта, а не как переживания.",
    "В новостном контексте слово «{lexeme}» относится к описанию ситуации.",
    "Фраза про «{lexeme}» является пересказом или классификацией, а не эмоциональным высказыванием.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic neutral lexeme templates for CEDR boundary correction.")
    parser.add_argument("--name", default="cedr_synthetic_neutral_lexeme_templates_2400")
    parser.add_argument("--count", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=1103)
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

    per_group = args.count // len(LEXEMES)
    remainder = args.count % len(LEXEMES)
    records = []
    for group_index, (group, lexemes) in enumerate(LEXEMES.items()):
        target = per_group + (1 if group_index < remainder else 0)
        examples = []
        for i in range(target):
            lexeme = lexemes[i % len(lexemes)]
            template = NEUTRAL_TEMPLATES[(i // len(lexemes)) % len(NEUTRAL_TEMPLATES)]
            examples.append(template.format(lexeme=lexeme))
        for i, text in enumerate(examples):
            positives = [candidate for candidate in examples if candidate != text]
            negative_groups = [group] + [other for other in emotion_pools if other != group]
            negatives = []
            for negative_group in negative_groups:
                pool = emotion_pools[negative_group]
                if pool:
                    negatives.append(rng.choice(pool))
            records.append(
                {
                    "source": "synthetic:cedr_neutral_lexeme_boundary",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + text,
                    "positive": CEDR_PREFIX + rng.choice(positives),
                    "negatives": negatives,
                    "metadata": {
                        "group": "neutral",
                        "trigger_group": group,
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
            "per_group": {group: sum(1 for row in records if row["metadata"]["trigger_group"] == group) for group in LEXEMES},
            "go_path": str(args.go_path),
            "construction": "synthetic neutral CEDR-style sentences with emotion lexemes; same-trigger emotion negatives from clean GoEmotions-RU",
            "contamination_policy": "synthetic templates only; no CEDR records used",
        },
    )
    print(f"prepared {out}: {len(records)} rows")


if __name__ == "__main__":
    main()
