from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_cedr_neutral_lexical_distractors import read_jsonl
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]

LEXEMES = {
    "joy": [
        "смех",
        "радость",
        "счастье",
        "веселый",
        "удача",
        "прекрасный",
        "забавный",
        "любовь",
    ],
    "sadness": [
        "слезы",
        "плач",
        "грусть",
        "печаль",
        "сожаление",
        "беда",
        "несчастье",
        "тоска",
    ],
    "surprise": [
        "странный",
        "удивительный",
        "неожиданный",
        "недоумение",
        "необычный",
        "изумление",
        "загадочный",
        "удивление",
    ],
    "fear": [
        "страх",
        "опасность",
        "угроза",
        "тревога",
        "паника",
        "напуганный",
        "опасение",
        "пугливый",
    ],
    "anger": [
        "злость",
        "гнев",
        "ярость",
        "агрессия",
        "возмущение",
        "раздражение",
        "мстительный",
        "ненависть",
    ],
}

OBJECTS = [
    "случай",
    "комментарий",
    "описание",
    "заявление",
    "сюжет",
    "эпизод",
    "фрагмент",
    "материал",
    "пример",
    "формулировка",
]

NEUTRAL_TEMPLATES = [
    "В материале слово «{lexeme}» используется как описание, а не как личное переживание автора.",
    "Автор сообщает о реакции людей на {obj}, связанной со словом «{lexeme}», в нейтральном стиле.",
    "Фраза фиксирует, что {obj} назвали «{lexeme}», но не выражает эмоцию говорящего.",
    "В тексте пересказывается чужая оценка «{lexeme}» без эмоциональной позиции автора.",
    "Сообщение классифицирует {obj} как «{lexeme}» и остается информационным.",
    "Здесь «{lexeme}» является темой обсуждения, а не эмоциональным состоянием автора.",
    "Новостной пересказ упоминает «{lexeme}» как характеристику ситуации.",
    "Слово «{lexeme}» относится к описываемому {obj}, а не к чувству рассказчика.",
    "В справочном контексте говорится о слове «{lexeme}» без выраженной эмоции.",
    "Текст сообщает, что у наблюдателей возникла реакция «{lexeme}», но сам остается нейтральным.",
    "Эксперт назвал {obj} словом «{lexeme}», это передано как факт.",
    "В цитате встречается оценка «{lexeme}», однако весь фрагмент является пересказом.",
]


def load_emotion_pools(path: Path) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(path):
        group = row.get("metadata", {}).get("group")
        query = row.get("query")
        if group in LEXEMES and isinstance(query, str) and query.strip():
            pools[group].append(query)
    return pools


def make_neutral_texts(*, count_per_group: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    grouped = {}
    for group, lexemes in LEXEMES.items():
        rows = []
        for index in range(count_per_group):
            lexeme = lexemes[index % len(lexemes)]
            template = NEUTRAL_TEMPLATES[(index // len(lexemes)) % len(NEUTRAL_TEMPLATES)]
            obj = OBJECTS[(index + len(group)) % len(OBJECTS)]
            rows.append(CEDR_PREFIX + template.format(lexeme=lexeme, obj=obj))
        rng.shuffle(rows)
        grouped[group] = rows
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic reported-neutral pair-score CEDR boundary data.")
    parser.add_argument("--name", default="cedr_synthetic_reported_neutral_pairscore_9000")
    parser.add_argument("--count-per-group", type=int, default=360)
    parser.add_argument("--negative-score", type=float, default=0.02)
    parser.add_argument("--positive-score", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument(
        "--emotion-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    neutral = make_neutral_texts(count_per_group=args.count_per_group, seed=args.seed)
    emotion = load_emotion_pools(args.emotion_path)
    records = []
    for group, neutral_rows in neutral.items():
        for index, text in enumerate(neutral_rows):
            positive_pool = [candidate for candidate in neutral_rows if candidate != text]
            records.append(
                {
                    "source": "synthetic:reported_neutral_pairscore",
                    "objective": "pair_score",
                    "sentence1": text,
                    "sentence2": rng.choice(positive_pool),
                    "score": args.positive_score,
                    "metadata": {"group": "neutral", "trigger_group": group, "kind": "neutral_positive"},
                }
            )
            for negative_group in [group, *[other for other in LEXEMES if other != group]][:3]:
                records.append(
                    {
                        "source": "synthetic:reported_neutral_pairscore",
                        "objective": "pair_score",
                        "sentence1": text,
                        "sentence2": rng.choice(emotion[negative_group]),
                        "score": args.negative_score,
                        "metadata": {
                            "group": "neutral",
                            "trigger_group": group,
                            "negative_group": negative_group,
                            "kind": "emotion_negative",
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
            "count_per_group": args.count_per_group,
            "positive_score": args.positive_score,
            "negative_score": args.negative_score,
            "record_kinds": dict(Counter(row["metadata"]["kind"] for row in records)),
            "trigger_counts": dict(Counter(row["metadata"]["trigger_group"] for row in records)),
            "emotion_path": str(args.emotion_path),
            "construction": "synthetic neutral reported/emotion-word templates; pair-score pulls same-trigger neutral templates together and pushes clean emotion examples away",
            "contamination_policy": "synthetic neutral texts only; negative examples inherit CEDR-overlap filtering from clean GoEmotions-RU component; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
