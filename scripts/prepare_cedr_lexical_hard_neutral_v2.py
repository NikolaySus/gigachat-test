from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX  # noqa: E402
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (  # noqa: E402
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_json,
    write_jsonl,
)


GROUP_TERMS = {
    "joy": ["веселый", "радующий", "счастливый", "интересные и веселые", "радостный"],
    "sadness": ["печальный", "грустный", "скорбный", "тоскливый", "меланхоличный"],
    "surprise": ["странный", "удивительный", "изумительный", "неожиданный", "поразительный"],
    "fear": ["страх", "опасения", "тревога", "угроза", "испуг"],
    "anger": ["ярость", "злость", "гнев", "озлобленный", "агрессивный"],
}

NEUTRAL_TEMPLATES = [
    "В названии раздела встречается слово «{term}», но дальше идет сухое описание фактов.",
    "Автор использует выражение «{term}» как литературную характеристику предмета.",
    "В справке указано понятие «{term}» и приведены даты публикации.",
    "Фраза «{term}» относится к внешнему виду объекта, а не к переживанию автора.",
    "В заметке обсуждается термин «{term}» как часть классификации материалов.",
    "Слово «{term}» входит в цитату из документа, где перечисляются обстоятельства дела.",
    "В тексте говорится о теме «{term}», но сообщение остается информационным.",
    "Выражение «{term}» описывает стиль произведения, без личной эмоциональной реакции.",
    "В учебном примере слово «{term}» используется для разбора значения.",
    "Журналист сообщает о заявлении по теме «{term}» и не выражает собственных чувств.",
    "В аннотации сказано, что глава посвящена теме «{term}».",
    "Формулировка «{term}» дана как заголовок пункта в списке.",
]

OBJECT_CONTEXTS = [
    " Рядом указаны источник, номер страницы и краткое пояснение.",
    " После этого перечислены участники, места и хронология.",
    " Остальная часть текста состоит из нейтрального описания.",
    " Это похоже на энциклопедическую или новостную формулировку.",
    " В предложении нет прямого указания на состояние говорящего.",
    " Контекст относится к предмету, событию или документу.",
]

EMOTION_TEMPLATES = {
    "joy": [
        "Мне радостно читать это сообщение.",
        "Я улыбаюсь и чувствую настоящую радость.",
        "Автор явно радуется происходящему.",
    ],
    "sadness": [
        "Мне грустно после этих слов.",
        "Автор пишет с заметной печалью.",
        "В сообщении прямо выражена тоска.",
    ],
    "surprise": [
        "Я удивлен таким поворотом.",
        "Автор явно не ожидал такого результата.",
        "В комментарии слышно искреннее удивление.",
    ],
    "fear": [
        "Мне страшно думать об этом.",
        "Автор тревожится и боится последствий.",
        "В сообщении выражен настоящий страх.",
    ],
    "anger": [
        "Меня злит эта ситуация.",
        "Автор явно раздражен и сердится.",
        "В комментарии выражена злость.",
    ],
}


def make_text(group: str, rng: random.Random) -> str:
    term = rng.choice(GROUP_TERMS[group])
    return rng.choice(NEUTRAL_TEMPLATES).format(term=term) + rng.choice(OBJECT_CONTEXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CEDR-like hard neutral lexical distractor component.")
    parser.add_argument("--per-trigger", type=int, default=700)
    parser.add_argument("--seed", type=int, default=1003)
    parser.add_argument("--name", default="cedr_lexical_hard_neutral_v2_3500")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    rows_by_group: dict[str, list[str]] = {group: [] for group in GROUP_TERMS}
    seen: set[str] = set()
    skipped = Counter()
    for group in GROUP_TERMS:
        attempts = 0
        while len(rows_by_group[group]) < args.per_trigger and attempts < args.per_trigger * 200:
            attempts += 1
            text = make_text(group, rng)
            key = normalize_text(text).lower()
            if key in seen:
                skipped["duplicate"] += 1
                continue
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            seen.add(key)
            rows_by_group[group].append(text)
        if len(rows_by_group[group]) < args.per_trigger:
            raise RuntimeError(f"Only generated {len(rows_by_group[group])} for {group}")

    records = []
    for group, rows in rows_by_group.items():
        for index, text in enumerate(rows):
            positive = rows[(index + rng.randrange(1, len(rows))) % len(rows)]
            negatives = [CEDR_PREFIX + rng.choice(EMOTION_TEMPLATES[group])]
            for other in GROUP_TERMS:
                if other != group:
                    negatives.append(CEDR_PREFIX + rng.choice(EMOTION_TEMPLATES[other]))
            rng.shuffle(negatives)
            records.append(
                {
                    "source": "synthetic:cedr_lexical_hard_neutral_v2",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + text,
                    "positive": CEDR_PREFIX + positive,
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
            "records": len(records),
            "per_trigger": args.per_trigger,
            "selected_by_trigger_group": dict(Counter(row["metadata"]["trigger_group"] for row in records)),
            "skipped": dict(skipped),
            "construction": "synthetic neutral CEDR-like lexical hard cases with direct-emotion negatives",
            "contamination_policy": "synthetic rows checked against exact/near CEDR overlap; no CEDR records used",
            "seed": args.seed,
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
