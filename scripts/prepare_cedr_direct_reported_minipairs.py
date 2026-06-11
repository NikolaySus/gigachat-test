from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, is_contaminated, load_cedr_index, normalize_text


GROUP_LEXEMES = {
    "joy": ["радость", "счастье", "восторг", "улыбка", "порадовало"],
    "sadness": ["грусть", "печаль", "тоска", "огорчение", "сожаление"],
    "surprise": ["удивление", "изумление", "неожиданность", "сюрприз", "поразило"],
    "fear": ["страх", "тревога", "опасение", "угроза", "испуг"],
    "anger": ["злость", "гнев", "ярость", "возмущение", "раздражение"],
}

DIRECT_TEMPLATES = [
    "Мне сейчас очень {term}, я не могу спокойно об этом думать.",
    "Я чувствую {term} после этого сообщения.",
    "По тону видно: автор испытывает {term}.",
    "Меня накрыло ощущение: {term}.",
    "В этом комментарии прямо выражено чувство: {term}.",
    "Я пишу это не нейтрально, во мне заметна {term}.",
    "После этой новости у меня появилась сильная {term}.",
    "Если коротко описать мое состояние, это {term}.",
]

REPORTED_TEMPLATES = [
    "В статье упоминается слово «{term}» как тема исследования.",
    "Доклад содержит раздел «{term}», но написан в справочном стиле.",
    "В новости говорится, что участники обсуждали «{term}» как социальное явление.",
    "Автор перечисляет причины, по которым термин «{term}» появился в отчете.",
    "Материал описывает понятие «{term}» без личного переживания автора.",
    "В заголовке есть «{term}», а основной текст сообщает факты и даты.",
    "Эксперты прокомментировали тему «{term}» в нейтральной формулировке.",
    "В документе слово «{term}» используется как название категории.",
]

NEUTRAL_TEMPLATES = [
    "Комиссия опубликовала отчет с датами и условиями.",
    "В таблице приведены параметры оборудования и сроки поставки.",
    "Участники встречи согласовали порядок следующих действий.",
    "На странице перечислены разделы, источники и технические детали.",
    "Исследователи сравнили несколько методов обработки данных.",
    "Компания сообщила о плановом обновлении сервиса.",
    "В документе описана последовательность этапов проекта.",
    "Пользователь уточнил правила заполнения формы.",
]

OPENERS = ["", "В коротком комментарии: ", "По смыслу сообщения: ", "В тексте сказано: "]
ENDINGS = ["", ".", "!", "...", " — именно так.", " без дополнительных пояснений."]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_text(template: str, term: str, rng: random.Random) -> str:
    text = rng.choice(OPENERS) + template.format(term=term) + rng.choice(ENDINGS)
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean CEDR direct-vs-reported emotion minimal pairs.")
    parser.add_argument("--per-emotion", type=int, default=700)
    parser.add_argument("--neutral-count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=963)
    parser.add_argument("--name", default="cedr_direct_reported_minipairs_5000")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    seen: set[str] = set()
    pools: dict[str, list[str]] = defaultdict(list)
    reported_by_group: dict[str, list[str]] = defaultdict(list)
    skipped = Counter()

    def add_text(group: str, text: str, *, reported: bool = False) -> bool:
        norm = normalize_text(text)
        if norm in seen:
            skipped["duplicate"] += 1
            return False
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            return False
        seen.add(norm)
        if reported:
            reported_by_group[group].append(text)
        else:
            pools[group].append(text)
        return True

    for group, terms in GROUP_LEXEMES.items():
        attempts = 0
        while len(pools[group]) < args.per_emotion and attempts < args.per_emotion * 100:
            attempts += 1
            add_text(group, make_text(rng.choice(DIRECT_TEMPLATES), rng.choice(terms), rng))
        attempts = 0
        while len(reported_by_group[group]) < args.per_emotion and attempts < args.per_emotion * 100:
            attempts += 1
            add_text(group, make_text(rng.choice(REPORTED_TEMPLATES), rng.choice(terms), rng), reported=True)
        if len(pools[group]) < args.per_emotion or len(reported_by_group[group]) < args.per_emotion:
            raise RuntimeError(f"Could not generate enough rows for {group}")

    attempts = 0
    while len(pools["neutral"]) < args.neutral_count and attempts < args.neutral_count * 100:
        attempts += 1
        text = rng.choice(OPENERS) + rng.choice(NEUTRAL_TEMPLATES) + rng.choice(ENDINGS)
        add_text("neutral", " ".join(text.split()))

    groups = list(GROUP_LEXEMES)
    records = []
    for group in groups:
        direct = pools[group]
        reported = reported_by_group[group]
        for index, query in enumerate(direct):
            positives = direct
            positive = positives[(index + rng.randrange(1, len(positives))) % len(positives)]
            negatives = [CEDR_PREFIX + rng.choice(reported)]
            negatives.extend(CEDR_PREFIX + rng.choice(pools["neutral"]) for _ in range(2))
            for other in groups:
                if other != group:
                    negatives.append(CEDR_PREFIX + rng.choice(pools[other]))
            rng.shuffle(negatives)
            records.append(
                {
                    "source": "synthetic:cedr_direct_reported_minipairs",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + query,
                    "positive": CEDR_PREFIX + positive,
                    "negatives": negatives,
                    "metadata": {"group": group, "synthetic": True, "construction": "direct_emotion"},
                }
            )
        for index, query in enumerate(reported):
            positive = reported[(index + rng.randrange(1, len(reported))) % len(reported)]
            negatives = [CEDR_PREFIX + rng.choice(direct)]
            negatives.extend(CEDR_PREFIX + rng.choice(pools[other]) for other in groups if other != group)
            rng.shuffle(negatives)
            records.append(
                {
                    "source": "synthetic:cedr_direct_reported_minipairs",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + query,
                    "positive": CEDR_PREFIX + positive,
                    "negatives": negatives,
                    "metadata": {"group": "neutral", "trigger_group": group, "synthetic": True, "construction": "reported_neutral"},
                }
            )

    neutral = pools["neutral"]
    for index, query in enumerate(neutral):
        positive = neutral[(index + rng.randrange(1, len(neutral))) % len(neutral)]
        negatives = [CEDR_PREFIX + rng.choice(pools[group]) for group in groups]
        records.append(
            {
                "source": "synthetic:cedr_direct_reported_minipairs",
                "objective": "contrastive",
                "query": CEDR_PREFIX + query,
                "positive": CEDR_PREFIX + positive,
                "negatives": negatives,
                "metadata": {"group": "neutral", "synthetic": True, "construction": "plain_neutral"},
            }
        )

    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    summary = {
        "records": len(records),
        "direct_per_emotion": args.per_emotion,
        "reported_per_emotion": args.per_emotion,
        "neutral_count": args.neutral_count,
        "label_counts": dict(Counter(row["metadata"]["group"] for row in records)),
        "construction_counts": dict(Counter(row["metadata"]["construction"] for row in records)),
        "skipped": dict(skipped),
        "seed": args.seed,
        "contamination_policy": "synthetic only; exact and near CEDR overlap removed",
    }
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {out.relative_to(Path.cwd())}: {len(records)} rows")


if __name__ == "__main__":
    main()
