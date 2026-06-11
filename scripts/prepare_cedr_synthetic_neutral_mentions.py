from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, is_contaminated, load_cedr_index, normalize_text


GROUP_TERMS = {
    "joy": ["радость", "счастье", "улыбка", "веселье"],
    "sadness": ["грусть", "печаль", "тоска", "сожаление"],
    "surprise": ["удивление", "изумление", "неожиданность", "странность"],
    "fear": ["страх", "опасения", "угроза", "тревога"],
    "anger": ["злость", "гнев", "ярость", "агрессия"],
}

NEUTRAL_TEMPLATES = [
    "В тексте упоминается термин «{term}», но сообщение носит справочный характер.",
    "Раздел с названием «{term}» содержит только краткое описание темы.",
    "Автор перечисляет факты о понятии «{term}» без выраженной личной оценки.",
    "В документе слово «{term}» используется как часть заголовка и не задает тон сообщения.",
    "Комментарий сообщает, что тема «{term}» будет рассмотрена отдельно.",
    "На странице приведено нейтральное объяснение слова «{term}».",
    "Фраза про «{term}» относится к классификации материалов, а не к настроению автора.",
    "В заметке указано, что пункт «{term}» добавлен в общий список.",
    "Сообщение фиксирует наличие темы «{term}» в статье без эмоционального вывода.",
    "В описании встречается слово «{term}», но остальная часть текста информационная.",
]

CONTEXTS = [
    " Дополнительно указаны дата публикации и источник.",
    " Рядом приведены технические детали и ссылки на разделы.",
    " Далее перечислены причины, параметры и ограничения.",
    " После этого идет обычное описание последовательности событий.",
    " Остальной текст состоит из нейтральных пояснений.",
    " Формулировка похожа на краткую аннотацию или справку.",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_text(term: str, rng: random.Random) -> str:
    return rng.choice(NEUTRAL_TEMPLATES).format(term=term) + rng.choice(CONTEXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic neutral CEDR mentions of emotion lexemes.")
    parser.add_argument("--go-path", type=Path, default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl")
    parser.add_argument("--per-trigger", type=int, default=400)
    parser.add_argument("--seed", type=int, default=873)
    parser.add_argument("--name", default="cedr_synthetic_neutral_mentions_2000")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    go_rows = read_jsonl(args.go_path)
    emotion_pools: dict[str, list[str]] = {group: [] for group in GROUP_TERMS}
    for row in go_rows:
        group = row.get("metadata", {}).get("group")
        if group in emotion_pools:
            emotion_pools[group].append(row["query"])

    rows_by_trigger: dict[str, list[str]] = {group: [] for group in GROUP_TERMS}
    seen: set[str] = set()
    skipped = Counter()
    for group, terms in GROUP_TERMS.items():
        attempts = 0
        while len(rows_by_trigger[group]) < args.per_trigger and attempts < args.per_trigger * 100:
            attempts += 1
            text = make_text(rng.choice(terms), rng)
            norm = normalize_text(text)
            if norm in seen:
                skipped["duplicate"] += 1
                continue
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            seen.add(norm)
            rows_by_trigger[group].append(text)
        if len(rows_by_trigger[group]) < args.per_trigger:
            raise RuntimeError(f"Could only generate {len(rows_by_trigger[group])} rows for {group}")

    records = []
    for group, texts in rows_by_trigger.items():
        for index, query_text in enumerate(texts):
            positive_text = texts[(index + rng.randrange(1, len(texts))) % len(texts)]
            negatives = [
                rng.choice(emotion_pools[negative_group])
                for negative_group in ["joy", "sadness", "surprise", "fear", "anger"]
                if emotion_pools[negative_group]
            ]
            records.append(
                {
                    "source": "synthetic:cedr_neutral_emotion_lexeme_mentions",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + query_text,
                    "positive": CEDR_PREFIX + positive_text,
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
    summary = {
        "records": len(records),
        "per_trigger": args.per_trigger,
        "selected_by_trigger_group": dict(Counter(row["metadata"]["trigger_group"] for row in records)),
        "skipped": dict(skipped),
        "go_path": str(args.go_path),
        "construction": "synthetic neutral emotion-lexeme mentions with GoEmotions-RU emotion negatives",
        "contamination_policy": "no CEDR records used; generated texts checked against exact/near CEDR overlap index",
        "seed": args.seed,
    }
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {out.relative_to(Path.cwd())}: {len(records)} rows")


if __name__ == "__main__":
    main()
