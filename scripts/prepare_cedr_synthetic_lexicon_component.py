#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
)


PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "

LEXICON = {
    "joy": [
        "мне очень радостно от этой новости",
        "я прямо сияю от счастья сегодня",
        "какая приятная неожиданность, настроение отличное",
        "улыбаюсь весь вечер, все получилось",
        "это меня вдохновило и сильно порадовало",
        "ура, наконец-то хорошие новости",
        "сердце радуется, когда такое читаешь",
        "это было так мило, я довольна",
        "ахаха, день стал заметно лучше",
        "спасибо, я счастлив и спокоен",
    ],
    "sadness": [
        "мне тоскливо и совсем не хочется говорить",
        "очень жаль, что все закончилось именно так",
        "настроение упало, грустно до слез",
        "снова это чувство пустоты и усталости",
        "печально смотреть на такие события",
        "я расстроился сильнее, чем ожидал",
        "весь день какой-то тяжелый и унылый",
        "не могу не сожалеть об этом решении",
        "так обидно, что ничего не изменилось",
        "после этих слов стало совсем мрачно",
    ],
    "surprise": [
        "я совсем не ожидал такого поворота",
        "ничего себе, вот это неожиданная история",
        "удивительно, как быстро все поменялось",
        "я в полном недоумении после этой новости",
        "это застало меня врасплох",
        "странно, но результат оказался другим",
        "вот это сюрприз, даже не верится",
        "поразительно, что никто раньше не заметил",
        "я удивлена такой реакцией людей",
        "хм, неожиданно видеть такое решение",
    ],
    "fear": [
        "мне страшно думать, чем это закончится",
        "я переживаю и боюсь ошибиться",
        "от этих слов стало тревожно",
        "опасаюсь, что ситуация станет хуже",
        "меня пугает такая неопределенность",
        "внутри тревога, будто скоро случится беда",
        "страшновато идти туда одному",
        "я нервничаю перед этой встречей",
        "не по себе от такого сообщения",
        "боюсь, что уже поздно что-то менять",
    ],
    "anger": [
        "меня злит такое отношение",
        "я раздражен и не хочу это терпеть",
        "какая наглость, просто возмутительно",
        "сердит после этих слов уже весь день",
        "это вызывает злость и усталость",
        "достало, сколько можно так поступать",
        "я в бешенстве от этой несправедливости",
        "раздражает, когда обещают и не делают",
        "обидно и зло берет от происходящего",
        "такой поступок невозможно спокойно принять",
    ],
    "neutral": [
        "сообщение содержит только описание фактов",
        "в документе приведены сроки и условия",
        "участники встречи обсудили рабочий график",
        "в отчете указаны основные параметры устройства",
        "поезд прибыл на станцию согласно расписанию",
        "компания опубликовала техническое обновление",
        "исследователи сравнили несколько методов анализа",
        "на странице перечислены доступные разделы",
        "в таблице представлены результаты измерений",
        "пользователь задал уточняющий вопрос",
    ],
}

MODIFIERS = [
    "",
    " если честно",
    " прямо сейчас",
    " после вчерашнего разговора",
    " и это заметно по тону",
    " хотя внешне все спокойно",
    " без лишних подробностей",
    " в конце дня",
    " когда читаю такие новости",
    " в обычном комментарии",
]

OPENERS = [
    "",
    "по ощущениям, ",
    "в комментарии звучит так: ",
    "коротко говоря, ",
    "мне кажется, ",
    "по тону сообщения, ",
    "если передать настроение, ",
    "автор пишет, что ",
    "в этой ситуации ",
    "сегодня ",
]

CONTEXTS = [
    "",
    " из-за новости",
    " после сообщения",
    " в разговоре",
    " на фоне событий",
    " в личной переписке",
    " после такого ответа",
    " при чтении поста",
    " в конце обсуждения",
    " из-за этой ситуации",
]

PUNCT = [".", "!", "...", " :)", " (", ")", "!!", ""]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "open_ru_1r_nc_cedr_synthetic_lexicon_3600.jsonl")
    parser.add_argument("--per-group", type=int, default=600)
    parser.add_argument("--seed", type=int, default=821)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    texts_by_group: dict[str, list[str]] = defaultdict(list)
    skipped = Counter()
    seen = set()

    for group, bases in LEXICON.items():
        attempts = 0
        while len(texts_by_group[group]) < args.per_group and attempts < args.per_group * 50:
            attempts += 1
            base = rng.choice(bases)
            text = rng.choice(OPENERS) + base + rng.choice(CONTEXTS) + rng.choice(MODIFIERS) + rng.choice(PUNCT)
            text = " ".join(text.split())
            norm = normalize_text(text)
            if norm in seen:
                skipped["duplicate"] += 1
                continue
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            seen.add(norm)
            texts_by_group[group].append(text)
        if len(texts_by_group[group]) < args.per_group:
            raise RuntimeError(f"Could only generate {len(texts_by_group[group])} rows for {group}")

    records = []
    groups = list(LEXICON)
    for group in groups:
        pool = texts_by_group[group]
        for idx, query_text in enumerate(pool):
            positive_text = pool[(idx + rng.randrange(1, len(pool))) % len(pool)]
            negatives = []
            for negative_group in groups:
                if negative_group == group:
                    continue
                negatives.append(PREFIX + rng.choice(texts_by_group[negative_group]))
            rng.shuffle(negatives)
            records.append(
                {
                    "source": "synthetic:cedr_lexicon_no_overlap",
                    "objective": "contrastive",
                    "query": PREFIX + query_text,
                    "positive": PREFIX + positive_text,
                    "negatives": negatives,
                    "metadata": {"group": group, "synthetic": True},
                }
            )
    rng.shuffle(records)
    write_jsonl(args.output_path, records)
    summary = {
        "records": len(records),
        "per_group": args.per_group,
        "label_counts": dict(Counter(row["metadata"]["group"] for row in records)),
        "skipped": dict(skipped),
        "seed": args.seed,
        "contamination_policy": "exact and near CEDR overlap removed",
    }
    args.output_path.with_suffix(args.output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
