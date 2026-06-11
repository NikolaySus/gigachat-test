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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
CONFIG_DIR = ROOT / "configs" / "experiments"
CACHE_DIR = ROOT / "results" / "mteb_cache"
CONTAM_DIR = ROOT / "results" / "contamination" / "cedr_candidates"
RAW_DIR = ROOT / "data" / "raw" / "rusentiment"

CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
GROUPS = ["joy", "sadness", "anger", "fear", "surprise", "no_emotion"]
EMOTION_GROUPS = ["joy", "sadness", "anger", "fear", "surprise"]
GROUP_RU = {
    "joy": "радость",
    "sadness": "грусть",
    "anger": "злость",
    "fear": "страх",
    "surprise": "удивление",
    "no_emotion": "без явной эмоции",
}
RUSENTIMENT_URLS = {
    "preselected": "https://raw.githubusercontent.com/strawberrypie/rusentiment/master/Dataset/rusentiment_preselected_posts.csv",
    "random": "https://raw.githubusercontent.com/strawberrypie/rusentiment/master/Dataset/rusentiment_random_posts.csv",
    "test": "https://raw.githubusercontent.com/strawberrypie/rusentiment/master/Dataset/rusentiment_test.csv",
}

GROUP_LEXICON = {
    "joy": [
        "ахах",
        "ахаха",
        "класс",
        "круто",
        "ура",
        "рад",
        "рада",
        "радость",
        "люблю",
        "обожаю",
        "счаст",
        "весел",
        "улыб",
        "кайф",
        "мил",
        "супер",
        "топ",
        "вау",
    ],
    "sadness": [
        "плачу",
        "плак",
        "рыда",
        "груст",
        "печал",
        "тоск",
        "боль",
        "болит",
        "одинок",
        "устал",
        "устала",
        "жаль",
        "обид",
        "депресс",
        "скуч",
        "слез",
    ],
    "anger": [
        "бесит",
        "злит",
        "злюсь",
        "ненавиж",
        "ярост",
        "гнев",
        "мудак",
        "сука",
        "твар",
        "бля",
        "хуй",
        "пизд",
        "еба",
        "нах",
        "дерьм",
        "говн",
    ],
    "fear": [
        "страш",
        "боюсь",
        "страх",
        "ужас",
        "паник",
        "тревож",
        "тревог",
        "опас",
        "кошмар",
        "жутк",
        "испуг",
    ],
    "surprise": [
        "шок",
        "офиг",
        "охрен",
        "неожидан",
        "удив",
        "серьезно",
        "реально",
        "вау",
        "ничего себе",
        "капец",
    ],
}

EMOJI_GROUPS = {
    "joy": set("😀😃😄😁😆😂🤣😊☺️😍😘😎😋🙂🙃😉👍❤️💖💕🔥"),
    "sadness": set("😢😭😞😔😟🙁☹️💔😿"),
    "anger": set("😡😠🤬👿💢"),
    "fear": set("😱😨😰😥😧😦"),
    "surprise": set("😮😯😲😳🤯"),
}

EMOTICON_RE = re.compile(r"(?:(?:[:;=xX8][-~]?[)D(Pp(/\\])|(?:\){2,})|(?:\({2,}))")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[\w]+", re.U)


def normalize_text(value: Any, *, strip_social: bool = True) -> str:
    text = str(value or "").replace("ё", "е").replace("Ё", "Е")
    text = URL_RE.sub(" ", text)
    if strip_social:
        text = MENTION_RE.sub(" ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_key(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.U)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value: str) -> set[str]:
    return set(TOKEN_RE.findall(normalized_key(value)))


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
            text = normalized_key(row["text"])
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
    normalized = normalized_key(text)
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
        if overlap >= 0.78 and SequenceMatcher(None, normalized, cedr_index["texts"][index]).ratio() >= 0.9:
            return True
    return False


def load_flagged(name: str) -> set[tuple[str, int]]:
    path = CONTAM_DIR / f"{name}_flagged_rows.json"
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    flagged = set()
    for row in rows:
        split = row.get("source_split", row.get("split"))
        index = row.get("source_index", row.get("index"))
        if split is not None and index is not None:
            flagged.add((str(split), int(index)))
    return flagged


def feature_scores(text: str) -> dict[str, float | int | str | dict[str, int]]:
    raw = str(text)
    normalized = normalize_text(raw, strip_social=False)
    readable = normalize_text(raw)
    tokens = TOKEN_RE.findall(readable.lower().replace("ё", "е"))
    cyr = len(re.findall(r"[А-Яа-я]", readable))
    alpha = len(re.findall(r"[A-Za-zА-Яа-я]", readable))
    lexicon_hits: Counter[str] = Counter()
    low = readable.lower()
    for group, words in GROUP_LEXICON.items():
        for word in words:
            if word in low:
                lexicon_hits[group] += 1
    emoji_hits: Counter[str] = Counter()
    for group, symbols in EMOJI_GROUPS.items():
        emoji_hits[group] = sum(raw.count(symbol) for symbol in symbols)
    emoticon_count = len(EMOTICON_RE.findall(raw))
    punctuation = int("!" in raw) + int("?" in raw) + int(bool(re.search(r"[!?]{2,}", raw)))
    informal = sum(lexicon_hits.values()) + sum(emoji_hits.values()) + emoticon_count + punctuation
    return {
        "text": readable,
        "token_count": len(tokens),
        "char_count": len(readable),
        "cyr_ratio": cyr / alpha if alpha else 0.0,
        "url_count": len(URL_RE.findall(raw)),
        "mention_count": len(MENTION_RE.findall(raw)),
        "emoticon_count": emoticon_count,
        "emoji_count": sum(emoji_hits.values()),
        "lexicon_count": sum(lexicon_hits.values()),
        "punctuation_count": punctuation,
        "informal_score": informal,
        "lexicon_hits": dict(lexicon_hits),
        "emoji_hits": dict(emoji_hits),
    }


def choose_group(source_label: str, text: str, features: dict[str, Any]) -> tuple[str | None, float, dict[str, Any]]:
    label = str(source_label).strip().lower()
    votes: Counter[str] = Counter()
    votes.update(features["lexicon_hits"])
    votes.update({group: count * 2 for group, count in features["emoji_hits"].items() if count})

    if label in {"joy", "sadness", "anger", "fear", "surprise"}:
        votes[label] += 3
    elif label in {"positive", "положительный"}:
        votes["joy"] += 2
    elif label in {"negative", "негативный"}:
        # Negative polarity is too broad for CEDR, so keep it only when a sharper signal exists.
        for group in ("anger", "sadness", "fear"):
            votes[group] += 1
    elif label in {"neutral", "нейтральный"}:
        if sum(votes.values()) == 0:
            return "no_emotion", 0.72, {"votes": dict(votes), "source_label": source_label}

    if not votes:
        return None, 0.0, {"votes": {}, "source_label": source_label}
    group, score = votes.most_common(1)[0]
    if group == "surprise" and label == "positive" and score < 3:
        group = "joy"
    confidence = min(0.98, 0.45 + score * 0.12)
    return group, confidence, {"votes": dict(votes), "source_label": source_label}


def quality_ok(text: str, features: dict[str, Any], *, keep_informal: bool) -> bool:
    if features["char_count"] < 24 or features["char_count"] > 220:
        return False
    if features["token_count"] < 4 or features["token_count"] > 40:
        return False
    if features["cyr_ratio"] < 0.6:
        return False
    if features["url_count"] and features["token_count"] < 7:
        return False
    if features["mention_count"] >= 3 and features["token_count"] < 8:
        return False
    if re.search(r"(.)\1{6,}", normalize_text(text, strip_social=False)):
        return False
    if keep_informal and features["informal_score"] > 0:
        return True
    return features["token_count"] >= 5


def add_candidate(
    rows: list[dict[str, Any]],
    seen: set[str],
    *,
    source_dataset: str,
    split: str,
    index: int,
    text: str,
    label: str,
    cedr_index: dict[str, Any],
    keep_informal: bool,
    skipped: Counter,
) -> None:
    display_text = normalize_text(text)
    key = normalized_key(display_text)
    if not key or key in seen:
        skipped["duplicate_or_empty"] += 1
        return
    features = feature_scores(text)
    if not quality_ok(text, features, keep_informal=keep_informal):
        skipped["quality"] += 1
        return
    if is_contaminated(display_text, cedr_index):
        skipped["cedr_overlap"] += 1
        return
    group, confidence, guide = choose_group(label, display_text, features)
    if group is None:
        skipped["unmapped_or_low_signal"] += 1
        return
    if group != "no_emotion" and confidence < 0.57:
        skipped["low_confidence"] += 1
        return
    seen.add(key)
    style = "informal" if features["informal_score"] > 0 else "ordinary"
    rows.append(
        {
            "source_dataset": source_dataset,
            "split": split,
            "index": index,
            "text": display_text,
            "group": group,
            "source_label": label,
            "confidence": round(confidence, 6),
            "style": style,
            "quality_score": round(
                confidence
                + min(1.0, features["informal_score"] / 4.0) * 0.35
                + min(1.0, features["token_count"] / 16.0) * 0.15,
                6,
            ),
            "features": {key: value for key, value in features.items() if key != "text"},
            "guide": guide,
        }
    )


def download_rusentiment_file(name: str, url: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"rusentiment_{name}.csv"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return path


def load_primary_sources(cedr_index: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = Counter()

    rusentitweet = load_dataset("psytechlab/RuSentiTweet", cache_dir=str(CACHE_DIR))
    for split, ds in rusentitweet.items():
        for index, row in enumerate(ds):
            add_candidate(
                rows,
                seen,
                source_dataset="psytechlab/RuSentiTweet",
                split=split,
                index=index,
                text=row["text"],
                label=row["label"],
                cedr_index=cedr_index,
                keep_informal=True,
                skipped=skipped,
            )

    for split_name, url in RUSENTIMENT_URLS.items():
        path = download_rusentiment_file(split_name, url)
        text = path.read_text(encoding="utf-8-sig")
        for index, row in enumerate(csv.DictReader(io.StringIO(text))):
            add_candidate(
                rows,
                seen,
                source_dataset="text-machine-lab/rusentiment",
                split=split_name,
                index=index,
                text=row.get("text", ""),
                label=row.get("label", ""),
                cedr_index=cedr_index,
                keep_informal=True,
                skipped=skipped,
            )

    meta = {
        "rows": len(rows),
        "skipped": dict(skipped),
        "by_group": dict(Counter(row["group"] for row in rows)),
        "by_source": dict(Counter(row["source_dataset"] for row in rows)),
        "by_style": dict(Counter(row["style"] for row in rows)),
    }
    return rows, meta


def add_secondary_rows(
    rows: list[dict[str, Any]],
    cedr_index: dict[str, Any],
    *,
    source_names: list[str],
    cap_per_source: int,
) -> dict[str, Any]:
    seen = {normalized_key(row["text"]) for row in rows}
    skipped = Counter()
    added_by_source = Counter()

    def maybe_add(source: str, split: str, index: int, text: str, label: str) -> None:
        before = len(rows)
        add_candidate(
            rows,
            seen,
            source_dataset=source,
            split=split,
            index=index,
            text=text,
            label=label,
            cedr_index=cedr_index,
            keep_informal=True,
            skipped=skipped,
        )
        if len(rows) > before:
            added_by_source[source] += 1

    if "brighter" in source_names:
        flagged = load_flagged("brighter")
        dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", cache_dir=str(CACHE_DIR))
        label_names = ["anger", "fear", "joy", "sadness", "surprise"]
        for split, ds in dataset.items():
            for index, row in enumerate(ds):
                if (split, index) in flagged:
                    skipped["brighter_flagged"] += 1
                    continue
                labels = [label for label in label_names if int(row.get(label, 0)) == 1]
                if len(labels) == 1:
                    maybe_add("brighter-dataset/BRIGHTER-emotion-categories:rus", split, index, row["text"], labels[0])

    if "ruemotions" in source_names:
        flagged = load_flagged("ruemotions")
        ruemotions_map = {
            "радость": "joy",
            "восторг": "joy",
            "восхищение": "joy",
            "веселье": "joy",
            "смех": "joy",
            "любовь": "joy",
            "грусть": "sadness",
            "печаль": "sadness",
            "тоска": "sadness",
            "одиночество": "sadness",
            "злость": "anger",
            "гнев": "anger",
            "ярость": "anger",
            "ненависть": "anger",
            "обида": "anger",
            "раздражение": "anger",
            "удивление": "surprise",
            "шок": "surprise",
            "страх": "fear",
            "тревога": "fear",
            "ужас": "fear",
            "паника": "fear",
        }
        dataset = load_dataset("Darkester/RuEmotions", cache_dir=str(CACHE_DIR))
        for split, ds in dataset.items():
            for index, row in enumerate(ds):
                if (split, index) in flagged:
                    skipped["ruemotions_flagged"] += 1
                    continue
                label = ruemotions_map.get(str(row["emotion"]).strip().lower())
                if label:
                    maybe_add("Darkester/RuEmotions", split, index, row["text"], label)

    if "twitter" in source_names:
        flagged = load_flagged("twitter_emotions_ekman")
        dataset = load_dataset("AiLab-IMCS-UL/twitter_emotions-ru", "simplified_ekman", cache_dir=str(CACHE_DIR))
        for split, ds in dataset.items():
            label_feature = ds.features["labels_ekman"]
            for index, row in enumerate(ds):
                if added_by_source["AiLab-IMCS-UL/twitter_emotions-ru"] >= cap_per_source:
                    break
                if (split, index) in flagged:
                    skipped["twitter_flagged"] += 1
                    continue
                label = label_feature.int2str(row["labels_ekman"])
                if label in EMOTION_GROUPS:
                    maybe_add("AiLab-IMCS-UL/twitter_emotions-ru", split, index, row["ru_text"], label)

    if "go_ekman" in source_names:
        flagged = load_flagged("go_ekman")
        dataset = load_dataset("SkyWater21/ru_go_emotions_ekman", "simplified_ekman", cache_dir=str(CACHE_DIR))
        label_names = dataset["train"].features["labels_ekman"].feature.names
        for split, ds in dataset.items():
            for index, row in enumerate(ds):
                if added_by_source["SkyWater21/ru_go_emotions_ekman"] >= cap_per_source:
                    break
                if (split, index) in flagged:
                    skipped["go_ekman_flagged"] += 1
                    continue
                labels = [label_names[label] for label in row["labels_ekman"] if label_names[label] in EMOTION_GROUPS]
                if len(labels) == 1:
                    maybe_add("SkyWater21/ru_go_emotions_ekman", split, index, row["ru_text"], labels[0])

    return {
        "skipped": dict(skipped),
        "added_by_source": dict(added_by_source),
        "total_rows_after_secondary": len(rows),
    }


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    mode: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    if mode == "polarity":
        base_targets = {"joy": 360, "anger": 260, "sadness": 220, "fear": 120, "surprise": 80, "no_emotion": 160}
    elif mode == "informal":
        base_targets = {"joy": 450, "anger": 340, "sadness": 300, "fear": 180, "surprise": 130, "no_emotion": 200}
    else:
        base_targets = {"joy": 420, "anger": 300, "sadness": 290, "fear": 180, "surprise": 150, "no_emotion": 260}
    base_total = sum(base_targets.values())
    targets = {group: math.floor(count * value / base_total) for group, value in base_targets.items()}
    remainder = count - sum(targets.values())
    order = sorted(
        base_targets,
        key=lambda group: (count * base_targets[group] / base_total) - targets[group],
        reverse=True,
    )
    for group in order[:remainder]:
        targets[group] += 1
    if sum(targets.values()) != count:
        raise ValueError(f"{mode}: target count mismatch")

    selected = []
    for group, target in targets.items():
        pool = by_group[group][:]
        if mode == "informal":
            pool.sort(key=lambda row: (row["style"] == "informal", row["quality_score"], row["confidence"]), reverse=True)
        else:
            pool.sort(key=lambda row: (row["quality_score"], row["style"] == "informal", row["confidence"]), reverse=True)
        top = pool[: max(target * 4, target)]
        rng.shuffle(top)
        top.sort(key=lambda row: (row["quality_score"], row["style"] == "informal"), reverse=True)
        if len(top) < target:
            raise ValueError(f"{mode}: need {target} rows for {group}, got {len(top)}")
        selected.extend(top[:target])
    rng.shuffle(selected)
    return selected


def row_id(row: dict[str, Any]) -> str:
    return f"{row['source_dataset']}::{row['split']}::{row['index']}"


def mine_contrastive_records(
    selected: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    *,
    source_name: str,
    seed: int,
    episodic: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    pool = sorted(all_rows, key=lambda row: row["quality_score"], reverse=True)[:40000]
    selected_ids = {row_id(row) for row in selected}
    for row in selected:
        if row_id(row) not in {row_id(item) for item in pool}:
            pool.append(row)

    texts = [row["text"] for row in pool]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    labels = [row["group"] for row in pool]
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    index_by_id = {row_id(row): idx for idx, row in enumerate(pool)}
    selected_indices = [index_by_id[row_id(row)] for row in selected if row_id(row) in index_by_id]
    neighbors = NearestNeighbors(n_neighbors=min(len(pool), 260), metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix[selected_indices], return_distance=True)

    records = []
    stats = Counter()
    positive_sims = []
    negative_sims = []
    for item_idx, item_distances, item_neighbors in zip(selected_indices, distances, indices, strict=True):
        item = pool[item_idx]
        same = []
        for distance, idx in zip(item_distances, item_neighbors, strict=True):
            idx = int(idx)
            sim = float(1.0 - distance)
            if idx == item_idx or labels[idx] != item["group"]:
                continue
            if pool[idx]["text"] == item["text"]:
                continue
            if not (0.06 <= sim <= 0.70):
                continue
            if abs(pool[idx]["confidence"] - item["confidence"]) > 0.35:
                continue
            same.append((idx, sim))
        if not same:
            relaxed_same = []
            for distance, idx in zip(item_distances, item_neighbors, strict=True):
                idx = int(idx)
                sim = float(1.0 - distance)
                if idx != item_idx and labels[idx] == item["group"] and pool[idx]["text"] != item["text"] and sim <= 0.82:
                    relaxed_same.append((idx, sim))
            if relaxed_same:
                same = relaxed_same
                stats["positive_relaxed"] += 1
            else:
                fallback = [idx for idx in by_label[item["group"]] if idx != item_idx and pool[idx]["text"] != item["text"]]
                if fallback:
                    fallback.sort(key=lambda idx: pool[idx]["quality_score"], reverse=True)
                    same = [(rng.choice(fallback[: min(32, len(fallback))]), 0.0)]
                    stats["positive_random_fallback"] += 1
        if not same:
            stats["no_positive"] += 1
            continue
        same.sort(key=lambda pair: (pair[1], pool[pair[0]]["quality_score"]), reverse=True)
        pos_idx, pos_sim = same[min(len(same) - 1, rng.randrange(min(3, len(same))))]

        negatives = []
        for distance, idx in zip(item_distances, item_neighbors, strict=True):
            idx = int(idx)
            sim = float(1.0 - distance)
            if labels[idx] == item["group"] or labels[idx] == "no_emotion" == item["group"]:
                continue
            if sim > max(0.82, pos_sim + 0.18):
                stats["false_negative_guard"] += 1
                continue
            if sim < 0.025:
                continue
            negatives.append((idx, sim))
        negatives.sort(key=lambda pair: (pair[1], pool[pair[0]]["quality_score"]), reverse=True)

        picked = []
        used_labels = set()
        for idx, sim in negatives:
            label = labels[idx]
            if label in used_labels and len(used_labels) < 4:
                continue
            picked.append((idx, sim))
            used_labels.add(label)
            if len(picked) >= 5:
                break
        if len(picked) < 4:
            stats["too_few_negatives"] += 1
            continue

        positive_sims.append(pos_sim)
        negative_sims.extend(sim for _, sim in picked)
        query_text = item["text"]
        positive_text = pool[pos_idx]["text"]
        if episodic:
            group_name = GROUP_RU[item["group"]]
            query_text = f"Комментарий с эмоциональным сигналом `{group_name}`: {query_text}"
            positive_text = f"Другой комментарий с похожим эмоциональным сигналом `{group_name}`: {positive_text}"
        records.append(
            {
                "source": source_name,
                "objective": "contrastive",
                "query": CEDR_PREFIX + query_text,
                "positive": CEDR_PREFIX + positive_text,
                "negatives": [CEDR_PREFIX + pool[idx]["text"] for idx, _ in picked],
                "metadata": {
                    "group": item["group"],
                    "source_dataset": item["source_dataset"],
                    "split": item["split"],
                    "index": item["index"],
                    "style": item["style"],
                    "confidence": item["confidence"],
                    "quality_score": item["quality_score"],
                    "positive_group": pool[pos_idx]["group"],
                    "positive_source_dataset": pool[pos_idx]["source_dataset"],
                    "positive_similarity": round(pos_sim, 6),
                    "negative_groups": [pool[idx]["group"] for idx, _ in picked],
                    "negative_similarities": [round(sim, 6) for _, sim in picked],
                    "features": item["features"],
                },
            }
        )

    rng.shuffle(records)
    summary = {
        "records": len(records),
        "selected_ids": len(selected_ids),
        "selected_by_group": dict(Counter(record["metadata"]["group"] for record in records)),
        "selected_by_source": dict(Counter(record["metadata"]["source_dataset"] for record in records)),
        "selected_by_style": dict(Counter(record["metadata"]["style"] for record in records)),
        "positive_similarity_mean": sum(positive_sims) / len(positive_sims) if positive_sims else None,
        "negative_similarity_mean": sum(negative_sims) / len(negative_sims) if negative_sims else None,
        "stats": dict(stats),
        "episodic": episodic,
    }
    return records, summary


def sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Need {count} records, got {len(records)}")
    rows = records[:]
    random.Random(seed).shuffle(rows)
    return rows[:count]


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
        "cedr_sentiment_mined_v2": addon_records,
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
        },
    )
    write_json(
        CONFIG_DIR / f"exp01r_nc_{name}_4096_eager_frozenrepro.json",
        {
            "name": f"exp01r_nc_{name}_4096_eager_frozenrepro",
            "description": f"CEDR sentiment-mined-v2 ablation: {name}",
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


def build_variant(
    *,
    name: str,
    mode: str,
    count: int,
    seed: int,
    include_secondary: bool,
    episodic: bool,
) -> None:
    cedr_index = load_cedr_index()
    primary_rows, primary_meta = load_primary_sources(cedr_index)
    rows = primary_rows[:]
    secondary_meta = {}
    if include_secondary:
        secondary_meta = add_secondary_rows(
            rows,
            cedr_index,
            source_names=["brighter", "ruemotions", "twitter", "go_ekman"],
            cap_per_source=12000,
        )
    selected_count = math.ceil(count * 1.18)
    selected = select_candidates(rows, count=selected_count, seed=seed, mode=mode)
    records, mining_summary = mine_contrastive_records(
        selected,
        rows,
        source_name=f"cedr_sentiment_mined_v2:{name}",
        seed=seed + 19,
        episodic=episodic,
    )
    if len(records) < count:
        raise ValueError(f"{name}: only mined {len(records)} records from requested {count}")
    before_trim = len(records)
    records = records[:count]
    mining_summary["records_before_trim"] = before_trim
    mining_summary["records"] = len(records)
    mining_summary["selected_by_group"] = dict(Counter(record["metadata"]["group"] for record in records))
    mining_summary["selected_by_source"] = dict(Counter(record["metadata"]["source_dataset"] for record in records))
    mining_summary["selected_by_style"] = dict(Counter(record["metadata"]["style"] for record in records))
    mining_summary["positive_similarity_mean"] = (
        sum(record["metadata"]["positive_similarity"] for record in records) / len(records) if records else None
    )
    all_negative_sims = [sim for record in records for sim in record["metadata"]["negative_similarities"]]
    mining_summary["negative_similarity_mean"] = (
        sum(all_negative_sims) / len(all_negative_sims) if all_negative_sims else None
    )
    component_path = DATA_DIR / f"open_ru_1r_nc_{name}_component.jsonl"
    write_jsonl(component_path, records)
    summary = {
        "construction": "slang/emoticon-preserving CEDR sentiment mining with semi-hard cross-label negatives",
        "name": name,
        "mode": mode,
        "requested_count": count,
        "used_count": len(records),
        "include_secondary": include_secondary,
        "primary_loader": primary_meta,
        "secondary_loader": secondary_meta,
        "mining": mining_summary,
        "component_path": str(component_path.relative_to(ROOT)),
        "contamination_policy": "exact and near CEDR overlap removed before mining; no CEDR-trained teacher used",
        "slang_emoticon_policy": "emoji, emoticons, slang, profanity, and expressive punctuation increase preservation score when text is readable",
    }
    write_json(component_path.with_name(component_path.stem + "_summary.json"), summary)
    make_mix(name, records, summary, seed=seed)
    print(f"prepared {name}: {len(records)} addon rows")


def main() -> None:
    build_variant(
        name="cedr_sentiment_mined_v2_polarity_1200",
        mode="polarity",
        count=1200,
        seed=411,
        include_secondary=False,
        episodic=False,
    )
    build_variant(
        name="cedr_sentiment_mined_v2_informal_1600",
        mode="informal",
        count=1600,
        seed=421,
        include_secondary=False,
        episodic=False,
    )
    build_variant(
        name="cedr_sentiment_mined_v2_mixed_1600",
        mode="mixed",
        count=1600,
        seed=431,
        include_secondary=True,
        episodic=True,
    )


if __name__ == "__main__":
    main()
