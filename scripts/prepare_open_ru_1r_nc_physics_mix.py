from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


INVALID_JSON_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")
JSON_STRING_RE = r"((?:\\.|[^\"\\])*)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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


def parse_text_field(raw: str) -> str:
    try:
        parsed = json.loads(INVALID_JSON_ESCAPE_RE.sub(r"\\\\", raw))
        if isinstance(parsed, dict) and isinstance(parsed.get("text"), list):
            parts = [
                str(item.get("text", "")).strip()
                for item in parsed["text"]
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            return " ".join(parts)
    except Exception:
        pass
    return raw.strip()


def parse_answer_field(raw: str) -> str:
    sanitized = INVALID_JSON_ESCAPE_RE.sub(r"\\\\", raw)
    try:
        parsed = json.loads(sanitized)
        if isinstance(parsed, list):
            parts = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                condition = str(item.get("condition", "")).strip()
                solution = item.get("solution", {})
                if isinstance(solution, dict):
                    solution_text = str(solution.get("text", "")).strip()
                else:
                    solution_text = str(solution).strip()
                text = " ".join(part for part in (condition, solution_text) if part)
                if text:
                    parts.append(text)
            return " ".join(parts)
    except Exception:
        pass
    parts = []
    for pattern in (
        rf'"condition"\s*:\s*"{JSON_STRING_RE}"',
        rf'"solution"\s*:\s*\{{\s*"text"\s*:\s*"{JSON_STRING_RE}"',
        rf'"answers"\s*:\s*\[\s*"{JSON_STRING_RE}"',
    ):
        for match in re.finditer(pattern, sanitized, flags=re.DOTALL):
            value = match.group(1)
            try:
                value = json.loads(f'"{value}"')
            except Exception:
                pass
            value = str(value).strip()
            if value:
                parts.append(value)
    if parts:
        return " ".join(parts)
    return raw.strip()


def parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed]
        return [str(parsed)]
    except Exception:
        return [str(raw)]


def make_physics_records(*, seed: int, target_count: int, min_problem_chars: int, min_answer_chars: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    dataset = load_dataset("Vikhrmodels/physics_big", split="train")
    base_records: list[dict[str, Any]] = []
    for row in dataset:
        problem = parse_text_field(row.get("text") or "")
        answer = parse_answer_field(row.get("answer") or "")
        tags = parse_tags(row.get("tags"))
        if len(problem) < min_problem_chars or len(answer) < min_answer_chars:
            continue
        base_records.append(
            {
                "source": "Vikhrmodels/physics_big",
                "query": (
                    "Instruct: Given a physics problem, retrieve the solution or a closely related physics problem\n"
                    f"Query: {problem}"
                ),
                "positive": answer,
                "negatives": [],
                "metadata": {
                    "tags": tags,
                    "problem_chars": len(problem),
                    "answer_chars": len(answer),
                },
                "objective": "contrastive",
            }
        )
    if not base_records:
        raise RuntimeError("No physics_big records passed filtering")

    rng.shuffle(base_records)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_count:
        selected.extend(base_records)
    selected = selected[:target_count]
    rng.shuffle(selected)
    return selected


def sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Need {count} records, got {len(records)}")
    records = records[:]
    random.Random(seed).shuffle(records)
    return records[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 1R-NC GeRaCl:Habr:DeepVK:physics mix.")
    parser.add_argument("--geracl-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument(
        "--habr-path",
        type=Path,
        default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"),
    )
    parser.add_argument(
        "--deepvk-path",
        type=Path,
        default=Path("data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"),
    )
    parser.add_argument(
        "--physics-out",
        type=Path,
        default=Path("data/contrastive/open_ru_1r_nc_physics_big_problem_solution_3200.jsonl"),
    )
    parser.add_argument(
        "--mix-out",
        type=Path,
        default=Path("data/contrastive/open_ru_1r_nc_mixe_geracl2_habr1_deepvk1_physics1_16000.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--physics-count", type=int, default=3200)
    parser.add_argument("--min-problem-chars", type=int, default=80)
    parser.add_argument("--min-answer-chars", type=int, default=120)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    physics = make_physics_records(
        seed=args.seed,
        target_count=args.physics_count,
        min_problem_chars=args.min_problem_chars,
        min_answer_chars=args.min_answer_chars,
    )
    write_jsonl(args.physics_out, physics)

    geracl = read_jsonl(args.geracl_path)
    habr = read_jsonl(args.habr_path)
    deepvk = read_jsonl(args.deepvk_path)

    selected = {
        "geracl": sample(geracl, count=6400, seed=args.seed),
        "habr_harder": sample(habr, count=3200, seed=args.seed + 1),
        "deepvk_filtered": sample(deepvk, count=3200, seed=args.seed + 2),
        "physics_big": physics,
    }
    mixed = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(args.seed + 3).shuffle(mixed)
    write_jsonl(args.mix_out, mixed)

    summary_path = args.summary_out or args.mix_out.with_name(args.mix_out.stem + "_summary.json")
    write_json(
        summary_path,
        {
            "output": str(args.mix_out),
            "seed": args.seed,
            "ratio": {
                "geracl": 2,
                "habr_harder": 1,
                "deepvk_ru_hnp_filtered": 1,
                "physics_big": 1,
            },
            "counts": {
                "geracl_source": len(geracl),
                "geracl_used": len(selected["geracl"]),
                "habr_harder_source": len(habr),
                "habr_harder_used": len(selected["habr_harder"]),
                "deepvk_filtered_source": len(deepvk),
                "deepvk_used": len(selected["deepvk_filtered"]),
                "physics_used": len(selected["physics_big"]),
                "total": len(mixed),
            },
            "batch_size": 4,
            "max_steps_1x": len(mixed) // 4,
            "source_paths": {
                "geracl": str(args.geracl_path),
                "habr_harder": str(args.habr_path),
                "deepvk_filtered": str(args.deepvk_path),
                "physics_big_hf": "Vikhrmodels/physics_big",
                "physics_prepared": str(args.physics_out),
            },
            "physics_filter": {
                "min_problem_chars": args.min_problem_chars,
                "min_answer_chars": args.min_answer_chars,
                "note": "Physics examples are problem-to-solution contrastive pairs with in-batch negatives only.",
            },
        },
    )
    print(f"Wrote {args.physics_out}")
    print(f"Wrote {args.mix_out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
