from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rawpara", type=Path, required=True)
    parser.add_argument("--geo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rawpara-per-step", type=int, default=4)
    parser.add_argument("--geo-per-step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2831)
    args = parser.parse_args()

    rawpara = read_jsonl(ROOT / args.rawpara)
    geo = read_jsonl(ROOT / args.geo)
    rng = random.Random(args.seed)
    rng.shuffle(rawpara)

    # Keep Georeview records in source order because the source file is already
    # grouped into SupCon-friendly label-balanced batches.
    needed_rawpara = args.steps * args.rawpara_per_step
    needed_geo = args.steps * args.geo_per_step
    if len(rawpara) < needed_rawpara:
        raise ValueError(f"Not enough raw paraphrase records: {len(rawpara)} < {needed_rawpara}")
    if len(geo) < needed_geo:
        raise ValueError(f"Not enough Georeview records: {len(geo)} < {needed_geo}")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for step in range(args.steps):
        start_raw = step * args.rawpara_per_step
        start_geo = step * args.geo_per_step
        records.extend(rawpara[start_raw : start_raw + args.rawpara_per_step])
        records.extend(geo[start_geo : start_geo + args.geo_per_step])

    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "records": len(records),
        "steps": args.steps,
        "batch_size": args.rawpara_per_step + args.geo_per_step,
        "rawpara_per_step": args.rawpara_per_step,
        "geo_per_step": args.geo_per_step,
        "rawpara_source": str(args.rawpara),
        "geo_source": str(args.geo),
        "seed": args.seed,
    }
    with output.with_name(output.stem + "_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
