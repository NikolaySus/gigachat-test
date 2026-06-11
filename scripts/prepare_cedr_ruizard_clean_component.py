from __future__ import annotations

import argparse
from pathlib import Path

from prepare_open_ru_1r_nc_mixh_habrfull_ruizard import (
    DATA_DIR,
    ROOT,
    build_ruizard_records,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean RuIzard CEDR-compatible emotion component.")
    parser.add_argument("--count", type=int, default=6400)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--name", default="cedr_ruizard_clean_emotion_contrastive_6400")
    args = parser.parse_args()

    records, summary = build_ruizard_records(count=args.count, seed=args.seed, ignore_overlap=False)
    path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(path, records)
    write_json(path.with_name(path.stem + "_summary.json"), summary | {"name": args.name})
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
