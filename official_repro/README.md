# Official RuMTEB Reproduction Attempt

This folder isolates an evaluation path for reproducing the public `mteb/results`
entry for `ai-sage/Giga-Embeddings-instruct`.

Pinned target:

- model: `ai-sage/Giga-Embeddings-instruct`
- model revision: `40b27667b9ad586d7812675df76e5062ccc80b0e`
- MTEB version: `1.38.30`
- benchmark: `MTEB(rus, v1)`

The working isolated environment currently uses the versions in
`official_repro/requirements.txt`. `datasets==2.21.0` is intentional: newer
`datasets` releases reject or break several script-based datasets still used by
MTEB 1.38 Russian tasks.

Run from the repository root:

```bash
official_repro/.venv/bin/python official_repro/run_official_rumteb.py \
  --output-folder results/official_repro/giga_mteb138_rus_v1 \
  --batch-size 12 \
  --max-length 4096 \
  --attn-implementation eager \
  --overwrite-results
```

Compare against the public result files:

```bash
official_repro/.venv/bin/python official_repro/compare_official_rumteb.py \
  results/official_repro/giga_mteb138_rus_v1 \
  --write-md results/official_repro/giga_mteb138_rus_v1_comparison.md
```

For a quick smoke test, pass one or more task names:

```bash
official_repro/.venv/bin/python official_repro/run_official_rumteb.py \
  --output-folder results/official_repro/giga_mteb138_smoke_terra \
  --tasks TERRa \
  --batch-size 12 \
  --max-length 4096 \
  --attn-implementation eager \
  --overwrite-results
```
