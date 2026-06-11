from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
CHECKPOINT_ROOT = ROOT / "experiments" / "exp01_reinit_fair" / "checkpoints"
RESULTS_ROOT = ROOT / "results" / "official_repro"

CLEAN_BASE = (
    "experiments/exp01_reinit_fair/checkpoints/"
    "open_ru_1r_nc_mixh_habrfull_plus_geracl_remaining_4096_eager_frozenrepro/latest.pt"
)
GO9000 = "data/contrastive/open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl"
LENTA_NEUTRAL = "data/contrastive/open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl"
MIXH_HABRFULL = (
    "data/contrastive/"
    "open_ru_1r_nc_mixh_habrfull_geracl6400_habr4369_deepvk3200_grandmaster3200_17169.jsonl"
)
HABR_DEEPVK = "data/contrastive/cedr_a050_recovery_habr1x_deepvk1x_11977.jsonl"
RUSTS_32K = "data/contrastive/rusts_external_cointegrated_diverse_32000.jsonl"
CEDR_SETFIT_CONTRASTIVE = "data/contrastive/open_ru_1r_nc_cedr_setfit_proto_v1_contrastive.jsonl"
CEDR_SETFIT_PROTOTYPE = "data/contrastive/open_ru_1r_nc_cedr_setfit_proto_v1_prototype.jsonl"
CEDR_SETFIT_KNN = "data/contrastive/open_ru_1r_nc_cedr_setfit_proto_v1_knn.jsonl"
CEDR_SETFIT_MIXED = "data/contrastive/open_ru_1r_nc_cedr_setfit_proto_v1_mixed.jsonl"
CEDR_LABELED_SUPCON = "data/contrastive/open_ru_1r_nc_cedr_labeled_metric_v1_supcon.jsonl"
CEDR_LABELED_CIRCLE = "data/contrastive/open_ru_1r_nc_cedr_labeled_metric_v1_circle.jsonl"
CEDR_LABELED_MS = "data/contrastive/open_ru_1r_nc_cedr_labeled_metric_v1_multi_similarity.jsonl"
CEDR_MULTILABEL_SUPPORT_P4 = "data/contrastive/open_ru_1r_nc_cedr_multilabel_support_v1_p4_7100.jsonl"
CEDR_MULTILABEL_SUPPORT_P8 = "data/contrastive/open_ru_1r_nc_cedr_multilabel_support_v1_p8_7100.jsonl"
NO_GERACL_DIRECTION = (
    "experiments/exp01_reinit_fair/checkpoints/"
    "mixh_habrfull_leave1out_no_geracl_4096_eager_frozenrepro/latest.pt"
)

GATE5_TASKS = [
    "CEDRClassification",
    "GeoreviewClassification",
    "RuSciBenchOECDClassification",
    "RuSTSBenchmarkSTS",
    "GeoreviewClusteringP2P",
]

OFFICIAL = {
    "avg": 0.657981,
    "CEDRClassification": 0.685069,
    "GeoreviewClassification": 0.547510,
    "RuSciBenchOECDClassification": 0.545068,
    "RuSTSBenchmarkSTS": 0.835900,
    "GeoreviewClusteringP2P": 0.676357,
}


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    stages: list[dict]


VARIANTS = {
    "cedr_go600_lenta100": Variant(
        name="cedr_go600_lenta100",
        description="Clean-base CEDR correction: GoEmotions-RU neutral prior, then strict Lenta neutral boundary.",
        stages=[
            {
                "name": "go9000_neutral404",
                "data_path": GO9000,
                "max_steps": 600,
                "batch_size": 3,
                "learning_rate": 1e-6,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "save_every": 300,
            },
            {
                "name": "lenta_reported_neutral",
                "data_path": LENTA_NEUTRAL,
                "max_steps": 100,
                "batch_size": 3,
                "learning_rate": 2e-7,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "rehearsal_loss_weight": 1.0,
                "save_every": 50,
            },
        ],
    ),
    "cedr_go900_lenta150": Variant(
        name="cedr_go900_lenta150",
        description="Longer clean-base CEDR correction with the same two CEDR-focused stages.",
        stages=[
            {
                "name": "go9000_neutral404",
                "data_path": GO9000,
                "max_steps": 900,
                "batch_size": 3,
                "learning_rate": 1e-6,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "save_every": 300,
            },
            {
                "name": "lenta_reported_neutral",
                "data_path": LENTA_NEUTRAL,
                "max_steps": 150,
                "batch_size": 3,
                "learning_rate": 2e-7,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "rehearsal_loss_weight": 1.0,
                "save_every": 50,
            },
        ],
    ),
    "cedr_go900_lenta150_broad150": Variant(
        name="cedr_go900_lenta150_broad150",
        description="CEDR correction followed by low-LR broad Mix-H repair, all through optimizer updates.",
        stages=[
            {
                "name": "go9000_neutral404",
                "data_path": GO9000,
                "max_steps": 900,
                "batch_size": 3,
                "learning_rate": 1e-6,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "save_every": 300,
            },
            {
                "name": "lenta_reported_neutral",
                "data_path": LENTA_NEUTRAL,
                "max_steps": 150,
                "batch_size": 3,
                "learning_rate": 2e-7,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "rehearsal_loss_weight": 1.0,
                "save_every": 50,
            },
            {
                "name": "mixh_broad_repair",
                "data_path": MIXH_HABRFULL,
                "max_steps": 150,
                "batch_size": 2,
                "learning_rate": 5e-8,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "rehearsal_loss_weight": 0.5,
                "save_every": 50,
            },
        ],
    ),
    "cedr_go600_lenta100_sts600": Variant(
        name="cedr_go600_lenta100_sts600",
        description="Short CEDR correction followed by light STS repair with strong retention.",
        stages=[
            {
                "name": "go9000_neutral404",
                "data_path": GO9000,
                "max_steps": 600,
                "batch_size": 3,
                "learning_rate": 1e-6,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "save_every": 300,
            },
            {
                "name": "lenta_reported_neutral",
                "data_path": LENTA_NEUTRAL,
                "max_steps": 100,
                "batch_size": 3,
                "learning_rate": 2e-7,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "rehearsal_loss_weight": 1.0,
                "save_every": 50,
            },
            {
                "name": "rusts_pair_repair",
                "data_path": RUSTS_32K,
                "max_steps": 600,
                "batch_size": 4,
                "learning_rate": 3e-6,
                "weight_decay": 0.001,
                "temperature": 0.02,
                "pair_score_loss_weight": 5.0,
                "parameter_anchor_weight": 1.0,
                "rehearsal_loss_weight": 0.75,
                "save_every": 300,
            },
        ],
    ),
    "cedr_go600_lenta100_habrdeepvk800": Variant(
        name="cedr_go600_lenta100_habrdeepvk800",
        description="Short CEDR correction followed by Habr+DeepVK broad recovery, no arithmetic.",
        stages=[
            {
                "name": "go9000_neutral404",
                "data_path": GO9000,
                "max_steps": 600,
                "batch_size": 3,
                "learning_rate": 1e-6,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "save_every": 300,
            },
            {
                "name": "lenta_reported_neutral",
                "data_path": LENTA_NEUTRAL,
                "max_steps": 100,
                "batch_size": 3,
                "learning_rate": 2e-7,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "rehearsal_loss_weight": 1.0,
                "save_every": 50,
            },
            {
                "name": "habr_deepvk_repair",
                "data_path": HABR_DEEPVK,
                "max_steps": 800,
                "batch_size": 2,
                "learning_rate": 2e-6,
                "temperature": 0.02,
                "parameter_anchor_weight": 1.0,
                "rehearsal_loss_weight": 0.75,
                "save_every": 400,
            },
        ],
    ),
    "negdir_lenta150_w001": Variant(
        name="negdir_lenta150_w001",
        description=(
            "Train-only direction probe: strict Lenta CEDR boundary while moving away from the "
            "no-GeRaCl trained direction. This is an optimizer loss term, not weight arithmetic."
        ),
        stages=[
            {
                "name": "lenta_negdir",
                "data_path": LENTA_NEUTRAL,
                "max_steps": 150,
                "batch_size": 3,
                "learning_rate": 1e-7,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "rehearsal_loss_weight": 1.0,
                "negative_direction_weight": 0.001,
                "save_every": 50,
            },
        ],
    ),
    "negdir_lenta150_w01": Variant(
        name="negdir_lenta150_w01",
        description=(
            "Stronger train-only direction probe away from no-GeRaCl, with the same Lenta CEDR "
            "boundary objective and clean-base anchor."
        ),
        stages=[
            {
                "name": "lenta_negdir",
                "data_path": LENTA_NEUTRAL,
                "max_steps": 150,
                "batch_size": 3,
                "learning_rate": 1e-7,
                "temperature": 0.02,
                "parameter_anchor_weight": 10.0,
                "rehearsal_loss_weight": 1.0,
                "negative_direction_weight": 0.1,
                "save_every": 50,
            },
        ],
    ),
    "setfit_contrastive600": Variant(
        name="setfit_contrastive600",
        description="CEDR SetFit-style hard-negative contrastive stage from the clean base.",
        stages=[
            {
                "name": "setfit_hardneg",
                "data_path": CEDR_SETFIT_CONTRASTIVE,
                "max_steps": 600,
                "batch_size": 4,
                "learning_rate": 2e-6,
                "temperature": 0.02,
                "parameter_anchor_weight": 1.0,
                "rehearsal_loss_weight": 0.5,
                "save_every": 200,
            },
        ],
    ),
    "prototype300": Variant(
        name="prototype300",
        description="CEDR prototype-classification episodes from the clean base.",
        stages=[
            {
                "name": "prototype",
                "data_path": CEDR_SETFIT_PROTOTYPE,
                "max_steps": 300,
                "batch_size": 2,
                "learning_rate": 5e-7,
                "temperature": 0.05,
                "parameter_anchor_weight": 5.0,
                "rehearsal_loss_weight": 0.5,
                "save_every": 100,
            },
        ],
    ),
    "knn300": Variant(
        name="knn300",
        description="CEDR kNN-style class support episodes from the clean base.",
        stages=[
            {
                "name": "knn",
                "data_path": CEDR_SETFIT_KNN,
                "max_steps": 300,
                "batch_size": 2,
                "learning_rate": 5e-7,
                "temperature": 0.05,
                "parameter_anchor_weight": 5.0,
                "rehearsal_loss_weight": 0.5,
                "save_every": 100,
            },
        ],
    ),
    "setfit_then_proto": Variant(
        name="setfit_then_proto",
        description="Hard-negative SetFit contrastive stage followed by compact prototype episodes.",
        stages=[
            {
                "name": "setfit_hardneg",
                "data_path": CEDR_SETFIT_CONTRASTIVE,
                "max_steps": 400,
                "batch_size": 4,
                "learning_rate": 2e-6,
                "temperature": 0.02,
                "parameter_anchor_weight": 1.0,
                "rehearsal_loss_weight": 0.5,
                "save_every": 200,
            },
            {
                "name": "prototype",
                "data_path": CEDR_SETFIT_PROTOTYPE,
                "max_steps": 200,
                "batch_size": 2,
                "learning_rate": 3e-7,
                "temperature": 0.05,
                "parameter_anchor_weight": 5.0,
                "rehearsal_loss_weight": 0.5,
                "save_every": 100,
            },
        ],
    ),
    "radical_supcon600": Variant(
        name="radical_supcon600",
        description="Batch-level supervised contrastive loss on balanced CEDR emotion groups.",
        stages=[
            {
                "name": "supcon",
                "data_path": CEDR_LABELED_SUPCON,
                "max_steps": 600,
                "batch_size": 10,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "save_every": 200,
                "shuffle": False,
                "temperature": 0.05,
                "parameter_anchor_weight": 0.5,
                "rehearsal_loss_weight": 0.75,
            }
        ],
    ),
    "radical_circle600": Variant(
        name="radical_circle600",
        description="Circle Loss on balanced CEDR emotion groups; stronger pair reweighting than SupCon.",
        stages=[
            {
                "name": "circle",
                "data_path": CEDR_LABELED_CIRCLE,
                "max_steps": 600,
                "batch_size": 10,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "save_every": 200,
                "shuffle": False,
                "temperature": 0.05,
                "parameter_anchor_weight": 0.5,
                "rehearsal_loss_weight": 0.75,
            }
        ],
    ),
    "radical_ms600": Variant(
        name="radical_ms600",
        description="Multi-Similarity Loss on balanced CEDR emotion groups; mines informative pairs inside each batch.",
        stages=[
            {
                "name": "multi_similarity",
                "data_path": CEDR_LABELED_MS,
                "max_steps": 600,
                "batch_size": 10,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "save_every": 200,
                "shuffle": False,
                "temperature": 0.05,
                "parameter_anchor_weight": 0.5,
                "rehearsal_loss_weight": 0.75,
            }
        ],
    ),
    "radical_circle_lr5_t10_loanchor600": Variant(
        name="radical_circle_lr5_t10_loanchor600",
        description=(
            "Circle Loss with higher LR, lower weight decay, warmer temperature, and weaker retention "
            "to test whether the CEDR plateau is caused by over-constrained late repair."
        ),
        stages=[
            {
                "name": "circle",
                "data_path": CEDR_LABELED_CIRCLE,
                "max_steps": 600,
                "batch_size": 10,
                "learning_rate": 5e-6,
                "weight_decay": 0.0001,
                "save_every": 600,
                "shuffle": False,
                "temperature": 0.10,
                "parameter_anchor_weight": 0.1,
                "rehearsal_loss_weight": 0.25,
            }
        ],
    ),
    "radical_circle_lr8_t15_noanchor600": Variant(
        name="radical_circle_lr8_t15_noanchor600",
        description=(
            "Circle Loss with aggressive LR, very low weight decay, warmer temperature, and no explicit "
            "anchor/rehearsal. This tests whether CEDR needs a larger move before later recovery."
        ),
        stages=[
            {
                "name": "circle",
                "data_path": CEDR_LABELED_CIRCLE,
                "max_steps": 600,
                "batch_size": 10,
                "learning_rate": 8e-6,
                "weight_decay": 0.0,
                "save_every": 600,
                "shuffle": False,
                "temperature": 0.15,
                "parameter_anchor_weight": 0.0,
                "rehearsal_loss_weight": 0.0,
            }
        ],
    ),
    "radical_supcon_lr5_t10_loanchor600": Variant(
        name="radical_supcon_lr5_t10_loanchor600",
        description=(
            "SupCon with higher LR, lower weight decay, warmer temperature, and weaker retention. "
            "This checks whether the best CEDR route was limited by optimizer settings."
        ),
        stages=[
            {
                "name": "supcon",
                "data_path": CEDR_LABELED_SUPCON,
                "max_steps": 600,
                "batch_size": 10,
                "learning_rate": 5e-6,
                "weight_decay": 0.0001,
                "save_every": 600,
                "shuffle": False,
                "temperature": 0.10,
                "parameter_anchor_weight": 0.1,
                "rehearsal_loss_weight": 0.25,
            }
        ],
    ),
    "radical_circle_lr5_t10_shuffle600": Variant(
        name="radical_circle_lr5_t10_shuffle600",
        description=(
            "Same as the lower-anchor high-LR Circle route, but with shuffled batches to test whether "
            "fixed class ordering in the labeled JSONL is limiting in-batch mining."
        ),
        stages=[
            {
                "name": "circle",
                "data_path": CEDR_LABELED_CIRCLE,
                "max_steps": 600,
                "batch_size": 10,
                "learning_rate": 5e-6,
                "weight_decay": 0.0001,
                "save_every": 600,
                "shuffle": True,
                "temperature": 0.10,
                "parameter_anchor_weight": 0.1,
                "rehearsal_loss_weight": 0.25,
            }
        ],
    ),
    "multilabel_support_p4_400": Variant(
        name="multilabel_support_p4_400",
        description=(
            "CEDR-structured multilabel support episodes. This mimics the benchmark's few-shot "
            "multilabel kNN setup, including empty-label neutral queries."
        ),
        stages=[
            {
                "name": "multilabel_support_p4",
                "data_path": CEDR_MULTILABEL_SUPPORT_P4,
                "max_steps": 400,
                "batch_size": 2,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "temperature": 0.08,
                "parameter_anchor_weight": 0.5,
                "rehearsal_loss_weight": 0.5,
                "save_every": 400,
            }
        ],
    ),
    "multilabel_support_p8_300": Variant(
        name="multilabel_support_p8_300",
        description=(
            "CEDR-structured multilabel support episodes with larger support sets per emotion class."
        ),
        stages=[
            {
                "name": "multilabel_support_p8",
                "data_path": CEDR_MULTILABEL_SUPPORT_P8,
                "max_steps": 300,
                "batch_size": 1,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "temperature": 0.08,
                "parameter_anchor_weight": 0.5,
                "rehearsal_loss_weight": 0.5,
                "save_every": 300,
            }
        ],
    ),
    "multilabel_support_p4_400_repair400": Variant(
        name="multilabel_support_p4_400_repair400",
        description=(
            "Multilabel CEDR support stage followed by a short Habr+DeepVK recovery stage to check "
            "whether CEDR movement can be retained without hurting STS and geography tasks."
        ),
        stages=[
            {
                "name": "multilabel_support_p4",
                "data_path": CEDR_MULTILABEL_SUPPORT_P4,
                "max_steps": 400,
                "batch_size": 2,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "temperature": 0.08,
                "parameter_anchor_weight": 0.5,
                "rehearsal_loss_weight": 0.5,
                "save_every": 400,
            },
            {
                "name": "habr_deepvk_repair",
                "data_path": HABR_DEEPVK,
                "max_steps": 400,
                "batch_size": 2,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "temperature": 0.02,
                "parameter_anchor_weight": 1.0,
                "rehearsal_loss_weight": 0.75,
                "save_every": 400,
            },
        ],
    ),
    "multilabel_support_margin_p4_400": Variant(
        name="multilabel_support_margin_p4_400",
        description=(
            "CEDR multilabel support episodes with thresholded nearest-support BCE logits. "
            "This fixes the raw-similarity BCE issue from the first support objective."
        ),
        stages=[
            {
                "name": "multilabel_support_margin_p4",
                "data_path": CEDR_MULTILABEL_SUPPORT_P4,
                "max_steps": 400,
                "batch_size": 2,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "temperature": 0.08,
                "parameter_anchor_weight": 0.5,
                "rehearsal_loss_weight": 0.5,
                "save_every": 400,
            }
        ],
    ),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def score_table(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Local avg:"):
            scores["avg"] = float(line.split(":", 1)[1].strip())
        if not line.startswith("| ") or line.startswith("| Task ") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        task, local, _official, delta = cells
        scores[task] = float(local)
        scores[f"{task}.delta"] = float(delta)
    return scores


def write_config(variant: Variant, *, seed: int) -> Path:
    run_name = f"nomanip_{variant.name}_4096_eager_frozenrepro"
    stages = []
    for stage in variant.stages:
        stage = dict(stage)
        stage.setdefault("weight_decay", 0.01)
        stages.append(stage)
    config = {
        "name": run_name,
        "description": f"No direct weight manipulation. {variant.description}",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": CLEAN_BASE,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": stages[0]["data_path"],
        "output_dir": str((CHECKPOINT_ROOT / run_name).relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": stages[0]["batch_size"],
        "learning_rate": stages[0]["learning_rate"],
        "weight_decay": stages[0].get("weight_decay", 0.01),
        "temperature": 0.02,
        "max_steps": sum(int(stage["max_steps"]) for stage in stages),
        "log_every": 50,
        "save_every": 10_000_000,
        "seed": seed,
        "retention": {
            "parameter_anchor_weight": 0.0,
            "rehearsal_data_path": GO9000,
            "rehearsal_batch_size": 3,
            "rehearsal_loss_weight": 0.0,
            "negative_direction_checkpoint": NO_GERACL_DIRECTION,
        },
        "stages": stages,
    }
    path = CONFIG_DIR / f"{run_name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def checkpoint_steps(output_dir: Path) -> list[Path]:
    checkpoints = []
    for path in output_dir.glob("step-*.pt"):
        try:
            int(path.stem.split("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append(path)
    return sorted(checkpoints, key=lambda path: int(path.stem.split("-", 1)[1]))


def eval_checkpoint(checkpoint: Path, *, run_name: str, tag: str) -> tuple[Path, dict[str, float]]:
    result_dir = RESULTS_ROOT / f"{run_name}_{tag}_gate5"
    comparison_path = RESULTS_ROOT / f"{run_name}_{tag}_gate5_comparison.md"
    run(
        [
            "official_repro/.venv/bin/python",
            "official_repro/run_official_rumteb.py",
            "--output-folder",
            str(result_dir.relative_to(ROOT)),
            "--tasks",
            *GATE5_TASKS,
            "--prompt-mode",
            "legacy_ru",
            "--batch-size",
            "8",
            "--seed",
            "8",
            "--attn-implementation",
            "eager",
            "--latent-checkpoint",
            str(checkpoint.relative_to(ROOT)),
            "--overwrite-results",
            "--reset-seed-per-task",
        ]
    )
    run(
        [
            "official_repro/.venv/bin/python",
            "official_repro/compare_official_rumteb.py",
            str(result_dir.relative_to(ROOT)),
            "--write-md",
            str(comparison_path.relative_to(ROOT)),
        ]
    )
    return comparison_path, score_table(comparison_path)


def cleanup_checkpoints(output_dir: Path, keep: set[Path]) -> None:
    for path in output_dir.glob("*.pt"):
        if path not in keep:
            path.unlink()
    for path in output_dir.glob("latest.*"):
        if path.with_suffix(".pt") not in keep:
            path.unlink(missing_ok=True)


def rank_key(row: dict[str, float]) -> tuple[float, float, float]:
    cedr_shortfall = max(0.0, OFFICIAL["CEDRClassification"] - row["CEDRClassification"])
    worst_delta = min(row[task] - OFFICIAL[task] for task in GATE5_TASKS)
    return (-cedr_shortfall, row["avg"], worst_delta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=["cedr_go600_lenta100"])
    parser.add_argument("--seed", type=int, default=1831)
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    args = parser.parse_args()

    rows = []
    for index, variant_name in enumerate(args.variants):
        if variant_name not in VARIANTS:
            raise SystemExit(f"Unknown variant {variant_name}. Available: {', '.join(VARIANTS)}")
        variant = VARIANTS[variant_name]
        config_path = write_config(variant, seed=args.seed + index)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_name = config["name"]
        output_dir = CHECKPOINT_ROOT / run_name

        if not checkpoint_steps(output_dir):
            run(["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path)])

        for checkpoint in checkpoint_steps(output_dir):
            tag = checkpoint.stem.replace("-", "")
            comparison_path, scores = eval_checkpoint(checkpoint, run_name=run_name, tag=tag)
            row = {
                "variant": variant_name,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "comparison": str(comparison_path.relative_to(ROOT)),
                **scores,
            }
            rows.append(row)

        if not args.keep_all_checkpoints:
            best_for_variant = max(
                [row for row in rows if row["variant"] == variant_name],
                key=rank_key,
            )
            cleanup_checkpoints(output_dir, {ROOT / best_for_variant["checkpoint"]})

    summary_path = RESULTS_ROOT / "nomanip_cedr_sts_gate5_summary.md"
    lines = [
        "# No-Manipulation CEDR/STS Gate Search",
        "",
        f"Clean base: `{CLEAN_BASE}`.",
        "No checkpoint averaging, subtraction, interpolation, or latent-state arithmetic is used in these runs.",
        "Evaluation: frozen official-reproduction wrapper, `legacy_ru`, eager attention, seed 8 reset per task.",
        "",
        "| Variant | Checkpoint | Avg | CEDR | GeoCls | OECDCls | RuSTS | GeoCluster | Worst delta vs released | Comparison |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=rank_key, reverse=True):
        worst_delta = min(row[task] - OFFICIAL[task] for task in GATE5_TASKS)
        lines.append(
            f"| `{row['variant']}` | `{row['checkpoint']}` | "
            f"{row['avg']:.6f} | {row['CEDRClassification']:.6f} | "
            f"{row['GeoreviewClassification']:.6f} | {row['RuSciBenchOECDClassification']:.6f} | "
            f"{row['RuSTSBenchmarkSTS']:.6f} | {row['GeoreviewClusteringP2P']:.6f} | "
            f"{worst_delta:+.6f} | `{row['comparison']}` |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
