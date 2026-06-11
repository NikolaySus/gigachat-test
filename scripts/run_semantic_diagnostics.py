from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from giga_model_utils import ModelLoadConfig, cosine_matrix, encode_texts, load_giga_embeddings, write_json


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    category: str
    anchor: str
    positive: str
    hard_negative: str
    note: str


CASES = [
    DiagnosticCase(
        "negation_medical",
        "negation",
        "Пациент сообщает, что аллергии на пенициллин нет.",
        "У пациента отсутствует аллергия на пенициллин.",
        "У пациента есть аллергия на пенициллин.",
        "Negation should dominate lexical overlap.",
    ),
    DiagnosticCase(
        "role_reversal_contract",
        "argument_roles",
        "Поставщик обязан вернуть предоплату покупателю при срыве сроков.",
        "Если поставщик нарушил сроки, покупатель получает предоплату обратно.",
        "Покупатель обязан вернуть предоплату поставщику при срыве сроков.",
        "Same entities, reversed obligation.",
    ),
    DiagnosticCase(
        "contrastive_policy",
        "contrast",
        "Сервис не хранит пароли, но сохраняет историю входов.",
        "Пароли не сохраняются; журнал авторизаций хранится.",
        "Сервис хранит пароли, но не сохраняет историю входов.",
        "But-clause facts are easy to blur under mean pooling.",
    ),
    DiagnosticCase(
        "hierarchy_tax",
        "hierarchy",
        "Налоговая льгота действует только для ИП на упрощенной системе без сотрудников.",
        "Льгота применима к предпринимателям на УСН, если у них нет работников.",
        "Льгота действует для всех предпринимателей независимо от режима и сотрудников.",
        "Tests nested constraints.",
    ),
    DiagnosticCase(
        "temporal_order",
        "temporal",
        "Сначала банк одобрил заявку, затем клиент отказался от кредита.",
        "Клиент отказался уже после того, как банк одобрил заявку.",
        "Банк одобрил заявку после отказа клиента от кредита.",
        "Same events, different order.",
    ),
    DiagnosticCase(
        "numeric_threshold",
        "numeric",
        "Скидка доступна при заказе от 5000 рублей и доставке по Москве.",
        "Для московской доставки скидка включается, если сумма заказа не ниже 5000 рублей.",
        "Скидка доступна при заказе до 5000 рублей и доставке по Москве.",
        "Threshold polarity should matter.",
    ),
    DiagnosticCase(
        "multi_hop_science",
        "multi_hop",
        "Если вещество растворяется в воде и проводит ток, раствор содержит ионы.",
        "Проводящий водный раствор указывает на наличие ионов.",
        "Если вещество растворяется в воде, оно обязательно не содержит ионов.",
        "Requires combining two conditions.",
    ),
    DiagnosticCase(
        "distractor_long",
        "distractor",
        "Главная мысль: договор продлевается автоматически. Детали про офис, логотип и цвет бумаги несущественны.",
        "Договор будет продлен без отдельного заявления сторон.",
        "Договор прекращается автоматически, а сведения об офисе важны.",
        "Checks whether salient fact survives distractors.",
    ),
]


def build_dataframe(vectors: np.ndarray, labels: list[str], cases: list[DiagnosticCase]) -> pd.DataFrame:
    sims = cosine_matrix(vectors)
    rows = []
    for idx, case in enumerate(cases):
        anchor_idx = 3 * idx
        pos_idx = anchor_idx + 1
        neg_idx = anchor_idx + 2
        pos_sim = float(sims[anchor_idx, pos_idx])
        neg_sim = float(sims[anchor_idx, neg_idx])
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "positive_similarity": pos_sim,
                "hard_negative_similarity": neg_sim,
                "margin": pos_sim - neg_sim,
                "passed": pos_sim > neg_sim,
                "note": case.note,
                "anchor": labels[anchor_idx],
                "positive": labels[pos_idx],
                "hard_negative": labels[neg_idx],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/semantic_diagnostics"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--latent-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    tokenizer, model = load_giga_embeddings(
        ModelLoadConfig(
            batch_size=args.batch_size,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
            latent_checkpoint=args.latent_checkpoint,
        )
    )
    texts = []
    for case in CASES:
        texts.extend([case.anchor, case.positive, case.hard_negative])

    vectors = encode_texts(
        texts,
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    df = build_dataframe(vectors, texts, CASES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "diagnostic_margins.csv", index=False)
    np.save(args.out_dir / "embeddings.npy", vectors)
    write_json(
        args.out_dir / "summary.json",
        {
            "cases": len(CASES),
            "passed": int(df["passed"].sum()),
            "failed": int((~df["passed"]).sum()),
            "mean_margin": float(df["margin"].mean()),
            "worst_cases": df.sort_values("margin").head(5)[
                ["case_id", "category", "margin", "positive_similarity", "hard_negative_similarity"]
            ].to_dict(orient="records"),
        },
    )
    print(df[["case_id", "category", "margin", "passed"]].to_string(index=False))
    print(f"Wrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
