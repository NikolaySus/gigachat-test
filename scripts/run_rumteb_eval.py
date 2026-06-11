from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from giga_model_utils import ModelLoadConfig, encode_texts, load_giga_embeddings
from rumteb_contamination import (
    contaminated_tasks_from_manifest,
    filter_tasks_by_scope,
    load_training_manifest,
)


class GigaMTEBWrapper:
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        local_files_only: bool,
        attn_implementation: str | None,
        use_prompts: bool,
        latent_checkpoint: Path | None,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_prompts = use_prompts
        self.tokenizer, self.model = load_giga_embeddings(
            ModelLoadConfig(
                batch_size=batch_size,
                max_length=max_length,
                local_files_only=local_files_only,
                attn_implementation=attn_implementation,
                latent_checkpoint=latent_checkpoint,
            )
        )
        from mteb.models.model_meta import ModelMeta

        self._mteb_model_meta = ModelMeta.create_empty(
            {
                "name": "ai-sage/Giga-Embeddings-instruct-local",
                "revision": "local",
                "framework": ["PyTorch", "Transformers"],
                "similarity_fn_name": "cosine",
                "languages": ["rus-Cyrl", "eng-Latn"],
                "max_tokens": max_length,
                "embed_dim": 2048,
                "use_instructions": True,
            }
        )

    def encode(self, sentences, **kwargs):
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        sentences = self._normalize_inputs(sentences)
        if self.use_prompts:
            sentences = self._apply_mteb_prompt(
                sentences,
                task_metadata=kwargs.get("task_metadata"),
                prompt_type=kwargs.get("prompt_type"),
            )
        embeddings = encode_texts(
            sentences,
            self.tokenizer,
            self.model,
            batch_size=batch_size,
            max_length=self.max_length,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def mteb_model_meta(self):
        return self._mteb_model_meta

    def similarity(self, embeddings1, embeddings2):
        embeddings1 = np.asarray(embeddings1, dtype=np.float32)
        embeddings2 = np.asarray(embeddings2, dtype=np.float32)
        return embeddings1 @ embeddings2.T

    def similarity_pairwise(self, embeddings1, embeddings2):
        embeddings1 = np.asarray(embeddings1, dtype=np.float32)
        embeddings2 = np.asarray(embeddings2, dtype=np.float32)
        return np.sum(embeddings1 * embeddings2, axis=1)

    @staticmethod
    def _normalize_inputs(inputs) -> list[str]:
        if isinstance(inputs, DataLoader):
            texts: list[str] = []
            for batch in inputs:
                texts.extend(GigaMTEBWrapper._normalize_inputs(batch))
            return texts

        if isinstance(inputs, dict):
            if "text" in inputs:
                return GigaMTEBWrapper._string_list(inputs["text"])
            if "query" in inputs:
                return GigaMTEBWrapper._string_list(inputs["query"])
            if "body" in inputs:
                titles = GigaMTEBWrapper._string_list(inputs.get("title", []))
                bodies = GigaMTEBWrapper._string_list(inputs["body"])
                if titles and len(titles) == len(bodies):
                    return [(title + " " + body).strip() for title, body in zip(titles, bodies, strict=True)]
                return bodies
            raise TypeError(f"Unsupported MTEB input keys: {sorted(inputs.keys())}")

        return GigaMTEBWrapper._string_list(inputs)

    @staticmethod
    def _apply_mteb_prompt(texts: list[str], *, task_metadata, prompt_type) -> list[str]:
        instruction = GigaMTEBWrapper._instruction_for(task_metadata, prompt_type)
        if instruction is None:
            return texts
        return [f"Instruct: {instruction}\nQuery: {text}" for text in texts]

    @staticmethod
    def _instruction_for(task_metadata, prompt_type) -> str | None:
        if task_metadata is None:
            return "Given a text, retrieve semantically similar text"

        task_type = str(getattr(task_metadata, "type", "") or "")
        raw_prompt = getattr(task_metadata, "prompt", None)
        prompt_type_value = str(getattr(prompt_type, "value", prompt_type) or "")

        if task_type in {"Retrieval", "Reranking"}:
            if prompt_type_value != "query":
                return None
            if isinstance(raw_prompt, dict):
                return raw_prompt.get("query") or raw_prompt.get("text")
            if isinstance(raw_prompt, str):
                return raw_prompt
            return "Given a question, retrieve relevant passages that answer the question"

        if isinstance(raw_prompt, str) and raw_prompt.strip():
            return raw_prompt

        if task_type in {"Classification", "MultilabelClassification"}:
            return "Classify the given text"
        if task_type == "Clustering":
            return "Identify semantically related texts for clustering"
        if task_type == "PairClassification":
            return "Given a text, retrieve semantically related text"
        if task_type == "STS":
            return "Given a text, retrieve semantically similar text"

        return "Given a text, retrieve semantically similar text"

    @staticmethod
    def _string_list(values) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, Iterable):
            return [str(value) for value in values]
        return [str(values)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GigaEmbeddings on Russian MTEB tasks.")
    parser.add_argument("--output-folder", type=Path, default=Path("results/rumteb"))
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional result subdirectory name. Defaults to max length, prompt mode, eval scope, and training manifest.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("results/mteb_cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--latent-checkpoint", type=Path, default=None)
    parser.add_argument("--no-prompts", action="store_true")
    parser.add_argument("--training-manifest", type=Path, default=None)
    parser.add_argument(
        "--eval-scope",
        choices=("all", "clean", "contaminated"),
        default="all",
        help="Evaluate all tasks, only tasks not contaminated by the training manifest, or only contaminated tasks.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional explicit MTEB task names. By default uses Russian v1.1 tasks when supported by installed mteb.",
    )
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MTEB_CACHE", str(args.cache_dir.resolve()))
    os.environ.setdefault("HF_HOME", str((args.cache_dir / "hf_home").resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str((args.cache_dir / "datasets").resolve()))

    try:
        import mteb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The `mteb` package is not installed in this environment. Install it first, for example: "
            "UV_CACHE_DIR=/tmp/uv-cache uv add mteb"
        ) from exc

    training_manifest = load_training_manifest(args.training_manifest)
    contaminated_tasks = contaminated_tasks_from_manifest(training_manifest)

    if args.tasks:
        selected_task_names = filter_tasks_by_scope(args.tasks, contaminated_tasks, args.eval_scope)
        if not selected_task_names:
            raise SystemExit(
                f"No tasks selected for eval scope `{args.eval_scope}`. "
                "Check --tasks and --training-manifest."
            )
        try:
            benchmark = mteb.get_benchmark("MTEB(rus, v1.1)")
            benchmark_task_by_name = {task.metadata.name: task for task in benchmark.tasks}
            resolved_tasks = [
                benchmark_task_by_name[name]
                for name in selected_task_names
                if name in benchmark_task_by_name
            ]
            missing_from_benchmark = [
                name for name in selected_task_names if name not in benchmark_task_by_name
            ]
            if missing_from_benchmark:
                resolved_tasks.extend(mteb.get_tasks(tasks=missing_from_benchmark))
        except Exception:
            resolved_tasks = mteb.get_tasks(tasks=selected_task_names)
        task_by_name = {task.metadata.name: task for task in resolved_tasks}
        missing_tasks = [name for name in selected_task_names if name not in task_by_name]
        if missing_tasks:
            raise SystemExit(f"Unknown MTEB task(s): {', '.join(missing_tasks)}")
        evaluator = mteb.MTEB(tasks=[task_by_name[name] for name in selected_task_names])
    else:
        try:
            benchmark = mteb.get_benchmark("MTEB(rus, v1.1)")
            task_by_name = {task.metadata.name: task for task in benchmark.tasks}
            selected_task_names = filter_tasks_by_scope(task_by_name.keys(), contaminated_tasks, args.eval_scope)
            if not selected_task_names:
                raise SystemExit(
                    f"No tasks selected for eval scope `{args.eval_scope}`. "
                    "Check --tasks and --training-manifest."
                )
            evaluator = mteb.MTEB(tasks=[task_by_name[name] for name in selected_task_names])
        except Exception:
            tasks = mteb.get_tasks(languages=["rus"])
            task_by_name = {task.metadata.name: task for task in tasks}
            selected_task_names = filter_tasks_by_scope(task_by_name.keys(), contaminated_tasks, args.eval_scope)
            if not selected_task_names:
                raise SystemExit(
                    f"No tasks selected for eval scope `{args.eval_scope}`. "
                    "Check --tasks and --training-manifest."
                )
            evaluator = mteb.MTEB(tasks=[task_by_name[name] for name in selected_task_names])

    model = GigaMTEBWrapper(
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        use_prompts=not args.no_prompts,
        latent_checkpoint=args.latent_checkpoint,
    )

    manifest_suffix = training_manifest["name"] if training_manifest is not None else "no-train-manifest"
    run_name = args.run_name or (
        f"maxlen-{args.max_length}-{'noprompt' if args.no_prompts else 'prompt'}-{args.eval_scope}-{manifest_suffix}"
    )
    output_folder = args.output_folder / run_name
    output_folder.mkdir(parents=True, exist_ok=True)
    evaluation_manifest = {
        "model": "ai-sage/Giga-Embeddings-instruct-local",
        "max_length": args.max_length,
        "prompting": not args.no_prompts,
        "eval_scope": args.eval_scope,
        "training_manifest_path": str(args.training_manifest) if args.training_manifest else None,
        "training_manifest_name": training_manifest["name"] if training_manifest else None,
        "latent_checkpoint": str(args.latent_checkpoint) if args.latent_checkpoint else None,
        "included_tasks": selected_task_names,
        "excluded_contaminated_tasks": [
            task for task in contaminated_tasks if task not in selected_task_names
        ],
        "contaminated_tasks": contaminated_tasks,
    }
    (output_folder / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evaluator.run(
        model,
        output_folder=str(output_folder),
        encode_kwargs={"batch_size": args.batch_size},
    )
    print(f"Wrote MTEB results to {output_folder}")


if __name__ == "__main__":
    main()
