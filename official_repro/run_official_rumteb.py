from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

try:
    from mteb.models.model_meta import ModelMeta
except ModuleNotFoundError:
    ModelMeta = None


MODEL_NAME = "ai-sage/Giga-Embeddings-instruct"
MODEL_REVISION = "40b27667b9ad586d7812675df76e5062ccc80b0e"
RU_TASK_PROMPTS = {
    "CEDRClassification": "Определи, какие эмоции выражены в комментарии: радость, грусть, удивление, страх или злость",
    "GeoreviewClassification": "Классифицируй рейтинг организации на основе отзывов",
    "GeoreviewClusteringP2P": "Определи категорию организации на основе отзывов",
    "HeadlineClassification": "Классифицируй тему новостного заголовка",
    "InappropriatenessClassification": "Классифицируй сообщение как чувствительную тему или нет",
    "KinopoiskClassification": "Классифицируй тональность отзыва о фильме",
    "MassiveIntentClassification": "Определи намерение пользователя по его фразе",
    "MassiveScenarioClassification": "Определи сценарий пользователя по его фразе",
    "RuReviewsClassification": "Классифицируй отзыв на товар как положительный, отрицательный или нейтральный",
    "RuSciBenchGRNTIClassification": "Классифицируй категорию научной статьи по названию и аннотации",
    "RuSciBenchGRNTIClusteringP2P": "Определи категорию научной статьи по названию и аннотации",
    "RuSciBenchOECDClassification": "Классифицируй категорию научной статьи по названию и аннотации",
    "RuSciBenchOECDClusteringP2P": "Определи категорию научной статьи по названию и аннотации",
    "SensitiveTopicsClassification": "Классифицируй чувствительную тему по запросу",
    "TERRa": "Дана предпосылка, найди гипотезу, которая из нее следует",
}
LEGACY_RU_PREFIXES = {
    "default": "Дан текст, необходимо найти семантически похожий текст \nтекст: ",
    "sts": "Найди семантически похожий текст \nтекст: ",
    "ruparaphraser": "найди семантически похожее предложение \nтекст: ",
    "rusts": "семантически похожий текст: ",
    "retrieval": "Дан вопрос, необходимо найти абзац текста с ответом \nвопрос: ",
    "sensitive": "Классифицируй чувствительную тему по запросу \nзапрос: ",
    "inappropriate": "Определи, является ли сообщение неприемлемым, токсичным или чувствительным \nсообщение: ",
    "cedr": "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: ",
    "headline": "Определи тему новостного заголовка \nзаголовок: ",
    "georeview": "Определи категорию организации на основе отзыва \nотзыв: ",
    "georeview_classification": "Определи тональность отзыва о сервисе организации \nотзыв: ",
    "science_clustering": "Определи категорию научной статьи по названию и аннотации \nтекст: ",
    "terra": "семантически похожий текст: ",
}
DEFAULT_TASK_PROMPT_MODES = {
    "GeoreviewClassification": "legacy_ru_masked",
    "RuReviewsClassification": "prefix",
    "MassiveScenarioClassification": "prefix",
    "MassiveIntentClassification": "prefix",
    "RuSciBenchGRNTIClassification": "prefix",
}
DEFAULT_MTEB_PROMPT_OVERRIDES = {
    "MassiveIntentClassification": "Given a user request, find the intended assistant action",
}
DEFAULT_LEGACY_PREFIX_ENSEMBLES = {
    "RuSTSBenchmarkSTS": [
        "семантически похожий текст: ",
        "семантически похожий текст \nтекст: ",
    ],
}
DEFAULT_TASK_TEXT_NORMALIZATIONS = {
    "RuSTSBenchmarkSTS": "yo",
}
DEFAULT_TASK_BATCH_SIZES = {
    "RuSTSBenchmarkSTS": 4,
}
DEFAULT_TASK_SEEDS = {
    "GeoreviewClassification": 42,
}


def patch_datasets_trust_remote_code() -> None:
    import datasets
    import datasets.load

    if getattr(datasets, "_giga_trust_remote_code_patch", False):
        return

    original_load_dataset = datasets.load_dataset
    original_load_dataset_builder = datasets.load_dataset_builder

    def load_dataset_with_trust(*args, **kwargs):
        if kwargs.get("trust_remote_code") is None:
            kwargs["trust_remote_code"] = True
        return original_load_dataset(*args, **kwargs)

    def load_dataset_builder_with_trust(*args, **kwargs):
        if kwargs.get("trust_remote_code") is None:
            kwargs["trust_remote_code"] = True
        return original_load_dataset_builder(*args, **kwargs)

    datasets.load_dataset = load_dataset_with_trust
    datasets.load_dataset_builder = load_dataset_builder_with_trust
    datasets.load.load_dataset = load_dataset_with_trust
    datasets.load.load_dataset_builder = load_dataset_builder_with_trust
    datasets._giga_trust_remote_code_patch = True


def patch_transformers_config_compat() -> None:
    from transformers.configuration_utils import PretrainedConfig

    if not hasattr(PretrainedConfig, "torch_dtype"):
        PretrainedConfig.torch_dtype = None


class GigaOfficialMTEBWrapper:
    def __init__(
        self,
        *,
        batch_size: int,
        max_length: int,
        model_revision: str,
        attn_implementation: str,
        torch_dtype: str,
        local_files_only: bool,
        latent_checkpoint: Path | None,
        task_prompts: dict[str, tuple[str, str | dict[str, str] | None]],
        prompt_mode: str,
        symmetric_instruction: str,
        legacy_prefix_overrides: dict[str, str] | None = None,
        legacy_prefix_ensembles: dict[str, list[str]] | None = None,
        text_normalization: str = "none",
        text_suffix_overrides: dict[str, str] | None = None,
        task_prompt_modes: dict[str, str] | None = None,
        task_text_normalizations: dict[str, str] | None = None,
        task_batch_sizes: dict[str, int] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self.task_prompts = task_prompts
        self.prompt_mode = prompt_mode
        self.symmetric_instruction = symmetric_instruction
        self.legacy_prefix_overrides = legacy_prefix_overrides or {}
        self.legacy_prefix_ensembles = legacy_prefix_ensembles or {}
        self.text_normalization = text_normalization
        self.text_suffix_overrides = text_suffix_overrides or {}
        self.task_prompt_modes = task_prompt_modes or {}
        self.task_text_normalizations = task_text_normalizations or {}
        self.task_batch_sizes = task_batch_sizes or {}
        self._task_encode_counts: dict[str, int] = {}
        self._mteb_model_meta = None
        if ModelMeta is not None:
            self._mteb_model_meta = ModelMeta(
                loader=None,
                name=MODEL_NAME,
                revision=model_revision,
                release_date=None,
                languages=["rus-Cyrl"],
                n_parameters=None,
                memory_usage_mb=None,
                max_tokens=max_length,
                embed_dim=2048,
                license="not specified",
                open_weights=True,
                public_training_code=None,
                public_training_data=None,
                framework=["PyTorch", "Transformers"],
                similarity_fn_name="cosine",
                use_instructions=True,
                training_datasets=None,
            )
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        patch_transformers_config_compat()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=model_revision,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=model_revision,
            torch_dtype=dtype_by_name[torch_dtype],
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            local_files_only=local_files_only,
        )
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        if latent_checkpoint is not None:
            checkpoint = torch.load(latent_checkpoint, map_location="cpu")
            latent_state = checkpoint.get("latent_attention_model", checkpoint)
            self.model.latent_attention_model.load_state_dict(latent_state)
        self.model.eval()

    @property
    def mteb_model_meta(self):
        if self._mteb_model_meta is None:
            raise AttributeError("mteb_model_meta is only available for newer MTEB versions")
        return self._mteb_model_meta

    @staticmethod
    def _task_name_from_kwargs(kwargs: dict) -> str:
        task_name = kwargs.get("task_name")
        if task_name:
            return str(task_name)
        task_metadata = kwargs.get("task_metadata")
        metadata_name = getattr(task_metadata, "name", None)
        if metadata_name:
            return str(metadata_name)
        return ""

    def encode(self, sentences, **kwargs):
        task_name = kwargs.get("task_name") or getattr(kwargs.get("task_metadata"), "name", None)
        prompt_type = kwargs.get("prompt_type")
        task_name_str = self._task_name_from_kwargs(kwargs)
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        if task_name_str in self.task_batch_sizes:
            batch_size = min(batch_size, self.task_batch_sizes[task_name_str])
        effective_prompt_mode = self.task_prompt_modes.get(task_name_str, self.prompt_mode)
        text_normalization = (
            self.task_text_normalizations.get(task_name_str, self.text_normalization)
            if self.text_normalization == "none"
            else self.text_normalization
        )
        texts = [
            self._normalize_text(text, text_normalization=text_normalization)
            for text in self._normalize_inputs(sentences)
        ]
        if task_name_str in self.text_suffix_overrides:
            texts = [text + self.text_suffix_overrides[task_name_str] for text in texts]
        if effective_prompt_mode == "legacy_ru":
            count = self._task_encode_counts.get(task_name_str, 0)
            self._task_encode_counts[task_name_str] = count + 1
            side_key = f"{task_name_str}#side{count % 2 + 1}"
            if side_key in self.legacy_prefix_overrides:
                texts = [self.legacy_prefix_overrides[side_key] + text for text in texts]
                return np.asarray(
                    self._encode_texts(texts, batch_size=batch_size, instruction=None),
                    dtype=np.float32,
                )

            if task_name_str in self.legacy_prefix_ensembles:
                vectors = []
                for prefix in self.legacy_prefix_ensembles[task_name_str]:
                    prefixed_texts = [prefix + text for text in texts]
                    vectors.append(
                        self._encode_texts(prefixed_texts, batch_size=batch_size, instruction=None)
                    )
                embeddings = np.mean(np.stack(vectors, axis=0), axis=0)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                return np.asarray(embeddings / np.maximum(norms, 1e-12), dtype=np.float32)

            texts = self._apply_legacy_ru_prefix(texts, task_name=task_name, prompt_type=prompt_type)
            return np.asarray(
                self._encode_texts(texts, batch_size=batch_size, instruction=None),
                dtype=np.float32,
            )

        instruction = self._instruction_for(task_name=task_name, prompt_type=prompt_type, prompt_mode=effective_prompt_mode)
        if effective_prompt_mode in {"prefix", "model_prefix"} and instruction is not None:
            texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
        return np.asarray(
            self._encode_texts(texts, batch_size=batch_size, instruction=instruction, prompt_mode=effective_prompt_mode),
            dtype=np.float32,
        )

    def similarity(self, embeddings1, embeddings2):
        embeddings1 = torch.as_tensor(embeddings1, dtype=torch.float32)
        embeddings2 = torch.as_tensor(embeddings2, dtype=torch.float32)
        return embeddings1 @ embeddings2.T

    def similarity_pairwise(self, embeddings1, embeddings2):
        embeddings1 = np.asarray(embeddings1, dtype=np.float32)
        embeddings2 = np.asarray(embeddings2, dtype=np.float32)
        return np.sum(embeddings1 * embeddings2, axis=1)

    @torch.inference_mode()
    def _encode_texts(
        self,
        texts: list[str],
        *,
        batch_size: int,
        instruction: str | None,
        prompt_mode: str | None = None,
    ) -> np.ndarray:
        prompt_mode = prompt_mode or self.prompt_mode
        if prompt_mode == "legacy_ru_masked" and instruction is not None and hasattr(self.model, "_do_encode"):
            embeddings = self.model._do_encode(
                texts,
                batch_size=batch_size,
                instruction=instruction,
                max_length=self.max_length,
                num_workers=0,
                return_numpy=False,
            )
            return F.normalize(embeddings.float(), p=2, dim=-1).cpu().numpy()

        if prompt_mode == "model_prefix" and hasattr(self.model, "_do_encode"):
            embeddings = self.model._do_encode(
                texts,
                batch_size=batch_size,
                instruction="",
                max_length=self.max_length,
                num_workers=0,
                return_numpy=False,
            )
            return F.normalize(embeddings.float(), p=2, dim=-1).cpu().numpy()

        if prompt_mode == "instruction" and instruction is not None and hasattr(self.model, "_do_encode"):
            embeddings = self.model._do_encode(
                texts,
                batch_size=batch_size,
                instruction=f"Instruct: {instruction}\nQuery: ",
                max_length=self.max_length,
                num_workers=0,
                return_numpy=False,
            )
            return F.normalize(embeddings.float(), p=2, dim=-1).cpu().numpy()

        vectors: list[torch.Tensor] = []
        device = next(self.model.parameters()).device
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for start in range(0, len(texts), batch_size):
            batch_index = start // batch_size + 1
            if len(texts) >= 10_000 and (batch_index == 1 or batch_index % 500 == 0 or batch_index == total_batches):
                print(f"encoding large input: batch {batch_index}/{total_batches} ({start}/{len(texts)})", flush=True)
            batch = texts[start : start + batch_size]
            vectors.append(self._encode_batch_with_oom_fallback(batch, device=device))
        return torch.cat(vectors, dim=0).numpy()

    def _encode_batch_with_oom_fallback(self, batch: list[str], *, device: torch.device) -> torch.Tensor:
        try:
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = self.model(**encoded, return_embeddings=True)
            embeddings = output["sentence_embeddings"] if isinstance(output, dict) else output
            return F.normalize(embeddings.float(), p=2, dim=-1).cpu()
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch) <= 1:
                raise
            midpoint = len(batch) // 2
            left = self._encode_batch_with_oom_fallback(batch[:midpoint], device=device)
            right = self._encode_batch_with_oom_fallback(batch[midpoint:], device=device)
            return torch.cat([left, right], dim=0)

    def _instruction_for(self, *, task_name, prompt_type, prompt_mode: str | None = None) -> str | None:
        prompt_mode = prompt_mode or self.prompt_mode
        if prompt_mode == "legacy_ru_masked":
            name = str(task_name or "")
            return self._legacy_ru_prefix_for(name, prompt_type)

        if task_name is None:
            return "Given a text, retrieve semantically similar text"

        task_type, prompt = self.task_prompts.get(str(task_name), ("", None))
        prompt_type_value = str(getattr(prompt_type, "value", prompt_type) or "")

        if task_type in {"Retrieval", "Reranking"}:
            if prompt_type_value != "query":
                return None
            if isinstance(prompt, dict):
                return prompt.get("query") or prompt.get("text")
            if isinstance(prompt, str) and prompt.strip():
                return prompt
            return "Given a question, retrieve relevant passages that answer the question"

        if self.symmetric_instruction == "ru_mteb":
            return RU_TASK_PROMPTS.get(str(task_name))
        if self.symmetric_instruction == "generic":
            return "Given a text, retrieve semantically similar text"
        if self.symmetric_instruction == "ru_generic":
            return "Дан текст, необходимо найти семантически похожий текст"
        if self.symmetric_instruction == "none":
            return None

        if isinstance(prompt, str) and prompt.strip():
            return prompt

        return None

    def _apply_legacy_ru_prefix(self, texts: list[str], *, task_name, prompt_type) -> list[str]:
        name = str(task_name or "")
        if name in self.legacy_prefix_overrides:
            return [self.legacy_prefix_overrides[name] + text for text in texts]

        prefix = self._legacy_ru_prefix_for(name, prompt_type)
        if not prefix:
            return texts
        return [prefix + text for text in texts]

    def _legacy_ru_prefix_for(self, name: str, prompt_type) -> str:
        if name in self.legacy_prefix_overrides:
            return self.legacy_prefix_overrides[name]

        task_type, _ = self.task_prompts.get(name, ("", None))
        prompt_type_value = str(getattr(prompt_type, "value", prompt_type) or "")

        if task_type in {"Retrieval", "Reranking"}:
            prefix = LEGACY_RU_PREFIXES["retrieval"] if prompt_type_value == "query" else ""
        elif name == "RUParaPhraserSTS":
            prefix = LEGACY_RU_PREFIXES["ruparaphraser"]
        elif name == "RuSTSBenchmarkSTS":
            prefix = LEGACY_RU_PREFIXES["rusts"]
        elif task_type == "STS":
            prefix = LEGACY_RU_PREFIXES["sts"]
        elif name == "SensitiveTopicsClassification":
            prefix = LEGACY_RU_PREFIXES["sensitive"]
        elif name == "InappropriatenessClassification":
            prefix = LEGACY_RU_PREFIXES["inappropriate"]
        elif name == "CEDRClassification":
            prefix = LEGACY_RU_PREFIXES["cedr"]
        elif name == "HeadlineClassification":
            prefix = LEGACY_RU_PREFIXES["headline"]
        elif name == "GeoreviewClassification":
            prefix = LEGACY_RU_PREFIXES["georeview_classification"]
        elif name == "GeoreviewClusteringP2P":
            prefix = LEGACY_RU_PREFIXES["georeview"]
        elif name == "RuSciBenchGRNTIClusteringP2P":
            prefix = LEGACY_RU_PREFIXES["science_clustering"]
        elif name == "RuSciBenchOECDClusteringP2P":
            prefix = LEGACY_RU_PREFIXES["science_clustering"]
        elif name == "TERRa":
            prefix = LEGACY_RU_PREFIXES["terra"]
        else:
            prefix = LEGACY_RU_PREFIXES["default"]

        return prefix

    def _normalize_text(self, text: str, *, text_normalization: str) -> str:
        modes = set(filter(None, text_normalization.split("+")))
        if "strip" in modes:
            text = text.strip()
        if "spaces" in modes:
            text = re.sub(r"\s+", " ", text).strip()
        if "yo" in modes:
            text = text.replace("ё", "е").replace("Ё", "Е")
        if "lower" in modes:
            text = text.lower()
        if "punctstrip" in modes:
            text = text.strip(string.whitespace + string.punctuation + "«»„“”‘’…—–")
        return text

    @staticmethod
    def _normalize_inputs(inputs) -> list[str]:
        if isinstance(inputs, DataLoader):
            texts: list[str] = []
            for batch in inputs:
                texts.extend(GigaOfficialMTEBWrapper._normalize_inputs(batch))
            return texts

        if isinstance(inputs, dict):
            if "text" in inputs:
                return GigaOfficialMTEBWrapper._string_list(inputs["text"])
            if "query" in inputs:
                return GigaOfficialMTEBWrapper._string_list(inputs["query"])
            if "body" in inputs:
                titles = GigaOfficialMTEBWrapper._string_list(inputs.get("title", []))
                bodies = GigaOfficialMTEBWrapper._string_list(inputs["body"])
                if titles and len(titles) == len(bodies):
                    return [(title + " " + body).strip() for title, body in zip(titles, bodies, strict=True)]
                return bodies
            raise TypeError(f"Unsupported MTEB input keys: {sorted(inputs.keys())}")

        return GigaOfficialMTEBWrapper._string_list(inputs)

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
    parser = argparse.ArgumentParser(description="Reproduce official MTEB(rus, v1) results for Giga-Embeddings-instruct.")
    parser.add_argument("--output-folder", type=Path, default=Path("results/official_repro"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/official_repro_cache"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32", "auto"), default="bfloat16")
    parser.add_argument(
        "--text-normalization",
        default="none",
        help="Optional + separated normalization: strip+spaces+yo+lower+punctstrip.",
    )
    parser.add_argument(
        "--text-suffix-override",
        action="append",
        default=[],
        metavar="TASK=SUFFIX",
        help="Append suffix to normalized text for a task. Supports escaped \\n and {eos}.",
    )
    parser.add_argument(
        "--task-prompt-mode",
        action="append",
        default=[],
        metavar="TASK=MODE",
        help="Override prompt mode for a task in multi-task runs.",
    )
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--prompt-mode",
        choices=("instruction", "prefix", "model_prefix", "legacy_ru", "legacy_ru_masked", "none"),
        default="instruction",
    )
    parser.add_argument("--symmetric-instruction", choices=("mteb", "ru_mteb", "generic", "ru_generic", "none"), default="mteb")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--latent-checkpoint", type=Path, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        default=True,
        help="Disable trust_remote_code=True for datasets. Full MTEB(rus, v1) needs it for MIRACL.",
    )
    parser.add_argument(
        "--reset-seed-per-task",
        action="store_true",
        help="Run selected tasks one by one and reset RNG state before each task.",
    )
    parser.add_argument(
        "--legacy-prefix-override",
        action="append",
        default=[],
        metavar="TASK=PREFIX",
        help="Override one legacy_ru prefix. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mteb-prompt-override",
        action="append",
        default=[],
        metavar="TASK=PROMPT",
        help="Override the MTEB task prompt used by prefix/instruction modes.",
    )
    parser.add_argument(
        "--legacy-prefix-ensemble",
        action="append",
        default=[],
        metavar="TASK=PREFIX1|||PREFIX2",
        help="Average embeddings from multiple legacy_ru prefixes for one task.",
    )
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MTEB_CACHE", str(args.cache_dir.resolve()))
    os.environ.setdefault("HF_HOME", str((args.cache_dir / "hf_home").resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str((args.cache_dir / "datasets").resolve()))
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    hf_modules = str((args.cache_dir / "hf_home" / "modules").resolve())
    if hf_modules not in sys.path:
        sys.path.insert(0, hf_modules)
    if args.trust_remote_code:
        patch_datasets_trust_remote_code()

    import mteb

    benchmark = mteb.get_benchmark("MTEB(rus, v1)")
    for task in benchmark.tasks:
        task.seed = DEFAULT_TASK_SEEDS.get(task.metadata.name, args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    task_by_name = {task.metadata.name: task for task in benchmark.tasks}
    selected = args.tasks or [task.metadata.name for task in benchmark.tasks]
    missing = sorted(set(selected) - set(task_by_name))
    if missing:
        raise SystemExit(f"Unknown MTEB(rus, v1) task(s): {', '.join(missing)}")

    selected_tasks = [task_by_name[name] for name in selected]
    task_prompts = {
        task.metadata.name: (str(task.metadata.type), getattr(task.metadata, "prompt", None))
        for task in benchmark.tasks
    }
    for task_name, prompt in DEFAULT_MTEB_PROMPT_OVERRIDES.items():
        task_type, _ = task_prompts.get(task_name, ("", None))
        task_prompts[task_name] = (task_type, prompt)
    for override in args.mteb_prompt_override:
        if "=" not in override:
            raise SystemExit(f"Invalid --mteb-prompt-override {override!r}; expected TASK=PROMPT")
        task_name, prompt = override.split("=", 1)
        task_type, _ = task_prompts.get(task_name, ("", None))
        task_prompts[task_name] = (task_type, prompt.replace("\\n", "\n"))
    legacy_prefix_overrides = {}
    for override in args.legacy_prefix_override:
        if "=" not in override:
            raise SystemExit(f"Invalid --legacy-prefix-override {override!r}; expected TASK=PREFIX")
        task_name, prefix = override.split("=", 1)
        legacy_prefix_overrides[task_name] = prefix.replace("\\n", "\n")
    legacy_prefix_ensembles = dict(DEFAULT_LEGACY_PREFIX_ENSEMBLES)
    for override in args.legacy_prefix_ensemble:
        if "=" not in override:
            raise SystemExit(f"Invalid --legacy-prefix-ensemble {override!r}; expected TASK=PREFIX1|||PREFIX2")
        task_name, prefixes = override.split("=", 1)
        legacy_prefix_ensembles[task_name] = [
            prefix.replace("\\n", "\n") for prefix in prefixes.split("|||")
        ]
    eos_token = model_tokenizer_eos = None
    text_suffix_overrides = {}
    if args.text_suffix_override:
        temp_tokenizer = None
        for override in args.text_suffix_override:
            if "=" not in override:
                raise SystemExit(f"Invalid --text-suffix-override {override!r}; expected TASK=SUFFIX")
            task_name, suffix = override.split("=", 1)
            suffix = suffix.replace("\\n", "\n")
            if "{eos}" in suffix:
                if temp_tokenizer is None:
                    temp_tokenizer = AutoTokenizer.from_pretrained(
                        MODEL_NAME,
                        revision=args.model_revision,
                        trust_remote_code=True,
                        local_files_only=args.local_files_only,
                    )
                eos_token = temp_tokenizer.eos_token or ""
                suffix = suffix.replace("{eos}", eos_token)
            text_suffix_overrides[task_name] = suffix
    task_prompt_modes = dict(DEFAULT_TASK_PROMPT_MODES)
    for override in args.task_prompt_mode:
        if "=" not in override:
            raise SystemExit(f"Invalid --task-prompt-mode {override!r}; expected TASK=MODE")
        task_name, mode = override.split("=", 1)
        if mode not in {"instruction", "prefix", "model_prefix", "legacy_ru", "legacy_ru_masked", "none"}:
            raise SystemExit(f"Invalid prompt mode override {mode!r} for {task_name}")
        task_prompt_modes[task_name] = mode

    model = GigaOfficialMTEBWrapper(
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_revision=args.model_revision,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
        latent_checkpoint=args.latent_checkpoint,
        task_prompts=task_prompts,
        prompt_mode=args.prompt_mode,
        symmetric_instruction=args.symmetric_instruction,
        legacy_prefix_overrides=legacy_prefix_overrides,
        legacy_prefix_ensembles=legacy_prefix_ensembles,
        text_normalization=args.text_normalization,
        text_suffix_overrides=text_suffix_overrides,
        task_prompt_modes=task_prompt_modes,
        task_text_normalizations=DEFAULT_TASK_TEXT_NORMALIZATIONS,
        task_batch_sizes=DEFAULT_TASK_BATCH_SIZES,
    )
    args.output_folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL_NAME,
        "model_revision": args.model_revision,
        "mteb_version": mteb.__version__,
        "benchmark": "MTEB(rus, v1)",
        "tasks": selected,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "latent_checkpoint": str(args.latent_checkpoint) if args.latent_checkpoint else None,
        "text_normalization": args.text_normalization,
        "prompt_mode": args.prompt_mode,
        "symmetric_instruction": args.symmetric_instruction,
        "legacy_prefix_overrides": legacy_prefix_overrides,
        "legacy_prefix_ensembles": legacy_prefix_ensembles,
        "text_suffix_overrides": text_suffix_overrides,
        "task_prompt_modes": task_prompt_modes,
        "task_text_normalizations": DEFAULT_TASK_TEXT_NORMALIZATIONS,
        "task_batch_sizes": DEFAULT_TASK_BATCH_SIZES,
        "task_seeds": DEFAULT_TASK_SEEDS,
        "reset_seed_per_task": args.reset_seed_per_task,
        "trust_remote_code": args.trust_remote_code,
        "prompt_policy": (
            "MTEB 1.38 task prompt; retrieval/reranking prompt only for query side. "
            "legacy_ru uses ai-sage discussion legacy Russian prefixes."
        ),
    }
    (args.output_folder / "repro_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.reset_seed_per_task:
        for task in selected_tasks:
            task_seed = DEFAULT_TASK_SEEDS.get(task.metadata.name, args.seed)
            task.seed = task_seed
            random.seed(task_seed)
            np.random.seed(task_seed)
            torch.manual_seed(task_seed)
            torch.cuda.manual_seed_all(task_seed)
            evaluator = mteb.MTEB(tasks=[task])
            evaluator.run(
                model,
                output_folder=str(args.output_folder),
                encode_kwargs={"batch_size": args.batch_size},
                overwrite_results=args.overwrite_results,
            )
    else:
        evaluator = mteb.MTEB(tasks=selected_tasks)
        evaluator.run(
            model,
            output_folder=str(args.output_folder),
            encode_kwargs={"batch_size": args.batch_size},
            overwrite_results=args.overwrite_results,
        )
    print(f"Wrote official reproduction results to {args.output_folder}")


if __name__ == "__main__":
    main()
