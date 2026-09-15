from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset


CHOICE_LABELS = ["A", "B", "C", "D"]
WMDP_DESCRIPTIONS = {
    "wmdp-cyber": "The following are multiple choice questions (with answers) about cybersecurity.\n\n",
    "wmdp-bio": "The following are multiple choice questions (with answers) about biology.\n\n",
    "wmdp-chem": "The following are multiple choice questions (with answers) about chemistry.\n\n",
}


def patch_transformers_for_deepseek() -> None:
    """Make DeepSeek remote code load with eager attention when flash-attn is absent."""
    import transformers.dynamic_module_utils as dynamic_module_utils
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    import_utils.is_flash_attn_2_available = lambda: False
    import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
    transformers_utils.is_flash_attn_2_available = lambda: False
    transformers_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False

    if getattr(dynamic_module_utils.check_imports, "_r2u_patched", False):
        return

    original = dynamic_module_utils.check_imports

    def patched(filename):
        try:
            return original(filename)
        except ImportError:
            return []

    patched._r2u_patched = True
    dynamic_module_utils.check_imports = patched


def _resolve_model_path(model_or_checkpoint: str) -> str:
    path = Path(model_or_checkpoint).expanduser()
    if path.is_absolute() or model_or_checkpoint.startswith("."):
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        return str(path.resolve())
    if path.exists():
        return str(path.resolve())
    return model_or_checkpoint


def load_model_and_tokenizer(model_or_checkpoint: str, dtype: str = "bf16", device_map: str | dict = "auto"):
    patch_transformers_for_deepseek()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_or_checkpoint = _resolve_model_path(model_or_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_or_checkpoint, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype_map = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}
    torch_dtype = dtype_map.get(dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_or_checkpoint,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def format_prompt(question: str, choices: list[str], description: str = "") -> str:
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, choices))
    return f"{description}{str(question).strip()}\n{options}\nAnswer:"


def mmlu_description(subject: str | None) -> str:
    if not subject:
        return ""
    clean_subject = str(subject).replace("_", " ")
    return f"The following are multiple choice questions (with answers) about {clean_subject}.\n\n"


def answer_to_index(answer: Any) -> int:
    if isinstance(answer, str):
        return CHOICE_LABELS.index(answer.strip().upper())
    return int(answer)


def parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    parsed = []
    for item in str(value).split(","):
        cleaned = item.strip()
        if cleaned:
            parsed.append(int(cleaned))
    return sorted(set(parsed))


def _coerce_experts(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        cleaned = value.lower().replace("expert", "").replace("[", "").replace("]", "")
        return parse_int_list(cleaned)
    if isinstance(value, dict):
        for key in ("expert", "expert_id", "id", "idx", "experts", "selected_experts"):
            if key in value:
                return _coerce_experts(value[key])
    out = []
    for item in value:
        out.extend(_coerce_experts(item))
    return sorted(set(out))


def parse_layer_expert_map(
    value: str | Path | dict | None,
    default_layers: list[int] | None = None,
    default_experts: list[int] | None = None,
) -> dict[int, list[int]]:
    default_layers = default_layers or []
    default_experts = default_experts or []
    if isinstance(value, (str, Path)) and Path(str(value)).exists():
        with open(value, encoding="utf-8") as handle:
            value = json.load(handle)

    out: dict[int, set[int]] = {}

    def add(layer: int, experts: list[int]) -> None:
        if experts:
            out.setdefault(int(layer), set()).update(int(expert) for expert in experts)

    if value is None or value == "":
        for layer in default_layers:
            add(layer, default_experts)
        return {layer: sorted(experts) for layer, experts in out.items()}

    if isinstance(value, str):
        bare_experts = []
        for token in [part.strip() for part in value.split(",") if part.strip()]:
            if ":" not in token:
                bare_experts.extend(_coerce_experts(token))
                continue
            layer_text, expert_text = token.split(":", 1)
            layer_text = layer_text.lower().replace("layer", "").strip()
            if not layer_text.isdigit():
                raise ValueError(f"Could not parse layer from expert spec: {token}")
            add(int(layer_text), _coerce_experts(expert_text))
        for layer in default_layers:
            add(layer, bare_experts)
        return {layer: sorted(experts) for layer, experts in out.items()}

    if isinstance(value, dict):
        for wrapper in ("target_experts", "force_experts", "recovery_experts", "expert_config"):
            if wrapper in value:
                return parse_layer_expert_map(value[wrapper], default_layers, default_experts)
        layers = value.get("layers") or value.get("layer_indices")
        experts = value.get("experts") or value.get("selected_experts")
        if layers is not None and experts is not None:
            for layer in _coerce_experts(layers):
                add(layer, _coerce_experts(experts))
        for key, item in value.items():
            key_text = str(key).lower().replace("layer_", "").replace("layer", "")
            if key_text.isdigit():
                add(int(key_text), _coerce_experts(item))

    if not out:
        for layer in default_layers:
            add(layer, default_experts)
    return {layer: sorted(experts) for layer, experts in out.items()}


def _limit(items: list[Any], max_examples: int | None) -> list[Any]:
    if max_examples is None or max_examples <= 0:
        return items
    return items[:max_examples]


def mcq_item_id(question: str, choices: list[str]) -> str:
    """Stable per-item ID, independent of dataset row order.

    Used to join item-level results across separate eval passes (e.g. different
    checkpoints or route interventions) run at different times.
    """
    payload = question + "\x1f" + "\x1e".join(choices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_mcq_examples(dataset: str, max_examples: int | None = None) -> list[dict[str, Any]]:
    name = dataset.replace("_", "-")
    if name in {"wmdp-cyber", "wmdp-bio", "wmdp-chem"}:
        rows = load_dataset("cais/wmdp", name, split="test")
        description = WMDP_DESCRIPTIONS[name]
        examples = [
            {
                "item_id": mcq_item_id(row["question"], row["choices"]),
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "prompt": format_prompt(row["question"], row["choices"], description),
                "prompt_description": description,
            }
            for row in rows
        ]
        return _limit(examples, max_examples)
    if name == "mmlu":
        rows = load_dataset("cais/mmlu", "all", split="test")
        examples = []
        for row in rows:
            description = mmlu_description(row.get("subject"))
            examples.append(
                {
                    "item_id": mcq_item_id(row["question"], row["choices"]),
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                    "subject": row.get("subject"),
                    "prompt": format_prompt(row["question"], row["choices"], description),
                    "prompt_description": description,
                }
            )
        return _limit(examples, max_examples)
    raise ValueError(f"Unknown MCQ dataset: {dataset}")


def load_text_prompts(dataset: str, split: str = "forget", max_examples: int | None = None) -> tuple[list[str], dict[str, Any]]:
    name = dataset.replace("-", "_")
    split = split.replace("-", "_")
    if name in {"wmdp_cyber", "wmdp"} and split in {"forget", "member"}:
        ds = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")
        return _limit([row["text"] for row in ds], max_examples), {"source": "cais/wmdp-corpora/cyber-forget-corpus"}
    if name in {"wmdp_cyber", "wmdp"} and split in {"retain", "nonmember", "dn"}:
        ds = load_dataset("cais/wmdp-corpora", "cyber-retain-corpus", split="train")
        return _limit([row["text"] for row in ds], max_examples), {"source": "cais/wmdp-corpora/cyber-retain-corpus"}
    if name in {"wmdp_cyber", "wmdp_bio", "wmdp_chem", "mmlu"}:
        examples = load_mcq_examples(name, max_examples)
        return [row["prompt"] for row in examples], {"source": name, "split": "test"}

    path = Path(dataset).expanduser()
    if not path.exists():
        raise ValueError(f"Unknown dataset or file: {dataset}")
    if path.suffix == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row.get("text") or row.get("prompt") or row.get("question") or line.strip())
        return _limit(rows, max_examples), {"source": str(path.resolve())}
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        rows = [row.get("text") or row.get("prompt") or row.get("question") or str(row) for row in data]
        return _limit(rows, max_examples), {"source": str(path.resolve())}
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return _limit(rows, max_examples), {"source": str(path.resolve())}


def checkpoint_fingerprint(model_path: str) -> dict[str, Any]:
    if "/" in model_path and not Path(model_path).expanduser().exists():
        return {"model_path_realpath": model_path, "checkpoint_file": None, "checkpoint_sha256": None}
    root = Path(model_path).expanduser()
    candidates = [
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
        root / "model.safetensors",
        root / "pytorch_model.bin",
        root / "config.json",
    ]
    chosen = next((path for path in candidates if path.exists()), None)
    digest = None
    if chosen is not None:
        h = hashlib.sha256()
        with open(chosen, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
    return {
        "model_path_realpath": str(root.resolve()),
        "checkpoint_file": str(chosen.resolve()) if chosen is not None else None,
        "checkpoint_sha256": digest,
    }
