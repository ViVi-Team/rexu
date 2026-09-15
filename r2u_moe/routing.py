from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from math import log
from types import MethodType
from typing import Any
import weakref

import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_gate(model, layer_idx: int):
    try:
        mlp = model.model.layers[layer_idx].mlp
        gate = mlp.gate
        if hasattr(gate, "_modules") and "_r2u_parent_moe" in gate._modules:
            del gate._modules["_r2u_parent_moe"]
        if not hasattr(gate, "_r2u_parent_moe_ref"):
            object.__setattr__(gate, "_r2u_parent_moe_ref", weakref.ref(mlp))
        return gate
    except AttributeError as exc:
        raise AttributeError(f"Could not find model.model.layers[{layer_idx}].mlp.gate") from exc


def list_router_layers(model) -> list[int]:
    layers = getattr(getattr(model, "model", None), "layers", [])
    return [idx for idx, layer in enumerate(layers) if hasattr(getattr(layer, "mlp", None), "gate")]


@dataclass
class RouterInterventionConfig:
    mode: str
    layers: list[int]
    target_experts: dict[int, list[int]] = field(default_factory=dict)
    force_experts: dict[int, list[int]] = field(default_factory=dict)
    gate_noise_std: float = 0.0
    router_temperature: float = 1.0
    uniform_mix: float = 0.0


def _gate_logits(module, hidden_states: torch.Tensor) -> torch.Tensor:
    prefix = hidden_states.shape[:-1]
    hidden = hidden_states.shape[-1]
    flat = hidden_states.reshape(-1, hidden)
    bias = getattr(module, "bias", None)
    bias = None if bias is None else bias.float()
    return F.linear(flat.float(), module.weight.float(), bias).view(*prefix, -1)


def _moe_attr(module, name: str, default: Any = None) -> Any:
    if hasattr(module, name):
        return getattr(module, name)
    parent_ref = getattr(module, "_r2u_parent_moe_ref", None)
    parent = parent_ref() if callable(parent_ref) else None
    if parent is not None and hasattr(parent, name):
        return getattr(parent, name)
    return default


def _num_experts(module, fallback: int = 0) -> int:
    value = _moe_attr(module, "n_routed_experts", None)
    if value is None:
        value = _moe_attr(module, "num_experts", None)
    if value is None and hasattr(module, "out_features"):
        value = getattr(module, "out_features")
    if value is None and hasattr(module, "weight"):
        value = int(module.weight.shape[0])
    value = int(value if value is not None else fallback)
    return max(0, value)


def _top_k(module, n_experts: int) -> int:
    value = _moe_attr(module, "top_k", None)
    if value is None:
        value = _moe_attr(module, "num_experts_per_tok", None)
    value = int(value if value is not None else 1)
    return max(1, min(value, max(1, int(n_experts))))


def _norm_topk_prob(module) -> bool:
    return bool(_moe_attr(module, "norm_topk_prob", True))


def _routed_scaling_factor(module) -> float:
    return float(_moe_attr(module, "routed_scaling_factor", 1.0))


def _scoring_func(module) -> str:
    return str(_moe_attr(module, "scoring_func", "softmax"))


def _logits_gate(module) -> bool:
    # Qwen2-MoE uses a plain Linear gate and the sparse block computes top-k
    # from returned logits. DeepSeek's custom gate returns (topk_idx, weight, aux).
    return not hasattr(module, "topk_method")


def _flat_topk_gate(module) -> bool:
    # Qwen3-Next's router returns (router_logits, routing_weights, selected_experts)
    # from already-flattened (N, hidden) input -- distinct from both DeepSeek's
    # (topk_idx, weight, aux_loss) convention and Qwen2-MoE's plain-logits gate.
    return type(module).__name__ == "Qwen3NextTopKRouter"


def _topk_from_scores(module, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_experts = scores.shape
    flat = scores.reshape(bsz * seq_len, n_experts)
    top_k = _top_k(module, n_experts)
    topk_method = str(_moe_attr(module, "topk_method", "greedy"))
    if topk_method == "greedy":
        topk_weight, topk_idx = torch.topk(flat, k=top_k, dim=-1, sorted=False)
    elif topk_method == "group_limited_greedy":
        n_group = int(_moe_attr(module, "n_group"))
        topk_group = int(_moe_attr(module, "topk_group"))
        group_scores = flat.view(bsz * seq_len, n_group, -1).max(dim=-1).values
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, n_group, n_experts // n_group)
            .reshape(bsz * seq_len, -1)
        )
        topk_weight, topk_idx = torch.topk(flat.masked_fill(~score_mask.bool(), 0.0), k=top_k, dim=-1, sorted=False)
    else:
        raise NotImplementedError(f"Unsupported top-k method: {topk_method}")

    if top_k > 1 and _norm_topk_prob(module):
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        topk_weight = topk_weight * _routed_scaling_factor(module)
    return topk_idx, topk_weight


def _forced_topk(module, scores: torch.Tensor, experts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_experts = scores.shape
    flat = scores.reshape(bsz * seq_len, n_experts)
    top_k = _top_k(module, n_experts)
    forced = []
    for expert in experts:
        expert = int(expert)
        if 0 <= expert < n_experts and expert not in forced:
            forced.append(expert)
        if len(forced) >= top_k:
            break
    if not forced:
        return _topk_from_scores(module, scores)
    forced_ids = torch.tensor(forced, dtype=torch.long, device=scores.device)
    if len(forced) < top_k:
        masked = flat.clone()
        masked.index_fill_(dim=-1, index=forced_ids, value=-torch.inf)
        filler = torch.topk(masked, k=top_k - len(forced), dim=-1, sorted=False).indices
        topk_idx = torch.cat([forced_ids.unsqueeze(0).expand(flat.shape[0], -1), filler], dim=-1)
    else:
        topk_idx = forced_ids.unsqueeze(0).expand(flat.shape[0], -1).clone()
    topk_weight = flat.gather(dim=-1, index=topk_idx).clamp_min(0.0)
    empty = topk_weight.sum(dim=-1, keepdim=True) <= 1e-20
    if empty.any():
        topk_weight = torch.where(empty, torch.full_like(topk_weight, 1.0 / top_k), topk_weight)
    if top_k > 1 and _norm_topk_prob(module):
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        topk_weight = topk_weight * _routed_scaling_factor(module)
    return topk_idx, topk_weight


def _force_logits(module, logits: torch.Tensor, experts: list[int]) -> torch.Tensor:
    n_experts = logits.shape[-1]
    top_k = _top_k(module, n_experts)
    forced: list[int] = []
    for expert in experts:
        expert = int(expert)
        if 0 <= expert < n_experts and expert not in forced:
            forced.append(expert)
        if len(forced) >= top_k:
            break
    if not forced:
        return logits
    flat = logits.reshape(-1, n_experts).clone()
    finite = flat.masked_fill(~torch.isfinite(flat), -1e9)
    high = finite.max(dim=-1, keepdim=True).values + float(len(forced) + 1)
    for rank, expert in enumerate(forced):
        flat[:, expert] = (high[:, 0] - float(rank)).to(flat.dtype)
    return flat.view_as(logits)


def extract_gate_topk(module, output) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return flattened (topk_idx, topk_weight, n_experts) for DeepSeek/Qwen gates."""
    if isinstance(output, (tuple, list)) and len(output) >= 2 and torch.is_tensor(output[0]):
        first = output[0]
        if first.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            idx = first.detach().long().reshape(-1)
            weight = output[1].detach().float().reshape(-1)
            fallback = int(idx.max().item()) + 1 if idx.numel() else 0
            return idx, weight, _num_experts(module, fallback)
    logits = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(logits):
        raise TypeError(f"Unsupported gate output type: {type(output)}")
    n_experts = _num_experts(module, int(logits.shape[-1]))
    k = _top_k(module, n_experts)
    scores = logits.detach().float().softmax(dim=-1)
    weight, idx = torch.topk(scores, k=k, dim=-1, sorted=False)
    if k > 1 and _norm_topk_prob(module):
        weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)
    return idx.long().reshape(-1), weight.reshape(-1), n_experts


def _patched_gate_forward(module, hidden_states, layer_idx: int, config: RouterInterventionConfig):
    logits = _gate_logits(module, hidden_states)
    if config.gate_noise_std > 0:
        logits = logits + torch.randn_like(logits) * float(config.gate_noise_std)
    if config.mode == "mask_target":
        experts = config.target_experts.get(layer_idx, [])
        if experts:
            idx = torch.tensor(experts, dtype=torch.long, device=logits.device)
            logits = logits.index_fill(dim=-1, index=idx, value=-torch.inf)
    if config.router_temperature != 1.0:
        logits = logits / float(config.router_temperature)
    if _scoring_func(module) != "softmax":
        raise NotImplementedError(f"Unsupported gate scoring function: {_scoring_func(module)}")
    forced = config.force_experts.get(layer_idx, []) if config.mode == "force_expert" else []
    if _flat_topk_gate(module):
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        if config.uniform_mix > 0:
            uniform = torch.full_like(scores, 1.0 / scores.shape[-1])
            scores = (1.0 - float(config.uniform_mix)) * scores + float(config.uniform_mix) * uniform
        # _topk_from_scores/_forced_topk only use scores.shape to derive bsz*seq_len via
        # reshape, then return already-flat (N, top_k) tensors -- a fake leading batch
        # dim of 1 satisfies the 3-value unpack without changing the output shape.
        scores_3d = scores.unsqueeze(0)
        topk_idx, topk_weight = _forced_topk(module, scores_3d, forced) if forced else _topk_from_scores(module, scores_3d)
        return logits, topk_weight.to(logits.dtype), topk_idx
    if _logits_gate(module):
        if config.uniform_mix > 0:
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            uniform = torch.full_like(scores, 1.0 / scores.shape[-1])
            scores = (1.0 - float(config.uniform_mix)) * scores + float(config.uniform_mix) * uniform
            logits = scores.clamp_min(1e-20).log()
        if forced:
            logits = _force_logits(module, logits, forced)
        return logits
    scores = logits.softmax(dim=-1, dtype=torch.float32)
    if config.uniform_mix > 0:
        uniform = torch.full_like(scores, 1.0 / scores.shape[-1])
        scores = (1.0 - float(config.uniform_mix)) * scores + float(config.uniform_mix) * uniform
    topk_idx, topk_weight = _forced_topk(module, scores, forced) if forced else _topk_from_scores(module, scores)
    aux_loss = logits.new_zeros(())
    return topk_idx, topk_weight, aux_loss


@contextmanager
def apply_router_intervention(model, config: RouterInterventionConfig | dict[str, Any] | None):
    if config is None:
        yield
        return
    if isinstance(config, dict):
        config = RouterInterventionConfig(**config)
    originals = []
    for layer_idx in config.layers:
        gate = get_gate(model, layer_idx)
        originals.append((gate, gate.forward))

        def forward(self, hidden_states, layer_idx=layer_idx, config=config):
            return _patched_gate_forward(self, hidden_states, layer_idx, config)

        gate.forward = MethodType(forward, gate)
    try:
        yield
    finally:
        for gate, original in originals:
            gate.forward = original


@torch.inference_mode()
def collect_topk_histogram(model, tokenizer, prompts: list[str], layers: list[int], max_length: int, batch_size: int) -> dict[int, torch.Tensor]:
    input_device = next(model.parameters()).device
    caches: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    n_experts: dict[int, int] = {}

    def make_hook(layer_idx: int):
        def hook(module, _inputs, output):
            topk, _weight, n = extract_gate_topk(module, output)
            topk = topk.detach().cpu().long().reshape(-1)
            valid = (topk >= 0) & (topk < int(n))
            caches[layer_idx].append(topk[valid])
            n_experts[layer_idx] = int(n)
            return None

        return hook

    handles = [get_gate(model, layer_idx).register_forward_hook(make_hook(layer_idx)) for layer_idx in layers]
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Retain route census"):
            batch = prompts[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            _ = model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    histograms: dict[int, torch.Tensor] = {}
    for layer in layers:
        counts = torch.zeros(n_experts.get(layer, 0), dtype=torch.long)
        for topk in caches[layer]:
            if counts.numel() and topk.numel():
                counts.scatter_add_(0, topk, torch.ones_like(topk, dtype=torch.long))
        histograms[layer] = counts
    return histograms


def select_never_selected_experts(histograms: dict[int, torch.Tensor], exclude: dict[int, list[int]], per_layer: int) -> dict[int, list[int]]:
    selected = {}
    for layer, counts in histograms.items():
        excluded = set(int(expert) for expert in exclude.get(layer, []))
        order = sorted(range(int(counts.numel())), key=lambda expert: (int(counts[expert]), expert))
        chosen = []
        for expert in order:
            if expert in excluded:
                continue
            chosen.append(int(expert))
            if len(chosen) >= per_layer:
                break
        selected[int(layer)] = chosen
    return selected


@torch.inference_mode()
def collect_routing_traces(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int],
    max_length: int = 1024,
    batch_size: int = 1,
    target_experts: dict[int, list[int]] | None = None,
    save_token_level: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_device = next(model.parameters()).device
    caches: dict[int, list[dict[str, torch.Tensor]]] = {layer: [] for layer in layers}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            logits = _gate_logits(module, inputs[0]).detach().cpu()
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            bsz, seq_len = logits.shape[:2]
            caches[layer_idx].append(
                {
                    "router_probs": probs,
                    "topk_indices": output[0].detach().cpu().view(bsz, seq_len, -1),
                    "topk_probs": output[1].detach().cpu().view(bsz, seq_len, -1),
                }
            )
            return None

        return hook

    handles = [get_gate(model, layer_idx).register_forward_hook(make_hook(layer_idx)) for layer_idx in layers]
    examples = []
    try:
        for start in tqdm(range(0, len(prompts), batch_size), desc="Collect routing traces"):
            batch = prompts[start : start + batch_size]
            for layer_cache in caches.values():
                layer_cache.clear()
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded["attention_mask"].to(input_device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for local_idx in range(len(batch)):
                mask = attention_mask[local_idx].detach().cpu().bool()
                item = {"index": start + local_idx, "layers": {}}
                for layer_idx in layers:
                    capture = caches[layer_idx][0]
                    probs = capture["router_probs"][local_idx][mask]
                    topk = capture["topk_indices"][local_idx][mask]
                    topk_probs = capture["topk_probs"][local_idx][mask]
                    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    n_experts = probs.shape[-1]
                    hist = torch.zeros(n_experts, dtype=torch.float32)
                    if topk.numel():
                        hist.scatter_add_(0, topk.reshape(-1), torch.ones(topk.numel()))
                        hist = hist / hist.sum().clamp_min(1.0)
                    target_mass = None
                    target_overlap = None
                    if target_experts and layer_idx in target_experts:
                        target = torch.tensor(target_experts[layer_idx], dtype=torch.long)
                        target_mass = probs.index_select(dim=-1, index=target).sum(dim=-1)
                        target_overlap = (topk.unsqueeze(-1) == target).any(dim=-1).any(dim=-1).float()
                    payload: dict[str, Any] = {
                        "sequence": {
                            "num_tokens": int(mask.sum()),
                            "n_experts": int(n_experts),
                            "entropy": float(entropy.mean()),
                            "kl_to_uniform": float(log(float(n_experts)) - float(entropy.mean())),
                            "mean_router_probs": probs.mean(dim=0),
                            "topk_expert_histogram": hist,
                        }
                    }
                    if target_mass is not None:
                        payload["sequence"]["target_mass"] = float(target_mass.mean())
                        payload["sequence"]["target_overlap"] = float(target_overlap.mean())
                    if save_token_level:
                        payload["topk_indices"] = topk
                        payload["topk_probs"] = topk_probs
                        payload["entropy"] = entropy
                    item["layers"][str(layer_idx)] = payload
                examples.append(item)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "schema_version": 1,
        "metadata": {
            **(metadata or {}),
            "num_examples": len(examples),
            "layers": layers,
            "target_experts": {str(k): v for k, v in (target_experts or {}).items()},
            "max_length": max_length,
            "batch_size": batch_size,
        },
        "examples": examples,
    }
