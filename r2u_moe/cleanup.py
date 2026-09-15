"""Core weight-baked output-activation cleanup (factored for in-memory reuse in the adversarial loop)."""
from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from .routing import RouterInterventionConfig, apply_router_intervention


class _OutputCollector:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.chunks: list[torch.Tensor] = []
        self.count = 0

    def add(self, tensor: torch.Tensor) -> None:
        if self.count >= self.max_tokens:
            return
        flat = tensor.detach().float().cpu().reshape(-1, tensor.shape[-1])
        keep = min(flat.shape[0], self.max_tokens - self.count)
        if keep > 0:
            self.chunks.append(flat[:keep])
            self.count += int(keep)

    def tensor(self) -> torch.Tensor:
        if not self.chunks:
            raise RuntimeError("No expert outputs collected. Check layer/expert and prompts.")
        return torch.cat(self.chunks, dim=0)


def get_expert_module(model, layer: int, expert: int):
    try:
        return model.model.layers[layer].mlp.experts[expert]
    except Exception as exc:
        raise AttributeError(
            f"Could not find model.model.layers[{layer}].mlp.experts[{expert}]"
        ) from exc


def collect_expert_outputs(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
    expert: int,
    max_tokens: int,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    collector = _OutputCollector(max_tokens=max_tokens)
    module = get_expert_module(model, layer, expert)

    def hook(_module, _inputs, output):
        collector.add(output)

    handle = module.register_forward_hook(hook)
    config = RouterInterventionConfig(
        mode="force_expert", layers=[layer], force_experts={layer: [expert]},
    )
    input_device = next(model.parameters()).device
    try:
        with apply_router_intervention(model, config):
            for start in tqdm(range(0, len(prompts), batch_size), desc=f"collect_l{layer}_e{expert}"):
                if collector.count >= max_tokens:
                    break
                batch = prompts[start : start + batch_size]
                encoded = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                )
                encoded = {k: v.to(input_device) for k, v in encoded.items()}
                with torch.inference_mode():
                    _ = model(**encoded, use_cache=False)
    finally:
        handle.remove()
    return collector.tensor()


def build_orthonormal_basis(
    forget_outputs: torch.Tensor,
    retain_outputs: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rank-r orthonormal basis for the forget-minus-retain output-activation subspace."""
    mean_diff = forget_outputs.mean(dim=0) - retain_outputs.mean(dim=0)
    if float(mean_diff.norm()) <= 1e-8:
        raise RuntimeError(
            "Forget/retain expert-output mean difference is near zero; "
            "cleanup subspace is undefined."
        )
    directions = [mean_diff / mean_diff.norm()]

    if rank > 1:
        n = min(len(forget_outputs), len(retain_outputs))
        contrast = (
            (forget_outputs[:n] - forget_outputs[:n].mean(dim=0, keepdim=True))
            - (retain_outputs[:n] - retain_outputs[:n].mean(dim=0, keepdim=True))
        )
        _, _s, vh = torch.linalg.svd(contrast.float(), full_matrices=False)
        for row in vh:
            vec = row.clone()
            for existing in directions:
                vec = vec - torch.dot(vec, existing) * existing
            norm = vec.norm()
            if float(norm) > 1e-8:
                directions.append(vec / norm)
            if len(directions) >= rank:
                break

    basis = torch.stack(directions, dim=1)
    return basis, {
        "rank_requested": rank,
        "rank_used": int(basis.shape[1]),
        "mean_diff_norm": float(mean_diff.norm()),
        "forget_tokens": int(forget_outputs.shape[0]),
        "retain_tokens": int(retain_outputs.shape[0]),
    }


@torch.no_grad()
def bake_down_proj(expert_module, basis: torch.Tensor) -> dict[str, float]:
    """Project forget subspace out of down_proj in-place. ASSERTS residual ~= 0."""
    if not hasattr(expert_module, "down_proj"):
        raise AttributeError("Expert module has no down_proj; cannot bake output-space projection.")
    weight = expert_module.down_proj.weight
    basis = basis.to(device=weight.device, dtype=torch.float32)
    w_fp32 = weight.data.float()
    removed_before = basis.T @ w_fp32
    before_norm = float(removed_before.norm().detach().cpu())
    projected = w_fp32 - basis @ removed_before
    weight.data.copy_(projected.to(dtype=weight.dtype))
    removed_after = (basis.T @ weight.data.float()).norm()
    after_norm = float(removed_after.detach().cpu())
    threshold = max(1e-3, before_norm * 1e-3)
    if after_norm > threshold:
        raise RuntimeError(
            f"Projection bake failed: removed component norm {before_norm:.4f} -> {after_norm:.6f} "
            f"(threshold {threshold:.6f})"
        )
    return {"removed_component_norm_before": before_norm, "removed_component_norm_after": after_norm}


def bake_cleanup_experts(
    model,
    tokenizer,
    recovery_experts: dict[str, list[int]],
    forget_prompts: list[str],
    retain_prompts: list[str],
    rank: int = 4,
    max_tokens: int = 4096,
    max_length: int = 1024,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """
    Bake output-activation cleanup into recovery experts in-place.
    Returns list of per-expert audit rows.
    """
    rows: list[dict[str, Any]] = []
    for layer_str, experts in recovery_experts.items():
        layer = int(layer_str)
        for expert in experts:
            f_out = collect_expert_outputs(
                model, tokenizer, forget_prompts, layer, expert,
                max_tokens, max_length, batch_size,
            )
            r_out = collect_expert_outputs(
                model, tokenizer, retain_prompts, layer, expert,
                max_tokens, max_length, batch_size,
            )
            basis, stats = build_orthonormal_basis(f_out, r_out, rank=max(1, rank))
            bake = bake_down_proj(get_expert_module(model, layer, expert), basis)
            print(
                f"  Baked l{layer}_e{expert}: "
                f"removed_norm {bake['removed_component_norm_before']:.4f} -> "
                f"{bake['removed_component_norm_after']:.6f}",
                flush=True,
            )
            rows.append({"layer": layer, "expert": expert, "subspace": stats, "bake_assertion": bake})
    return rows
