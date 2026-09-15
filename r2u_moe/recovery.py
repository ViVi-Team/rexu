"""Core recovery-expert discovery (factored for in-memory reuse in the adversarial loop)."""
from __future__ import annotations

from typing import Any

import torch

from .eval import evaluate_mcq_dataset
from .routing import (
    RouterInterventionConfig,
    apply_router_intervention,
    collect_topk_histogram,
    select_never_selected_experts,
)


def discover_recovery_experts(
    model,
    tokenizer,
    input_device,
    dataset: str,
    layers: list[int],
    target_experts: dict[int, list[int]],
    retain_prompts: list[str],
    candidate_limit: int = 8,
    delta: float = 0.02,
    max_eval_samples: int | None = 256,
    batch_size: int = 4,
    max_length: int = 1024,
) -> tuple[float, dict[str, list[int]], float, list[dict[str, Any]]]:
    """
    Returns (baseline_fe, recovery_experts, worst_route_fe, rows).

    Expert e is a recovery expert if forcing routing to it raises FE > baseline_fe + delta.
    Uses A3 never-selected strategy: candidates are experts with zero retain top-k count.
    """
    normal = evaluate_mcq_dataset(
        model, tokenizer, dataset, input_device,
        max_examples=max_eval_samples, batch_size=batch_size, desc="normal_FE",
    )
    baseline_fe = float(normal["accuracy"])

    histograms = collect_topk_histogram(
        model, tokenizer, retain_prompts, layers, max_length, batch_size=1,
    )
    candidates = select_never_selected_experts(histograms, target_experts, max(1, candidate_limit))

    rows: list[dict[str, Any]] = []
    forced_fes: list[float] = [baseline_fe]

    for layer in layers:
        for expert in candidates.get(layer, []):
            config = RouterInterventionConfig(
                mode="force_expert", layers=[layer], force_experts={layer: [expert]},
            )
            with apply_router_intervention(model, config):
                score = evaluate_mcq_dataset(
                    model, tokenizer, dataset, input_device,
                    max_examples=max_eval_samples, batch_size=batch_size,
                    desc=f"force_l{layer}_e{expert}",
                )
            forced_fe = float(score["accuracy"])
            forced_fes.append(forced_fe)
            row: dict[str, Any] = {
                "layer": int(layer),
                "expert": int(expert),
                "normal_FE": baseline_fe,
                "forced_FE": forced_fe,
                "delta_FE": forced_fe - baseline_fe,
                "is_recovery_expert": forced_fe > baseline_fe + delta,
                "retain_topk_count": int(histograms[layer][expert]) if layer in histograms else None,
            }
            rows.append(row)

    recovery_experts: dict[str, list[int]] = {}
    for row in rows:
        if row["is_recovery_expert"]:
            recovery_experts.setdefault(str(row["layer"]), []).append(int(row["expert"]))
    recovery_experts = {layer_s: sorted(set(exps)) for layer_s, exps in recovery_experts.items()}

    worst_route_fe = max(forced_fes)
    return baseline_fe, recovery_experts, worst_route_fe, rows
