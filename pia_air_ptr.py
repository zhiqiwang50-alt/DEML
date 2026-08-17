import argparse
import ast
import csv
import inspect
import json
import math
import os
import statistics
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pia_alpha_gm import alpha_nn_initialization, stage_a_dummy_proxy
from pia_attention_guided import random_public_embeddings
from pia_masked_server_attn_pia import (
    SAWConfig,
    bounded_mean_one_weights,
    build_variable_mask,
    load_model_and_data,
    server_attention_bundle,
    server_forward_with_attention,
    stage_b_optimize,
    variable_mask_audit_payload,
    weighted_variable_activation_loss,
)
from pia_tinyllama import (
    MODEL_NAME,
    TOTAL_BLOCKS,
    bleu_score,
    capture_prefix_activation,
    inferred_boundary_special_tokens,
    json_dump,
    jsonl_append,
    load_dataset_prompts,
    naive_discretization,
    optional_nerr,
    peak_memory_mb,
    set_seed,
    text_from_ids,
    token_accuracy,
    token_texts,
)


EXPERIMENT_TITLE = "AIR-PTR: Attention-Initialized Residual Prompt Inversion with Projection-Time Repair"
METHODS = ("B0", "B0V", "ASINIT", "LAR", "LAR_PTR", "FINAL")
PTR_METHODS = {"LAR_PTR", "FINAL"}
ASINIT_METHODS = {"ASINIT", "FINAL"}
LAR_METHODS = {"LAR", "LAR_PTR", "FINAL"}
BANNED_ATTACK_NAMES = {
    "prompt",
    "prompt_text",
    "original_ids",
    "original_tokens",
    "ground_truth_ids",
    "ground_truth_embedding",
    "reference_token",
    "reference_prompt",
}


@dataclass(frozen=True)
class PTRConfig:
    eta: float = 0.10
    repair_fraction: float = 0.20
    max_positions_per_pass: int = 8
    max_passes: int = 2
    epsilon_accept: float = 1e-6


@dataclass
class PTRResult:
    token_ids: List[int]
    stats: Dict[str, Any]
    events: List[Dict[str, Any]]


@dataclass
class AttackResult:
    recovered_ids: List[int]
    pre_ptr_ids: List[int]
    losses: Dict[str, Any]
    attention_rollout: Dict[str, Any]
    alpha_init_stats: Dict[str, Any]
    layer_adaptive_stats: Dict[str, Any]
    ptr_stats: Dict[str, Any]
    variable_mask_audit: Dict[str, Any]
    stage_a_history: List[Dict[str, Any]]
    stage_b_history: List[Dict[str, Any]]


def validate_strict_top1(*, k: int, y: int, semantic_speculation: bool) -> None:
    if int(k) != 1:
        raise ValueError(f"AIR-PTR requires strict Top-1: K=1, got {k}")
    if int(y) != 0:
        raise ValueError(f"AIR-PTR requires strict Top-1: Y=0, got {y}")
    if bool(semantic_speculation):
        raise ValueError("AIR-PTR requires semantic_speculation=false")


def late_schedule(step_index: int, epoch: int, start_ratio: float = 0.50, full_ratio: float = 0.70) -> float:
    total = max(1, int(epoch))
    progress = min(1.0, max(0.0, float(step_index) / float(total)))
    start = min(1.0, max(0.0, float(start_ratio)))
    full = min(1.0, max(start, float(full_ratio)))
    if progress <= start:
        return 0.0
    if progress >= full or full <= start:
        return 1.0
    return (progress - start) / max(1e-12, full - start)


def layer_adaptive_weights(
    *,
    normalized_importance: torch.Tensor,
    variable_mask: torch.Tensor,
    step_index: int,
    epoch: int,
    rho: float,
    weight_min: float,
    weight_max: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not (0.0 <= float(rho) <= 1.0):
        raise ValueError("rho must be in [0, 1]")
    variable = variable_mask.to(normalized_importance.device).bool()
    if int(variable.sum().item()) == 0:
        raise RuntimeError("variable_mask has no recoverable positions")
    schedule = late_schedule(step_index, epoch)
    raw = normalized_importance.detach().float().clamp_min(1e-12)
    target = torch.zeros_like(raw)
    target[variable] = 1.0 + schedule * float(rho) * (raw[variable] - 1.0)
    weights = bounded_mean_one_weights(target, variable, float(weight_min), float(weight_max))
    values = weights[variable]
    audit = {
        "formula": "w_i(t,l) = 1 + s(t) * rho_l * (r_i - 1)",
        "schedule": float(schedule),
        "rho": float(rho),
        "mean_variable_weight": float(values.mean().item()),
        "min_variable_weight": float(values.min().item()),
        "max_variable_weight": float(values.max().item()),
        "fixed_public_final_weight_sum": float(weights[~variable].sum().item()),
        "negative_weight_count": int((weights < 0).sum().item()),
    }
    return weights.detach(), audit


def _nearest_cosine_id(vector: torch.Tensor, embedding_weight: torch.Tensor, chunk_size: int = 4096) -> int:
    query = F.normalize(vector.detach().float().view(1, -1), dim=-1)
    weight = embedding_weight.detach().float()
    best_id = 0
    best_value = -float("inf")
    for start in range(0, weight.shape[0], chunk_size):
        scores = query @ F.normalize(weight[start : start + chunk_size], dim=-1).t()
        value, index = torch.max(scores[0], dim=0)
        scalar = float(value.item())
        if scalar > best_value:
            best_value = scalar
            best_id = start + int(index.item())
    return best_id


def projection_time_repair(
    *,
    initial_ids: Sequence[int],
    embedding_weight: torch.Tensor,
    variable_mask: torch.Tensor,
    objective_from_embeddings: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    eta: float,
    repair_fraction: float,
    max_positions_per_pass: int,
    max_passes: int,
    epsilon_accept: float,
) -> PTRResult:
    current_ids = [int(x) for x in initial_ids]
    variable = variable_mask.detach().bool().to(embedding_weight.device)
    if len(current_ids) != int(variable.numel()):
        raise ValueError("initial_ids and variable_mask lengths differ")
    variable_positions = [int(x) for x in torch.nonzero(variable, as_tuple=False).flatten().tolist()]
    with torch.no_grad():
        initial_embeds = embedding_weight[torch.tensor(current_ids, device=embedding_weight.device)].unsqueeze(0)
        initial_loss_tensor, _ = objective_from_embeddings(initial_embeds)
        initial_loss = float(initial_loss_tensor.detach().cpu())
    events: List[Dict[str, Any]] = []
    proposed_positions: List[int] = []
    accepted_positions: List[int] = []
    attempted = set()
    accepted_count = 0
    rejected_count = 0
    unchanged_count = 0
    proposal_count = 0
    pass_count = 0

    if float(eta) > 0.0 and int(max_passes) > 0 and variable_positions:
        per_pass_limit = min(
            int(max_positions_per_pass),
            max(1, int(math.ceil(float(repair_fraction) * len(variable_positions)))),
        )
        for pass_index in range(int(max_passes)):
            accepted_this_pass = 0
            attempted_this_pass = 0
            for _ in range(per_pass_limit):
                available = [pos for pos in variable_positions if pos not in attempted]
                if not available:
                    break
                ids_tensor = torch.tensor(current_ids, dtype=torch.long, device=embedding_weight.device)
                current_embeds = embedding_weight[ids_tensor].detach().unsqueeze(0).requires_grad_(True)
                current_loss_tensor, residual_tensor = objective_from_embeddings(current_embeds)
                if not torch.isfinite(current_loss_tensor):
                    raise RuntimeError("NaN/Inf in PTR activation objective")
                current_loss_tensor.backward()
                if current_embeds.grad is None:
                    raise RuntimeError("PTR objective produced no embedding gradient")
                residual = residual_tensor.detach().reshape(-1)
                position = max(available, key=lambda pos: float(residual[pos].detach().cpu()))
                attempted.add(position)
                attempted_this_pass += 1
                proposal_count += 1
                proposed_positions.append(position)
                grad = current_embeds.grad[0, position].detach().float()
                norm = grad.norm()
                old_id = int(current_ids[position])
                old_loss = float(current_loss_tensor.detach().cpu())
                if float(norm.detach().cpu()) <= 1e-12:
                    new_id = old_id
                else:
                    proposal_vector = embedding_weight[old_id].detach().float() - float(eta) * grad / norm
                    new_id = _nearest_cosine_id(proposal_vector, embedding_weight)
                if new_id == old_id:
                    unchanged_count += 1
                    events.append(
                        {
                            "pass": pass_index + 1,
                            "position": position,
                            "old_token_id": old_id,
                            "proposal_token_id": new_id,
                            "candidate_count": 1,
                            "accepted": False,
                            "unchanged_proposal": True,
                            "old_activation_loss": old_loss,
                            "new_activation_loss": old_loss,
                        }
                    )
                    continue
                candidate_ids = list(current_ids)
                candidate_ids[position] = int(new_id)
                with torch.no_grad():
                    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=embedding_weight.device)
                    candidate_embeds = embedding_weight[candidate_tensor].unsqueeze(0)
                    candidate_loss_tensor, _ = objective_from_embeddings(candidate_embeds)
                candidate_loss = float(candidate_loss_tensor.detach().cpu())
                accepted = candidate_loss < old_loss - float(epsilon_accept)
                if accepted:
                    current_ids = candidate_ids
                    accepted_count += 1
                    accepted_this_pass += 1
                    accepted_positions.append(position)
                else:
                    rejected_count += 1
                events.append(
                    {
                        "pass": pass_index + 1,
                        "position": position,
                        "old_token_id": old_id,
                        "proposal_token_id": int(new_id),
                        "candidate_count": 1,
                        "accepted": bool(accepted),
                        "unchanged_proposal": False,
                        "old_activation_loss": old_loss,
                        "new_activation_loss": candidate_loss,
                    }
                )
            if attempted_this_pass == 0:
                break
            pass_count += 1
            if accepted_this_pass == 0:
                break

    with torch.no_grad():
        final_tensor = torch.tensor(current_ids, dtype=torch.long, device=embedding_weight.device)
        final_loss_tensor, _ = objective_from_embeddings(embedding_weight[final_tensor].unsqueeze(0))
        final_loss = float(final_loss_tensor.detach().cpu())
    stats = {
        "eta": float(eta),
        "repair_fraction": float(repair_fraction),
        "max_positions_per_pass": int(max_positions_per_pass),
        "max_passes": int(max_passes),
        "epsilon_accept": float(epsilon_accept),
        "pass_count": pass_count,
        "proposal_count": proposal_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "unchanged_proposal_count": unchanged_count,
        "proposed_positions": proposed_positions,
        "accepted_positions": accepted_positions,
        "pre_ptr_activation_loss": initial_loss,
        "post_ptr_activation_loss": final_loss,
        "activation_improvement": initial_loss - final_loss,
        "single_nn_proposal_per_position": len(proposed_positions) == len(set(proposed_positions)),
        "uses_ground_truth_for_accept_reject": False,
    }
    return PTRResult(token_ids=current_ids, stats=stats, events=events)


def _make_ptr_objective(
    model: torch.nn.Module,
    target_layer: int,
    observed_activation: torch.Tensor,
    variable_mask: torch.Tensor,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    seq_len = int(observed_activation.shape[1])
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=observed_activation.device)
    target = observed_activation.detach()

    def objective(embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = capture_prefix_activation(
            model,
            target_layer,
            inputs_embeds=embeds.to(dtype=model.get_input_embeddings().weight.dtype),
            attention_mask=attention_mask,
        )
        per_token = torch.mean((hidden.float() - target.to(hidden.device).float()) ** 2, dim=-1)[0]
        active = variable_mask.to(per_token.device).bool()
        return per_token[active].mean(), per_token

    return objective


def _fixed_public_audit(weights: Optional[torch.Tensor], variable_mask: torch.Tensor) -> Dict[str, Any]:
    if weights is None:
        return {}
    variable = variable_mask.to(weights.device).bool()
    values = weights[variable]
    return {
        "mean_variable_final_weight": float(values.mean().detach().cpu()),
        "std_variable_final_weight": float(values.std(unbiased=False).detach().cpu()) if values.numel() > 1 else 0.0,
        "min_variable_final_weight": float(values.min().detach().cpu()),
        "max_variable_final_weight": float(values.max().detach().cpu()),
        "fixed_public_final_weight_sum": float(weights[~variable].sum().detach().cpu()),
    }


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: SAWConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    ptr_config: Optional[PTRConfig] = None,
) -> AttackResult:
    method = str(cfg.method)
    if method not in METHODS:
        raise ValueError(f"unknown AIR-PTR method {method!r}")
    validate_strict_top1(
        k=cfg.top_k_embedding,
        y=cfg.top_y_semantic,
        semantic_speculation=cfg.semantic_speculation,
    )
    ptr_config = ptr_config or PTRConfig()
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    variable_audit = build_variable_mask(
        attention_mask=attention_mask,
        fixed_public=fixed_public,
        special_token_ids=getattr(tokenizer, "all_special_ids", None),
        token_ids=None,
    )
    mask_payload = variable_mask_audit_payload(variable_audit)
    stage_a_history: List[Dict[str, Any]] = []
    alpha_init_stats: Dict[str, Any] = {
        "used": False,
        "gamma": None,
        "A_self_is_dummy_attention_proxy": True,
        "uses_true_prompt_attention": False,
    }
    attention_rollout: Dict[str, Any] = {"used": False}
    layer_stats: Dict[str, Any] = {
        "used": False,
        "rho": float(cfg.residual_alpha_rho),
        "formula": "w_i(t,l) = 1 + s(t) * rho_l * (r_i - 1)",
    }

    if method in ASINIT_METHODS:
        alpha_cfg = replace(cfg, method="alpha_nn_init_residual")
        _dummy, a_self, attention_stats, stage_a_history = stage_a_dummy_proxy(
            model,
            tokenizer,
            alpha_cfg,
            observed_activation,
            seq_len,
            device,
            fixed_public,
        )
        init_embeds, init_ids, _h_init, init_audit = alpha_nn_initialization(
            model,
            tokenizer,
            alpha_cfg,
            observed_activation,
            a_self,
        )
        alpha_init_stats = {
            "used": True,
            "gamma": float(cfg.gamma),
            "A_self_is_dummy_attention_proxy": True,
            "uses_true_prompt_attention": False,
            "stage_a_uses_ground_truth": False,
            "init_token_ids": init_ids,
            "attention_proxy_stats": attention_stats,
            "initialization_audit": init_audit,
        }
    else:
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)

    server_weights: Optional[torch.Tensor] = None
    if method == "B0":
        optimize_cfg = replace(cfg, method="original_pia_baseline")
    elif method in {"B0V", "ASINIT"}:
        optimize_cfg = replace(cfg, method="variable_only_uniform")
    else:
        optimize_cfg = replace(
            cfg,
            method="attn_linear_weighted_residual_schedule",
            weight_source="mean_query",
            weight_power=0.5,
            beta=1.0,
            alpha_min=0.5,
            alpha_max=1.5,
            attention_start_ratio=0.50,
            attention_full_ratio=0.70,
            residual_rollout=True,
            server_rollout_depth="all",
        )
        server_weights, server_stats, rollout_stats, weight_stats = server_attention_bundle(
            model,
            observed_activation,
            attention_mask,
            optimize_cfg,
            fixed_public,
            variable_audit,
        )
        attention_rollout = {
            "used": True,
            "weight_source": "mean_query",
            "weight_power": 0.5,
            "residual_rollout": True,
            "rollout_depth": "all",
            "server_attention_stats": server_stats,
            "rollout_stats": rollout_stats,
        }
        layer_stats.update(weight_stats)
        layer_stats.update(_fixed_public_audit(server_weights, variable_audit.variable_mask))
        layer_stats.update(
            {
                "used": True,
                "rho": float(cfg.residual_alpha_rho),
                "attention_start_ratio": 0.50,
                "attention_full_ratio": 0.70,
                "weight_min": 0.5,
                "weight_max": 1.5,
            }
        )

    z, losses, stage_b_history = stage_b_optimize(
        model,
        tokenizer,
        optimize_cfg,
        observed_activation,
        seq_len,
        device,
        fixed_public,
        variable_audit,
        init_embeds,
        server_weights,
        a_context=None,
    )
    pre_ptr_ids = naive_discretization(z, embed_layer.weight)
    for position, token_id in fixed_public.items():
        if 0 <= position < len(pre_ptr_ids):
            pre_ptr_ids[position] = int(token_id)
    recovered_ids = list(pre_ptr_ids)
    ptr_stats: Dict[str, Any] = {
        "used": False,
        "proposal_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "unchanged_proposal_count": 0,
        "accepted_positions": [],
        "uses_ground_truth_for_accept_reject": False,
        "events": [],
    }
    if method in PTR_METHODS:
        objective = _make_ptr_objective(
            model,
            cfg.target_layer,
            observed_activation,
            variable_audit.variable_mask,
        )
        repair = projection_time_repair(
            initial_ids=pre_ptr_ids,
            embedding_weight=embed_layer.weight,
            variable_mask=variable_audit.variable_mask,
            objective_from_embeddings=objective,
            eta=ptr_config.eta,
            repair_fraction=ptr_config.repair_fraction,
            max_positions_per_pass=ptr_config.max_positions_per_pass,
            max_passes=ptr_config.max_passes,
            epsilon_accept=ptr_config.epsilon_accept,
        )
        recovered_ids = repair.token_ids
        ptr_stats = {"used": True, **repair.stats, "events": repair.events}

    losses = {
        **losses,
        "air_method": method,
        "strict_top1": True,
        "top_k_embedding": 1,
        "top_y_semantic": 0,
        "semantic_speculation": False,
    }
    return AttackResult(
        recovered_ids=recovered_ids,
        pre_ptr_ids=pre_ptr_ids,
        losses=losses,
        attention_rollout=attention_rollout,
        alpha_init_stats=alpha_init_stats,
        layer_adaptive_stats=layer_stats,
        ptr_stats=ptr_stats,
        variable_mask_audit=mask_payload,
        stage_a_history=stage_a_history,
        stage_b_history=stage_b_history,
    )


def validate_attack_api() -> Dict[str, Any]:
    checked = [invert_observed, projection_time_repair, _make_ptr_objective]
    signature_hits: Dict[str, List[str]] = {}
    ast_hits: List[Dict[str, str]] = []
    for function in checked:
        names = list(inspect.signature(function).parameters)
        hits = [name for name in names if name in BANNED_ATTACK_NAMES]
        if hits:
            signature_hits[function.__name__] = hits
        source = inspect.getsource(function)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in BANNED_ATTACK_NAMES:
                ast_hits.append({"function": function.__name__, "name": node.id})
    return {
        "passes": not signature_hits and not ast_hits,
        "signature_hits": signature_hits,
        "ast_hits": ast_hits,
        "checked_functions": [fn.__name__ for fn in checked],
    }


def _mean_std(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "std": None}
    return {
        "mean": float(statistics.mean(clean)),
        "std": float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0,
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "token_accuracy": _mean_std([row["token_accuracy"] for row in rows]),
        "bleu": _mean_std([row["bleu"] for row in rows]),
        "runtime_seconds": _mean_std([row["runtime_seconds"] for row in rows]),
        "peak_gpu_memory_mb": _mean_std([row.get("peak_gpu_memory_mb") for row in rows]),
        "sample_count": len(rows),
    }


def _ensure_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _evaluate_ptr(pre_ids: Sequence[int], post_ids: Sequence[int], truth_ids: Sequence[int]) -> Dict[str, int]:
    fixes = 0
    harms = 0
    neutral = 0
    for before, after, truth in zip(pre_ids, post_ids, truth_ids):
        if int(before) == int(after):
            neutral += 1
        elif int(before) != int(truth) and int(after) == int(truth):
            fixes += 1
        elif int(before) == int(truth) and int(after) != int(truth):
            harms += 1
        else:
            neutral += 1
    return {"ptr_fix": fixes, "ptr_harm": harms, "ptr_neutral": neutral}


def run_one_config(cfg: SAWConfig, ptr_config: PTRConfig, resume: bool) -> Dict[str, Any]:
    validate_strict_top1(
        k=cfg.top_k_embedding,
        y=cfg.top_y_semantic,
        semantic_speculation=cfg.semantic_speculation,
    )
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    complete_path = output / "COMPLETE"
    if resume and complete_path.exists() and (output / "metrics.json").exists():
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    json_dump(
        output / "config.json",
        {
            "title": EXPERIMENT_TITLE,
            **cfg.__dict__,
            "ptr": ptr_config.__dict__,
            "strict_top1_definition": "K=1, Y=0, semantic_speculation=false; one embedding NN token only",
        },
    )
    tokenizer, model, device = load_model_and_data(cfg)
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(output / "dataset_meta.json", dataset_meta)
    predictions_path = output / "predictions.jsonl"
    failures_path = output / "failures.jsonl"
    if not resume:
        for path in (predictions_path, failures_path):
            if path.exists():
                path.unlink()
    _ensure_jsonl(predictions_path)
    _ensure_jsonl(failures_path)
    rows: List[Dict[str, Any]] = []
    completed_ids = set()
    if resume:
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                completed_ids.add(int(row["prompt_id"]))
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed_ids.add(int(json.loads(line)["prompt_id"]))
    failures: List[Dict[str, Any]] = []
    all_masks: List[Dict[str, Any]] = []
    all_rollouts: List[Dict[str, Any]] = []
    all_alpha: List[Dict[str, Any]] = []
    all_lar: List[Dict[str, Any]] = []
    all_ptr: List[Dict[str, Any]] = []
    all_losses: List[Dict[str, Any]] = []

    for prompt_id, prompt in enumerate(prompts):
        if prompt_id in completed_ids:
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.time()
            tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            if int(input_ids.shape[1]) > cfg.max_token_len:
                raise RuntimeError(
                    f"prompt_id={prompt_id} has {input_ids.shape[1]} tokens > max_token_len={cfg.max_token_len}"
                )
            evaluation_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
            with torch.no_grad():
                observed = capture_prefix_activation(
                    model,
                    cfg.target_layer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).detach()
            attack = invert_observed(
                model,
                tokenizer,
                cfg,
                observed,
                int(input_ids.shape[1]),
                device,
                ptr_config,
            )
            elapsed = time.time() - started
            pre_text = text_from_ids(tokenizer, attack.pre_ptr_ids)
            recovered_text = text_from_ids(tokenizer, attack.recovered_ids)
            original_text = text_from_ids(tokenizer, evaluation_ids)
            ptr_eval = _evaluate_ptr(attack.pre_ptr_ids, attack.recovered_ids, evaluation_ids)
            row = {
                "method": cfg.method,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "original_text": original_text,
                "pre_ptr_text": pre_text,
                "recovered_text": recovered_text,
                "original_token_ids": evaluation_ids,
                "pre_ptr_token_ids": attack.pre_ptr_ids,
                "recovered_token_ids": attack.recovered_ids,
                "original_tokens": token_texts(tokenizer, evaluation_ids),
                "recovered_tokens": token_texts(tokenizer, attack.recovered_ids),
                "token_accuracy": token_accuracy(tokenizer, evaluation_ids, attack.recovered_ids),
                "bleu": bleu_score(tokenizer, evaluation_ids, attack.recovered_ids),
                "pre_ptr_token_accuracy": token_accuracy(tokenizer, evaluation_ids, attack.pre_ptr_ids),
                "pre_ptr_bleu": bleu_score(tokenizer, evaluation_ids, attack.pre_ptr_ids),
                "nerr": optional_nerr(original_text, recovered_text),
                "runtime_seconds": elapsed,
                "peak_gpu_memory_mb": peak_memory_mb(),
                "prompt_token_count": len(evaluation_ids),
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "K": 1,
                "Y": 0,
                "semantic_speculation": False,
                "losses": attack.losses,
                "ptr": {**attack.ptr_stats, **ptr_eval},
                "attack_receives_ground_truth": False,
            }
            jsonl_append(predictions_path, row)
            rows.append(row)
            all_masks.append({"prompt_id": prompt_id, **attack.variable_mask_audit})
            all_rollouts.append({"prompt_id": prompt_id, **attack.attention_rollout})
            all_alpha.append({"prompt_id": prompt_id, **attack.alpha_init_stats})
            all_lar.append({"prompt_id": prompt_id, **attack.layer_adaptive_stats})
            all_ptr.append({"prompt_id": prompt_id, **attack.ptr_stats, **ptr_eval})
            all_losses.append(
                {
                    "prompt_id": prompt_id,
                    "final": attack.losses,
                    "stage_a_history": attack.stage_a_history,
                    "stage_b_history": attack.stage_b_history,
                }
            )
            print(
                f"method={cfg.method} prompt_id={prompt_id} acc={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} ptr_accept={attack.ptr_stats.get('accepted_count', 0)} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )
        except Exception as error:
            failure = {
                "method": cfg.method,
                "prompt_id": prompt_id,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
            }
            jsonl_append(failures_path, failure)
            failures.append(failure)
            print(f"failure={json.dumps(failure, ensure_ascii=True)}", flush=True)

    metrics = summarize_rows(rows)
    ptr_proposals = sum(int(row.get("ptr", {}).get("proposal_count", 0)) for row in rows)
    ptr_accepts = sum(int(row.get("ptr", {}).get("accepted_count", 0)) for row in rows)
    ptr_fixes = sum(int(row.get("ptr", {}).get("ptr_fix", 0)) for row in rows)
    ptr_harms = sum(int(row.get("ptr", {}).get("ptr_harm", 0)) for row in rows)
    metrics.update(
        {
            "title": EXPERIMENT_TITLE,
            "method": cfg.method,
            "dataset": cfg.dataset_name,
            "dataset_meta": dataset_meta,
            "seed": cfg.seed,
            "target_layer": cfg.target_layer,
            "K": 1,
            "Y": 0,
            "semantic_speculation": False,
            "completed_sample_count": len(rows),
            "failed_sample_count": len(failures),
            "nan_count": sum(
                1
                for row in rows
                if not math.isfinite(float(row["token_accuracy"])) or not math.isfinite(float(row["bleu"]))
            ),
            "ptr_proposal_count": ptr_proposals,
            "ptr_accepted_count": ptr_accepts,
            "ptr_fix": ptr_fixes,
            "ptr_harm": ptr_harms,
            "repair_precision": ptr_fixes / max(1, ptr_fixes + ptr_harms),
            "net_repaired_tokens": ptr_fixes - ptr_harms,
            "output_dir": cfg.output_dir,
        }
    )
    json_dump(output / "metrics.json", metrics)
    json_dump(output / "variable_mask_audit.json", {"samples": all_masks})
    json_dump(output / "attention_rollout.json", {"samples": all_rollouts})
    json_dump(output / "alpha_init_stats.json", {"samples": all_alpha})
    json_dump(output / "layer_adaptive_stats.json", {"samples": all_lar})
    json_dump(output / "ptr_stats.json", {"samples": all_ptr})
    json_dump(output / "loss_breakdown.json", {"samples": all_losses})
    complete_path.write_text("complete\n", encoding="utf-8")
    return metrics


def _bootstrap_ci(values: Sequence[float], iterations: int = 10000, seed: int = 2026) -> Tuple[float, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(data, size=(iterations, data.size), replace=True).mean(axis=1)
    lower, upper = np.percentile(sampled, [2.5, 97.5])
    return float(lower), float(upper)


def write_comparison(root: Path) -> List[Dict[str, Any]]:
    metrics_rows: List[Dict[str, Any]] = []
    predictions: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method in METHODS:
        candidates = sorted(root.glob(f"**/{method}/metrics.json"))
        if not candidates:
            candidates = sorted(root.glob(f"**/{method}_*/metrics.json"))
        if not candidates:
            continue
        metrics_path = candidates[-1]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_rows.append(
            {
                "method": method,
                "Token Accuracy": metrics.get("token_accuracy", {}).get("mean"),
                "BLEU": metrics.get("bleu", {}).get("mean"),
                "completed": metrics.get("completed_sample_count"),
                "failed": metrics.get("failed_sample_count"),
                "PTR proposal count": metrics.get("ptr_proposal_count", 0),
                "PTR accepted count": metrics.get("ptr_accepted_count", 0),
                "PTR fix": metrics.get("ptr_fix", 0),
                "PTR harm": metrics.get("ptr_harm", 0),
                "runtime": metrics.get("runtime_seconds", {}).get("mean"),
                "peak GPU memory": metrics.get("peak_gpu_memory_mb", {}).get("mean"),
                "metrics_path": str(metrics_path),
            }
        )
        pred_path = metrics_path.with_name("predictions.jsonl")
        predictions[method] = {}
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    predictions[method][int(row["prompt_id"])] = row
    by_method = {row["method"]: row for row in metrics_rows}
    for row in metrics_rows:
        for baseline in ("B0", "B0V"):
            base = by_method.get(baseline)
            row[f"delta Acc vs {baseline}"] = (
                float(row["Token Accuracy"]) - float(base["Token Accuracy"])
                if base and row["Token Accuracy"] is not None and base["Token Accuracy"] is not None
                else None
            )
            row[f"delta BLEU vs {baseline}"] = (
                float(row["BLEU"]) - float(base["BLEU"])
                if base and row["BLEU"] is not None and base["BLEU"] is not None
                else None
            )
    fields = list(metrics_rows[0]) if metrics_rows else []
    if metrics_rows:
        with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metrics_rows)
        lines = ["# AIR-PTR Comparison", "", "| " + " | ".join(fields[:-1]) + " |", "| " + " | ".join(["---"] * (len(fields) - 1)) + " |"]
        for row in metrics_rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields[:-1]) + " |")
        (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    paired_rows: List[Dict[str, Any]] = []
    b0v = predictions.get("B0V", {})
    for method in METHODS:
        if method == "B0V" or method not in predictions:
            continue
        ids = sorted(set(b0v) & set(predictions[method]))
        acc_delta = [predictions[method][i]["token_accuracy"] - b0v[i]["token_accuracy"] for i in ids]
        bleu_delta = [predictions[method][i]["bleu"] - b0v[i]["bleu"] for i in ids]
        if not ids:
            continue
        wins = sum(value > 1e-12 for value in acc_delta)
        losses = sum(value < -1e-12 for value in acc_delta)
        ties = len(ids) - wins - losses
        acc_ci = _bootstrap_ci(acc_delta)
        bleu_ci = _bootstrap_ci(bleu_delta)
        paired_rows.append(
            {
                "method": method,
                "paired_count": len(ids),
                "mean_token_accuracy_delta_vs_B0V": float(statistics.mean(acc_delta)),
                "token_accuracy_ci95_low": acc_ci[0],
                "token_accuracy_ci95_high": acc_ci[1],
                "mean_bleu_delta_vs_B0V": float(statistics.mean(bleu_delta)),
                "bleu_ci95_low": bleu_ci[0],
                "bleu_ci95_high": bleu_ci[1],
                "win": wins,
                "tie": ties,
                "loss": losses,
            }
        )
    if paired_rows:
        fields = list(paired_rows[0])
        with (root / "paired_vs_b0v.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(paired_rows)
        lines = ["# Paired Results vs B0V", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in paired_rows:
            lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
        (root / "paired_vs_b0v.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ptr_rows = [
        {
            "method": row["method"],
            "proposal": row["PTR proposal count"],
            "accepted": row["PTR accepted count"],
            "fix": row["PTR fix"],
            "harm": row["PTR harm"],
        }
        for row in metrics_rows
        if row["method"] in PTR_METHODS
    ]
    if ptr_rows:
        fields = list(ptr_rows[0])
        with (root / "ptr_stats.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ptr_rows)
        lines = ["# PTR Statistics", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in ptr_rows:
            lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
        (root / "ptr_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_rows


def config_from_args(args: argparse.Namespace, method: str, output_dir: str, rho: Optional[float] = None) -> SAWConfig:
    target = int(args.target_layer)
    return SAWConfig(
        method=method,
        run_name=Path(output_dir).name,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=int(args.dataset_len),
        seed=int(args.seed),
        participant_number=4,
        attacker_position=4,
        inverted_block_count=target + 1,
        target_layer=target,
        epoch=int(args.epoch),
        stage_a_epoch=int(args.stage_a_epoch if args.stage_a_epoch is not None else args.epoch),
        lr=float(args.lr),
        lambda_vocab=float(args.lambda_vocab),
        lambda_dummy=float(args.lambda_dummy),
        lambda_context=0.0,
        top_k_embedding=int(args.k),
        top_y_semantic=int(args.y),
        gamma=float(args.gamma),
        max_token_len=int(args.max_token_len),
        grad_clip=float(args.grad_clip),
        weight_source="mean_query",
        weight_floor=0.0,
        weight_power=0.5,
        weight_min=0.5,
        weight_max=1.5,
        alpha_min=0.5,
        alpha_max=1.5,
        attention_start_ratio=0.50,
        attention_full_ratio=0.70,
        uncertainty_fraction=0.0,
        adaptive_min_beta=0.0,
        refine_epoch=0,
        lambda_projection=0.0,
        refine_lr_scale=1.0,
        beta=1.0,
        last_window_size=10,
        residual_rollout=True,
        server_rollout_depth="all",
        adaptive_discretization=False,
        semantic_speculation=not bool(args.disable_semantic_speculation),
        local_files_only=bool(args.local_files_only),
        residual_alpha_rho=float(args.rho if rho is None else rho),
        fix_boundary_specials=True,
    )


def ptr_config_from_args(args: argparse.Namespace) -> PTRConfig:
    return PTRConfig(
        eta=float(args.ptr_eta),
        repair_fraction=float(args.repair_fraction),
        max_positions_per_pass=int(args.max_positions_per_pass),
        max_passes=int(args.max_passes),
        epsilon_accept=float(args.epsilon_accept),
    )


def run_single(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir or str(
        Path(args.output_root) / f"seed{args.seed}_layer{args.target_layer}" / str(args.method)
    )
    cfg = config_from_args(args, args.method, output_dir)
    return run_one_config(cfg, ptr_config_from_args(args), args.resume)


def run_smoke(args: argparse.Namespace) -> None:
    base = Path(args.output_root) / "smoke"
    for method in METHODS:
        cfg = config_from_args(args, method, str(base / method))
        run_one_config(cfg, ptr_config_from_args(args), args.resume)
    write_comparison(base)


def run_rho_ablation(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    rows: List[Dict[str, Any]] = []
    for layer in args.target_layers:
        for rho in args.rhos:
            layer_args = argparse.Namespace(**vars(args))
            layer_args.target_layer = int(layer)
            output_dir = root / "rho_ablation" / f"layer{layer}" / f"rho{str(rho).replace('.', 'p')}"
            cfg = config_from_args(layer_args, "LAR", str(output_dir), rho=float(rho))
            metrics = run_one_config(cfg, ptr_config_from_args(args), args.resume)
            rows.append(
                {
                    "layer": int(layer),
                    "rho": float(rho),
                    "Token Accuracy": metrics["token_accuracy"]["mean"],
                    "BLEU": metrics["bleu"]["mean"],
                    "completed": metrics["completed_sample_count"],
                    "failed": metrics["failed_sample_count"],
                    "NaN": metrics["nan_count"],
                    "output_dir": str(output_dir),
                }
            )
    with (root / "dev_rho_ablation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# AIR-PTR rho ablation", "", "| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in rows[0]) + " |")
    (root / "dev_rho_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected = []
    for layer in args.target_layers:
        valid = [
            row
            for row in rows
            if row["layer"] == int(layer)
            and row["completed"] == int(args.dataset_len)
            and row["failed"] == 0
            and row["NaN"] == 0
        ]
        valid.sort(key=lambda row: (-float(row["Token Accuracy"]), -float(row["BLEU"]), float(row["rho"])))
        if not valid:
            raise RuntimeError(f"no valid rho candidate for layer {layer}")
        selected.append(valid[0])
    with (root / "layer_rho_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    global_candidates = []
    for rho in args.rhos:
        matching = [row for row in rows if float(row["rho"]) == float(rho)]
        if len(matching) == len(args.target_layers):
            global_candidates.append(
                {
                    "rho": float(rho),
                    "mean_Token_Accuracy": float(statistics.mean(row["Token Accuracy"] for row in matching)),
                    "mean_BLEU": float(statistics.mean(row["BLEU"] for row in matching)),
                }
            )
    global_candidates.sort(key=lambda row: (-row["mean_Token_Accuracy"], -row["mean_BLEU"], row["rho"]))
    with (root / "global_rho_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(global_candidates[0]))
        writer.writeheader()
        writer.writerows(global_candidates)
    frozen = {
        "gamma": float(args.gamma),
        "rho_by_layer": {str(row["layer"]): float(row["rho"]) for row in selected},
        "rho_global": float(global_candidates[0]["rho"]),
        "weight_source": "mean_query",
        "weight_power": 0.5,
        "residual_rollout": True,
        "rollout_depth": "all",
        "attention_start_ratio": 0.50,
        "attention_full_ratio": 0.70,
        "weight_min": 0.5,
        "weight_max": 1.5,
        "ptr": ptr_config_from_args(args).__dict__,
    }
    json_dump(root / "frozen_layer_config.json", frozen)


def run_dev(args: argparse.Namespace) -> None:
    frozen_path = Path(args.output_root) / "frozen_layer_config.json"
    if not frozen_path.exists():
        raise RuntimeError("run rho-ablation before dev")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    rho = float(frozen["rho_by_layer"][str(args.target_layer)])
    root = Path(args.output_root) / "dev"
    for method in METHODS:
        cfg = config_from_args(args, method, str(root / method), rho=rho)
        run_one_config(cfg, ptr_config_from_args(args), args.resume)
    rows = write_comparison(root)
    by_method = {row["method"]: row for row in rows}
    final = by_method.get("FINAL")
    b0v = by_method.get("B0V")
    if not final or not b0v:
        raise RuntimeError("dev comparison is missing FINAL or B0V")
    delta = float(final["Token Accuracy"]) - float(b0v["Token Accuracy"])
    ptr_negative = int(final["PTR harm"]) > int(final["PTR fix"])
    decision = [
        "# AIR-PTR Development Decision",
        "",
        f"- FINAL Token Accuracy delta vs B0V: {delta:.6f}",
        f"- FINAL BLEU delta vs B0V: {float(final['BLEU']) - float(b0v['BLEU']):.6f}",
        f"- PTR fix/harm: {final['PTR fix']}/{final['PTR harm']}",
        f"- PTR negative component: {ptr_negative}",
        f"- Reached +2pp target: {delta >= 0.020 and not ptr_negative}",
        f"- May expand automatically: {delta >= 0.010 and not ptr_negative}",
        "",
        "Dev results alone are not evidence of stable or significant improvement.",
    ]
    Path("analysis/air_ptr_dev_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")


def run_verify(args: argparse.Namespace) -> Dict[str, Any]:
    validate_strict_top1(k=args.k, y=args.y, semantic_speculation=not args.disable_semantic_speculation)
    api = validate_attack_api()
    variable = torch.tensor([False, True, True, True])
    importance = torch.tensor([0.0, 0.5, 1.0, 1.5])
    rho_zero, _ = layer_adaptive_weights(
        normalized_importance=importance,
        variable_mask=variable,
        step_index=100,
        epoch=100,
        rho=0.0,
        weight_min=0.5,
        weight_max=1.5,
    )
    ptr_zero = projection_time_repair(
        initial_ids=[1],
        embedding_weight=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        variable_mask=torch.tensor([True]),
        objective_from_embeddings=lambda embeds: (
            ((embeds - torch.tensor([[[0.0, 1.0]]])) ** 2).mean(),
            ((embeds[0] - torch.tensor([[0.0, 1.0]])) ** 2).mean(dim=-1),
        ),
        eta=0.0,
        repair_fraction=1.0,
        max_positions_per_pass=1,
        max_passes=2,
        epsilon_accept=1e-6,
    )
    cfg = config_from_args(args, "ASINIT", str(Path(args.output_root) / "verification" / "sample"))
    cfg.dataset_len = 1
    cfg.epoch = min(2, cfg.epoch)
    cfg.stage_a_epoch = min(2, cfg.stage_a_epoch)
    tokenizer, model, device = load_model_and_data(cfg)
    prompts, _ = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, 1, cfg.seed)
    tokenized = tokenizer(prompts[0], add_special_tokens=True, truncation=False, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    with torch.no_grad():
        observed = capture_prefix_activation(
            model,
            cfg.target_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).detach()
        full = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        server_final, server_logits, _ = server_forward_with_attention(
            model,
            cfg.target_layer + 1,
            observed.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=True,
            rollout_depth="all",
        )
    boundary_diff = float((observed.float() - full.hidden_states[cfg.target_layer + 1].float()).abs().max().cpu())
    final_diff = float((server_final.float() - full.hidden_states[-1].float()).abs().max().cpu())
    logits_diff = float((server_logits.float() - full.logits.float()).abs().max().cpu())
    attack = invert_observed(model, tokenizer, cfg, observed, int(input_ids.shape[1]), device, PTRConfig(max_passes=0))
    result = {
        "strict_top1": args.k == 1 and args.y == 0 and args.disable_semantic_speculation,
        "attack_api": api,
        "manual_prefix_max_diff": boundary_diff,
        "server_final_max_diff": final_diff,
        "server_logits_max_diff": logits_diff,
        "A_self_uses_dummy_x_only": bool(attack.alpha_init_stats.get("used"))
        and not bool(attack.alpha_init_stats.get("uses_true_prompt_attention")),
        "rho_zero_equals_B0V": torch.equal(rho_zero, torch.tensor([0.0, 1.0, 1.0, 1.0])),
        "ptr_eta_zero_no_change": ptr_zero.token_ids == [1],
        "accepted_repair_decreases_activation_loss": all(
            event["new_activation_loss"] < event["old_activation_loss"]
            for event in attack.ptr_stats.get("events", [])
            if event.get("accepted")
        ),
    }
    result["passes"] = (
        result["strict_top1"]
        and api["passes"]
        and boundary_diff < 1e-5
        and final_diff < 1e-5
        and logits_diff < 1e-5
        and result["A_self_uses_dummy_x_only"]
        and result["rho_zero_equals_B0V"]
        and result["ptr_eta_zero_no_change"]
        and result["accepted_repair_decreases_activation_loss"]
    )
    json_dump(Path(args.output_root) / "leakage_verification.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    if not result["passes"]:
        raise RuntimeError("AIR-PTR verification failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "single", "smoke", "rho-ablation", "dev", "summarize"], required=True)
    parser.add_argument("--method", choices=METHODS, default="FINAL")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", default="runs/air_ptr")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-name", default="Skytrax-28")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--target-layers", type=int, nargs="*", default=[11, 17, 19])
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--stage-a-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--lambda-dummy", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--rhos", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--disable-semantic-speculation", action="store_true", default=True)
    parser.add_argument("--max-token-len", type=int, default=896)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ptr-eta", type=float, default=0.10)
    parser.add_argument("--repair-fraction", type=float, default=0.20)
    parser.add_argument("--max-positions-per-pass", type=int, default=8)
    parser.add_argument("--max-passes", type=int, default=2)
    parser.add_argument("--epsilon-accept", type=float, default=1e-6)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_strict_top1(
        k=args.k,
        y=args.y,
        semantic_speculation=not args.disable_semantic_speculation,
    )
    if args.mode == "verify":
        run_verify(args)
    elif args.mode == "single":
        run_single(args)
    elif args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "rho-ablation":
        run_rho_ablation(args)
    elif args.mode == "dev":
        run_dev(args)
    elif args.mode == "summarize":
        write_comparison(Path(args.output_root))
    else:
        raise NotImplementedError(args.mode)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
