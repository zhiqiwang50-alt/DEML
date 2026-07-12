import argparse
import ast
import csv
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_CACHE", "/data/zhiqi/hf_cache/transformers")

from pia_attention_guided import enforce_embedding_constraints, fixed_embedding_tensor, random_public_embeddings, variable_view
from pia_masked_server_attn_pia import (
    GpuMemory,
    build_mts_token_weights,
    build_variable_mask,
    plan_auto_batch_jobs,
    query_gpu_memory,
    server_attention_bundle,
    uniform_all_token_activation_loss,
    variable_mask_audit_payload,
    weighted_activation_loss,
    weighted_variable_activation_loss,
)
from pia_tinyllama import (
    MODEL_NAME,
    TOTAL_BLOCKS,
    activation_calibrated_discretization,
    bleu_score,
    capture_prefix_activation,
    embedding_bounds,
    embedding_candidates,
    inferred_boundary_special_tokens,
    json_dump,
    jsonl_append,
    load_dataset_prompts,
    load_tinyllama,
    naive_discretization,
    nearest_embedding_loss,
    optional_nerr,
    peak_memory_mb,
    set_seed,
    text_from_ids,
    token_accuracy,
    token_texts,
)


EXPERIMENT_TITLE = "Uncertainty-Calibrated Server Attention Weighted PIA"
OUTPUT_ROOT_DEFAULT = "runs/attention_weighted_pia"
METHODS = ["B0", "B0V", "B0A", "B0VA", "B0VAG"]


@dataclass
class AWPConfig:
    method: str
    run_name: str
    output_dir: str
    dataset_name: str
    dataset_path: str
    dataset_len: int
    seed: int
    participant_number: int
    attacker_position: int
    target_layer: int
    epoch: int
    lr: float
    lambda_vocab: float
    top_k_embedding: int
    top_y_semantic: int
    max_token_len: int
    grad_clip: float
    beta: float
    weight_power: float
    weight_min: float
    weight_max: float
    last_window_size: int
    attention_start_ratio: float
    uncertainty_fraction: float
    residual_rollout: bool
    server_rollout_depth: str
    adaptive_discretization: bool
    semantic_speculation: bool
    local_files_only: bool
    fix_boundary_specials: bool = True


def canonical_method(method: str) -> str:
    aliases = {
        "original_pia_baseline": "B0",
        "variable_only_uniform": "B0V",
        "b0_attention": "B0A",
        "b0v_attention": "B0VA",
        "b0v_attention_gate": "B0VAG",
    }
    return aliases.get(method, method)


def scheduled_weights(
    attention_weights: torch.Tensor,
    variable_mask: torch.Tensor,
    step: int,
    epoch: int,
    start_ratio: float,
    gate: Optional[torch.Tensor],
) -> torch.Tensor:
    variable = variable_mask.to(attention_weights.device).bool()
    if step < int(math.ceil(max(1, epoch) * float(start_ratio))):
        return variable.float()
    weights = attention_weights.to(attention_weights.device).float() * variable.float()
    if gate is not None:
        gate_mask = gate.to(attention_weights.device).bool() & variable
        weights = torch.where(gate_mask, weights, variable.float())
    return weights * variable.float()


def all_valid_weights_from_variable_weights(variable_weights: torch.Tensor, valid_mask: torch.Tensor, variable_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(variable_weights.device).bool()
    variable = variable_mask.to(variable_weights.device).bool() & valid
    weights = valid.float()
    weights = torch.where(variable, variable_weights.to(weights.device).float(), weights)
    return weights * valid.float()


def embedding_uncertainty_gate(
    z: torch.Tensor,
    embed_weight: torch.Tensor,
    variable_mask: torch.Tensor,
    fraction: float,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    variable = variable_mask.to(z.device).bool()
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    gate = torch.zeros(z.shape[1], dtype=torch.bool, device=z.device)
    if positions.numel() == 0 or fraction <= 0:
        return gate, {"gate_position_count": 0, "fraction": float(fraction), "threshold": None}
    vectors = z[0, positions].detach().float()
    weight = embed_weight.detach().float()
    top1 = torch.full((vectors.shape[0],), float("inf"), device=z.device)
    top2 = torch.full((vectors.shape[0],), float("inf"), device=z.device)
    vnorm = torch.sum(vectors * vectors, dim=1, keepdim=True)
    with torch.no_grad():
        for start in range(0, weight.shape[0], chunk_size):
            chunk = weight[start : start + chunk_size].to(z.device)
            dists = vnorm + torch.sum(chunk * chunk, dim=1).unsqueeze(0) - 2 * vectors @ chunk.t()
            vals = torch.topk(dists, k=min(2, dists.shape[1]), largest=False, dim=1).values
            combined = torch.cat([top1.unsqueeze(1), top2.unsqueeze(1), vals], dim=1)
            best2 = torch.topk(combined, k=2, largest=False, dim=1).values
            top1, top2 = best2[:, 0], best2[:, 1]
    margins = top2 - top1
    count = max(1, int(math.ceil(float(positions.numel()) * float(fraction))))
    selected_local = torch.topk(margins, k=min(count, margins.numel()), largest=False).indices
    gate[positions[selected_local]] = True
    threshold = float(margins[selected_local].max().detach().cpu()) if selected_local.numel() else None
    return gate, {
        "gate_position_count": int(gate.sum().detach().cpu()),
        "fraction": float(fraction),
        "threshold": threshold,
        "margin_mean": float(margins.mean().detach().cpu()) if margins.numel() else None,
        "margin_min": float(margins.min().detach().cpu()) if margins.numel() else None,
        "margin_max": float(margins.max().detach().cpu()) if margins.numel() else None,
    }


def load_model(cfg: AWPConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected {TOTAL_BLOCKS} TinyLlama blocks, got {len(model.model.layers)}")
    return tokenizer, model, device


def build_server_weights(model, cfg: AWPConfig, observed_activation: torch.Tensor, attention_mask: torch.Tensor, fixed_public: Dict[int, int], variable_audit: Any):
    class WeightCfg:
        pass

    wcfg = WeightCfg()
    wcfg.method = "mts_last_window"
    wcfg.target_layer = cfg.target_layer
    wcfg.weight_source = "last_window_mean"
    wcfg.weight_power = cfg.weight_power
    wcfg.weight_min = cfg.weight_min
    wcfg.weight_max = cfg.weight_max
    wcfg.beta = cfg.beta
    wcfg.last_window_size = cfg.last_window_size
    wcfg.residual_rollout = cfg.residual_rollout
    wcfg.server_rollout_depth = cfg.server_rollout_depth
    return server_attention_bundle(model, observed_activation, attention_mask, wcfg, fixed_public, variable_audit)


def stage_b_optimize(model, tokenizer, cfg: AWPConfig, observed_activation: torch.Tensor, seq_len: int, device, fixed_public: Dict[int, int], variable_audit: Any, server_weights: Optional[torch.Tensor]):
    method = canonical_method(cfg.method)
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _ = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public).detach().clone().float().requires_grad_(True)
    optimizer = torch.optim.AdamW([z], lr=cfg.lr)
    target = observed_activation.detach()
    valid_mask = attention_mask[0].bool()
    gate = None
    gate_stats: Dict[str, Any] = {"enabled": method == "B0VAG"}
    history: List[Dict[str, Any]] = []
    final: Dict[str, Any] = {}
    for step in range(max(1, cfg.epoch)):
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        if method == "B0VAG" and server_weights is not None and gate is None and step >= int(math.ceil(cfg.epoch * cfg.attention_start_ratio)):
            gate, gate_stats = embedding_uncertainty_gate(z, embed_layer.weight, variable_audit.variable_mask, cfg.uncertainty_fraction)
            gate_stats["enabled"] = True
        hidden = capture_prefix_activation(
            model,
            cfg.target_layer,
            inputs_embeds=z.to(dtype=embed_layer.weight.dtype),
            attention_mask=attention_mask,
        )
        all_uniform = uniform_all_token_activation_loss(hidden, target, attention_mask)
        var_uniform = weighted_variable_activation_loss(hidden, target, variable_audit.variable_mask, None)
        if server_weights is None:
            active_var_weights = None
            active_all_weights = None
        else:
            active_var_weights = scheduled_weights(server_weights, variable_audit.variable_mask, step, cfg.epoch, cfg.attention_start_ratio, gate)
            active_all_weights = all_valid_weights_from_variable_weights(active_var_weights, valid_mask, variable_audit.variable_mask)
        weighted_var = weighted_variable_activation_loss(hidden, target, variable_audit.variable_mask, active_var_weights)
        weighted_all = weighted_activation_loss(hidden, target, active_all_weights) if active_all_weights is not None else all_uniform
        if method == "B0":
            act_loss = all_uniform
        elif method == "B0V":
            act_loss = var_uniform
        elif method == "B0A":
            act_loss = weighted_all
        elif method in {"B0VA", "B0VAG"}:
            act_loss = weighted_var
        else:
            raise ValueError(f"unknown method={cfg.method!r}")
        vocab_loss = nearest_embedding_loss(variable_view(z, fixed_positions), embed_layer.weight)
        total = act_loss + float(cfg.lambda_vocab) * vocab_loss
        cosine = F.cosine_similarity(hidden.float(), target.to(hidden.device).float(), dim=-1).mean()
        if not torch.isfinite(total) or not torch.isfinite(cosine):
            raise RuntimeError(f"NaN/Inf at optimization step {step + 1}")
        optimizer.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip)
        optimizer.step()
        final = {
            "activation_loss": float(act_loss.detach().cpu()),
            "all_token_uniform_loss": float(all_uniform.detach().cpu()),
            "variable_uniform_loss": float(var_uniform.detach().cpu()),
            "weighted_all_loss": float(weighted_all.detach().cpu()),
            "weighted_variable_loss": float(weighted_var.detach().cpu()),
            "vocab_loss": float(vocab_loss.detach().cpu()),
            "optimization_loss": float(total.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
            "attention_active": bool(server_weights is not None and step >= int(math.ceil(cfg.epoch * cfg.attention_start_ratio))),
        }
        if (step + 1) % max(1, min(50, cfg.epoch)) == 0 or step == cfg.epoch - 1:
            history.append({"step": step + 1, **final})
            print(
                f"method={method} step={step + 1} act={final['activation_loss']:.6f} "
                f"all={final['all_token_uniform_loss']:.6f} var={final['variable_uniform_loss']:.6f} "
                f"wvar={final['weighted_variable_loss']:.6f} vocab={final['vocab_loss']:.6f} "
                f"total={final['optimization_loss']:.6f} cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    return z.detach().float(), final, history, gate_stats


def invert_observed(model, tokenizer, cfg: AWPConfig, observed_activation: torch.Tensor, seq_len: int, device):
    method = canonical_method(cfg.method)
    cfg.method = method
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    variable_audit = build_variable_mask(
        attention_mask=attention_mask,
        fixed_public=fixed_public,
        special_token_ids=getattr(tokenizer, "all_special_ids", None),
        token_ids=None,
    )
    server_weights = None
    attention_stats: Dict[str, Any] = {"server_attention_used": False}
    rollout_stats: Dict[str, Any] = {}
    weight_stats: Dict[str, Any] = {}
    if method in {"B0A", "B0VA", "B0VAG"}:
        server_weights, attention_stats, rollout_stats, weight_stats = build_server_weights(
            model, cfg, observed_activation, attention_mask, fixed_public, variable_audit
        )
        attention_stats["server_attention_used"] = True
    else:
        valid_count = int(variable_audit.valid_mask.sum().detach().cpu())
        weight_stats = {
            "source": "uniform",
            "mean": 1.0,
            "std": 0.0,
            "min": 1.0,
            "max": 1.0,
            "final_fixed_weight_sum": float((variable_audit.fixed_public_mask & variable_audit.valid_mask).sum().detach().cpu()) if method == "B0" else 0.0,
            "final_bos_weight": 1.0 if method == "B0" and valid_count > 0 else 0.0,
        }
    weight_stats["variable_mask_audit"] = variable_mask_audit_payload(variable_audit)
    z, losses, history, gate_stats = stage_b_optimize(model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public, variable_audit, server_weights)
    embed_sets = embedding_candidates(z, model.get_input_embeddings().weight, max(1, cfg.top_k_embedding))
    recovered_ids = naive_discretization(z, model.get_input_embeddings().weight)
    for pos, token_id in fixed_public.items():
        if 0 <= pos < len(recovered_ids):
            recovered_ids[pos] = int(token_id)
    calibration_score = None
    if cfg.adaptive_discretization:
        recovered_ids, calibration_score = activation_calibrated_discretization(
            model,
            cfg.target_layer,
            attention_mask,
            observed_activation,
            recovered_ids,
            embed_sets,
            sorted(fixed_public),
            cfg,
        )
    return recovered_ids, losses, attention_stats, rollout_stats, weight_stats, gate_stats, history, calibration_score


def validate_attack_api() -> Dict[str, Any]:
    sig = inspect.signature(invert_observed)
    names = list(sig.parameters)
    banned = ["reference", "original", "input_ids", "token_ids", "prompt", "text", "ground", "dummy"]
    signature_hits = [name for name in names if any(piece in name for piece in banned)]
    checked = [invert_observed, stage_b_optimize, build_server_weights, embedding_uncertainty_gate]
    global_hits: List[Dict[str, str]] = []
    for fn in checked:
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in {"original_ids", "original_tokens", "ground_truth_ids", "dummy_x", "dummy_attention"}:
                global_hits.append({"function": fn.__name__, "name": node.id})
    return {"signature": str(sig), "parameter_names": names, "banned_signature_hits": signature_hits, "banned_global_reference_hits": global_hits, "passes": not signature_hits and not global_hits}


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def mean_std(key: str) -> Dict[str, Optional[float]]:
        vals = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        return {"mean": float(statistics.mean(vals)) if vals else None, "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None}

    return {
        "token_accuracy": mean_std("token_accuracy"),
        "bleu": mean_std("bleu"),
        "runtime_seconds": mean_std("runtime_seconds"),
        "peak_gpu_memory_mb": mean_std("peak_gpu_memory_mb"),
        "sample_count": len(rows),
        "nerr": {"mean": None, "std": None, "status": "NERR unavailable"},
    }


def run_one_config(cfg: AWPConfig, resume: bool) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "COMPLETE"
    if resume and done.exists():
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    tokenizer, model, device = load_model(cfg)
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    predictions_path = out / "predictions.jsonl"
    failures_path = out / "failures.jsonl"
    if not resume:
        for path in [predictions_path, failures_path]:
            if path.exists():
                path.unlink()
    predictions_path.touch(exist_ok=True)
    failures_path.touch(exist_ok=True)
    completed = set()
    rows: List[Dict[str, Any]] = []
    if resume:
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add(int(row["prompt_id"]))
                rows.append(row)
    failures: List[Dict[str, Any]] = []
    attention_samples = []
    rollout_samples = []
    weight_samples = []
    loss_samples = []
    gate_samples = []
    api_check = validate_attack_api()
    for prompt_id, prompt in enumerate(prompts):
        if prompt_id in completed:
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            if input_ids.shape[1] > cfg.max_token_len:
                raise RuntimeError(f"prompt_id={prompt_id} has {input_ids.shape[1]} tokens > max_token_len={cfg.max_token_len}")
            original_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
            with torch.no_grad():
                observed = capture_prefix_activation(model, cfg.target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
            preflight = {
                "model_name": MODEL_NAME,
                "block_count": len(model.model.layers),
                "target_layer_index": cfg.target_layer,
                "server_start_layer": cfg.target_layer + 1,
                "observed_activation_shape": list(observed.shape),
                "sequence_length": int(input_ids.shape[1]),
                "attack_receives_ground_truth": False,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
            }
            print(f"preflight={json.dumps(preflight, ensure_ascii=True)}", flush=True)
            recovered_ids, losses, attn_stats, rollout_stats, weight_stats, gate_stats, history, calibration_score = invert_observed(
                model, tokenizer, cfg, observed, int(input_ids.shape[1]), device
            )
            elapsed = time.time() - start
            original_text = text_from_ids(tokenizer, original_ids)
            recovered_text = text_from_ids(tokenizer, recovered_ids)
            row = {
                "title": EXPERIMENT_TITLE,
                "method": cfg.method,
                "prompt_id": int(prompt_id),
                "prompt": prompt,
                "original_text": original_text,
                "recovered_text": recovered_text,
                "original_token_ids": original_ids,
                "recovered_token_ids": recovered_ids,
                "original_tokens": token_texts(tokenizer, original_ids),
                "recovered_tokens": token_texts(tokenizer, recovered_ids),
                "token_accuracy": token_accuracy(tokenizer, original_ids, recovered_ids),
                "bleu": bleu_score(tokenizer, original_ids, recovered_ids),
                "nerr": optional_nerr(original_text, recovered_text),
                **losses,
                "calibration_score": calibration_score,
                "runtime_seconds": elapsed,
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "top_k_embedding": cfg.top_k_embedding,
                "top_y_semantic": cfg.top_y_semantic,
                "peak_gpu_memory_mb": peak_memory_mb(),
                "preflight": preflight,
                "recovery_uses_ground_truth_tokens": False,
            }
            jsonl_append(predictions_path, row)
            rows.append(row)
            attention_samples.append({"prompt_id": int(prompt_id), **attn_stats})
            rollout_samples.append({"prompt_id": int(prompt_id), **rollout_stats})
            weight_samples.append({"prompt_id": int(prompt_id), **weight_stats})
            loss_samples.append({"prompt_id": int(prompt_id), "history": history, "final": losses})
            gate_samples.append({"prompt_id": int(prompt_id), **gate_stats})
            print(f"method={cfg.method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} bleu={row['bleu']:.6f} elapsed={elapsed:.2f}s", flush=True)
        except Exception as exc:
            failure = {"method": cfg.method, "prompt_id": int(prompt_id), "error": repr(exc), "traceback": traceback.format_exc()}
            jsonl_append(failures_path, failure)
            failures.append(failure)
            print(f"failure={json.dumps(failure, ensure_ascii=True)}", flush=True)
    metrics = summarize_rows(rows)
    metrics.update(
        {
            "title": EXPERIMENT_TITLE,
            "method": cfg.method,
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "target_layer": cfg.target_layer,
            "top_k_embedding": cfg.top_k_embedding,
            "top_y_semantic": cfg.top_y_semantic,
            "completed_sample_count": len(rows),
            "failed_sample_count": len(failures),
            "dataset_meta": dataset_meta,
            "output_dir": cfg.output_dir,
        }
    )
    json_dump(out / "metrics.json", metrics)
    json_dump(out / "server_attention_stats.json", {"samples": attention_samples})
    json_dump(out / "attention_rollout.json", {"samples": rollout_samples})
    json_dump(out / "token_weight_stats.json", {"samples": weight_samples})
    json_dump(out / "loss_breakdown.json", {"samples": loss_samples})
    json_dump(out / "uncertainty_gate_stats.json", {"samples": gate_samples})
    json_dump(out / "leakage_verification.json", {"attack_api_check": api_check, "attack_api_passes": bool(api_check.get("passes"))})
    done.write_text("complete\n", encoding="utf-8")
    return metrics


def tag_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def config_from_args(args: argparse.Namespace, method: str, layer: int, topk_label: str) -> AWPConfig:
    k = 1 if topk_label == "top1" else 10
    y = 0 if topk_label == "top1" else 10
    naive = topk_label == "top1"
    run_name = f"{method}_layer{layer}_{topk_label}_seed{args.seed}_epoch{args.epoch}"
    return AWPConfig(
        method=method,
        run_name=run_name,
        output_dir=str(Path(args.output_root) / "stage1" / run_name),
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=args.dataset_len,
        seed=args.seed,
        participant_number=args.participant_number,
        attacker_position=args.attacker_position,
        target_layer=layer,
        epoch=args.epoch,
        lr=args.lr,
        lambda_vocab=args.lambda_vocab,
        top_k_embedding=k,
        top_y_semantic=y,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        beta=args.beta,
        weight_power=args.weight_power,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        last_window_size=args.last_window_size,
        attention_start_ratio=args.attention_start_ratio,
        uncertainty_fraction=args.uncertainty_fraction,
        residual_rollout=not args.no_residual_rollout,
        server_rollout_depth=args.server_rollout_depth,
        adaptive_discretization=not naive,
        semantic_speculation=not naive,
        local_files_only=args.local_files_only,
    )


def build_task_command(args: argparse.Namespace, method: str, layer: int, topk_label: str) -> List[str]:
    return [
        sys.executable,
        "pia_attention_weighted_pia.py",
        "--mode",
        "single",
        "--method",
        method,
        "--target-layer",
        str(layer),
        "--topk-label",
        topk_label,
        "--output-root",
        args.output_root,
        "--dataset-name",
        args.dataset_name,
        "--dataset-path",
        args.dataset_path,
        "--dataset-len",
        str(args.dataset_len),
        "--seed",
        str(args.seed),
        "--participant-number",
        str(args.participant_number),
        "--attacker-position",
        str(args.attacker_position),
        "--epoch",
        str(args.epoch),
        "--lr",
        str(args.lr),
        "--lambda-vocab",
        str(args.lambda_vocab),
        "--max-token-len",
        str(args.max_token_len),
        "--grad-clip",
        str(args.grad_clip),
        "--beta",
        str(args.beta),
        "--weight-power",
        str(args.weight_power),
        "--weight-min",
        str(args.weight_min),
        "--weight-max",
        str(args.weight_max),
        "--last-window-size",
        str(args.last_window_size),
        "--attention-start-ratio",
        str(args.attention_start_ratio),
        "--uncertainty-fraction",
        str(args.uncertainty_fraction),
        "--server-rollout-depth",
        args.server_rollout_depth,
    ] + (["--local-files-only"] if args.local_files_only else []) + (["--resume"] if args.resume else [])


def run_stage1(args: argparse.Namespace) -> None:
    memories = query_gpu_memory()
    plan = plan_auto_batch_jobs(memories, args.batch_free_mb_per_job, args.batch_reserve_free_mb, args.max_jobs_per_gpu, args.max_parallel_jobs)
    slots: List[int] = []
    for gpu, count in sorted(plan.jobs_by_gpu.items()):
        slots.extend([int(gpu)] * int(count))
    if not slots:
        slots = [0]
    tasks: List[Tuple[str, int, str, int]] = []
    idx = 0
    for layer in args.target_layers:
        for topk_label in ["topk10", "top1"]:
            for method in METHODS:
                tasks.append((method, int(layer), topk_label, slots[idx % len(slots)]))
                idx += 1
    log_dir = Path(args.output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"gpu_memories": [m.__dict__ for m in memories], "plan": plan.__dict__, "task_count": len(tasks)}, ensure_ascii=False), flush=True)
    if args.dry_run:
        for method, layer, topk_label, gpu in tasks:
            print(f"CUDA_VISIBLE_DEVICES={gpu} {' '.join(build_task_command(args, method, layer, topk_label))}")
        return
    queue = list(tasks)
    active: List[Tuple[subprocess.Popen, Tuple[str, int, str, int], Any, Path]] = []
    capacity: Dict[int, int] = {gpu: slots.count(gpu) for gpu in set(slots)}
    while queue or active:
        launched = True
        while queue and launched:
            launched = False
            active_by_gpu: Dict[int, int] = {}
            for _proc, task, _handle, _path in active:
                active_by_gpu[task[3]] = active_by_gpu.get(task[3], 0) + 1
            for qidx, task in enumerate(queue):
                if active_by_gpu.get(task[3], 0) < capacity.get(task[3], 0):
                    method, layer, topk_label, gpu = queue.pop(qidx)
                    env = dict(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    command = build_task_command(args, method, layer, topk_label)
                    log_path = log_dir / f"{method}_layer{layer}_{topk_label}_gpu{gpu}.log"
                    handle = log_path.open("w", encoding="utf-8")
                    print(json.dumps({"launch": command, "gpu": gpu, "log": str(log_path)}, ensure_ascii=False), flush=True)
                    proc = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT)
                    active.append((proc, (method, layer, topk_label, gpu), handle, log_path))
                    launched = True
                    break
        next_active: List[Tuple[subprocess.Popen, Tuple[str, int, str, int], Any, Path]] = []
        for proc, task, handle, log_path in active:
            code = proc.poll()
            if code is None:
                next_active.append((proc, task, handle, log_path))
                continue
            handle.close()
            if code != 0:
                print(json.dumps({"task_failed": task, "log": str(log_path), "code": code}), flush=True)
        active = next_active
        if active:
            time.sleep(10)
    summarize_stage1(args.output_root)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric_mean(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return None


def summarize_stage1(output_root: str) -> None:
    root = Path(output_root)
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted((root / "stage1").glob("*/metrics.json")):
        m = read_json(metrics_path)
        rows.append(
            {
                "method": m.get("method"),
                "run_name": m.get("run_name"),
                "layer": m.get("target_layer"),
                "top_k_embedding": m.get("top_k_embedding"),
                "token_accuracy": metric_mean(m, "token_accuracy"),
                "bleu": metric_mean(m, "bleu"),
                "completed_sample_count": m.get("completed_sample_count"),
                "failed_sample_count": m.get("failed_sample_count"),
                "output_dir": m.get("output_dir"),
            }
        )
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "stage1_summary.csv"
    keys = ["method", "run_name", "layer", "top_k_embedding", "token_accuracy", "bleu", "completed_sample_count", "failed_sample_count", "output_dir"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    baselines = {(row["layer"], row["top_k_embedding"], row["method"]): row for row in rows}
    delta_rows = []
    for row in rows:
        b0 = baselines.get((row["layer"], row["top_k_embedding"], "B0"))
        b0v = baselines.get((row["layer"], row["top_k_embedding"], "B0V"))
        delta_rows.append(
            {
                **row,
                "dAcc_vs_B0": None if not b0 or row["token_accuracy"] is None or b0["token_accuracy"] is None else row["token_accuracy"] - b0["token_accuracy"],
                "dBLEU_vs_B0": None if not b0 or row["bleu"] is None or b0["bleu"] is None else row["bleu"] - b0["bleu"],
                "dAcc_vs_B0V": None if not b0v or row["token_accuracy"] is None or b0v["token_accuracy"] is None else row["token_accuracy"] - b0v["token_accuracy"],
                "dBLEU_vs_B0V": None if not b0v or row["bleu"] is None or b0v["bleu"] is None else row["bleu"] - b0v["bleu"],
            }
        )
    with (report_dir / "stage1_delta_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        keys2 = list(delta_rows[0].keys()) if delta_rows else keys
        writer = csv.DictWriter(handle, fieldnames=keys2)
        writer.writeheader()
        writer.writerows(delta_rows)
    lines = ["# Attention Weighted PIA Stage1 Summary", "", "| layer | top_k | method | Acc | BLEU | dAcc vs B0 | dAcc vs B0V | done/fail |", "|---:|---:|---|---:|---:|---:|---:|---|"]
    for row in delta_rows:
        lines.append(
            f"| {row['layer']} | {row['top_k_embedding']} | {row['method']} | {row['token_accuracy']} | {row['bleu']} | "
            f"{row['dAcc_vs_B0']} | {row['dAcc_vs_B0V']} | {row['completed_sample_count']}/{row['failed_sample_count']} |"
        )
    (report_dir / "stage1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_from_single_args(args: argparse.Namespace) -> AWPConfig:
    return config_from_args(args, args.method, args.target_layer, args.topk_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "single", "stage1", "summarize"], default="verify")
    parser.add_argument("--method", choices=METHODS, default="B0")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--dataset-name", default="Skytrax-28")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--target-layers", type=int, nargs="*", default=[11, 17, 19])
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--topk-label", choices=["topk10", "top1"], default="topk10")
    parser.add_argument("--max-token-len", type=int, default=896)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--weight-min", type=float, default=0.5)
    parser.add_argument("--weight-max", type=float, default=2.0)
    parser.add_argument("--last-window-size", type=int, default=16)
    parser.add_argument("--attention-start-ratio", type=float, default=0.7)
    parser.add_argument("--uncertainty-fraction", type=float, default=0.3)
    parser.add_argument("--no-residual-rollout", action="store_true")
    parser.add_argument("--server-rollout-depth", choices=["all", "last2"], default="all")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-free-mb-per-job", type=int, default=10000)
    parser.add_argument("--batch-reserve-free-mb", type=int, default=2048)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--max-parallel-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "verify":
        print(json.dumps({"attack_api": validate_attack_api()}, ensure_ascii=False))
    elif args.mode == "single":
        run_one_config(config_from_single_args(args), args.resume)
    elif args.mode == "stage1":
        run_stage1(args)
    elif args.mode == "summarize":
        summarize_stage1(args.output_root)


if __name__ == "__main__":
    main()
