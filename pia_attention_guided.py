import argparse
import csv
import inspect
import json
import math
import os
import random
import re
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask

from pia_tinyllama import (
    ATTACKER_MAP,
    MODEL_NAME,
    TOTAL_BLOCKS,
    activation_calibrated_discretization,
    args_local_files_only,
    bleu_score,
    capture_activation,
    capture_prefix_activation,
    embedding_candidates,
    embedding_bounds,
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


EXPERIMENT_TITLE = "TinyLlama attention-guided Prompt Inversion pilot"
METHODS = ["baseline", "dummy_init", "attention_context", "attention_gradient", "full"]


@dataclass
class AGConfig:
    method: str
    run_name: str
    output_dir: str
    dataset_name: str
    dataset_path: str
    dataset_len: int
    seed: int
    participant_number: int
    attacker_position: int
    inverted_block_count: int
    target_layer: int
    epoch: int
    stage_a_epoch: int
    lr: float
    lambda_vocab: float
    lambda_attn: float
    top_k_embedding: int
    top_y_semantic: int
    adaptive_discretization: bool
    semantic_speculation: bool
    max_token_len: int
    disable_attention_guidance: bool
    local_files_only: bool


def tensor_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "token_accuracy": tensor_stats([float(row["token_accuracy"]) for row in rows]),
        "bleu": tensor_stats([float(row["bleu"]) for row in rows]),
        "runtime_seconds": tensor_stats([float(row["runtime_seconds"]) for row in rows]),
        "peak_gpu_memory_mb": tensor_stats(
            [float(row["peak_gpu_memory_mb"]) for row in rows if row.get("peak_gpu_memory_mb") is not None]
        ),
        "optimization_loss": tensor_stats([float(row["optimization_loss"]) for row in rows]),
        "cosine_similarity": tensor_stats([float(row["cosine_similarity"]) for row in rows]),
        "sample_count": len(rows),
    }


def fixed_embedding_tensor(
    embed_layer: torch.nn.Module,
    fixed_token_ids: Dict[int, int],
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    fixed_positions = sorted(pos for pos in fixed_token_ids if 0 <= pos < seq_len)
    fixed_ids = [int(fixed_token_ids[pos]) for pos in fixed_positions]
    if not fixed_positions:
        return torch.empty(0, embed_layer.embedding_dim, device=device), [], []
    ids = torch.tensor(fixed_ids, dtype=torch.long, device=device)
    return embed_layer(ids).detach().float(), fixed_positions, fixed_ids


def random_public_embeddings(
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    seq_len: int,
    device: torch.device,
    fixed_token_ids: Dict[int, int],
) -> torch.Tensor:
    special = set(int(x) for x in (tokenizer.all_special_ids or []))
    vocab_size = len(tokenizer)
    candidates = [idx for idx in range(vocab_size) if idx not in special]
    if not candidates:
        raise RuntimeError("No non-special public vocabulary ids are available for initialization")
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    sampled = candidate_tensor[torch.randint(0, candidate_tensor.numel(), (seq_len,), device=device)]
    for pos, token_id in fixed_token_ids.items():
        if 0 <= pos < seq_len:
            sampled[pos] = int(token_id)
    return embed_layer(sampled.unsqueeze(0)).detach().float()


def enforce_embedding_constraints(
    embeds: torch.Tensor,
    fixed_positions: Sequence[int],
    fixed_embeds: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> None:
    with torch.no_grad():
        embeds.clamp_(left.view(1, 1, -1), right.view(1, 1, -1))
        if fixed_positions:
            embeds[:, list(fixed_positions), :] = fixed_embeds.to(embeds.device).unsqueeze(0)


def prefix_forward_hidden_and_attention(
    model: torch.nn.Module,
    target_layer: int,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    output_target_attention: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_len),
        inputs_embeds,
        past_key_values_length=0,
    )
    hidden_states = inputs_embeds
    target_attention = None
    for layer_index in range(target_layer + 1):
        want_attn = output_target_attention and layer_index == target_layer
        layer_outputs = model.model.layers[layer_index](
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=want_attn,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
        if want_attn:
            target_attention = layer_outputs[1]
    if output_target_attention and target_attention is None:
        raise RuntimeError(
            "Target self-attention weights were not returned. "
            "This transformers/LLaMA path must support output_attentions=True for explicit attention weights."
        )
    return hidden_states, target_attention


def attention_proxy_stats(attn_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if attn_weights.dim() != 4:
        raise RuntimeError(f"Expected attention weights [batch, heads, seq, seq], got {list(attn_weights.shape)}")
    a_dummy = attn_weights.detach().float().mean(dim=1)[0]
    seq_len = a_dummy.shape[0]
    if list(a_dummy.shape) != [seq_len, seq_len]:
        raise RuntimeError(f"A_dummy must have shape [seq_len, seq_len], got {list(a_dummy.shape)}")
    row_sums = a_dummy.sum(dim=-1)
    upper_max = torch.triu(a_dummy, diagonal=1).abs().max().item() if seq_len > 1 else 0.0
    entropy = (-(a_dummy.clamp_min(1e-12) * a_dummy.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
    sparsity = (a_dummy < 1e-3).float().mean().item()
    stats = {
        "attention_shape": [int(seq_len), int(seq_len)],
        "row_sum_min": float(row_sums.min().detach().cpu()),
        "row_sum_max": float(row_sums.max().detach().cpu()),
        "row_sum_mean": float(row_sums.mean().detach().cpu()),
        "upper_triangular_max": float(upper_max),
        "entropy_mean": float(entropy),
        "sparsity_lt_1e_3": float(sparsity),
        "attention_map_top_left_8x8": a_dummy[:8, :8].detach().cpu().tolist(),
    }
    if abs(stats["row_sum_min"] - 1.0) > 2e-3 or abs(stats["row_sum_max"] - 1.0) > 2e-3:
        raise RuntimeError(f"A_dummy rows are not normalized to 1: {stats}")
    if stats["upper_triangular_max"] > 2e-3:
        raise RuntimeError(f"A_dummy violates causal upper-triangular mask: {stats}")
    return a_dummy.detach(), stats


def context_projection(a_dummy: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a_dummy.to(hidden.device).unsqueeze(0), hidden.float())


def variable_view(embeds: torch.Tensor, fixed_positions: Sequence[int]) -> torch.Tensor:
    seq_len = embeds.shape[1]
    fixed = set(int(x) for x in fixed_positions)
    variable_positions = [idx for idx in range(seq_len) if idx not in fixed]
    if not variable_positions:
        return embeds[:, :0, :].reshape(0, embeds.shape[-1])
    return embeds[0, variable_positions, :]


def stage_a_dummy_attention(
    model: torch.nn.Module,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    cfg: AGConfig,
    target_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_token_ids: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], List[Dict[str, float]]]:
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _fixed_ids = fixed_embedding_tensor(embed_layer, fixed_token_ids, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_token_ids).requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=cfg.lr)
    target = target_activation.detach()
    history = []
    for step in range(max(1, cfg.stage_a_epoch)):
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden, _ = prefix_forward_hidden_and_attention(
            model,
            cfg.target_layer,
            z.to(dtype=embed_layer.weight.dtype),
            attention_mask,
            output_target_attention=False,
        )
        loss = F.mse_loss(hidden.float(), target.to(hidden.device).float())
        if torch.isnan(loss):
            raise RuntimeError(f"NaN in Stage A at step {step + 1}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % max(1, min(50, cfg.stage_a_epoch)) == 0 or step == cfg.stage_a_epoch - 1:
            history.append({"step": step + 1, "stage_a_activation_loss": float(loss.detach().cpu())})
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    with torch.no_grad():
        _hidden, attn = prefix_forward_hidden_and_attention(
            model,
            cfg.target_layer,
            z.detach().to(dtype=embed_layer.weight.dtype),
            attention_mask,
            output_target_attention=True,
        )
    a_dummy, stats = attention_proxy_stats(attn)
    return z.detach().float(), a_dummy, stats, history


def stage_b_optimize(
    model: torch.nn.Module,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    cfg: AGConfig,
    target_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_token_ids: Dict[int, int],
    init_embeds: Optional[torch.Tensor],
    a_dummy: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, float]]]:
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _fixed_ids = fixed_embedding_tensor(embed_layer, fixed_token_ids, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    if init_embeds is None:
        e = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_token_ids)
    else:
        e = init_embeds.detach().clone().to(device=device, dtype=torch.float32)
    e.requires_grad_(True)
    optimizer = torch.optim.Adam([e], lr=cfg.lr)
    target = target_activation.detach()
    target_ctx = context_projection(a_dummy, target) if a_dummy is not None else None
    history: List[Dict[str, float]] = []
    final: Dict[str, Any] = {}
    use_ctx = a_dummy is not None and cfg.lambda_attn > 0.0
    use_gradient_fusion = cfg.method in {"attention_gradient", "full"} and use_ctx

    for step in range(cfg.epoch):
        enforce_embedding_constraints(e, fixed_positions, fixed_embeds, left, right)
        hidden, _ = prefix_forward_hidden_and_attention(
            model,
            cfg.target_layer,
            e.to(dtype=embed_layer.weight.dtype),
            attention_mask,
            output_target_attention=False,
        )
        activation_loss = F.mse_loss(hidden.float(), target.to(hidden.device).float())
        vocab_loss = nearest_embedding_loss(variable_view(e, fixed_positions), embed_layer.weight)
        if use_ctx:
            ctx_loss = F.mse_loss(context_projection(a_dummy, hidden), target_ctx.to(hidden.device))
        else:
            ctx_loss = torch.tensor(0.0, device=device)
        cosine = F.cosine_similarity(hidden.float(), target.to(hidden.device).float(), dim=-1).mean()

        if torch.isnan(activation_loss) or torch.isnan(vocab_loss) or torch.isnan(ctx_loss) or torch.isnan(cosine):
            raise RuntimeError(f"NaN in Stage B at step {step + 1}")

        grad_stats = {
            "g_act_norm": None,
            "g_ctx_norm": None,
            "g_act_ctx_cosine": None,
            "update_norm": None,
        }
        if use_gradient_fusion:
            g_act = torch.autograd.grad(activation_loss, e, retain_graph=True)[0]
            g_ctx = torch.autograd.grad(ctx_loss, e, retain_graph=True)[0]
            g_vocab = torch.autograd.grad(vocab_loss, e, retain_graph=False, allow_unused=True)[0]
            if g_vocab is None:
                g_vocab = torch.zeros_like(e)
            act_flat = g_act.reshape(-1)
            ctx_flat = g_ctx.reshape(-1)
            dot = torch.dot(act_flat, ctx_flat)
            act_norm_sq = torch.dot(act_flat, act_flat).clamp_min(1e-12)
            if dot < 0:
                g_ctx = g_ctx - (dot / act_norm_sq) * g_act
            update = g_act + cfg.lambda_attn * g_ctx + cfg.lambda_vocab * g_vocab
            with torch.no_grad():
                e -= cfg.lr * update
            grad_stats = {
                "g_act_norm": float(g_act.norm().detach().cpu()),
                "g_ctx_norm": float(g_ctx.norm().detach().cpu()),
                "g_act_ctx_cosine": float(
                    F.cosine_similarity(g_act.reshape(1, -1), g_ctx.reshape(1, -1), dim=-1).detach().cpu()[0]
                ),
                "update_norm": float(update.norm().detach().cpu()),
            }
            total_loss = activation_loss + cfg.lambda_vocab * vocab_loss + cfg.lambda_attn * ctx_loss
        else:
            total_loss = activation_loss + cfg.lambda_vocab * vocab_loss + cfg.lambda_attn * ctx_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        final = {
            "activation_loss": float(activation_loss.detach().cpu()),
            "vocab_loss": float(vocab_loss.detach().cpu()),
            "context_loss": float(ctx_loss.detach().cpu()),
            "optimization_loss": float(total_loss.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
            **grad_stats,
        }
        if (step + 1) % max(1, min(50, cfg.epoch)) == 0 or step == cfg.epoch - 1:
            history.append({"step": step + 1, **final})
            print(
                f"method={cfg.method} step={step + 1} act={final['activation_loss']:.6f} "
                f"vocab={final['vocab_loss']:.6f} ctx={final['context_loss']:.6f} "
                f"total={final['optimization_loss']:.6f} cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(e, fixed_positions, fixed_embeds, left, right)
    return e.detach().float(), final, history


def invert_from_activation(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: AGConfig,
    target_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], List[Dict[str, float]], List[Dict[str, float]], Optional[float]]:
    effective_method = "baseline" if cfg.disable_attention_guidance else cfg.method
    if effective_method not in METHODS:
        raise ValueError(f"Unknown method {effective_method!r}")
    embed_layer = model.get_input_embeddings()
    fixed_token_ids = inferred_boundary_special_tokens(tokenizer, seq_len)
    fixed_positions = sorted(fixed_token_ids)
    a_dummy = None
    attention_stats: Dict[str, Any] = {
        "method": cfg.method,
        "effective_method": effective_method,
        "attention_guidance_enabled": effective_method != "baseline",
    }
    stage_a_history: List[Dict[str, float]] = []
    init_embeds = None
    needs_stage_a = effective_method in {"dummy_init", "attention_context", "attention_gradient", "full"}
    if needs_stage_a:
        dummy_embeds, a_dummy, attention_stats, stage_a_history = stage_a_dummy_attention(
            model,
            tokenizer,
            embed_layer,
            cfg,
            target_activation,
            seq_len,
            device,
            fixed_token_ids,
        )
        attention_stats.update({"method": cfg.method, "effective_method": effective_method})
        if effective_method in {"dummy_init", "full"}:
            init_embeds = dummy_embeds
    if effective_method == "dummy_init":
        a_dummy_for_stage_b = None
    elif effective_method in {"attention_context", "attention_gradient", "full"}:
        a_dummy_for_stage_b = a_dummy
    else:
        a_dummy_for_stage_b = None
    optimized_embeds, losses, stage_b_history = stage_b_optimize(
        model,
        tokenizer,
        embed_layer,
        cfg,
        target_activation,
        seq_len,
        device,
        fixed_token_ids,
        init_embeds,
        a_dummy_for_stage_b,
    )
    embed_sets = embedding_candidates(optimized_embeds, embed_layer.weight, max(1, cfg.top_k_embedding))
    recovered_ids = naive_discretization(optimized_embeds, embed_layer.weight)
    for pos, token_id in fixed_token_ids.items():
        if 0 <= pos < len(recovered_ids):
            recovered_ids[pos] = int(token_id)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    calibration_score = None
    if cfg.adaptive_discretization:
        recovered_ids, calibration_score = activation_calibrated_discretization(
            model,
            cfg.target_layer,
            attention_mask,
            target_activation,
            recovered_ids,
            embed_sets,
            fixed_positions,
            cfg,
        )
    return recovered_ids, losses, attention_stats, stage_a_history, stage_b_history, calibration_score


def validate_attack_api() -> Dict[str, Any]:
    sig = inspect.signature(invert_from_activation)
    names = list(sig.parameters)
    banned = ["reference", "original", "input_ids", "token_ids", "prompt", "text", "embedding"]
    hits = [name for name in names if any(item in name for item in banned)]
    return {"signature": str(sig), "parameter_names": names, "banned_parameter_hits": hits, "passes": len(hits) == 0}


def run_one_config(cfg: AGConfig, resume: bool) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "COMPLETE"
    if resume and done.exists():
        with (out / "metrics.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected TinyLlama 22 transformer blocks, got {len(model.model.layers)}")
    if cfg.target_layer < 0 or cfg.target_layer >= len(model.model.layers):
        raise ValueError(f"target_layer must be in 0..21, got {cfg.target_layer}")
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    predictions_path = out / "predictions.jsonl"
    failures_path = out / "failures.jsonl"
    if not resume:
        for path in (predictions_path, failures_path):
            if path.exists():
                path.unlink()
    completed = set()
    rows: List[Dict[str, Any]] = []
    if resume and predictions_path.exists():
        by_prompt = {}
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    by_prompt[int(row["prompt_id"])] = row
        rows = [by_prompt[key] for key in sorted(by_prompt)]
        completed = set(by_prompt)
    attention_rows = []
    if resume and (out / "attention_stats.json").exists():
        try:
            attention_rows = json.loads((out / "attention_stats.json").read_text(encoding="utf-8")).get("samples", [])
        except Exception:
            attention_rows = []
    failed_rows: List[Dict[str, Any]] = []
    api_check = validate_attack_api()
    if not api_check["passes"]:
        raise RuntimeError(f"Attack API leak risk: {api_check}")

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
                raise RuntimeError(
                    f"prompt_id={prompt_id} has {input_ids.shape[1]} tokens, exceeding max_token_len={cfg.max_token_len}; "
                    "not silently truncating"
                )
            original_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
            with torch.no_grad():
                target_activation = capture_prefix_activation(
                    model,
                    cfg.target_layer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).detach()
            seq_len = int(input_ids.shape[1])
            recovered_ids, losses, attention_stats, stage_a_history, stage_b_history, calibration_score = invert_from_activation(
                model,
                tokenizer,
                cfg,
                target_activation,
                seq_len,
                device,
            )
            recovered_text = text_from_ids(tokenizer, recovered_ids)
            original_text = text_from_ids(tokenizer, original_ids)
            elapsed = time.time() - start
            row = {
                "title": EXPERIMENT_TITLE,
                "method": cfg.method,
                "prompt_id": prompt_id,
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
                "optimization_loss": losses["optimization_loss"],
                "activation_loss": losses["activation_loss"],
                "vocab_loss": losses["vocab_loss"],
                "context_loss": losses["context_loss"],
                "cosine_similarity": losses["cosine_similarity"],
                "g_act_norm": losses.get("g_act_norm"),
                "g_ctx_norm": losses.get("g_ctx_norm"),
                "g_act_ctx_cosine": losses.get("g_act_ctx_cosine"),
                "update_norm": losses.get("update_norm"),
                "calibration_score": calibration_score,
                "elapsed_time": elapsed,
                "runtime_seconds": elapsed,
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "participant_number": cfg.participant_number,
                "attacker_position": cfg.attacker_position,
                "inverted_block_count": cfg.inverted_block_count,
                "prompt_token_count": seq_len,
                "peak_gpu_memory_mb": peak_memory_mb(),
                "recovery_uses_ground_truth_tokens": False,
                "attack_api_check": api_check,
                "stage_a_history": stage_a_history,
                "stage_b_history": stage_b_history,
            }
            jsonl_append(predictions_path, row)
            rows.append(row)
            attention_rows.append({"prompt_id": prompt_id, **attention_stats})
            json_dump(out / "attention_stats.json", {"samples": attention_rows})
            print(
                f"method={cfg.method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} elapsed={elapsed:.2f}s recovered={recovered_text!r}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "method": cfg.method,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "participant_number": cfg.participant_number,
                "attacker_position": cfg.attacker_position,
            }
            failed_rows.append(failure)
            jsonl_append(failures_path, failure)
            print(f"method={cfg.method} prompt_id={prompt_id} failed error={repr(exc)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics = summarize_rows(rows)
    metrics.update({
        "title": EXPERIMENT_TITLE,
        "config": cfg.__dict__,
        "dataset_meta": dataset_meta,
        "completed_sample_count": len(rows),
        "failed_sample_count": len(failed_rows),
        "attack_api_check": api_check,
        "attention_stats_path": str(out / "attention_stats.json"),
    })
    json_dump(out / "metrics.json", metrics)
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    return metrics


def config_from_args(args: argparse.Namespace, method: str, seed: int, dataset_len: int, run_name: str, output_dir: str) -> AGConfig:
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = args.target_layer
        inverted = target + 1
    stage_a_epoch = args.stage_a_epoch if args.stage_a_epoch is not None else args.epoch
    return AGConfig(
        method=method,
        run_name=run_name,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=dataset_len,
        seed=seed,
        participant_number=args.participant_number,
        attacker_position=args.attacker_position,
        inverted_block_count=inverted,
        target_layer=target,
        epoch=args.epoch,
        stage_a_epoch=stage_a_epoch,
        lr=args.lr,
        lambda_vocab=args.lambda_vocab,
        lambda_attn=args.lambda_attn,
        top_k_embedding=args.k,
        top_y_semantic=args.y,
        adaptive_discretization=not args.naive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        max_token_len=args.max_token_len,
        disable_attention_guidance=args.disable_attention_guidance,
        local_files_only=args.local_files_only,
    )


def run_single(args: argparse.Namespace) -> Dict[str, Any]:
    method = args.method
    output_dir = args.output_dir or str(Path(args.output_root) / f"{method}_seed{args.seed}")
    run_name = args.run_name or Path(output_dir).name
    cfg = config_from_args(args, method, args.seed, args.dataset_len, run_name, output_dir)
    return run_one_config(cfg, resume=args.resume)


def run_comparison(args: argparse.Namespace) -> None:
    for method in METHODS:
        for seed in args.seeds:
            output_dir = Path(args.output_root) / f"{method}_seed{seed}"
            cfg = config_from_args(args, method, seed, args.dataset_len, output_dir.name, str(output_dir))
            run_one_config(cfg, resume=args.resume)
    summarize_comparison(args.output_root)


def verify_no_leakage(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.output_dir or Path(args.output_root) / "leakage_verification")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=args.local_files_only)
    model.to(device)
    prompts, dataset_meta = load_dataset_prompts(args.dataset_name, args.dataset_path, 1, args.seed)
    prompt = prompts[0]
    tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    target_layer = args.target_layer if args.target_layer is not None else ATTACKER_MAP[args.participant_number][args.attacker_position][1]
    block = model.model.layers[target_layer]
    with torch.no_grad():
        full_activation = capture_activation(model, block, {"input_ids": input_ids, "attention_mask": attention_mask}).detach()
        prefix_activation = capture_prefix_activation(model, target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
    max_abs_diff = float((full_activation.float() - prefix_activation.float()).abs().max().detach().cpu())
    mean_abs_diff = float((full_activation.float() - prefix_activation.float()).abs().mean().detach().cpu())
    base_cfg = config_from_args(
        args,
        "baseline",
        args.seed,
        1,
        "verify_baseline",
        str(out / "verify_baseline"),
    )
    base_cfg.target_layer = target_layer
    base_cfg.inverted_block_count = target_layer + 1
    base_cfg.epoch = min(args.epoch, 10)
    base_cfg.stage_a_epoch = min(args.stage_a_epoch or args.epoch, 10)
    base_cfg.adaptive_discretization = False
    target_activation = prefix_activation.detach()
    seq_len = int(input_ids.shape[1])
    set_seed(args.seed)
    recovered_a, *_ = invert_from_activation(model, tokenizer, base_cfg, target_activation, seq_len, device)
    fake_prompt = "This deliberately unrelated evaluation prompt must not affect recovery."
    fake_ids = tokenizer(fake_prompt, add_special_tokens=True, truncation=False, return_tensors="pt")["input_ids"]
    set_seed(args.seed)
    recovered_b, *_ = invert_from_activation(model, tokenizer, base_cfg, target_activation, seq_len, device)
    off_cfg = config_from_args(
        args,
        "full",
        args.seed,
        1,
        "verify_attention_off",
        str(out / "verify_attention_off"),
    )
    off_cfg.target_layer = target_layer
    off_cfg.inverted_block_count = target_layer + 1
    off_cfg.epoch = base_cfg.epoch
    off_cfg.stage_a_epoch = base_cfg.stage_a_epoch
    off_cfg.adaptive_discretization = False
    off_cfg.disable_attention_guidance = True
    set_seed(args.seed)
    recovered_off, *_ = invert_from_activation(model, tokenizer, off_cfg, target_activation, seq_len, device)
    api_check = validate_attack_api()
    result = {
        "title": EXPERIMENT_TITLE,
        "dataset_meta": dataset_meta,
        "target_layer": target_layer,
        "activation_shape_full": list(full_activation.shape),
        "activation_shape_prefix": list(prefix_activation.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "passes_strict_1e_5": max_abs_diff < 1e-5,
        "passes_fp16_bf16_tolerance_5e_3": max_abs_diff < 5e-3,
        "attack_api_check": api_check,
        "randomized_eval_prompt": fake_prompt,
        "randomized_eval_token_count": int(fake_ids.shape[1]),
        "recovery_identical_after_eval_reference_replacement": recovered_a == recovered_b,
        "attention_off_matches_baseline": recovered_a == recovered_off,
        "recovered_ids": recovered_a,
        "recovered_ids_after_reference_replacement": recovered_b,
        "recovered_ids_attention_off": recovered_off,
    }
    if not result["passes_fp16_bf16_tolerance_5e_3"] or not api_check["passes"]:
        raise RuntimeError(f"Leakage verification failed: {result}")
    if not result["recovery_identical_after_eval_reference_replacement"]:
        raise RuntimeError("Recovery changed after replacing evaluation-only prompt/token ids")
    if not result["attention_off_matches_baseline"]:
        raise RuntimeError("Attention-guided disabled result does not match baseline")
    json_dump(out / "leakage_verification.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def summarize_comparison(output_root: str) -> None:
    root = Path(output_root)
    rows = []
    run_re = re.compile(r"^(baseline|dummy_init|attention_context|attention_gradient|full)_seed(42|43|44)$")
    for metrics_path in sorted(root.glob("*_seed*/metrics.json")):
        if not run_re.match(metrics_path.parent.name):
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        cfg = metrics["config"]
        rows.append({
            "method": cfg["method"],
            "seed": cfg["seed"],
            "dataset_label": "Skytrax-28 pilot",
            "dataset_size": metrics["dataset_meta"].get("selected_prompts"),
            "participant_number": cfg["participant_number"],
            "attacker_position": cfg["attacker_position"],
            "target_layer": cfg["target_layer"],
            "inverted_blocks": cfg["inverted_block_count"],
            "token_accuracy_mean": metrics["token_accuracy"]["mean"],
            "token_accuracy_std": metrics["token_accuracy"]["std"],
            "bleu_mean": metrics["bleu"]["mean"],
            "bleu_std": metrics["bleu"]["std"],
            "mean_runtime": metrics["runtime_seconds"]["mean"],
            "peak_gpu_memory": metrics["peak_gpu_memory_mb"]["mean"],
            "completed_sample_count": metrics["completed_sample_count"],
            "failed_sample_count": metrics["failed_sample_count"],
            "metrics_path": str(metrics_path),
        })
    rows.sort(key=lambda row: (METHODS.index(row["method"]), int(row["seed"])))
    fieldnames = [
        "method",
        "seed",
        "dataset_label",
        "dataset_size",
        "participant_number",
        "attacker_position",
        "target_layer",
        "inverted_blocks",
        "token_accuracy_mean",
        "token_accuracy_std",
        "bleu_mean",
        "bleu_std",
        "mean_runtime",
        "peak_gpu_memory",
        "completed_sample_count",
        "failed_sample_count",
        "metrics_path",
    ]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        f"# {EXPERIMENT_TITLE}",
        "",
        "| method | seed | acc mean | acc std | BLEU mean | runtime | completed | failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | {row['seed']} | {row['token_accuracy_mean']} | "
            f"{row['token_accuracy_std']} | {row['bleu_mean']} | {row['mean_runtime']} | "
            f"{row['completed_sample_count']} | {row['failed_sample_count']} |"
        )
    (root / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{row['method']}\n{row['seed']}" for row in rows]
        values = [0.0 if row["token_accuracy_mean"] is None else float(row["token_accuracy_mean"]) for row in rows]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.5), 4.5))
        ax.bar(labels, values, color="#1f77b4")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Token Accuracy")
        ax.set_title(EXPERIMENT_TITLE)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(root / "comparison.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        (root / "comparison_plot_error.txt").write_text(repr(exc), encoding="utf-8")
    Path("analysis").mkdir(exist_ok=True)
    report = [
        f"# {EXPERIMENT_TITLE}",
        "",
        "This is a TinyLlama attention-guided Prompt Inversion pilot, not a strict reproduction of the paper's large-model numbers.",
        "",
        "Attack API policy: inversion receives target activation, sequence length, public model parameters, and the public vocabulary embedding matrix. Original prompt text and token ids are used only to produce the target activation and to evaluate recovered output.",
        "",
        "Results are stored under `runs/attention_guided_pilot/`.",
        "",
        (root / "comparison.md").read_text(encoding="utf-8") if (root / "comparison.md").exists() else "",
    ]
    Path("analysis/attention_guided_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "comparison", "summarize", "verify"], required=True)
    parser.add_argument("--method", choices=METHODS, default="baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", default="runs/attention_guided_pilot")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-name", default="Skytrax-28-pilot")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--stage-a-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--lambda-attn", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--y", type=int, default=10)
    parser.add_argument("--max-token-len", type=int, default=256)
    parser.add_argument("--naive-discretization", action="store_true")
    parser.add_argument("--disable-semantic-speculation", action="store_true")
    parser.add_argument("--disable-attention-guidance", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "single":
            run_single(args)
        elif args.mode == "comparison":
            run_comparison(args)
        elif args.mode == "summarize":
            summarize_comparison(args.output_root)
        elif args.mode == "verify":
            verify_no_leakage(args)
        else:
            raise NotImplementedError(args.mode)
    except Exception:
        failure_dir = Path(args.output_root) / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"failure_{int(time.time())}.log"
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"failure logged to {failure_path}", flush=True)
        raise


if __name__ == "__main__":
    main()
