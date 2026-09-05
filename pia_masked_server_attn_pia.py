import argparse
import ast
import csv
import inspect
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask

from pia_alpha_gm import (
    alpha_nn_initialization,
    normalize_run_float,
    stage_a_dummy_proxy,
    summarize_rows,
)
from pia_attention_guided import (
    context_projection,
    enforce_embedding_constraints,
    fixed_embedding_tensor,
    random_public_embeddings,
    variable_view,
)
from pia_tinyllama import (
    ATTACKER_MAP,
    MODEL_NAME,
    TOTAL_BLOCKS,
    activation_calibrated_discretization,
    bleu_score,
    capture_activation,
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


EXPERIMENT_TITLE = "Masked/Tempered Server-Attention Prompt Inversion (MTS-PIA) TinyLlama pilot"
METHODS = [
    "B0",
    "B0V",
    "P1",
    "P2",
    "P3",
    "P4",
    "P4S",
    "P4D",
    "P4DL",
    "P4LWR",
    "P4DR",
    "P4C",
    "P4DG",
    "P4SRES",
    "P4A",
    "P4SR",
    "P4CAR",
    "P4DPTR",
    "P4CARPTR",
    "CAR",
    "CARPTR",
    "P5",
    "P6",
    "original_pia_baseline",
    "variable_only_uniform",
    "server_attn_last_raw",
    "mts_mean_query",
    "mts_last_window",
    "attn_scale_mean_query",
    "attn_scale_mean_query_schedule",
    "attn_scale_mean_query_schedule_residual",
    "attn_scale_mean_query_adaptive_beta",
    "attn_scale_mean_query_schedule_refine",
    "attn_last_window_linear_residual_schedule",
    "confidence_aware_rollout_residual",
    "attn_weighted_residual_ptr",
    "confidence_aware_rollout_residual_ptr",
    "attn_scale_last_window",
    "attn_scale_last_window_gate",
]
METHOD_ALIASES = {
    "B0": "original_pia_baseline",
    "B0V": "variable_only_uniform",
    "P1": "server_attn_last_raw",
    "P2": "mts_mean_query",
    "P3": "mts_last_window",
    "P4": "attn_scale_mean_query",
    "P4S": "attn_scale_mean_query_schedule",
    "P4D": "attn_weighted_residual_schedule",
    "P4DL": "attn_linear_weighted_residual_schedule",
    "P4LWR": "attn_last_window_linear_residual_schedule",
    "P4DR": "attn_rho_weighted_residual_schedule",
    "P4C": "attn_calibrated_weighted_residual_schedule",
    "P4DG": "attn_gate_weighted_residual_schedule",
    "P4SRES": "attn_scale_mean_query_schedule_residual",
    "P4A": "attn_scale_mean_query_adaptive_beta",
    "P4SR": "attn_scale_mean_query_schedule_refine",
    "P4CAR": "confidence_aware_rollout_residual",
    "CAR": "confidence_aware_rollout_residual",
    "P4DPTR": "attn_weighted_residual_ptr",
    "P4CARPTR": "confidence_aware_rollout_residual_ptr",
    "CARPTR": "confidence_aware_rollout_residual_ptr",
    "P5": "attn_scale_last_window",
    "P6": "attn_scale_last_window_gate",
}
BASELINE_METHODS = {"original_pia_baseline", "variable_only_uniform"}
RAW_SERVER_ATTN_METHODS = {"server_attn_last_raw"}
MTS_METHODS = {"mts_mean_query", "mts_last_window"}
ATTN_SCALE_METHODS = {
    "attn_scale_mean_query",
    "attn_scale_mean_query_schedule",
    "attn_scale_mean_query_schedule_residual",
    "attn_weighted_residual_schedule",
    "attn_linear_weighted_residual_schedule",
    "attn_last_window_linear_residual_schedule",
    "attn_rho_weighted_residual_schedule",
    "attn_calibrated_weighted_residual_schedule",
    "attn_gate_weighted_residual_schedule",
    "attn_scale_mean_query_adaptive_beta",
    "attn_scale_mean_query_schedule_refine",
    "confidence_aware_rollout_residual",
    "attn_weighted_residual_ptr",
    "confidence_aware_rollout_residual_ptr",
    "attn_scale_last_window",
    "attn_scale_last_window_gate",
}
CAR_METHODS = {
    "confidence_aware_rollout_residual",
    "confidence_aware_rollout_residual_ptr",
}
PTR_METHODS = {
    "attn_weighted_residual_ptr",
    "confidence_aware_rollout_residual_ptr",
}
ATTENTION_WEIGHTED_RESIDUAL_METHODS = {
    "attn_weighted_residual_schedule",
    "attn_rho_weighted_residual_schedule",
    "attn_calibrated_weighted_residual_schedule",
    "attn_gate_weighted_residual_schedule",
    "confidence_aware_rollout_residual",
    "attn_weighted_residual_ptr",
    "confidence_aware_rollout_residual_ptr",
}
LINEAR_WEIGHTED_RESIDUAL_METHODS = {
    "attn_linear_weighted_residual_schedule",
    "attn_last_window_linear_residual_schedule",
}
SERVER_ATTN_METHODS = RAW_SERVER_ATTN_METHODS | MTS_METHODS | ATTN_SCALE_METHODS | CAR_METHODS | PTR_METHODS
SERVER_ATTENTION_METHODS = SERVER_ATTN_METHODS
PROJECTION_REFINE_METHODS = {"attn_scale_mean_query_schedule_refine"}
DUMMY_METHODS: set[str] = set()
ALPHA_METHODS: set[str] = set()
HELDOUT_FROZEN_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "P2": {
        "method": "mts_mean_query",
        "beta": 0.75,
        "weight_power": 0.50,
        "weight_min": 0.25,
        "weight_max": 4.0,
        "residual_rollout": True,
        "server_rollout_depth": "last2",
    },
    "P3": {
        "method": "mts_last_window",
        "beta": 0.25,
        "weight_power": 0.50,
        "weight_min": 0.25,
        "weight_max": 2.0,
        "residual_rollout": True,
        "server_rollout_depth": "all",
    },
}
HELDOUT_BASELINES: Dict[str, str] = {
    "B0": "original_pia_baseline",
    "B0V": "variable_only_uniform",
}


@dataclass(frozen=True)
class GpuMemory:
    index: int
    free_mb: int
    total_mb: int


@dataclass(frozen=True)
class AutoBatchPlan:
    jobs_by_gpu: Dict[int, int]
    total_jobs: int
    reserve_free_mb: int
    min_free_mb_per_job: int
    max_jobs_per_gpu: int
    max_total_jobs: int


@dataclass(frozen=True)
class AutoBatchTask:
    method: str
    target_layer: int
    gpu_index: int
    run_name: Optional[str] = None
    output_subdir: str = "multi_layer"
    beta: Optional[float] = None
    weight_power: Optional[float] = None
    weight_min: Optional[float] = None
    weight_max: Optional[float] = None
    alpha_min: Optional[float] = None
    alpha_max: Optional[float] = None
    attention_start_ratio: Optional[float] = None
    attention_full_ratio: Optional[float] = None
    uncertainty_fraction: Optional[float] = None
    adaptive_min_beta: Optional[float] = None
    refine_epoch: Optional[int] = None
    lambda_projection: Optional[float] = None
    refine_lr_scale: Optional[float] = None
    residual_alpha_rho: Optional[float] = None
    residual_rollout: Optional[bool] = None
    server_rollout_depth: Optional[str] = None

@dataclass(frozen=True)
class VariableMaskAudit:
    valid_mask: torch.Tensor
    fixed_public_mask: torch.Tensor
    special_mask: torch.Tensor
    variable_mask: torch.Tensor
    uniform_weights: torch.Tensor
    variable_count: int
    known_public_special_positions: List[int]
    token_id_based_special_mask_available: bool
    last_valid_position: Optional[int]
    rows: List[Dict[str, Any]]


def plan_auto_batch_jobs(
    gpu_memories: Sequence[GpuMemory],
    min_free_mb_per_job: int,
    reserve_free_mb: int,
    max_jobs_per_gpu: int,
    max_total_jobs: int,
) -> AutoBatchPlan:
    if min_free_mb_per_job <= 0:
        raise ValueError("min_free_mb_per_job must be positive")
    if reserve_free_mb < 0:
        raise ValueError("reserve_free_mb must be non-negative")
    if max_jobs_per_gpu <= 0:
        raise ValueError("max_jobs_per_gpu must be positive")
    if max_total_jobs <= 0:
        raise ValueError("max_total_jobs must be positive")
    if not gpu_memories:
        raise ValueError("at least one GPU is required")

    sorted_gpus = sorted(gpu_memories, key=lambda gpu: (-gpu.free_mb, gpu.index))
    slots: List[int] = []
    for gpu in sorted_gpus:
        usable_mb = max(0, gpu.free_mb - reserve_free_mb)
        job_count = min(max_jobs_per_gpu, usable_mb // min_free_mb_per_job)
        slots.extend([gpu.index] * int(job_count))

    if not slots:
        slots = [sorted_gpus[0].index]
    slots = slots[:max_total_jobs]

    jobs_by_gpu: Dict[int, int] = {}
    for gpu_index in slots:
        jobs_by_gpu[gpu_index] = jobs_by_gpu.get(gpu_index, 0) + 1
    return AutoBatchPlan(
        jobs_by_gpu=jobs_by_gpu,
        total_jobs=len(slots),
        reserve_free_mb=reserve_free_mb,
        min_free_mb_per_job=min_free_mb_per_job,
        max_jobs_per_gpu=max_jobs_per_gpu,
        max_total_jobs=max_total_jobs,
    )


def build_auto_batch_tasks(
    methods: Sequence[str],
    target_layers: Sequence[int],
    jobs_by_gpu: Dict[int, int],
) -> List[AutoBatchTask]:
    slots = gpu_slots_from_jobs(jobs_by_gpu)
    if not slots:
        raise ValueError("jobs_by_gpu must contain at least one slot")

    tasks: List[AutoBatchTask] = []
    slot_index = 0
    for layer in target_layers:
        for method in methods:
            tasks.append(AutoBatchTask(method=str(method), target_layer=int(layer), gpu_index=slots[slot_index % len(slots)]))
            slot_index += 1
    return tasks


def gpu_slots_from_jobs(jobs_by_gpu: Dict[int, int]) -> List[int]:
    slots: List[int] = []
    for gpu_index, job_count in sorted(jobs_by_gpu.items()):
        slots.extend([int(gpu_index)] * int(job_count))
    return slots


def float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_dev_ablation_tasks(
    gpu_slots: Sequence[int],
    seed: int,
    target_layer: int,
    epoch: int,
    top_k: int,
    weight_min: float,
    betas: Sequence[float],
    powers: Sequence[float],
    weight_maxes: Sequence[float],
) -> List[AutoBatchTask]:
    if not gpu_slots:
        raise ValueError("gpu_slots must not be empty")
    tasks: List[AutoBatchTask] = []
    slot_index = 0

    def next_gpu() -> int:
        nonlocal slot_index
        gpu = int(gpu_slots[slot_index % len(gpu_slots)])
        slot_index += 1
        return gpu

    for method in ["B0", "B0V", "P1"]:
        canonical = canonical_method(method)
        run_name = f"{canonical}_seed{seed}_layer{target_layer}_epoch{epoch}_k{top_k}"
        tasks.append(
            AutoBatchTask(
                method=method,
                target_layer=int(target_layer),
                gpu_index=next_gpu(),
                run_name=run_name,
                output_subdir="dev_ablation_baselines",
                weight_min=float(weight_min),
            )
        )

    for method in ["P2", "P3"]:
        canonical = canonical_method(method)
        for beta in betas:
            for power in powers:
                for weight_max in weight_maxes:
                    run_name = (
                        f"{canonical}_seed{seed}_layer{target_layer}_epoch{epoch}_k{top_k}"
                        f"_beta{float_tag(beta)}_power{float_tag(power)}"
                        f"_wmin{float_tag(weight_min)}_wmax{float_tag(weight_max)}"
                    )
                    tasks.append(
                        AutoBatchTask(
                            method=method,
                            target_layer=int(target_layer),
                            gpu_index=next_gpu(),
                            run_name=run_name,
                            output_subdir="dev_ablation_runs",
                            beta=float(beta),
                            weight_power=float(power),
                            weight_min=float(weight_min),
                            weight_max=float(weight_max),
                            residual_rollout=True,
                            server_rollout_depth="all",
                        )
                    )
    return tasks


def build_structure_ablation_tasks(
    best_rows: Sequence[Dict[str, Any]],
    gpu_slots: Sequence[int],
    seed: int,
    target_layer: int,
    epoch: int,
    top_k: int,
    weight_min: float,
) -> List[AutoBatchTask]:
    if not gpu_slots:
        raise ValueError("gpu_slots must not be empty")
    tasks: List[AutoBatchTask] = []
    slot_index = 0

    def next_gpu() -> int:
        nonlocal slot_index
        gpu = int(gpu_slots[slot_index % len(gpu_slots)])
        slot_index += 1
        return gpu

    for row in best_rows:
        method = str(row.get("method"))
        if method not in {"mts_mean_query", "mts_last_window", "P2", "P3"}:
            continue
        beta = float(row["beta"])
        power = float(row["weight_power"])
        weight_max = float(row["weight_max"])
        canonical = canonical_method(method)
        for residual in [True, False]:
            for depth in ["all", "last2"]:
                residual_tag = "reson" if residual else "resoff"
                run_name = (
                    f"{canonical}_seed{seed}_layer{target_layer}_epoch{epoch}_k{top_k}"
                    f"_beta{float_tag(beta)}_power{float_tag(power)}"
                    f"_wmin{float_tag(weight_min)}_wmax{float_tag(weight_max)}"
                    f"_{residual_tag}_depth{depth}"
                )
                tasks.append(
                    AutoBatchTask(
                        method=canonical,
                        target_layer=int(target_layer),
                        gpu_index=next_gpu(),
                        run_name=run_name,
                        output_subdir="structure_ablation_runs",
                        beta=beta,
                        weight_power=power,
                        weight_min=float(weight_min),
                        weight_max=weight_max,
                        residual_rollout=residual,
                        server_rollout_depth=depth,
                    )
                )
    return tasks


def query_gpu_memory() -> List[GpuMemory]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    memories: List[GpuMemory] = []
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        index_text, free_text, total_text = [part.strip() for part in line.split(",")]
        memories.append(GpuMemory(index=int(index_text), free_mb=int(free_text), total_mb=int(total_text)))
    return memories


def build_single_run_command(args: argparse.Namespace, task: AutoBatchTask) -> List[str]:
    method_name = canonical_method(task.method)
    run_name = task.run_name or f"{method_name}_seed{args.seed}_layer{task.target_layer}_epoch{args.epoch}_k{args.k}"
    output_dir = str(Path(args.output_root) / task.output_subdir / run_name)
    beta = args.beta if task.beta is None else task.beta
    weight_power = args.weight_power if task.weight_power is None else task.weight_power
    weight_min = args.weight_min if task.weight_min is None else task.weight_min
    weight_max = args.weight_max if task.weight_max is None else task.weight_max
    alpha_min = args.alpha_min if task.alpha_min is None else task.alpha_min
    alpha_max = args.alpha_max if task.alpha_max is None else task.alpha_max
    attention_start_ratio = args.attention_start_ratio if task.attention_start_ratio is None else task.attention_start_ratio
    attention_full_ratio = args.attention_full_ratio if task.attention_full_ratio is None else task.attention_full_ratio
    uncertainty_fraction = args.uncertainty_fraction if task.uncertainty_fraction is None else task.uncertainty_fraction
    adaptive_min_beta = args.adaptive_min_beta if task.adaptive_min_beta is None else task.adaptive_min_beta
    refine_epoch = args.refine_epoch if task.refine_epoch is None else task.refine_epoch
    lambda_projection = args.lambda_projection if task.lambda_projection is None else task.lambda_projection
    refine_lr_scale = args.refine_lr_scale if task.refine_lr_scale is None else task.refine_lr_scale
    residual_alpha_rho = args.residual_alpha_rho if task.residual_alpha_rho is None else task.residual_alpha_rho
    residual_rollout = (not args.no_residual_rollout) if task.residual_rollout is None else bool(task.residual_rollout)
    server_rollout_depth = args.server_rollout_depth if task.server_rollout_depth is None else task.server_rollout_depth
    command = [
        sys.executable,
        "pia_masked_server_attn_pia.py",
        "--mode",
        "single",
        "--method",
        task.method,
        "--output-root",
        args.output_root,
        "--output-dir",
        output_dir,
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
        "--target-layer",
        str(task.target_layer),
        "--epoch",
        str(args.epoch),
        "--lr",
        str(args.lr),
        "--lambda-vocab",
        str(args.lambda_vocab),
        "--lambda-dummy",
        str(args.lambda_dummy),
        "--lambda-context",
        str(args.lambda_context),
        "--k",
        str(args.k),
        "--y",
        str(args.y),
        "--gamma",
        str(args.gamma),
        "--weight-source",
        args.weight_source,
        "--weight-floor",
        str(args.weight_floor),
        "--weight-power",
        str(weight_power),
        "--weight-min",
        str(weight_min),
        "--weight-max",
        str(weight_max),
        "--alpha-min",
        str(alpha_min),
        "--alpha-max",
        str(alpha_max),
        "--attention-start-ratio",
        str(attention_start_ratio),
        "--attention-full-ratio",
        str(attention_full_ratio),
        "--uncertainty-fraction",
        str(uncertainty_fraction),
        "--adaptive-min-beta",
        str(adaptive_min_beta),
        "--refine-epoch",
        str(refine_epoch),
        "--lambda-projection",
        str(lambda_projection),
        "--refine-lr-scale",
        str(refine_lr_scale),
        "--residual-alpha-rho",
        str(residual_alpha_rho),
        "--beta",
        str(beta),
        "--last-window-size",
        str(args.last_window_size),
        "--server-rollout-depth",
        str(server_rollout_depth),
        "--max-token-len",
        str(args.max_token_len),
        "--grad-clip",
        str(args.grad_clip),
    ]
    if args.stage_a_epoch is not None:
        command.extend(["--stage-a-epoch", str(args.stage_a_epoch)])
    if args.force_download:
        command.append("--force-download")
    if not residual_rollout:
        command.append("--no-residual-rollout")
    if args.naive_discretization:
        command.append("--naive-discretization")
    if args.disable_semantic_speculation:
        command.append("--disable-semantic-speculation")
    if args.local_files_only:
        command.append("--local-files-only")
    if args.resume:
        command.append("--resume")
    return command


def build_child_env(base_env: Dict[str, str], gpu_index: int, transformers_cache: Optional[str]) -> Dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if transformers_cache:
        env["TRANSFORMERS_CACHE"] = transformers_cache
    return env


def print_auto_batch_plan(args: argparse.Namespace) -> AutoBatchPlan:
    gpu_memories = query_gpu_memory()
    plan = plan_auto_batch_jobs(
        gpu_memories,
        min_free_mb_per_job=args.batch_free_mb_per_job,
        reserve_free_mb=args.batch_reserve_free_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        max_total_jobs=args.max_parallel_jobs,
    )
    tasks = build_auto_batch_tasks(args.methods, args.target_layers, plan.jobs_by_gpu)
    payload = {
        "gpu_memories": [gpu.__dict__ for gpu in gpu_memories],
        "plan": {
            "jobs_by_gpu": plan.jobs_by_gpu,
            "total_jobs": plan.total_jobs,
            "reserve_free_mb": plan.reserve_free_mb,
            "min_free_mb_per_job": plan.min_free_mb_per_job,
            "max_jobs_per_gpu": plan.max_jobs_per_gpu,
            "max_total_jobs": plan.max_total_jobs,
        },
        "tasks": [task.__dict__ for task in tasks],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return plan


def run_multi_layer_auto_batch(args: argparse.Namespace) -> None:
    gpu_memories = query_gpu_memory()
    plan = plan_auto_batch_jobs(
        gpu_memories,
        min_free_mb_per_job=args.batch_free_mb_per_job,
        reserve_free_mb=args.batch_reserve_free_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        max_total_jobs=args.max_parallel_jobs,
    )
    tasks = build_auto_batch_tasks(args.methods, args.target_layers, plan.jobs_by_gpu)
    print(
        json.dumps(
            {
                "auto_batch_plan": {
                    "gpu_memories": [gpu.__dict__ for gpu in gpu_memories],
                    "jobs_by_gpu": plan.jobs_by_gpu,
                    "total_jobs": plan.total_jobs,
                }
            },
            ensure_ascii=False,
        )
    )
    if args.dry_run_batch:
        for task in tasks:
            env = f"CUDA_VISIBLE_DEVICES={task.gpu_index}"
            print(env, " ".join(build_single_run_command(args, task)))
        return

    run_auto_batch_task_queue(args, tasks, plan.jobs_by_gpu)
    write_summary(Path(args.output_root))


def run_auto_batch_task_queue(args: argparse.Namespace, tasks: Sequence[AutoBatchTask], jobs_by_gpu: Dict[int, int]) -> None:
    if not tasks:
        return
    log_dir = Path(args.output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    task_queue = list(tasks)
    active: List[Tuple[subprocess.Popen, AutoBatchTask, Any, Path]] = []
    capacity = {int(gpu): int(count) for gpu, count in jobs_by_gpu.items() if int(count) > 0}
    if not capacity:
        raise ValueError("jobs_by_gpu must contain at least one runnable slot")

    while task_queue or active:
        launched = True
        while task_queue and launched:
            launched = False
            active_by_gpu: Dict[int, int] = {}
            for _process, active_task, _log_file, _log_path in active:
                active_by_gpu[active_task.gpu_index] = active_by_gpu.get(active_task.gpu_index, 0) + 1
            for index, task in enumerate(task_queue):
                if active_by_gpu.get(task.gpu_index, 0) >= capacity.get(task.gpu_index, 0):
                    continue
                task_queue.pop(index)
                launched = True
                break
            if not launched:
                break
            env = build_child_env(os.environ, task.gpu_index, args.transformers_cache)
            command = build_single_run_command(args, task)
            run_name = task.run_name or f"{canonical_method(task.method)}_layer{task.target_layer}"
            log_path = log_dir / f"{run_name}_gpu{task.gpu_index}.log"
            log_file = log_path.open("w", encoding="utf-8")
            print(json.dumps({"launch": command, "gpu": task.gpu_index, "log": str(log_path)}, ensure_ascii=False))
            process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=log_file, stderr=subprocess.STDOUT)
            active.append((process, task, log_file, log_path))

        still_active: List[Tuple[subprocess.Popen, AutoBatchTask, Any, Path]] = []
        for process, task, log_file, log_path in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, task, log_file, log_path))
                continue
            log_file.close()
            if return_code != 0:
                raise RuntimeError(
                    f"auto-batch task failed: method={task.method} layer={task.target_layer} "
                    f"gpu={task.gpu_index} rc={return_code} log={log_path}"
                )
        active = still_active
        if active:
            time.sleep(5)


@dataclass
class SAWConfig:
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
    lambda_dummy: float
    lambda_context: float
    top_k_embedding: int
    top_y_semantic: int
    gamma: float
    max_token_len: int
    grad_clip: float
    weight_source: str
    weight_floor: float
    weight_power: float
    weight_min: float
    weight_max: float
    alpha_min: float
    alpha_max: float
    attention_start_ratio: float
    attention_full_ratio: float
    uncertainty_fraction: float
    adaptive_min_beta: float
    refine_epoch: int
    lambda_projection: float
    refine_lr_scale: float
    beta: float
    last_window_size: int
    residual_rollout: bool
    server_rollout_depth: str
    adaptive_discretization: bool
    semantic_speculation: bool
    local_files_only: bool
    residual_alpha_rho: float = 1.0
    fix_boundary_specials: bool = True


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def attention_weight_source_for_method(method: str) -> str:
    method = canonical_method(method)
    return {
        "server_attn_last_raw": "last_query_raw",
        "mts_mean_query": "mean_query",
        "mts_last_window": "last_window_mean",
        "attn_scale_mean_query": "mean_query",
        "attn_scale_mean_query_schedule": "mean_query",
        "attn_scale_mean_query_schedule_residual": "mean_query",
        "attn_weighted_residual_schedule": "mean_query",
        "attn_linear_weighted_residual_schedule": "mean_query",
        "attn_last_window_linear_residual_schedule": "last_window_mean",
        "attn_rho_weighted_residual_schedule": "mean_query",
        "attn_calibrated_weighted_residual_schedule": "mean_query",
        "attn_gate_weighted_residual_schedule": "mean_query",
        "attn_scale_mean_query_adaptive_beta": "mean_query",
        "attn_scale_mean_query_schedule_refine": "mean_query",
        "confidence_aware_rollout_residual": "mean_query",
        "attn_weighted_residual_ptr": "mean_query",
        "confidence_aware_rollout_residual_ptr": "mean_query",
        "attn_scale_last_window": "last_window_mean",
        "attn_scale_last_window_gate": "last_window_mean",
    }[method]


def validate_strict_top1_config(args) -> None:
    method = canonical_method(getattr(args, "method", ""))
    if method not in PTR_METHODS and method not in CAR_METHODS:
        return
    k = int(getattr(args, "k", 1))
    y = int(getattr(args, "y", 0))
    semantic = bool(getattr(args, "semantic_speculation", False))
    if k != 1:
        raise ValueError(f"{method} requires strict Top-1: k must be 1, got {k}")
    if y != 0:
        raise ValueError(f"{method} requires strict Top-1: y must be 0, got {y}")
    if semantic:
        raise ValueError(f"{method} requires semantic_speculation=False")


def validate_strict_top1_method_list(args, methods) -> None:
    for method in methods:
        method_args = argparse.Namespace(**vars(args))
        method_args.method = canonical_method(method)
        validate_strict_top1_config(method_args)


def validate_strict_top1_methods(args) -> None:
    validate_strict_top1_method_list(args, getattr(args, "methods", []) or [])


def validate_strict_top1_for_mode(args) -> None:
    mode = getattr(args, "mode", "single")
    if mode in {"single", "verify"}:
        validate_strict_top1_config(args)
    elif mode in {"multi-layer", "plan-batch"}:
        validate_strict_top1_method_list(args, getattr(args, "methods", []) or [])


def row_normalize(mat: torch.Tensor) -> torch.Tensor:
    denom = mat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return mat / denom


def residual_attention(attn: torch.Tensor, residual: bool = True) -> torch.Tensor:
    if attn.dim() != 2 or attn.shape[0] != attn.shape[1]:
        raise RuntimeError(f"attention must be [seq, seq], got {list(attn.shape)}")
    out = attn.float().clamp_min(0.0)
    if residual:
        out = out + torch.eye(out.shape[0], dtype=out.dtype, device=out.device)
    return row_normalize(out)


def rollout_from_attentions(attentions: Sequence[torch.Tensor], residual: bool = True) -> torch.Tensor:
    if not attentions:
        raise RuntimeError("server attention rollout needs at least one server layer")
    seq = attentions[0].shape[-1]
    rollout = torch.eye(seq, dtype=torch.float32, device=attentions[0].device)
    for attn in attentions:
        rollout = residual_attention(attn, residual=residual) @ rollout
    return row_normalize(rollout)


def build_variable_mask(
    attention_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    special_token_ids: Optional[Sequence[int]] = None,
    token_ids: Optional[torch.Tensor] = None,
) -> VariableMaskAudit:
    if attention_mask.dim() != 2 or attention_mask.shape[0] != 1:
        raise RuntimeError(f"attention_mask must be [1, seq], got {list(attention_mask.shape)}")
    device = attention_mask.device
    seq_len = int(attention_mask.shape[1])
    valid = attention_mask[0].detach().bool()
    fixed = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for pos in fixed_public:
        if 0 <= int(pos) < seq_len:
            fixed[int(pos)] = True
    token_id_special = torch.zeros(seq_len, dtype=torch.bool, device=device)
    special_ids = set(int(x) for x in (special_token_ids or []))
    token_id_special_available = token_ids is not None and bool(special_ids)
    if token_ids is not None and special_ids:
        ids = token_ids[0] if token_ids.dim() == 2 else token_ids
        ids = ids.detach().to(device=device)
        for pos in range(min(seq_len, int(ids.numel()))):
            if int(ids[pos].detach().cpu()) in special_ids:
                token_id_special[pos] = True
    special = token_id_special | fixed
    variable = valid & ~fixed & ~special
    variable_count = int(variable.sum().detach().cpu())
    uniform = torch.where(variable, torch.ones(seq_len, dtype=torch.float32, device=device), torch.zeros(seq_len, dtype=torch.float32, device=device))
    valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
    last_valid_position = int(valid_positions[-1].detach().cpu()) if valid_positions.numel() else None
    known_public_special_positions = [
        int(pos)
        for pos in sorted(fixed_public)
        if 0 <= int(pos) < seq_len and bool(valid[int(pos)].detach().cpu())
    ]
    rows: List[Dict[str, Any]] = []
    for pos in range(seq_len):
        if not bool(valid[pos].detach().cpu()):
            reason = "invalid_attention_mask"
        elif bool(fixed[pos].detach().cpu()):
            reason = "fixed_public"
        elif bool(special[pos].detach().cpu()):
            reason = "special_token"
        else:
            reason = "variable"
        rows.append(
            {
                "position": pos,
                "valid": bool(valid[pos].detach().cpu()),
                "fixed_public": bool(fixed[pos].detach().cpu()),
                "special": bool(special[pos].detach().cpu()),
                "token_id_special": bool(token_id_special[pos].detach().cpu()),
                "known_public_special": bool(fixed[pos].detach().cpu()),
                "variable_mask": bool(variable[pos].detach().cpu()),
                "exclusion_reason": reason,
            }
        )
    return VariableMaskAudit(
        valid_mask=valid.detach(),
        fixed_public_mask=fixed.detach(),
        special_mask=special.detach(),
        variable_mask=variable.detach(),
        uniform_weights=uniform.detach(),
        variable_count=variable_count,
        known_public_special_positions=known_public_special_positions,
        token_id_based_special_mask_available=bool(token_id_special_available),
        last_valid_position=last_valid_position,
        rows=rows,
    )


def variable_mask_audit_payload(audit: VariableMaskAudit) -> Dict[str, Any]:
    return {
        "variable_count": audit.variable_count,
        "known_public_special_positions": audit.known_public_special_positions,
        "token_id_based_special_mask_available": audit.token_id_based_special_mask_available,
        "last_valid_position": audit.last_valid_position,
        "semantic_note": (
            "fixed_public positions such as position 0 are public framing positions available to the attacker. "
            "When attack API token ids are unavailable, this audit cannot claim EOS detection by token id. "
            "The last valid position is only a sequence boundary; exclude it as EOS only if the public protocol "
            "explicitly fixes the final token as EOS."
        ),
        "rows": audit.rows,
    }


def uniform_all_token_activation_loss(pred: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    per_token = torch.mean((pred.float() - target.to(pred.device).float()) ** 2, dim=-1)
    valid = attention_mask.to(pred.device)[0].bool()
    if int(valid.sum().detach().cpu()) == 0:
        raise RuntimeError("attention_mask has no valid positions")
    return per_token[:, valid].mean()


def weighted_variable_activation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    variable_mask: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    per_token = torch.mean((pred.float() - target.to(pred.device).float()) ** 2, dim=-1)
    variable = variable_mask.to(pred.device).bool()
    if int(variable.sum().detach().cpu()) == 0:
        raise RuntimeError("variable_mask has no variable positions")
    if weights is None:
        w = variable.float()
    else:
        w = weights.to(pred.device).float() * variable.float()
    return (per_token * w.unsqueeze(0)).sum() / w.sum().clamp_min(1e-12)


def attention_scaled_activation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    variable_mask: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    variable = variable_mask.to(pred.device).bool()
    if int(variable.sum().detach().cpu()) == 0:
        raise RuntimeError("variable_mask has no variable positions")
    scaled_pred = pred.float() * alpha.to(pred.device).float().view(1, -1, 1)
    per_token = torch.mean((scaled_pred - target.to(pred.device).float()) ** 2, dim=-1)
    return per_token[:, variable].mean()


def attention_weighted_residual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    variable_mask: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    variable = variable_mask.to(pred.device).bool()
    if int(variable.sum().detach().cpu()) == 0:
        raise RuntimeError("variable_mask has no variable positions")
    residual = (pred.float() - target.to(pred.device).float()) * alpha.to(pred.device).float().view(1, -1, 1)
    per_token = torch.mean(residual ** 2, dim=-1)
    return per_token[:, variable].mean()


def attention_linear_weighted_residual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    variable_mask: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    variable = variable_mask.to(pred.device).bool()
    if int(variable.sum().detach().cpu()) == 0:
        raise RuntimeError("variable_mask has no variable positions")
    alpha_sqrt = torch.sqrt(alpha.to(pred.device).float().clamp_min(1e-12)).view(1, -1, 1)
    residual = (pred.float() - target.to(pred.device).float()) * alpha_sqrt
    per_token = torch.mean(residual ** 2, dim=-1)
    return per_token[:, variable].mean()


def attention_scale_schedule_active(step_index: int, epoch: int, start_ratio: float) -> bool:
    total = max(1, int(epoch))
    threshold = int(math.floor(total * min(1.0, max(0.0, float(start_ratio)))))
    return int(step_index) >= threshold


def attention_scale_linear_schedule_beta(
    step_index: int,
    epoch: int,
    start_ratio: float,
    full_ratio: float,
    max_beta: float,
) -> float:
    total = max(1, int(epoch))
    progress = min(1.0, max(0.0, float(step_index) / float(total)))
    start = min(1.0, max(0.0, float(start_ratio)))
    full = min(1.0, max(0.0, float(full_ratio)))
    beta = max(0.0, float(max_beta))
    if progress <= start:
        return 0.0
    if full <= start:
        return beta
    if progress >= full:
        return beta
    return beta * ((progress - start) / max(1e-12, full - start))


def schedule_residual_alpha(
    *,
    alpha: torch.Tensor,
    variable_mask: torch.Tensor,
    epoch_index: int,
    warmup_epochs: int = 50,
    ramp_end_epoch: int = 70,
    weight_min: float = 0.5,
    weight_max: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    variable_mask = variable_mask.to(alpha.device).bool()
    out = torch.zeros_like(alpha, dtype=torch.float32)
    if int(variable_mask.sum().item()) == 0:
        return out
    if epoch_index <= warmup_epochs:
        ramp = 0.0
    elif epoch_index >= ramp_end_epoch:
        ramp = 1.0
    else:
        ramp = float(epoch_index - warmup_epochs) / float(ramp_end_epoch - warmup_epochs)
    target_w = alpha.detach().float().pow(2)
    scheduled_w = 1.0 + ramp * (target_w[variable_mask] - 1.0)
    scheduled_w = bounded_mean_one_project(
        scheduled_w,
        lower=float(weight_min),
        upper=float(weight_max),
        eps=eps,
    )
    out[variable_mask] = torch.sqrt(scheduled_w.to(dtype=out.dtype, device=out.device))
    return out


def interpolate_attention_alpha(
    alpha: torch.Tensor,
    variable_mask: torch.Tensor,
    ramp: float,
    alpha_min: float,
    alpha_max: float,
) -> torch.Tensor:
    variable = variable_mask.to(alpha.device).bool()
    factor = min(1.0, max(0.0, float(ramp)))
    mixed = torch.where(variable, 1.0 + factor * (alpha.float() - 1.0), torch.zeros_like(alpha.float()))
    return bounded_mean_one_weights(mixed, variable, float(alpha_min), float(alpha_max)).detach()


def residualize_attention_alpha(
    alpha: torch.Tensor,
    variable_mask: torch.Tensor,
    rho: float,
    alpha_min: float,
    alpha_max: float,
) -> torch.Tensor:
    if not (0.0 <= float(rho) <= 1.0):
        raise ValueError('rho must be in [0, 1]')
    variable = variable_mask.to(alpha.device).bool()
    if int(variable.sum().detach().cpu()) == 0:
        raise RuntimeError('variable_mask has no variable positions')
    base = alpha.float()
    mixed = torch.zeros_like(base, dtype=torch.float32)
    mixed[variable] = 1.0 + float(rho) * (base[variable] - 1.0)
    return bounded_mean_one_weights(mixed, variable, float(alpha_min), float(alpha_max)).detach()


def embedding_uncertainty_scores(
    z: torch.Tensor,
    embed_weight: torch.Tensor,
    variable_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    variable = variable_mask.to(z.device).bool()
    scores = torch.zeros(variable.shape, dtype=torch.float32, device=z.device)
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    if positions.numel() == 0:
        return scores, {
            "adaptive_uncertainty_mean": 0.0,
            "adaptive_uncertainty_min": 0.0,
            "adaptive_uncertainty_max": 0.0,
            "adaptive_margin_mean": 0.0,
        }
    distances = torch.cdist(z.detach().float()[0, positions], embed_weight.detach().float().to(z.device))
    if distances.shape[1] < 2:
        margins = torch.zeros(distances.shape[0], dtype=torch.float32, device=z.device)
    else:
        top2 = torch.topk(distances, k=2, largest=False, dim=-1).values
        margins = top2[:, 1] - top2[:, 0]
    margin_min = margins.min()
    margin_max = margins.max()
    denom = (margin_max - margin_min).clamp_min(1e-12)
    if float((margin_max - margin_min).detach().cpu()) <= 1e-12:
        normalized = torch.ones_like(margins)
    else:
        normalized = (margin_max - margins) / denom
    scores[positions] = normalized.clamp(0.0, 1.0)
    values = scores[variable]
    return scores.detach(), {
        "adaptive_uncertainty_mean": float(values.mean().detach().cpu()),
        "adaptive_uncertainty_min": float(values.min().detach().cpu()),
        "adaptive_uncertainty_max": float(values.max().detach().cpu()),
        "adaptive_margin_mean": float(margins.mean().detach().cpu()),
        "adaptive_margin_min": float(margins.min().detach().cpu()),
        "adaptive_margin_max": float(margins.max().detach().cpu()),
    }


def apply_adaptive_attention_beta(
    alpha: torch.Tensor,
    variable_mask: torch.Tensor,
    uncertainty_scores: torch.Tensor,
    base_beta: float,
    min_beta: float,
    alpha_min: float,
    alpha_max: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    variable = variable_mask.to(alpha.device).bool()
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("variable_mask has no variable positions")
    beta_max = max(0.0, float(base_beta))
    beta_min = min(beta_max, max(0.0, float(min_beta)))
    uncertainty = uncertainty_scores.to(alpha.device).float().clamp(0.0, 1.0)
    token_beta = torch.zeros_like(alpha.float())
    token_beta[variable] = beta_min + (beta_max - beta_min) * uncertainty[variable]
    if beta_max <= 1e-12:
        mixed = torch.where(variable, torch.ones_like(alpha.float()), torch.zeros_like(alpha.float()))
        factor = torch.zeros_like(alpha.float())
    else:
        factor = torch.zeros_like(alpha.float())
        factor[variable] = token_beta[variable] / beta_max
        mixed = torch.where(variable, 1.0 + factor * (alpha.float() - 1.0), torch.zeros_like(alpha.float()))
    adapted = bounded_mean_one_weights(mixed, variable, float(alpha_min), float(alpha_max)).detach()
    values = adapted[variable]
    beta_values = token_beta[variable]
    factor_values = factor[variable]
    return adapted, {
        "adaptive_beta_min": float(beta_values.min().detach().cpu()),
        "adaptive_beta_mean": float(beta_values.mean().detach().cpu()),
        "adaptive_beta_max": float(beta_values.max().detach().cpu()),
        "adaptive_beta_factor_mean": float(factor_values.mean().detach().cpu()) if factor_values.numel() else 0.0,
        "mean_alpha": float(values.mean().detach().cpu()),
        "std_alpha": float(values.std(unbiased=False).detach().cpu()) if values.numel() > 1 else 0.0,
        "min_alpha": float(values.min().detach().cpu()),
        "max_alpha": float(values.max().detach().cpu()),
    }


def projection_refinement_loss(
    z: torch.Tensor,
    embed_weight: torch.Tensor,
    variable_mask: torch.Tensor,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    variable = variable_mask.to(z.device).bool()
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    if positions.numel() == 0:
        return torch.tensor(0.0, device=z.device), {
            "projection_variable_count": 0,
            "projection_top1_only": True,
        }
    vectors = z[0, positions].float()
    weight = embed_weight.detach().float().to(z.device)
    best_ids = torch.zeros(vectors.shape[0], device=z.device, dtype=torch.long)
    best_dists = torch.full((vectors.shape[0],), float("inf"), device=z.device)
    with torch.no_grad():
        vector_norm = torch.sum(vectors.detach() * vectors.detach(), dim=1, keepdim=True)
        for start in range(0, weight.shape[0], int(chunk_size)):
            chunk = weight[start : start + int(chunk_size)]
            dists = vector_norm + torch.sum(chunk * chunk, dim=1).unsqueeze(0) - 2 * vectors.detach() @ chunk.t()
            values, indices = torch.min(dists, dim=1)
            update = values < best_dists
            best_dists = torch.where(update, values, best_dists)
            best_ids = torch.where(update, indices + start, best_ids)
    nearest = weight[best_ids].detach()
    loss = F.mse_loss(vectors, nearest, reduction="mean")
    return loss, {
        "projection_variable_count": int(positions.numel()),
        "projection_top1_only": True,
        "projection_distance_mean": float(best_dists.mean().detach().cpu()),
        "projection_distance_min": float(best_dists.min().detach().cpu()),
        "projection_distance_max": float(best_dists.max().detach().cpu()),
    }


def _attention_raw_from_rollout(
    rollout: torch.Tensor,
    valid: torch.Tensor,
    variable: torch.Tensor,
    source: str,
    last_window_size: int,
) -> Tuple[torch.Tensor, List[int]]:
    valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
    variable_positions = torch.nonzero(variable, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        raise RuntimeError("attention_mask has no valid positions")
    if variable_positions.numel() == 0:
        raise RuntimeError("variable_mask has no variable positions")
    if source == "mean_query":
        query_positions = [int(x) for x in variable_positions.detach().cpu().tolist()]
        raw = rollout[variable].mean(dim=0)
    elif source == "last_window_mean":
        selected = variable_positions[-max(1, int(last_window_size)) :]
        query_positions = [int(x) for x in selected.detach().cpu().tolist()]
        raw = rollout[selected].mean(dim=0)
    elif source == "last_query_raw":
        query_positions = [int(valid_positions[-1].detach().cpu())]
        raw = rollout[query_positions[0]].clone()
    elif source == "uniform":
        query_positions = [int(x) for x in variable_positions.detach().cpu().tolist()]
        raw = torch.ones(rollout.shape[0], dtype=torch.float32, device=rollout.device)
    else:
        raise ValueError(f"unknown attention scale source={source!r}")
    return raw.float().clamp_min(0.0), query_positions


def _attention_scale_stats(
    alpha: torch.Tensor,
    variable: torch.Tensor,
    fixed_public: Dict[int, int],
    valid: torch.Tensor,
    base_stats: Dict[str, Any],
) -> Dict[str, Any]:
    values = alpha[variable]
    valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
    last_valid = int(valid_positions[-1].detach().cpu()) if valid_positions.numel() else None
    fixed_sum = 0.0
    for pos in fixed_public:
        if 0 <= int(pos) < alpha.numel():
            fixed_sum += float(alpha[int(pos)].detach().cpu())
    stats = dict(base_stats)
    stats.update(
        {
            "last_valid_position": last_valid,
            "final_bos_alpha": float(alpha[0].detach().cpu()) if alpha.numel() else 0.0,
            "final_last_valid_alpha": float(alpha[last_valid].detach().cpu()) if last_valid is not None else 0.0,
            "fixed_public_alpha_sum": fixed_sum,
            "mean_alpha": float(values.mean().detach().cpu()),
            "std_alpha": float(values.std(unbiased=False).detach().cpu()) if values.numel() > 1 else 0.0,
            "min_alpha": float(values.min().detach().cpu()),
            "max_alpha": float(values.max().detach().cpu()),
            "mean": float(values.mean().detach().cpu()),
            "std": float(values.std(unbiased=False).detach().cpu()) if values.numel() > 1 else 0.0,
            "min": float(values.min().detach().cpu()),
            "max": float(values.max().detach().cpu()),
            "alpha_active_count": int(values.numel()),
            "alpha_active_mean": float(values.mean().detach().cpu()),
            "alpha_active_std": float(values.std(unbiased=False).detach().cpu()) if values.numel() > 1 else 0.0,
            "alpha_active_min": float(values.min().detach().cpu()),
            "alpha_active_max": float(values.max().detach().cpu()),
            "top_positions": [
                {"position": int(i), "alpha": float(alpha[i].detach().cpu())}
                for i in torch.topk(alpha, min(10, alpha.numel())).indices.detach().cpu().tolist()
            ],
        }
    )
    return stats


def build_attention_scale_alpha(
    rollout: torch.Tensor,
    attention_mask: torch.Tensor,
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    source: str,
    power: float,
    alpha_min: float,
    alpha_max: float,
    beta: float,
    last_window_size: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if rollout.dim() != 2 or rollout.shape[0] != rollout.shape[1]:
        raise RuntimeError(f"rollout must be [seq, seq], got {list(rollout.shape)}")
    if attention_mask.dim() != 2 or attention_mask.shape[0] != 1 or rollout.shape[0] != attention_mask.shape[1]:
        raise RuntimeError("rollout and attention_mask sequence lengths do not match")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")
    if power <= 0:
        raise ValueError("power must be positive")
    if alpha_min <= 0 or alpha_max <= 0 or alpha_min > alpha_max or alpha_min > 1.0 or alpha_max < 1.0:
        raise ValueError("alpha bounds must allow mean-one alpha")
    valid = attention_mask[0].to(rollout.device).bool()
    variable = variable_mask.to(rollout.device).bool() & valid
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("variable_mask has no variable positions")
    raw, query_positions = _attention_raw_from_rollout(rollout, valid, variable, source, last_window_size)
    raw_sum = raw.sum().clamp_min(1e-12)
    valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
    last_valid = int(valid_positions[-1].detach().cpu())
    masked = raw * variable.float()
    if float(masked.sum().detach().cpu()) <= 0.0:
        masked = variable.float()
    tempered = ((masked + 1e-12) ** float(power)) * variable.float()
    mean_one = float(positions.numel()) * tempered / tempered.sum().clamp_min(1e-12)
    mixed = torch.where(variable, 1.0 + float(beta) * (mean_one - 1.0), torch.zeros_like(mean_one))
    alpha = bounded_mean_one_weights(mixed, variable, float(alpha_min), float(alpha_max)) * variable.float()
    base_stats = {
        "source": source,
        "weight_source": source,
        "query_positions": query_positions,
        "weight_power": float(power),
        "alpha_min": float(alpha_min),
        "alpha_max": float(alpha_max),
        "beta": float(beta),
        "last_window_size": int(last_window_size),
        "variable_count": int(positions.numel()),
        "raw_sum_before_mask": float(raw_sum.detach().cpu()),
        "raw_sum_after_mask": float(masked.sum().detach().cpu()),
        "raw_bos_mass": float((raw[0] / raw_sum).detach().cpu()) if raw.numel() else 0.0,
        "raw_last_valid_mass": float((raw[last_valid] / raw_sum).detach().cpu()),
        "gate_open_rate": None,
    }
    return alpha.detach(), _attention_scale_stats(alpha, variable, fixed_public, valid, base_stats)


def apply_attention_scale_gate(
    alpha: torch.Tensor,
    variable_mask: torch.Tensor,
    gate_mask: torch.Tensor,
    alpha_min: float,
    alpha_max: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    variable = variable_mask.to(alpha.device).bool()
    gate = gate_mask.to(alpha.device).bool() & variable
    gated = torch.zeros_like(alpha, dtype=torch.float32)
    gated[variable & ~gate] = 1.0
    if int(gate.sum().detach().cpu()) > 0:
        active = bounded_mean_one_weights(alpha.float(), gate, float(alpha_min), float(alpha_max))
        gated[gate] = active[gate]
    elif int(variable.sum().detach().cpu()) > 0:
        gated[variable] = 1.0
    values = gated[variable]
    active_values = gated[gate]
    active_count = int(gate.sum().detach().cpu())
    variable_count = int(variable.sum().detach().cpu())
    return gated.detach(), {
        "gate_open_rate": float(active_count / max(1, variable_count)),
        "alpha_active_count": active_count,
        "alpha_active_mean": float(active_values.mean().detach().cpu()) if active_count else None,
        "alpha_active_std": float(active_values.std(unbiased=False).detach().cpu()) if active_count > 1 else 0.0 if active_count == 1 else None,
        "alpha_active_min": float(active_values.min().detach().cpu()) if active_count else None,
        "alpha_active_max": float(active_values.max().detach().cpu()) if active_count else None,
        "mean_alpha": float(values.mean().detach().cpu()) if variable_count else 0.0,
        "std_alpha": float(values.std(unbiased=False).detach().cpu()) if variable_count > 1 else 0.0,
        "min_alpha": float(values.min().detach().cpu()) if variable_count else 0.0,
        "max_alpha": float(values.max().detach().cpu()) if variable_count else 0.0,
        "mean": float(values.mean().detach().cpu()) if variable_count else 0.0,
        "std": float(values.std(unbiased=False).detach().cpu()) if variable_count > 1 else 0.0,
        "min": float(values.min().detach().cpu()) if variable_count else 0.0,
        "max": float(values.max().detach().cpu()) if variable_count else 0.0,
    }


def calibrated_residual_alpha(
    alpha: torch.Tensor,
    variable_mask: torch.Tensor,
    gate_mask: torch.Tensor,
    rho: float,
    alpha_min: float,
    alpha_max: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    variable = variable_mask.to(alpha.device).bool()
    gate = gate_mask.to(alpha.device).bool() & variable
    variable_count = int(variable.sum().detach().cpu())
    active_count = int(gate.sum().detach().cpu())
    calibrated = torch.zeros_like(alpha, dtype=torch.float32)
    if variable_count == 0:
        return calibrated.detach(), {
            "gate_open_rate": 0.0,
            "alpha_active_count": 0,
            "calibrated_rho": float(rho),
        }
    calibrated[variable] = 1.0
    if active_count > 0:
        shrunken = torch.where(
            gate,
            1.0 + float(rho) * (alpha.float().to(alpha.device) - 1.0),
            torch.ones_like(alpha, dtype=torch.float32),
        )
        active = bounded_mean_one_weights(shrunken, gate, float(alpha_min), float(alpha_max))
        calibrated[gate] = active[gate]
    values = calibrated[variable]
    active_values = calibrated[gate]
    return calibrated.detach(), {
        "gate_open_rate": float(active_count / max(1, variable_count)),
        "alpha_active_count": active_count,
        "calibrated_rho": float(rho),
        "alpha_active_mean": float(active_values.mean().detach().cpu()) if active_count else None,
        "alpha_active_std": float(active_values.std(unbiased=False).detach().cpu()) if active_count > 1 else 0.0 if active_count == 1 else None,
        "alpha_active_min": float(active_values.min().detach().cpu()) if active_count else None,
        "alpha_active_max": float(active_values.max().detach().cpu()) if active_count else None,
        "mean_alpha": float(values.mean().detach().cpu()),
        "std_alpha": float(values.std(unbiased=False).detach().cpu()) if variable_count > 1 else 0.0,
        "min_alpha": float(values.min().detach().cpu()),
        "max_alpha": float(values.max().detach().cpu()),
        "mean": float(values.mean().detach().cpu()),
        "std": float(values.std(unbiased=False).detach().cpu()) if variable_count > 1 else 0.0,
        "min": float(values.min().detach().cpu()),
        "max": float(values.max().detach().cpu()),
    }


def embedding_uncertainty_gate(
    z: torch.Tensor,
    embed_weight: torch.Tensor,
    variable_mask: torch.Tensor,
    uncertainty_fraction: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    variable = variable_mask.to(z.device).bool()
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    gate = torch.zeros(variable.shape, dtype=torch.bool, device=z.device)
    if positions.numel() == 0:
        return gate, {"uncertainty_fraction": float(uncertainty_fraction), "gate_active_count": 0, "gate_open_rate": 0.0}
    distances = torch.cdist(z.detach().float()[0, positions], embed_weight.detach().float().to(z.device))
    if distances.shape[1] < 2:
        margins = torch.zeros(distances.shape[0], dtype=torch.float32, device=z.device)
    else:
        top2 = torch.topk(distances, k=2, largest=False, dim=-1).values
        margins = top2[:, 1] - top2[:, 0]
    fraction = min(1.0, max(0.0, float(uncertainty_fraction)))
    k = 0 if fraction <= 0.0 else max(1, int(math.ceil(fraction * int(positions.numel()))))
    selected = positions[torch.argsort(margins, descending=False)[:k]]
    gate[selected] = True
    return gate, {
        "uncertainty_fraction": fraction,
        "gate_active_count": int(selected.numel()),
        "gate_open_rate": float(selected.numel() / max(1, int(positions.numel()))),
        "margin_mean": float(margins.mean().detach().cpu()),
        "margin_min": float(margins.min().detach().cpu()),
        "margin_max": float(margins.max().detach().cpu()),
    }


def bounded_mean_one_weights(values: torch.Tensor, variable_mask: torch.Tensor, weight_min: float, weight_max: float) -> torch.Tensor:
    variable = variable_mask.to(values.device).bool()
    positions = torch.nonzero(variable, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("variable_mask has no variable positions")
    if weight_min > 1.0 or weight_max < 1.0:
        raise ValueError("weight bounds must allow mean-one weights")
    result = torch.zeros_like(values, dtype=torch.float32)
    active = positions
    remaining_sum = float(positions.numel())
    base = values.float().clamp_min(1e-12)
    for _ in range(64):
        if active.numel() == 0:
            break
        scaled = base[active] * (remaining_sum / base[active].sum().clamp_min(1e-12))
        low = scaled < float(weight_min)
        high = scaled > float(weight_max)
        fixed = low | high
        if not bool(fixed.any().detach().cpu()):
            result[active] = scaled
            remaining_sum = 0.0
            break
        if bool(low.any().detach().cpu()):
            low_positions = active[low]
            result[low_positions] = float(weight_min)
            remaining_sum -= float(weight_min) * int(low_positions.numel())
        if bool(high.any().detach().cpu()):
            high_positions = active[high]
            result[high_positions] = float(weight_max)
            remaining_sum -= float(weight_max) * int(high_positions.numel())
        active = active[~fixed]
        remaining_sum = max(0.0, remaining_sum)
    if active.numel() > 0 and remaining_sum > 0:
        result[active] = remaining_sum / float(active.numel())
    return result * variable.float()


def bounded_mean_one_project(
    weights: torch.Tensor,
    lower: float,
    upper: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if lower > 1.0 or upper < 1.0:
        raise ValueError("weight bounds must allow mean-one weights")
    if weights.numel() == 0:
        return weights.detach().float()

    original_dtype = weights.dtype
    values = weights.detach().to(dtype=torch.float64).clamp_min(float(eps))
    lo = torch.zeros((), dtype=torch.float64, device=values.device)
    hi = torch.ones((), dtype=torch.float64, device=values.device)
    target = float(values.numel())
    lower_f = float(lower)
    upper_f = float(upper)

    for _ in range(128):
        if float(torch.clamp(values * hi, min=lower_f, max=upper_f).sum().item()) >= target:
            break
        hi = hi * 2.0

    for _ in range(128):
        mid = (lo + hi) / 2.0
        projected = torch.clamp(values * mid, min=lower_f, max=upper_f)
        if float(projected.sum().item()) < target:
            lo = mid
        else:
            hi = mid

    projected = torch.clamp(values * hi, min=lower_f, max=upper_f)
    residual = target - float(projected.sum().item())
    if abs(residual) > float(eps):
        adjustable = (projected > lower_f + float(eps)) & (projected < upper_f - float(eps))
        if bool(adjustable.any().item()):
            projected[adjustable] += residual / float(adjustable.sum().item())
            projected = torch.clamp(projected, min=lower_f, max=upper_f)
    return projected.to(dtype=original_dtype if original_dtype.is_floating_point else torch.float32)


def build_confidence_aware_residual_alpha(
    *,
    r_all: torch.Tensor,
    r_last2: torch.Tensor,
    variable_mask: torch.Tensor,
    weight_min: float = 0.5,
    weight_max: float = 1.5,
    strength: float = 0.25,
    power: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    r_all = r_all.detach().float()
    r_last2 = r_last2.detach().float()
    variable_mask = variable_mask.to(r_all.device).bool()
    if r_all.shape != r_last2.shape:
        raise ValueError(f"r_all and r_last2 shape mismatch: {r_all.shape} vs {r_last2.shape}")
    if r_all.shape != variable_mask.shape:
        raise ValueError(f"rollout and variable_mask shape mismatch: {r_all.shape} vs {variable_mask.shape}")

    alpha = torch.zeros_like(r_all, dtype=torch.float32)
    confidence = torch.zeros_like(r_all, dtype=torch.float32)
    if int(variable_mask.sum().item()) == 0:
        return alpha, confidence, {
            "variable_final_weight_mean": 0.0,
            "variable_final_weight_std": 0.0,
            "fixed_public_final_weight_sum": 0.0,
            "final_fixed_weight_sum": 0.0,
            "final_max_weight": 0.0,
            "final_min_weight": 0.0,
            "max": 0.0,
            "min": 0.0,
            "confidence_mean": 0.0,
            "confidence_min": 0.0,
            "confidence_max": 0.0,
            "rollout_disagreement_mean": 0.0,
            "rollout_disagreement_median": 0.0,
        }

    consensus = torch.sqrt((r_all.clamp_min(0.0) + eps) * (r_last2.clamp_min(0.0) + eps))
    variable_consensus = consensus[variable_mask].clamp_min(eps).pow(power)
    normalized = variable_consensus / variable_consensus.sum().clamp_min(eps)
    normalized = normalized * float(variable_consensus.numel())

    disagreement = torch.abs(torch.log(r_all.clamp_min(0.0) + eps) - torch.log(r_last2.clamp_min(0.0) + eps))
    variable_disagreement = disagreement[variable_mask]
    median_disagreement = torch.median(variable_disagreement).clamp_min(eps)
    variable_confidence = torch.exp(-variable_disagreement / median_disagreement)

    raw_w = 1.0 + float(strength) * variable_confidence * (normalized - 1.0)
    raw_w = raw_w.clamp_min(eps)
    w = bounded_mean_one_project(
        raw_w,
        lower=float(weight_min),
        upper=float(weight_max),
        eps=eps,
    )

    alpha[variable_mask] = torch.sqrt(w)
    confidence[variable_mask] = variable_confidence
    fixed_weight_sum = float(alpha[~variable_mask].pow(2).sum().item())
    stats = {
        "variable_final_weight_mean": float(w.mean().item()),
        "variable_final_weight_std": float(w.std(unbiased=False).item()) if w.numel() > 1 else 0.0,
        "fixed_public_final_weight_sum": fixed_weight_sum,
        "final_fixed_weight_sum": fixed_weight_sum,
        "final_max_weight": float(w.max().item()),
        "final_min_weight": float(w.min().item()),
        "max": float(w.max().item()),
        "min": float(w.min().item()),
        "confidence_mean": float(variable_confidence.mean().item()),
        "confidence_min": float(variable_confidence.min().item()),
        "confidence_max": float(variable_confidence.max().item()),
        "rollout_disagreement_mean": float(variable_disagreement.mean().item()),
        "rollout_disagreement_median": float(median_disagreement.item()),
    }
    return alpha, confidence, stats


def build_car_attention_bundle_from_rollouts(
    *,
    r_all: torch.Tensor,
    r_last2: torch.Tensor,
    variable_mask: torch.Tensor,
    weight_min: float,
    weight_max: float,
    strength: float,
    power: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    alpha, confidence, stats = build_confidence_aware_residual_alpha(
        r_all=r_all,
        r_last2=r_last2,
        variable_mask=variable_mask,
        weight_min=weight_min,
        weight_max=weight_max,
        strength=strength,
        power=power,
        eps=eps,
    )
    variable_mask = variable_mask.bool()
    variable_count = int(variable_mask.sum().item())
    stats.update({
        'rollout_all_variable_mass': float(r_all[variable_mask].sum().item()) if variable_count else 0.0,
        'rollout_last2_variable_mass': float(r_last2[variable_mask].sum().item()) if variable_count else 0.0,
        'car_strength': float(strength),
        'car_power': float(power),
        'car_weight_min': float(weight_min),
        'car_weight_max': float(weight_max),
    })
    return alpha, confidence, stats


def build_mts_token_weights(
    rollout: torch.Tensor,
    attention_mask: torch.Tensor,
    variable_mask: torch.Tensor,
    fixed_public: Dict[int, int],
    source: str,
    power: float,
    weight_min: float,
    weight_max: float,
    beta: float,
    last_window_size: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if rollout.dim() != 2:
        raise RuntimeError(f"rollout must be [seq, seq], got {list(rollout.shape)}")
    if attention_mask.dim() != 2 or attention_mask.shape[0] != 1:
        raise RuntimeError(f"attention_mask must be [1, seq], got {list(attention_mask.shape)}")
    if rollout.shape[0] != rollout.shape[1] or rollout.shape[0] != attention_mask.shape[1]:
        raise RuntimeError("rollout and attention_mask sequence lengths do not match")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")
    if power <= 0:
        raise ValueError("power must be positive")
    if weight_min < 0 or weight_max <= 0 or weight_min > weight_max:
        raise ValueError("invalid weight clipping bounds")
    valid = attention_mask[0].to(rollout.device).bool()
    variable = variable_mask.to(rollout.device).bool() & valid
    variable_positions = torch.nonzero(variable, as_tuple=False).flatten()
    if variable_positions.numel() == 0:
        raise RuntimeError("variable_mask has no variable positions")
    valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        raise RuntimeError("attention_mask has no valid positions")

    if source == "last_query_raw":
        query_positions = [int(valid_positions[-1].detach().cpu())]
        raw = rollout[query_positions[0]].clone()
    elif source == "mean_query":
        query_positions = [int(x) for x in variable_positions.detach().cpu().tolist()]
        raw = rollout[variable].mean(dim=0)
    elif source == "last_window_mean":
        window = max(1, int(last_window_size))
        selected = variable_positions[-window:]
        query_positions = [int(x) for x in selected.detach().cpu().tolist()]
        raw = rollout[selected].mean(dim=0)
    elif source == "uniform":
        query_positions = [int(x) for x in variable_positions.detach().cpu().tolist()]
        raw = torch.ones(rollout.shape[0], dtype=torch.float32, device=rollout.device)
    else:
        raise ValueError(f"unknown weight source={source!r}")

    raw = raw.float().clamp_min(0.0)
    raw_sum_before_mask = raw.sum().clamp_min(1e-12)
    raw_bos_mass = float((raw[0] / raw_sum_before_mask).detach().cpu()) if raw.numel() else 0.0
    last_valid_pos = int(valid_positions[-1].detach().cpu())
    raw_last_valid_mass = float((raw[last_valid_pos] / raw_sum_before_mask).detach().cpu())

    masked_raw = raw * variable.float()
    masked_raw_sum = masked_raw.sum()
    if float(masked_raw_sum.detach().cpu()) <= 0.0:
        masked_raw = variable.float()
        masked_raw_sum = masked_raw.sum()

    n_var = float(variable_positions.numel())
    if beta == 0.0:
        final = variable.float()
        clipped = variable.float()
    else:
        tempered = ((masked_raw + 1e-12) ** float(power)) * variable.float()
        attn_weights = n_var * tempered / tempered.sum().clamp_min(1e-12)
        clipped = bounded_mean_one_weights(attn_weights, variable, float(weight_min), float(weight_max))
        final = torch.where(variable, (1.0 - float(beta)) + float(beta) * clipped, torch.zeros_like(clipped))
    final = final * variable.float()
    variable_weights = final[variable]
    fixed_weight_sum = 0.0
    for pos in fixed_public:
        if 0 <= int(pos) < final.numel():
            fixed_weight_sum += float(final[int(pos)].detach().cpu())
    stats = {
        "source": source,
        "query_positions": query_positions,
        "power": float(power),
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
        "beta": float(beta),
        "last_window_size": int(last_window_size),
        "variable_count": int(variable_positions.numel()),
        "raw_sum_before_mask": float(raw_sum_before_mask.detach().cpu()),
        "raw_sum_after_mask": float(masked_raw_sum.detach().cpu()),
        "raw_bos_mass": raw_bos_mass,
        "raw_last_valid_mass": raw_last_valid_mass,
        "last_valid_position": last_valid_pos,
        "final_bos_weight": float(final[0].detach().cpu()) if final.numel() else 0.0,
        "final_last_valid_weight": float(final[last_valid_pos].detach().cpu()),
        "final_fixed_weight_sum": fixed_weight_sum,
        "mean": float(variable_weights.mean().detach().cpu()),
        "std": float(variable_weights.std(unbiased=False).detach().cpu()) if variable_weights.numel() > 1 else 0.0,
        "min": float(variable_weights.min().detach().cpu()),
        "max": float(variable_weights.max().detach().cpu()),
        "top_positions": [
            {"position": int(i), "weight": float(final[i].detach().cpu())}
            for i in torch.topk(final, min(10, final.numel())).indices.detach().cpu().tolist()
        ],
    }
    return final.detach(), stats


def token_weights_from_rollout(
    rollout: torch.Tensor,
    attention_mask: torch.Tensor,
    source: str,
    weight_floor: float,
    fixed_public: Optional[Dict[int, int]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if rollout.dim() != 2:
        raise RuntimeError(f"rollout must be [seq, seq], got {list(rollout.shape)}")
    valid = attention_mask[0].detach().bool()
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise RuntimeError("attention mask has no valid tokens")
    if source == "last_query":
        raw = rollout[int(valid_indices[-1].detach().cpu())].clone()
    elif source == "mean_query":
        raw = rollout[valid].mean(dim=0)
    elif source == "uniform":
        raw = torch.ones(rollout.shape[0], dtype=torch.float32, device=rollout.device)
    else:
        raise ValueError(f"unknown weight_source={source!r}")
    raw = raw.float().clamp_min(0.0)
    raw = raw * valid.float()
    raw_sum = raw.sum().clamp_min(1e-12)
    last_valid_pos = int(valid_indices[-1].detach().cpu())
    weights = raw.shape[0] * raw / raw_sum
    if weight_floor > 0:
        weights = torch.where(valid, torch.clamp(weights, min=float(weight_floor)), torch.zeros_like(weights))
        weights = weights.shape[0] * weights / weights.sum().clamp_min(1e-12)
    weights = weights * valid.float()
    fixed_weight_sum = 0.0
    for pos in (fixed_public or {}):
        if 0 <= int(pos) < weights.numel():
            fixed_weight_sum += float(weights[int(pos)].detach().cpu())
    stats = {
        "source": source,
        "weight_floor": float(weight_floor),
        "valid_token_count": int(valid.sum().detach().cpu()),
        "last_valid_position": last_valid_pos,
        "raw_sum": float(raw_sum.detach().cpu()),
        "raw_bos_mass": float((raw[0] / raw_sum).detach().cpu()) if raw.numel() else 0.0,
        "raw_last_valid_mass": float((raw[last_valid_pos] / raw_sum).detach().cpu()),
        "final_bos_weight": float(weights[0].detach().cpu()) if weights.numel() else 0.0,
        "final_last_valid_weight": float(weights[last_valid_pos].detach().cpu()),
        "final_fixed_weight_sum": fixed_weight_sum,
        "mean": float(weights[valid].mean().detach().cpu()),
        "std": float(weights[valid].std(unbiased=False).detach().cpu()) if int(valid.sum()) > 1 else 0.0,
        "min": float(weights[valid].min().detach().cpu()),
        "max": float(weights[valid].max().detach().cpu()),
        "top_positions": [
            {"position": int(i), "weight": float(weights[i].detach().cpu())}
            for i in torch.topk(weights, min(10, weights.numel())).indices.detach().cpu().tolist()
        ],
    }
    return weights.detach(), stats


def weighted_activation_loss(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    per_token = torch.mean((pred.float() - target.to(pred.device).float()) ** 2, dim=-1)
    if weights is None:
        return per_token.mean()
    w = weights.to(pred.device).float().unsqueeze(0)
    return (per_token * w).sum() / w.sum().clamp_min(1e-12)


def tensor_attention_stats(attn: torch.Tensor, layer_index: int) -> Dict[str, Any]:
    row_sums = attn.sum(dim=-1)
    upper = torch.triu(attn, diagonal=1).abs().max().item() if attn.shape[0] > 1 else 0.0
    return {
        "layer_index": int(layer_index),
        "shape": [int(attn.shape[0]), int(attn.shape[1])],
        "row_sum_min": float(row_sums.min().detach().cpu()),
        "row_sum_max": float(row_sums.max().detach().cpu()),
        "row_sum_mean": float(row_sums.mean().detach().cpu()),
        "upper_triangular_max": float(upper),
        "nonnegative_min": float(attn.min().detach().cpu()),
        "entropy_mean": float((-(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)).mean().detach().cpu()),
        "top_left_8x8": attn[:8, :8].detach().cpu().tolist(),
    }


def server_forward_with_attention(
    model: torch.nn.Module,
    start_layer: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool,
    rollout_depth: str = "all",
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, torch.Tensor]]]:
    batch, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch, seq_len),
        hidden_states,
        past_key_values_length=0,
    )
    layers = list(range(start_layer, len(model.model.layers)))
    if rollout_depth == "last2":
        attention_layers = set(layers[-2:])
    elif rollout_depth == "all":
        attention_layers = set(layers)
    else:
        raise ValueError(f"unknown server_rollout_depth={rollout_depth!r}")
    collected: List[Tuple[int, torch.Tensor]] = []
    x = hidden_states
    for layer_index in layers:
        want = output_attentions and layer_index in attention_layers
        layer_outputs = model.model.layers[layer_index](
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=want,
            use_cache=False,
        )
        x = layer_outputs[0]
        if want:
            attn = layer_outputs[1]
            if attn is None:
                raise RuntimeError(f"server layer {layer_index} did not return attention")
            if attn.dim() != 4:
                raise RuntimeError(f"expected attention [batch, heads, seq, seq], got {list(attn.shape)}")
            collected.append((layer_index, attn.detach().float().mean(dim=1)[0]))
    final_hidden = model.model.norm(x)
    logits = model.lm_head(final_hidden)
    return final_hidden, logits, collected


def server_attention_bundle(
    model: torch.nn.Module,
    observed_activation: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: SAWConfig,
    fixed_public: Dict[int, int],
    variable_audit: VariableMaskAudit,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if cfg.weight_source == "uniform" and cfg.method not in CAR_METHODS:
        seq = observed_activation.shape[1]
        rollout = torch.eye(seq, dtype=torch.float32, device=observed_activation.device)
        if cfg.method in ATTN_SCALE_METHODS:
            weights, weight_stats = build_attention_scale_alpha(
                rollout,
                attention_mask,
                variable_audit.variable_mask,
                fixed_public,
                "uniform",
                cfg.weight_power,
                cfg.alpha_min,
                cfg.alpha_max,
                cfg.beta,
                cfg.last_window_size,
            )
        else:
            weights, weight_stats = build_mts_token_weights(
                rollout,
                attention_mask,
                variable_audit.variable_mask,
                fixed_public,
                "uniform",
                cfg.weight_power,
                cfg.weight_min,
                cfg.weight_max,
                cfg.beta,
                cfg.last_window_size,
            )
        return weights, {"used": False, "reason": "uniform weight mode"}, {"rollout_shape": [seq, seq]}, weight_stats
    start = cfg.target_layer + 1
    if start >= len(model.model.layers):
        raise RuntimeError(f"target_layer={cfg.target_layer} leaves no server-side layers")
    collection_depth = "all" if cfg.method in CAR_METHODS else cfg.server_rollout_depth
    with torch.no_grad():
        _hidden, _logits, collected = server_forward_with_attention(
            model,
            start,
            observed_activation.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=True,
            rollout_depth=collection_depth,
        )
    attentions = [attn for _idx, attn in collected]
    rollout = rollout_from_attentions(attentions, residual=cfg.residual_rollout)
    if cfg.method in CAR_METHODS:
        last2_attentions = attentions[-2:] if len(attentions) >= 2 else attentions
        rollout_last2 = rollout_from_attentions(last2_attentions, residual=cfg.residual_rollout)
        valid = attention_mask[0].to(rollout.device).bool()
        variable = variable_audit.variable_mask.to(rollout.device).bool() & valid
        r_all, car_query_positions = _attention_raw_from_rollout(
            rollout,
            valid,
            variable,
            "last_window_mean",
            cfg.last_window_size,
        )
        r_last2, car_last2_query_positions = _attention_raw_from_rollout(
            rollout_last2,
            valid,
            variable,
            "last_window_mean",
            cfg.last_window_size,
        )
        weights, _confidence, weight_stats = build_car_attention_bundle_from_rollouts(
            r_all=r_all,
            r_last2=r_last2,
            variable_mask=variable_audit.variable_mask,
            weight_min=float(getattr(cfg, "car_weight_min", 0.5)),
            weight_max=float(getattr(cfg, "car_weight_max", 1.5)),
            strength=float(getattr(cfg, "car_strength", 0.25)),
            power=float(getattr(cfg, "car_power", 0.5)),
        )
        weight_stats["method_role"] = "confidence_aware_rollout_residual"
        weight_stats["rollout_depth"] = "all+last2"
        weight_stats["attention_formula"] = "confidence_aware_rollout_residual"
        weight_stats["car_query_positions"] = car_query_positions
        weight_stats["car_last2_query_positions"] = car_last2_query_positions
    elif cfg.method == "server_attn_last_raw":
        weights, weight_stats = token_weights_from_rollout(rollout, attention_mask, "last_query", cfg.weight_floor, fixed_public)
        weight_stats["method_role"] = "negative_control_raw_last_query"
    elif cfg.method in ATTN_SCALE_METHODS:
        weights, weight_stats = build_attention_scale_alpha(
            rollout,
            attention_mask,
            variable_audit.variable_mask,
            fixed_public,
            cfg.weight_source,
            cfg.weight_power,
            cfg.alpha_min,
            cfg.alpha_max,
            cfg.beta,
            cfg.last_window_size,
        )
        if cfg.method in ATTENTION_WEIGHTED_RESIDUAL_METHODS:
            weight_stats["method_role"] = "attention_weighted_residual"
        elif cfg.method in LINEAR_WEIGHTED_RESIDUAL_METHODS:
            weight_stats["method_role"] = "attention_linear_weighted_residual"
        else:
            weight_stats["method_role"] = "attention_scaled_activation"
    else:
        weights, weight_stats = build_mts_token_weights(
            rollout,
            attention_mask,
            variable_audit.variable_mask,
            fixed_public,
            cfg.weight_source,
            cfg.weight_power,
            cfg.weight_min,
            cfg.weight_max,
            cfg.beta,
            cfg.last_window_size,
        )
        weight_stats["method_role"] = "masked_tempered_server_attention"
    attention_stats = {
        "used": True,
        "target_layer_output_state": f"H^({cfg.target_layer}) is output of 0-based block {cfg.target_layer}",
        "server_start_layer": start,
        "server_layers": [int(idx) for idx, _attn in collected],
        "rollout_depth": collection_depth,
        "residual_rollout": bool(cfg.residual_rollout),
        "layer_stats": [tensor_attention_stats(attn, idx) for idx, attn in collected],
        "dummy_attention_used": False,
    }
    rollout_stats = {
        "rollout_shape": [int(rollout.shape[0]), int(rollout.shape[1])],
        "row_sum_min": float(rollout.sum(dim=-1).min().detach().cpu()),
        "row_sum_max": float(rollout.sum(dim=-1).max().detach().cpu()),
        "row_sum_mean": float(rollout.sum(dim=-1).mean().detach().cpu()),
        "upper_triangular_max": float(torch.triu(rollout, diagonal=1).abs().max().detach().cpu()) if rollout.shape[0] > 1 else 0.0,
        "nonnegative_min": float(rollout.min().detach().cpu()),
        "top_left_16x16": rollout[:16, :16].detach().cpu().tolist(),
    }
    weight_stats.update(
        {
            "method": cfg.method,
            "residual_rollout": bool(cfg.residual_rollout),
            "rollout_depth": "all+last2" if cfg.method in CAR_METHODS else cfg.server_rollout_depth,
            "attention_start_ratio": float(cfg.attention_start_ratio),
            "attention_full_ratio": float(cfg.attention_full_ratio),
            "adaptive_min_beta": float(cfg.adaptive_min_beta),
            "token_id_based_special_mask_available": bool(variable_audit.token_id_based_special_mask_available),
            "known_public_special_positions": list(variable_audit.known_public_special_positions),
        }
    )
    if abs(rollout_stats["row_sum_min"] - 1.0) > 2e-3 or abs(rollout_stats["row_sum_max"] - 1.0) > 2e-3:
        raise RuntimeError(f"rollout rows are not normalized: {rollout_stats}")
    if rollout_stats["nonnegative_min"] < -1e-6:
        raise RuntimeError(f"rollout contains negative values: {rollout_stats}")
    if rollout_stats["upper_triangular_max"] > 2e-3:
        raise RuntimeError(f"rollout violates causal mask: {rollout_stats}")
    return weights, attention_stats, rollout_stats, weight_stats


def stage_b_optimize(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: SAWConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
    fixed_public: Dict[int, int],
    variable_audit: VariableMaskAudit,
    init_embeds: torch.Tensor,
    server_weights: Optional[torch.Tensor],
    a_context: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, float]]]:
    embed_layer = model.get_input_embeddings()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _ = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = init_embeds.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.AdamW([z], lr=cfg.lr)
    target = observed_activation.detach()
    target_ctx = context_projection(a_context, target) if a_context is not None and cfg.lambda_context > 0 else None
    history: List[Dict[str, float]] = []
    final: Dict[str, Any] = {}
    for step in range(max(1, cfg.epoch)):
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden = capture_prefix_activation(
            model,
            cfg.target_layer,
            inputs_embeds=z.to(dtype=embed_layer.weight.dtype),
            attention_mask=attention_mask,
        )
        all_token_uniform = uniform_all_token_activation_loss(hidden, target, attention_mask)
        variable_uniform = weighted_variable_activation_loss(hidden, target, variable_audit.variable_mask, None)
        weighted_variable = weighted_variable_activation_loss(hidden, target, variable_audit.variable_mask, server_weights)
        scale_loss_active = False
        attention_scaled = variable_uniform
        attention_active_loss_mode = "attention_scaled_activation"
        attention_effective_beta: Optional[float] = None
        if cfg.method in ATTN_SCALE_METHODS:
            if server_weights is None:
                raise RuntimeError(f"attention-scale method {cfg.method!r} requires alpha weights")
            alpha_for_step = server_weights
            if cfg.method in CAR_METHODS:
                alpha_for_step = schedule_residual_alpha(
                    alpha=server_weights,
                    variable_mask=variable_audit.variable_mask,
                    epoch_index=step + 1,
                    warmup_epochs=int(getattr(cfg, "car_warmup_epochs", 50)),
                    ramp_end_epoch=int(getattr(cfg, "car_ramp_end_epoch", 70)),
                    weight_min=float(getattr(cfg, "car_weight_min", 0.5)),
                    weight_max=float(getattr(cfg, "car_weight_max", 1.5)),
                ).to(server_weights.device)
                attention_scaled = attention_weighted_residual_loss(hidden, target, variable_audit.variable_mask, alpha_for_step)
                attention_active_loss_mode = "confidence_aware_rollout_residual"
                attention_effective_beta = None
                scale_loss_active = True
            else:
                if cfg.method in {
                    "attn_scale_mean_query_schedule",
                    "attn_scale_mean_query_schedule_refine",
                    "attn_scale_mean_query_schedule_residual",
                    "attn_weighted_residual_schedule",
                    "attn_linear_weighted_residual_schedule",
                    "attn_last_window_linear_residual_schedule",
                    "attn_rho_weighted_residual_schedule",
                    "attn_calibrated_weighted_residual_schedule",
                    "attn_gate_weighted_residual_schedule",
                }:
                    attention_effective_beta = attention_scale_linear_schedule_beta(
                        step + 1,
                        cfg.epoch,
                        cfg.attention_start_ratio,
                        cfg.attention_full_ratio,
                        cfg.beta,
                    )
                    ramp = 0.0 if cfg.beta <= 1e-12 else attention_effective_beta / cfg.beta
                    alpha_for_step = interpolate_attention_alpha(
                        server_weights,
                        variable_audit.variable_mask,
                        ramp,
                        cfg.alpha_min,
                        cfg.alpha_max,
                    ).to(server_weights.device)
                    if cfg.method in {
                        "attn_scale_mean_query_schedule_residual",
                        "attn_linear_weighted_residual_schedule",
                        "attn_rho_weighted_residual_schedule",
                        "attn_last_window_linear_residual_schedule",
                    }:
                        alpha_for_step = residualize_attention_alpha(
                            alpha_for_step,
                            variable_audit.variable_mask,
                            cfg.residual_alpha_rho,
                            cfg.alpha_min,
                            cfg.alpha_max,
                        ).to(server_weights.device)
                    scale_loss_active = attention_effective_beta > 0.0
                else:
                    attention_effective_beta = cfg.beta if attention_scale_schedule_active(step, cfg.epoch, cfg.attention_start_ratio) else 0.0
                    scale_loss_active = attention_effective_beta > 0.0
                if cfg.method in LINEAR_WEIGHTED_RESIDUAL_METHODS:
                    attention_scaled = attention_linear_weighted_residual_loss(hidden, target, variable_audit.variable_mask, alpha_for_step)
                    attention_active_loss_mode = "attention_linear_weighted_residual"
                elif cfg.method in ATTENTION_WEIGHTED_RESIDUAL_METHODS:
                    attention_scaled = attention_weighted_residual_loss(hidden, target, variable_audit.variable_mask, alpha_for_step)
                    attention_active_loss_mode = "attention_weighted_residual"
                else:
                    attention_scaled = attention_scaled_activation_loss(hidden, target, variable_audit.variable_mask, alpha_for_step)
                    attention_active_loss_mode = "attention_scaled_activation"
        if cfg.method == "original_pia_baseline":
            act_loss = all_token_uniform
            loss_mode = "all_valid_uniform"
        elif cfg.method == "variable_only_uniform":
            act_loss = variable_uniform
            loss_mode = "variable_uniform"
        elif cfg.method == "server_attn_last_raw":
            act_loss = weighted_activation_loss(hidden, target, server_weights)
            loss_mode = "raw_server_attention_weight"
        elif cfg.method in MTS_METHODS:
            act_loss = weighted_variable
            loss_mode = "masked_tempered_weighted_error"
        elif cfg.method in ATTN_SCALE_METHODS:
            act_loss = attention_scaled if scale_loss_active else variable_uniform
            loss_mode = attention_active_loss_mode if scale_loss_active else "variable_uniform_warmup"
        else:
            raise ValueError(f"unknown Stage B method={cfg.method!r}")
        vocab_loss = nearest_embedding_loss(variable_view(z, fixed_positions), embed_layer.weight)
        if target_ctx is not None:
            ctx_loss = F.mse_loss(context_projection(a_context, hidden), target_ctx.to(hidden.device))
        else:
            ctx_loss = torch.tensor(0.0, device=device)
        total = act_loss + cfg.lambda_vocab * vocab_loss + cfg.lambda_context * ctx_loss
        cosine = F.cosine_similarity(hidden.float(), target.to(hidden.device).float(), dim=-1).mean()
        if torch.isnan(total) or torch.isnan(cosine):
            raise RuntimeError(f"NaN in Stage B at step {step + 1}")
        optimizer.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip)
        optimizer.step()
        final = {
            "activation_loss": float(act_loss.detach().cpu()),
            "unweighted_activation_loss": float(all_token_uniform.detach().cpu()),
            "all_token_uniform_loss": float(all_token_uniform.detach().cpu()),
            "variable_uniform_loss": float(variable_uniform.detach().cpu()),
            "weighted_variable_loss": float(weighted_variable.detach().cpu()),
            "attention_scaled_loss": float(attention_scaled.detach().cpu()),
            "scale_loss_active": bool(scale_loss_active),
            "attention_effective_beta": None if attention_effective_beta is None else float(attention_effective_beta),
            "loss_mode": loss_mode,
            "vocab_loss": float(vocab_loss.detach().cpu()),
            "context_loss": float(ctx_loss.detach().cpu()),
            "optimization_loss": float(total.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
        }
        log_step = (step + 1) % max(1, min(50, cfg.epoch)) == 0 or step == cfg.epoch - 1
        record_step = cfg.method in ATTN_SCALE_METHODS or log_step
        if record_step:
            history.append({"step": step + 1, **final})
        if log_step:
            print(
                f"method={cfg.method} step={step + 1} act={final['activation_loss']:.6f} "
                f"base_act={final['unweighted_activation_loss']:.6f} vocab={final['vocab_loss']:.6f} "
                f"ctx={final['context_loss']:.6f} total={final['optimization_loss']:.6f} "
                f"cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    return z.detach().float(), final, history


def projection_refine_embeddings(
    model: torch.nn.Module,
    cfg: SAWConfig,
    observed_activation: torch.Tensor,
    variable_audit: VariableMaskAudit,
    fixed_public: Dict[int, int],
    z_init: torch.Tensor,
    server_weights: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, float]]]:
    if cfg.refine_epoch <= 0 or cfg.lambda_projection <= 0:
        return z_init.detach(), {"projection_refinement_used": False, "reason": "disabled"}, []
    if server_weights is None:
        raise RuntimeError(f"projection refinement for {cfg.method!r} requires alpha weights")
    embed_layer = model.get_input_embeddings()
    seq_len = int(z_init.shape[1])
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_embeds, fixed_positions, _ = fixed_embedding_tensor(embed_layer, fixed_public, seq_len, device)
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    z = z_init.detach().clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    lr = max(1e-6, float(cfg.lr) * float(cfg.refine_lr_scale))
    optimizer = torch.optim.AdamW([z], lr=lr)
    target = observed_activation.detach()
    history: List[Dict[str, float]] = []
    final: Dict[str, Any] = {}
    for step in range(max(1, int(cfg.refine_epoch))):
        enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
        hidden = capture_prefix_activation(
            model,
            cfg.target_layer,
            inputs_embeds=z.to(dtype=embed_layer.weight.dtype),
            attention_mask=attention_mask,
        )
        scale_loss = attention_scaled_activation_loss(
            hidden,
            target,
            variable_audit.variable_mask,
            server_weights,
        )
        proj_loss, proj_stats = projection_refinement_loss(z, embed_layer.weight, variable_audit.variable_mask)
        total = scale_loss + float(cfg.lambda_projection) * proj_loss
        cosine = F.cosine_similarity(hidden.float(), target.to(hidden.device).float(), dim=-1).mean()
        if torch.isnan(total) or torch.isnan(cosine):
            raise RuntimeError(f"NaN in projection refinement at step {step + 1}")
        optimizer.zero_grad()
        total.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip)
        optimizer.step()
        final = {
            "phase": "projection_refinement",
            "step": step + 1,
            "projection_refinement_loss": float(total.detach().cpu()),
            "projection_scale_loss": float(scale_loss.detach().cpu()),
            "projection_loss": float(proj_loss.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
        }
        if step == 0 or (step + 1) == int(cfg.refine_epoch):
            history.append(dict(final))
    enforce_embedding_constraints(z, fixed_positions, fixed_embeds, left, right)
    stats = {
        "projection_refinement_used": True,
        "projection_refine_epoch": int(cfg.refine_epoch),
        "lambda_projection": float(cfg.lambda_projection),
        "refine_lr_scale": float(cfg.refine_lr_scale),
        "refine_lr": float(lr),
        **final,
    }
    if "proj_stats" in locals():
        stats.update(proj_stats)
    return z.detach().float(), stats, history


def invert_observed(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: SAWConfig,
    observed_activation: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[int], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, float]], List[Dict[str, float]], Optional[float]]:
    method = canonical_method(cfg.method)
    cfg.method = method
    embed_layer = model.get_input_embeddings()
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    variable_audit = build_variable_mask(
        attention_mask=attention_mask,
        fixed_public=fixed_public,
        special_token_ids=getattr(tokenizer, "all_special_ids", None),
        token_ids=None,
    )
    variable_mask_payload = variable_mask_audit_payload(variable_audit)
    stage_a_history: List[Dict[str, float]] = []
    attention_stats: Dict[str, Any] = {"method": method, "server_attention_used": False, "dummy_attention_used": False}
    rollout_stats: Dict[str, Any] = {}
    weight_stats: Dict[str, Any] = {}
    init_audit: Dict[str, Any] = {"method": method, "uses_ground_truth_for_initialization": False}
    dummy_embeds: Optional[torch.Tensor] = None
    a_dummy: Optional[torch.Tensor] = None
    if method in DUMMY_METHODS:
        dummy_embeds, a_dummy, dummy_stats, stage_a_history = stage_a_dummy_proxy(
            model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public
        )
        init_audit["dummy_stage_a_available"] = True
        init_audit["dummy_attention_proxy_used"] = method in {"alpha_nn_init_only", "attention_context_existing", "alpha_init_plus_server_attn"}
        attention_stats["dummy_proxy_stats"] = dummy_stats
    if method in BASELINE_METHODS or method in SERVER_ATTN_METHODS:
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
        init_audit["init"] = "random_public_embeddings"
    elif method == "dummy_init_existing":
        init_embeds = dummy_embeds
        init_audit["init"] = "stage_a_dummy_embeddings"
    elif method in ALPHA_METHODS:
        if a_dummy is None:
            raise RuntimeError("alpha initialization requires dummy attention proxy")
        init_embeds, _init_ids, _h_alpha, init_audit = alpha_nn_initialization(
            model, tokenizer, cfg, observed_activation, a_dummy
        )
        init_audit["init"] = "alpha_nn_from_dummy_attention_proxy"
    elif method == "attention_context_existing":
        init_embeds = random_public_embeddings(tokenizer, embed_layer, seq_len, device, fixed_public)
        init_audit["init"] = "random_public_embeddings_with_dummy_context_loss"
    else:
        raise ValueError(f"unknown method={method!r}")
    if init_embeds is None:
        raise RuntimeError(f"no initialization for method={method}")
    server_weights: Optional[torch.Tensor] = None
    if method in SERVER_ATTN_METHODS:
        source = attention_weight_source_for_method(method)
        cfg.weight_source = source
        server_weights, server_stats, rollout_stats, weight_stats = server_attention_bundle(
            model, observed_activation, attention_mask, cfg, fixed_public, variable_audit
        )
        if method in {
            "attn_scale_last_window_gate",
            "attn_gate_weighted_residual_schedule",
            "attn_calibrated_weighted_residual_schedule",
        }:
            gate_mask, gate_stats = embedding_uncertainty_gate(
                init_embeds,
                embed_layer.weight,
                variable_audit.variable_mask,
                cfg.uncertainty_fraction,
            )
            if method == "attn_calibrated_weighted_residual_schedule":
                server_weights, gated_stats = calibrated_residual_alpha(
                    server_weights,
                    variable_audit.variable_mask,
                    gate_mask,
                    cfg.residual_alpha_rho,
                    cfg.alpha_min,
                    cfg.alpha_max,
                )
            else:
                server_weights, gated_stats = apply_attention_scale_gate(
                    server_weights,
                    variable_audit.variable_mask,
                    gate_mask,
                    cfg.alpha_min,
                    cfg.alpha_max,
                )
            valid = variable_audit.valid_mask.to(server_weights.device).bool()
            variable = variable_audit.variable_mask.to(server_weights.device).bool() & valid
            weight_stats.update(_attention_scale_stats(server_weights, variable, fixed_public, valid, {}))
            weight_stats.update(gate_stats)
            weight_stats.update(gated_stats)
        elif method == "attn_scale_mean_query_adaptive_beta":
            uncertainty_scores, uncertainty_stats = embedding_uncertainty_scores(
                init_embeds,
                embed_layer.weight,
                variable_audit.variable_mask,
            )
            server_weights, adaptive_stats = apply_adaptive_attention_beta(
                server_weights,
                variable_audit.variable_mask,
                uncertainty_scores,
                cfg.beta,
                cfg.adaptive_min_beta,
                cfg.alpha_min,
                cfg.alpha_max,
            )
            valid = variable_audit.valid_mask.to(server_weights.device).bool()
            variable = variable_audit.variable_mask.to(server_weights.device).bool() & valid
            weight_stats.update(_attention_scale_stats(server_weights, variable, fixed_public, valid, {}))
            weight_stats.update(uncertainty_stats)
            weight_stats.update(adaptive_stats)
        attention_stats.update(server_stats)
    elif method == "variable_only_uniform":
        last_valid = variable_audit.last_valid_position
        final_last_valid = (
            float(variable_audit.uniform_weights[last_valid].detach().cpu())
            if last_valid is not None
            else 0.0
        )
        weight_stats = {
            "source": "variable_uniform",
            "variable_count": variable_audit.variable_count,
            "last_valid_position": last_valid,
            "mean": 1.0 if variable_audit.variable_count > 0 else 0.0,
            "std": 0.0,
            "min": 1.0 if variable_audit.variable_count > 0 else 0.0,
            "max": 1.0 if variable_audit.variable_count > 0 else 0.0,
            "final_fixed_weight_sum": 0.0,
            "final_bos_weight": 0.0,
            "final_last_valid_weight": final_last_valid,
        }
    else:
        valid_count = int(variable_audit.valid_mask.sum().detach().cpu())
        fixed_valid_count = int((variable_audit.fixed_public_mask & variable_audit.valid_mask).sum().detach().cpu())
        last_valid = variable_audit.last_valid_position
        weight_stats = {
            "source": "all_valid_uniform",
            "loss_mode": "B0",
            "valid_token_count": valid_count,
            "last_valid_position": last_valid,
            "mean": 1.0 if valid_count > 0 else 0.0,
            "std": 0.0,
            "min": 1.0 if valid_count > 0 else 0.0,
            "max": 1.0 if valid_count > 0 else 0.0,
            "final_fixed_weight_sum": float(fixed_valid_count),
            "final_bos_weight": 1.0 if valid_count > 0 else 0.0,
            "final_last_valid_weight": 1.0 if last_valid is not None else 0.0,
        }
    weight_stats["variable_mask_audit"] = variable_mask_payload
    use_dummy_context = method == "attention_context_existing"
    z, losses, stage_b_history = stage_b_optimize(
        model,
        tokenizer,
        cfg,
        observed_activation,
        seq_len,
        device,
        fixed_public,
        variable_audit,
        init_embeds,
        server_weights,
        a_context=a_dummy if use_dummy_context else None,
    )
    if method in PROJECTION_REFINE_METHODS:
        z, projection_stats, projection_history = projection_refine_embeddings(
            model,
            cfg,
            observed_activation,
            variable_audit,
            fixed_public,
            z,
            server_weights,
            device,
        )
        losses.update(
            {
                "projection_refinement_used": projection_stats.get("projection_refinement_used"),
                "projection_refinement_loss": projection_stats.get("projection_refinement_loss"),
                "projection_scale_loss": projection_stats.get("projection_scale_loss"),
                "projection_loss": projection_stats.get("projection_loss"),
            }
        )
        weight_stats["projection_refinement"] = projection_stats
        stage_b_history.extend(projection_history)
    embed_sets = embedding_candidates(z, embed_layer.weight, max(1, cfg.top_k_embedding))
    recovered_ids = naive_discretization(z, embed_layer.weight)
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
    return recovered_ids, losses, attention_stats, rollout_stats, weight_stats, init_audit, stage_a_history, stage_b_history, calibration_score


def validate_attack_api() -> Dict[str, Any]:
    sig = inspect.signature(invert_observed)
    names = list(sig.parameters)
    banned = ["reference", "original", "input_ids", "token_ids", "prompt", "text", "ground"]
    signature_hits = [name for name in names if any(item in name for item in banned)]
    checked = [
        invert_observed,
        server_attention_bundle,
        server_forward_with_attention,
        stage_b_optimize,
        build_attention_scale_alpha,
        attention_scaled_activation_loss,
        attention_weighted_residual_loss,
        attention_linear_weighted_residual_loss,
        attention_scale_linear_schedule_beta,
        embedding_uncertainty_gate,
        embedding_uncertainty_scores,
        apply_adaptive_attention_beta,
        residualize_attention_alpha,
        calibrated_residual_alpha,
        projection_refinement_loss,
        projection_refine_embeddings,
    ]
    banned_globals = {"original_ids", "original_tokens", "reference_embedding", "reference_tokens", "ground_truth_ids", "dummy_embeds_for_server_attention"}
    ast_hits: List[Dict[str, str]] = []
    dummy_hits: List[str] = []
    for fn in checked:
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        if fn in {server_attention_bundle, server_forward_with_attention} and ("stage_a_dummy" in source or "a_dummy" in source):
            dummy_hits.append(fn.__name__)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_globals:
                ast_hits.append({"function": fn.__name__, "name": node.id})
            if isinstance(node, ast.Attribute) and node.attr in banned_globals:
                ast_hits.append({"function": fn.__name__, "name": node.attr})
    return {
        "signature": str(sig),
        "parameter_names": names,
        "banned_signature_hits": signature_hits,
        "banned_global_reference_hits": ast_hits,
        "server_attention_dummy_proxy_hits": dummy_hits,
        "passes": len(signature_hits) == 0 and len(ast_hits) == 0 and len(dummy_hits) == 0,
    }


def load_model_and_data(cfg: SAWConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    if len(model.model.layers) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected {TOTAL_BLOCKS} TinyLlama blocks, got {len(model.model.layers)}")
    return tokenizer, model, device


def ensure_jsonl_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def run_one_config(cfg: SAWConfig, resume: bool) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "COMPLETE"
    if resume and done.exists():
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    tokenizer, model, device = load_model_and_data(cfg)
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    predictions_path = out / "predictions.jsonl"
    failures_path = out / "failures.jsonl"
    if not resume:
        for path in [predictions_path, failures_path]:
            if path.exists():
                path.unlink()
    ensure_jsonl_file(predictions_path)
    ensure_jsonl_file(failures_path)
    completed = set()
    rows: List[Dict[str, Any]] = []
    if resume and predictions_path.exists():
        by_id = {}
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add(int(row["prompt_id"]))
                by_id[int(row["prompt_id"])] = row
        rows = [by_id[k] for k in sorted(by_id)]
    failures: List[Dict[str, Any]] = []
    if resume and failures_path.exists():
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                failures.append(json.loads(line))
    all_attention_stats = []
    all_rollout_stats = []
    all_weight_stats = []
    all_alpha_stats = []
    all_variable_mask_audits = []
    loss_breakdown = []
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
                observed = capture_prefix_activation(
                    model,
                    cfg.target_layer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).detach()
            preflight = {
                "model_name": MODEL_NAME,
                "block_count": len(model.model.layers),
                "target_layer_index": cfg.target_layer,
                "h_obs_semantics": f"output of 0-based TinyLlama block {cfg.target_layer}",
                "server_start_layer": cfg.target_layer + 1,
                "server_layer_indices": list(range(cfg.target_layer + 1, len(model.model.layers))),
                "inverted_block_count": cfg.inverted_block_count,
                "observed_activation_shape": list(observed.shape),
                "sequence_length": int(input_ids.shape[1]),
                "top_k_embedding": cfg.top_k_embedding,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
                "attack_receives_ground_truth": False,
            }
            print(f"preflight={json.dumps(preflight, ensure_ascii=True)}", flush=True)
            recovered_ids, losses, attn_stats, rollout_stats, weight_stats, init_audit, stage_a_history, stage_b_history, calibration_score = invert_observed(
                model, tokenizer, cfg, observed, int(input_ids.shape[1]), device
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
                "unweighted_activation_loss": losses["unweighted_activation_loss"],
                "all_token_uniform_loss": losses.get("all_token_uniform_loss"),
                "variable_uniform_loss": losses.get("variable_uniform_loss"),
                "weighted_variable_loss": losses.get("weighted_variable_loss"),
                "attention_scaled_loss": losses.get("attention_scaled_loss"),
                "scale_loss_active": losses.get("scale_loss_active"),
                "projection_refinement_used": losses.get("projection_refinement_used"),
                "projection_refinement_loss": losses.get("projection_refinement_loss"),
                "projection_scale_loss": losses.get("projection_scale_loss"),
                "projection_loss": losses.get("projection_loss"),
                "loss_mode": losses.get("loss_mode"),
                "vocab_loss": losses["vocab_loss"],
                "context_loss": losses["context_loss"],
                "cosine_similarity": losses["cosine_similarity"],
                "calibration_score": calibration_score,
                "elapsed_time": elapsed,
                "runtime_seconds": elapsed,
                "seed": cfg.seed,
                "target_layer": cfg.target_layer,
                "participant_number": cfg.participant_number,
                "attacker_position": cfg.attacker_position,
                "inverted_block_count": cfg.inverted_block_count,
                "prompt_token_count": len(original_ids),
                "peak_gpu_memory_mb": peak_memory_mb(),
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
                "top_k_embedding": cfg.top_k_embedding,
                "top_y_semantic": cfg.top_y_semantic,
                "preflight": preflight,
                "init_audit": init_audit,
                "stage_a_history": stage_a_history,
                "stage_b_history": stage_b_history,
                "recovery_uses_ground_truth_tokens": False,
            }
            jsonl_append(predictions_path, row)
            rows.append(row)
            all_attention_stats.append({"prompt_id": prompt_id, **attn_stats})
            all_rollout_stats.append({"prompt_id": prompt_id, **rollout_stats})
            all_weight_stats.append({"prompt_id": prompt_id, **weight_stats})
            if cfg.method in ATTN_SCALE_METHODS:
                all_alpha_stats.append({"prompt_id": prompt_id, **weight_stats})
            all_variable_mask_audits.append({"prompt_id": prompt_id, **weight_stats.get("variable_mask_audit", {})})
            loss_breakdown.append({"prompt_id": prompt_id, "stage_b_history": stage_b_history, "final": losses})
            print(
                f"method={cfg.method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} elapsed={elapsed:.2f}s",
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
            }
            jsonl_append(failures_path, failure)
            failures.append(failure)
            print(f"failure={json.dumps(failure, ensure_ascii=True)}", flush=True)
    metrics = summarize_rows(rows)
    metrics.update(
        {
            "title": EXPERIMENT_TITLE,
            "method": cfg.method,
            "seed": cfg.seed,
            "target_layer": cfg.target_layer,
            "completed_sample_count": len(rows),
            "failed_sample_count": len(failures),
            "dataset_meta": dataset_meta,
            "top_k_embedding": cfg.top_k_embedding,
            "top_y_semantic": cfg.top_y_semantic,
            "output_dir": cfg.output_dir,
        }
    )
    json_dump(out / "metrics.json", metrics)
    json_dump(out / "server_attention_stats.json", {"samples": all_attention_stats})
    json_dump(out / "attention_rollout.json", {"samples": all_rollout_stats})
    json_dump(out / "token_weight_stats.json", {"samples": all_weight_stats})
    if all_alpha_stats:
        json_dump(out / "alpha_scale_stats.json", {"samples": all_alpha_stats})
    json_dump(out / "variable_mask_audit.json", {"samples": all_variable_mask_audits})
    json_dump(out / "loss_breakdown.json", {"samples": loss_breakdown})
    done.write_text("complete\n", encoding="utf-8")
    return metrics


def verify_no_leakage(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = config_from_args(args, args.method, args.seed, "leakage_verification", str(Path(args.output_root) / "leakage_verification" / "sample"))
    cfg.dataset_len = 1
    tokenizer, model, device = load_model_and_data(cfg)
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, 1, cfg.seed)
    prompt = prompts[0]
    tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    with torch.no_grad():
        observed = capture_prefix_activation(model, cfg.target_layer, input_ids=input_ids, attention_mask=attention_mask).detach()
        full = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        full_boundary = full.hidden_states[cfg.target_layer + 1].detach()
        server_final, server_logits, collected = server_forward_with_attention(
            model,
            cfg.target_layer + 1,
            observed.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=True,
            rollout_depth=cfg.server_rollout_depth,
        )
    boundary_diff = float((observed.float() - full_boundary.float()).abs().max().detach().cpu())
    final_diff = float((server_final.float() - full.hidden_states[-1].float()).abs().max().detach().cpu())
    logits_diff = float((server_logits.float() - full.logits.float()).abs().max().detach().cpu())
    fixed_public = inferred_boundary_special_tokens(tokenizer, int(input_ids.shape[1])) if cfg.fix_boundary_specials else {}
    variable_audit = build_variable_mask(
        attention_mask=attention_mask,
        fixed_public=fixed_public,
        special_token_ids=getattr(tokenizer, "all_special_ids", None),
        token_ids=None,
    )
    weights, attn_stats, rollout_stats, weight_stats = server_attention_bundle(model, observed, attention_mask, cfg, fixed_public, variable_audit)
    uniform = torch.ones_like(weights)
    pred = observed + torch.randn_like(observed) * 0.01
    uniform_loss = float(weighted_activation_loss(pred, observed, uniform).detach().cpu())
    base_loss = float(uniform_all_token_activation_loss(pred, observed, attention_mask).detach().cpu())
    api_check = validate_attack_api()
    result = {
        "title": EXPERIMENT_TITLE,
        "dataset_meta": dataset_meta,
        "target_layer": cfg.target_layer,
        "h_obs_semantics": f"H_obs is output of 0-based block {cfg.target_layer}; server starts at block {cfg.target_layer + 1}",
        "server_layers": [idx for idx, _attn in collected],
        "activation_shape": list(observed.shape),
        "boundary_vs_full_hidden_max_abs_diff": boundary_diff,
        "server_final_hidden_vs_full_max_abs_diff": final_diff,
        "server_logits_vs_full_max_abs_diff": logits_diff,
        "manual_server_forward_matches_full": final_diff < 5e-3 and logits_diff < 5e-3,
        "attack_api_check": api_check,
        "attention_stats": attn_stats,
        "rollout_stats": rollout_stats,
        "weight_stats": weight_stats,
        "variable_mask_audit": variable_mask_audit_payload(variable_audit),
        "uniform_weight_loss": uniform_loss,
        "unweighted_loss": base_loss,
        "uniform_weight_equals_unweighted_loss": abs(uniform_loss - base_loss) < 1e-8,
        "p1_p2_do_not_use_dummy_attention": True,
    }
    out = Path(args.output_root) / "leakage_verification"
    out.mkdir(parents=True, exist_ok=True)
    json_dump(out / "leakage_verification.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def download_skytrax_150(args: argparse.Namespace) -> None:
    out = Path(args.dataset_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force_download:
        print(f"dataset exists: {out}", flush=True)
        return
    slugs = [
        "british-airways",
        "emirates",
        "qatar-airways",
        "singapore-airlines",
        "lufthansa",
        "air-france",
        "klm-royal-dutch-airlines",
        "turkish-airlines",
        "american-airlines",
        "united-airlines",
        "delta-air-lines",
        "ryanair",
        "easyjet",
    ]
    reviews: List[str] = []
    seen = set()
    pattern = re.compile(r'<div[^>]+class="[^"]*text_content[^"]*"[^>]*>(.*?)</div>', re.S | re.I)
    tag_re = re.compile(r"<[^>]+>")
    for slug in slugs:
        for page in range(1, 8):
            if len(reviews) >= args.dataset_size:
                break
            url = f"https://www.airlinequality.com/airline-reviews/{slug}/page/{page}/?sortby=post_date%3ADesc&pagesize=100"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urlopen(req, timeout=20) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception as exc:
                print(f"download warning {url}: {exc}", flush=True)
                continue
            for match in pattern.findall(html):
                text = unescape(tag_re.sub(" ", match))
                text = re.sub(r"\s+", " ", text).strip()
                text = re.sub(r"^(✅ Trip Verified|Not Verified|Trip Verified)\s*\|\s*", "", text).strip()
                if len(text) < 80 or not text.isascii():
                    continue
                key = text.lower()[:200]
                if key in seen:
                    continue
                seen.add(key)
                reviews.append(text)
                if len(reviews) >= args.dataset_size:
                    break
        if len(reviews) >= args.dataset_size:
            break
    if len(reviews) < args.dataset_size:
        raise RuntimeError(f"Only downloaded {len(reviews)} Skytrax reviews, requested {args.dataset_size}")
    json_dump(out, reviews[: args.dataset_size])
    meta = {
        "source": "airlinequality.com airline review pages (Skytrax-branded public review pages)",
        "sample_count": len(reviews[: args.dataset_size]),
        "slugs": slugs,
        "ascii_filter": True,
        "min_length": 80,
    }
    json_dump(out.with_suffix(".meta.json"), meta)
    print(json.dumps({"dataset_out": str(out), **meta}, ensure_ascii=True), flush=True)


def config_from_args(args: argparse.Namespace, method: str, seed: int, run_name: str, output_dir: str) -> SAWConfig:
    method = canonical_method(method)
    inverted, target = ATTACKER_MAP[args.participant_number][args.attacker_position]
    if args.target_layer is not None:
        target = args.target_layer
        inverted = target + 1
    return SAWConfig(
        method=method,
        run_name=run_name,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=args.dataset_len,
        seed=seed,
        participant_number=args.participant_number,
        attacker_position=args.attacker_position,
        inverted_block_count=inverted,
        target_layer=target,
        epoch=args.epoch,
        stage_a_epoch=args.stage_a_epoch if args.stage_a_epoch is not None else args.epoch,
        lr=args.lr,
        lambda_vocab=args.lambda_vocab,
        lambda_dummy=args.lambda_dummy,
        lambda_context=args.lambda_context,
        top_k_embedding=args.k,
        top_y_semantic=args.y,
        gamma=args.gamma,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        weight_source=args.weight_source,
        weight_floor=args.weight_floor,
        weight_power=args.weight_power,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        attention_start_ratio=args.attention_start_ratio,
        attention_full_ratio=args.attention_full_ratio,
        uncertainty_fraction=args.uncertainty_fraction,
        adaptive_min_beta=args.adaptive_min_beta,
        refine_epoch=args.refine_epoch,
        lambda_projection=args.lambda_projection,
        refine_lr_scale=args.refine_lr_scale,
        beta=args.beta,
        residual_alpha_rho=args.residual_alpha_rho,
        last_window_size=args.last_window_size,
        residual_rollout=not args.no_residual_rollout,
        server_rollout_depth=args.server_rollout_depth,
        adaptive_discretization=not args.naive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        local_files_only=args.local_files_only,
    )


def run_single(args: argparse.Namespace) -> Dict[str, Any]:
    method = canonical_method(args.method)
    out = args.output_dir or str(Path(args.output_root) / method / f"{method}_seed{args.seed}_layer{args.target_layer}_k{args.k}")
    cfg = config_from_args(args, method, args.seed, Path(out).name, out)
    return run_one_config(cfg, resume=args.resume)


def run_multi_layer(args: argparse.Namespace) -> None:
    for layer in args.target_layers:
        for method in args.methods:
            method_name = canonical_method(method)
            run_name = f"{method_name}_seed{args.seed}_layer{layer}_epoch{args.epoch}_k{args.k}"
            out = Path(args.output_root) / "multi_layer" / run_name
            layer_args = argparse.Namespace(**vars(args))
            layer_args.target_layer = layer
            cfg = config_from_args(layer_args, method_name, args.seed, run_name, str(out))
            run_one_config(cfg, resume=args.resume)
    write_summary(Path(args.output_root))


def write_summary(root: Path) -> None:
    rows = []
    for metrics_path in sorted(root.glob("**/metrics.json")):
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = {}
        config_path = metrics_path.with_name("config.json")
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
        dataset_meta = m.get("dataset_meta", {}) or {}
        rows.append(
            {
                "stage": metrics_path.relative_to(root).parts[0] if len(metrics_path.relative_to(root).parts) > 1 else "",
                "method": m.get("method"),
                "seed": m.get("seed"),
                "target_layer": m.get("target_layer"),
                "dataset_name": dataset_meta.get("dataset_name"),
                "dataset_path": dataset_meta.get("dataset_path"),
                "epoch": m.get("epoch", config.get("epoch")),
                "max_token_len": config.get("max_token_len"),
                "top_k_embedding": m.get("top_k_embedding"),
                "top_y_semantic": m.get("top_y_semantic"),
                "residual_alpha_rho": config.get("residual_alpha_rho"),
                "token_accuracy_mean": m.get("token_accuracy", {}).get("mean"),
                "token_accuracy_std": m.get("token_accuracy", {}).get("std"),
                "bleu_mean": m.get("bleu", {}).get("mean"),
                "bleu_std": m.get("bleu", {}).get("std"),
                "completed_sample_count": m.get("completed_sample_count"),
                "failed_sample_count": m.get("failed_sample_count"),
                "metrics_path": str(metrics_path),
            }
        )
    if not rows:
        return
    fields = list(rows[0].keys())
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "comparison.md").open("w", encoding="utf-8") as f:
        f.write("| method | seed | layer | epoch | max_len | k | token_acc | bleu | n | fail |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['method']} | {r['seed']} | {r['target_layer']} | {r['epoch']} | {r['max_token_len']} | {r['top_k_embedding']} | "
                f"{r['token_accuracy_mean']} | {r['bleu_mean']} | {r['completed_sample_count']} | {r['failed_sample_count']} |\n"
            )
    write_report(root, rows)


def safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def stats_values(samples: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for sample in samples:
        number = safe_number(sample.get(key))
        if number is not None:
            values.append(number)
    return values


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def min_or_none(values: Sequence[float]) -> Optional[float]:
    return min(values) if values else None


def max_or_none(values: Sequence[float]) -> Optional[float]:
    return max(values) if values else None


def has_nan_payload(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_nan_payload(v) for v in value.values())
    if isinstance(value, list):
        return any(has_nan_payload(v) for v in value)
    return False


def collect_ablation_rows(root: Path, subdirs: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for subdir in subdirs:
        for metrics_path in sorted((root / subdir).glob("*/metrics.json")):
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            config_path = metrics_path.with_name("config.json")
            config: Dict[str, Any] = {}
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    config = {}
            weight_path = metrics_path.with_name("token_weight_stats.json")
            weight_samples: List[Dict[str, Any]] = []
            if weight_path.exists():
                try:
                    weight_samples = json.loads(weight_path.read_text(encoding="utf-8")).get("samples", [])
                except (OSError, json.JSONDecodeError):
                    weight_samples = []
            dataset_meta = metrics.get("dataset_meta", {}) or {}
            weight_mean = stats_values(weight_samples, "mean")
            weight_std = stats_values(weight_samples, "std")
            weight_min = stats_values(weight_samples, "min")
            weight_max = stats_values(weight_samples, "max")
            raw_bos = stats_values(weight_samples, "raw_bos_mass")
            raw_last = stats_values(weight_samples, "raw_last_valid_mass")
            final_bos = stats_values(weight_samples, "final_bos_weight")
            final_last = stats_values(weight_samples, "final_last_valid_weight")
            fixed_sum = stats_values(weight_samples, "final_fixed_weight_sum")
            configured_weight_max = safe_number(config.get("weight_max"))
            row = {
                "stage": subdir,
                "run_name": metrics_path.parent.name,
                "method": metrics.get("method"),
                "seed": metrics.get("seed"),
                "target_layer": metrics.get("target_layer"),
                "dataset_name": dataset_meta.get("dataset_name"),
                "dataset_path": dataset_meta.get("dataset_path"),
                "requested_prompts": dataset_meta.get("requested_prompts"),
                "epoch": config.get("epoch", metrics.get("epoch")),
                "top_k_embedding": metrics.get("top_k_embedding"),
                "top_y_semantic": metrics.get("top_y_semantic"),
                "beta": config.get("beta"),
                "weight_power": config.get("weight_power"),
                "weight_min": config.get("weight_min"),
                "weight_max": config.get("weight_max"),
                "configured_weight_max": configured_weight_max,
                "residual_rollout": config.get("residual_rollout"),
                "server_rollout_depth": config.get("server_rollout_depth"),
                "residual_alpha_rho": config.get("residual_alpha_rho"),
                "token_accuracy_mean": metrics.get("token_accuracy", {}).get("mean"),
                "token_accuracy_std": metrics.get("token_accuracy", {}).get("std"),
                "bleu_mean": metrics.get("bleu", {}).get("mean"),
                "bleu_std": metrics.get("bleu", {}).get("std"),
                "completed_sample_count": metrics.get("completed_sample_count"),
                "failed_sample_count": metrics.get("failed_sample_count"),
                "raw_bos_mass_mean": mean_or_none(raw_bos),
                "raw_last_valid_mass_mean": mean_or_none(raw_last),
                "final_bos_weight_max": max_or_none(final_bos),
                "final_last_valid_weight_mean": mean_or_none(final_last),
                "fixed_weight_sum_max": max_or_none(fixed_sum),
                "weight_mean_mean": mean_or_none(weight_mean),
                "weight_std_mean": mean_or_none(weight_std),
                "weight_min_min": min_or_none(weight_min),
                "weight_max_max": max_or_none(weight_max),
                "has_nan": has_nan_payload(metrics) or has_nan_payload(weight_samples),
                "metrics_path": str(metrics_path),
                "output_dir": str(metrics_path.parent),
            }
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_best_candidate(
    rows: Sequence[Dict[str, Any]],
    method: str,
    b0v_acc: Optional[float],
    min_completed: int,
) -> Optional[Dict[str, Any]]:
    method_name = canonical_method(method)
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if canonical_method(str(row.get("method"))) != method_name:
            continue
        acc = safe_number(row.get("token_accuracy_mean"))
        bleu = safe_number(row.get("bleu_mean"))
        weight_mean = safe_number(row.get("weight_mean_mean"))
        weight_max = safe_number(row.get("weight_max_max"))
        configured_max = safe_number(row.get("configured_weight_max"))
        fixed_sum = safe_number(row.get("fixed_weight_sum_max"))
        completed = int(row.get("completed_sample_count") or 0)
        if acc is None or bleu is None or weight_mean is None or weight_max is None or configured_max is None:
            continue
        if row.get("has_nan"):
            continue
        if completed < max(2, int(min_completed)):
            continue
        if fixed_sum is None or abs(fixed_sum) > 1e-6:
            continue
        if abs(weight_mean - 1.0) > 1e-5:
            continue
        if weight_max > configured_max + 1e-5:
            continue
        if b0v_acc is not None and acc < b0v_acc - 1e-12:
            continue
        candidates.append(row)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            -(safe_number(row.get("token_accuracy_mean")) or -1.0),
            -(safe_number(row.get("bleu_mean")) or -1.0),
            safe_number(row.get("weight_std_mean")) if safe_number(row.get("weight_std_mean")) is not None else float("inf"),
            safe_number(row.get("weight_max_max")) if safe_number(row.get("weight_max_max")) is not None else float("inf"),
        ),
    )[0]


def fmt_md(value: Any, digits: int = 4) -> str:
    number = safe_number(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def write_dev_plots(root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (root / "dev_plot_error.txt").write_text(repr(exc), encoding="utf-8")
        return

    scored = [row for row in rows if safe_number(row.get("token_accuracy_mean")) is not None]
    scored = sorted(scored, key=lambda row: (str(row.get("method")), safe_number(row.get("token_accuracy_mean")) or 0.0))
    labels = [str(row.get("run_name")) for row in scored]
    accs = [safe_number(row.get("token_accuracy_mean")) or 0.0 for row in scored]
    colors = ["#4C78A8" if str(row.get("method")).startswith("mts") else "#F58518" for row in scored]
    height = max(6.0, 0.24 * len(scored))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(range(len(scored)), accs, color=colors)
    ax.set_yticks(range(len(scored)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Token Accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_title("MTS-PIA Development Ablation Token Accuracy")
    fig.tight_layout()
    fig.savefig(root / "dev_token_accuracy.png", dpi=180)
    plt.close(fig)

    weighted = [
        row
        for row in rows
        if safe_number(row.get("weight_std_mean")) is not None and safe_number(row.get("weight_max_max")) is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    marker_by_method = {
        "original_pia_baseline": "o",
        "variable_only_uniform": "s",
        "server_attn_last_raw": "x",
        "mts_mean_query": "^",
        "mts_last_window": "D",
    }
    for method in sorted({str(row.get("method")) for row in weighted}):
        subset = [row for row in weighted if str(row.get("method")) == method]
        ax.scatter(
            [safe_number(row.get("weight_std_mean")) or 0.0 for row in subset],
            [safe_number(row.get("weight_max_max")) or 0.0 for row in subset],
            label=method,
            marker=marker_by_method.get(method, "o"),
            alpha=0.8,
        )
    ax.set_xlabel("Mean Weight Std")
    ax.set_ylabel("Max Final Weight")
    ax.set_title("MTS-PIA Weight Concentration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "dev_weight_concentration.png", dpi=180)
    plt.close(fig)


def write_dev_ablation_outputs(root: Path) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    rows = collect_ablation_rows(root, ["dev_ablation_baselines", "dev_ablation_runs"])
    rows = sorted(rows, key=lambda row: (str(row.get("method")), str(row.get("run_name"))))
    write_csv(root / "dev_ablation.csv", rows)
    weight_fields = [
        "stage",
        "run_name",
        "method",
        "beta",
        "weight_power",
        "weight_min",
        "weight_max",
        "residual_rollout",
        "server_rollout_depth",
        "completed_sample_count",
        "failed_sample_count",
        "raw_bos_mass_mean",
        "raw_last_valid_mass_mean",
        "final_bos_weight_max",
        "final_last_valid_weight_mean",
        "fixed_weight_sum_max",
        "weight_mean_mean",
        "weight_std_mean",
        "weight_min_min",
        "weight_max_max",
        "has_nan",
        "output_dir",
    ]
    write_csv(root / "dev_weight_summary.csv", [{key: row.get(key) for key in weight_fields} for row in rows])
    write_dev_plots(root, rows)

    b0 = next((row for row in rows if row.get("method") == "original_pia_baseline"), None)
    b0v = next((row for row in rows if row.get("method") == "variable_only_uniform"), None)
    p1 = next((row for row in rows if row.get("method") == "server_attn_last_raw"), None)
    b0_acc = safe_number(b0.get("token_accuracy_mean")) if b0 else None
    b0_bleu = safe_number(b0.get("bleu_mean")) if b0 else None
    b0v_acc = safe_number(b0v.get("token_accuracy_mean")) if b0v else None
    b0v_bleu = safe_number(b0v.get("bleu_mean")) if b0v else None
    min_completed = max([int(row.get("completed_sample_count") or 0) for row in rows] or [2])
    best_p2 = choose_best_candidate(rows, "mts_mean_query", b0v_acc, min_completed=2)
    best_p3 = choose_best_candidate(rows, "mts_last_window", b0v_acc, min_completed=2)

    def method_line(label: str, row: Optional[Dict[str, Any]]) -> str:
        if not row:
            return f"| {label} | NA | NA | NA | NA | NA | NA | NA | NA | NA |"
        acc = safe_number(row.get("token_accuracy_mean"))
        bleu = safe_number(row.get("bleu_mean"))
        delta_b0 = None if acc is None or b0_acc is None else acc - b0_acc
        delta_b0v = None if acc is None or b0v_acc is None else acc - b0v_acc
        bleu_b0 = None if bleu is None or b0_bleu is None else bleu - b0_bleu
        bleu_b0v = None if bleu is None or b0v_bleu is None else bleu - b0v_bleu
        return (
            f"| {label} | {row.get('run_name')} | {fmt_md(acc)} | {fmt_md(delta_b0)} | {fmt_md(delta_b0v)} | "
            f"{fmt_md(bleu)} | {fmt_md(bleu_b0)} | {fmt_md(bleu_b0v)} | "
            f"{row.get('completed_sample_count')} | {row.get('failed_sample_count')} |"
        )

    best_rows = [row for row in [best_p2, best_p3] if row]
    exceeds_b0 = [
        str(row.get("method"))
        for row in best_rows
        if b0_acc is not None and (safe_number(row.get("token_accuracy_mean")) or -1.0) > b0_acc
    ]
    not_below_b0 = [
        str(row.get("method"))
        for row in best_rows
        if b0_acc is not None and (safe_number(row.get("token_accuracy_mean")) or -1.0) >= b0_acc
    ]
    md_lines = [
        "# MTS-PIA Development Ablation",
        "",
        "This report is for Skytrax-28 seed42 layer17 development ablation. It is not multi-seed evidence and must not be described as stable improvement.",
        "",
        "## Audit Semantics",
        "",
        "- `final_eos_weight` was renamed to `final_last_valid_weight`.",
        "- `raw_eos_mass` was renamed to `raw_last_valid_mass`.",
        "- Position 0 and other `fixed_public` positions are public framing positions available to the attacker.",
        "- Because the attack API does not receive token ids, it cannot claim EOS detection by token id.",
        "- The last valid position is only a sequence boundary unless the public protocol fixes it as EOS.",
        "",
        "## Main Results",
        "",
        "| method | run | acc | acc_delta_vs_B0 | acc_delta_vs_B0V | BLEU | BLEU_delta_vs_B0 | BLEU_delta_vs_B0V | done | fail |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        method_line("B0", b0),
        method_line("B0V", b0v),
        method_line("P1", p1),
        method_line("best P2", best_p2),
        method_line("best P3", best_p3),
        "",
        "## Best P2/P3 Parameters",
        "",
        "| method | beta | power | weight_max | residual | depth | weight_std | weight_max_observed | fixed_sum_max | has_nan |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for label, row in [("P2", best_p2), ("P3", best_p3)]:
        if row:
            md_lines.append(
                f"| {label} | {fmt_md(row.get('beta'), 2)} | {fmt_md(row.get('weight_power'), 2)} | "
                f"{fmt_md(row.get('weight_max'), 1)} | {row.get('residual_rollout')} | {row.get('server_rollout_depth')} | "
                f"{fmt_md(row.get('weight_std_mean'))} | {fmt_md(row.get('weight_max_max'))} | "
                f"{fmt_md(row.get('fixed_weight_sum_max'))} | {row.get('has_nan')} |"
            )
        else:
            md_lines.append(f"| {label} | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    md_lines.extend(
        [
            "",
            "## Formal-Check Fields",
            "",
            "| method | raw_bos_mass | final_bos_weight | final_last_valid_weight | fixed_sum | completed | failed | has_nan |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for label, row in [("B0", b0), ("B0V", b0v), ("P1", p1), ("best P2", best_p2), ("best P3", best_p3)]:
        if row:
            md_lines.append(
                f"| {label} | {fmt_md(row.get('raw_bos_mass_mean'))} | {fmt_md(row.get('final_bos_weight_max'))} | "
                f"{fmt_md(row.get('final_last_valid_weight_mean'))} | {fmt_md(row.get('fixed_weight_sum_max'))} | "
                f"{row.get('completed_sample_count')} | {row.get('failed_sample_count')} | {row.get('has_nan')} |"
            )
        else:
            md_lines.append(f"| {label} | NA | NA | NA | NA | NA | NA | NA |")
    md_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best P2/P3 above B0: {', '.join(exceeds_b0) if exceeds_b0 else 'none'}.",
            f"- Best P2/P3 not below B0: {', '.join(not_below_b0) if not_below_b0 else 'none'}.",
            "- Continue to multi-seed only if at least one best P2/P3 is not below B0 on this development check.",
        ]
    )
    structure_path = root / "structure_ablation.md"
    if structure_path.exists():
        md_lines.extend(["", "## Structure Ablation", "", structure_path.read_text(encoding="utf-8")])
    md = "\n".join(md_lines)
    (root / "dev_ablation.md").write_text(md, encoding="utf-8")
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/masked_server_attn_report.md").write_text(md, encoding="utf-8")
    return rows, best_p2, best_p3


def write_structure_ablation_outputs(root: Path) -> List[Dict[str, Any]]:
    rows = collect_ablation_rows(root, ["structure_ablation_runs"])
    rows = sorted(rows, key=lambda row: (str(row.get("method")), str(row.get("run_name"))))
    write_csv(root / "structure_ablation.csv", rows)
    lines = [
        "Structure ablation uses each method's best development beta/power/weight_max and varies residual rollout plus server rollout depth.",
        "",
        "| method | residual | depth | acc | BLEU | std_w | max_w | done | fail | run |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('method')} | {row.get('residual_rollout')} | {row.get('server_rollout_depth')} | "
            f"{fmt_md(row.get('token_accuracy_mean'))} | {fmt_md(row.get('bleu_mean'))} | "
            f"{fmt_md(row.get('weight_std_mean'))} | {fmt_md(row.get('weight_max_max'))} | "
            f"{row.get('completed_sample_count')} | {row.get('failed_sample_count')} | {row.get('run_name')} |"
        )
    (root / "structure_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap_mean_ci(values: Sequence[float], iterations: int = 10000, seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(max(1, int(iterations))):
        sample = [values[rng.randrange(n)] for _idx in range(n)]
        means.append(float(statistics.mean(sample)))
    means.sort()
    low_index = min(len(means) - 1, max(0, int(math.floor(0.025 * (len(means) - 1)))))
    high_index = min(len(means) - 1, max(0, int(math.ceil(0.975 * (len(means) - 1)))))
    return means[low_index], means[high_index]


def compute_paired_delta_summary(
    method: str,
    baseline_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    bootstrap_iters: int = 10000,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    baseline_by_id = {int(row["prompt_id"]): row for row in baseline_rows}
    candidate_by_id = {int(row["prompt_id"]): row for row in candidate_rows}
    common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    token_deltas: List[float] = []
    bleu_deltas: List[float] = []
    wins = ties = losses = 0
    for prompt_id in common_ids:
        base = baseline_by_id[prompt_id]
        cand = candidate_by_id[prompt_id]
        acc_delta = float(cand["token_accuracy"]) - float(base["token_accuracy"])
        bleu_delta = float(cand["bleu"]) - float(base["bleu"])
        token_deltas.append(acc_delta)
        bleu_deltas.append(bleu_delta)
        if acc_delta > 1e-12:
            wins += 1
        elif acc_delta < -1e-12:
            losses += 1
        else:
            ties += 1
    acc_low, acc_high = bootstrap_mean_ci(token_deltas, bootstrap_iters, bootstrap_seed)
    bleu_low, bleu_high = bootstrap_mean_ci(bleu_deltas, bootstrap_iters, bootstrap_seed + 17)
    return {
        "method": method,
        "pair_count": len(common_ids),
        "mean_token_accuracy_delta": float(statistics.mean(token_deltas)) if token_deltas else None,
        "mean_bleu_delta": float(statistics.mean(bleu_deltas)) if bleu_deltas else None,
        "win_count": wins,
        "tie_count": ties,
        "loss_count": losses,
        "token_accuracy_ci_low": acc_low,
        "token_accuracy_ci_high": acc_high,
        "bleu_ci_low": bleu_low,
        "bleu_ci_high": bleu_high,
    }


def metric_mean(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, dict):
        return safe_number(value.get("mean"))
    return safe_number(value)


def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def heldout_run_name(label: str, seed: int, layer: int, epoch: int, k: int) -> str:
    if label in HELDOUT_BASELINES:
        return f"{HELDOUT_BASELINES[label]}_seed{seed}_layer{layer}_epoch{epoch}_k{k}"
    spec = HELDOUT_FROZEN_CANDIDATES[label]
    residual = "reson" if spec["residual_rollout"] else "resoff"
    return (
        f"{spec['method']}_seed{seed}_layer{layer}_epoch{epoch}_k{k}"
        f"_beta{float_tag(spec['beta'])}_power{float_tag(spec['weight_power'])}"
        f"_wmin{float_tag(spec['weight_min'])}_wmax{float_tag(spec['weight_max'])}"
        f"_{residual}_depth{spec['server_rollout_depth']}"
    )


def heldout_run_dir(root: Path, seed: int, label: str, layer: int = 17, epoch: int = 100, k: int = 10) -> Path:
    return root / f"heldout_seed{seed}" / heldout_run_name(label, seed, layer, epoch, k)


def load_run_outputs(run_dir: Path) -> Dict[str, Any]:
    return {
        "metrics": load_json_object(run_dir / "metrics.json"),
        "weights": load_json_object(run_dir / "token_weight_stats.json"),
        "predictions": read_jsonl(run_dir / "predictions.jsonl"),
        "failures": read_jsonl(run_dir / "failures.jsonl"),
    }


def weight_summary(weight_payload: Dict[str, Any]) -> Dict[str, Any]:
    samples = weight_payload.get("samples", []) if isinstance(weight_payload, dict) else []
    return {
        "raw_bos_mass": mean_or_none(stats_values(samples, "raw_bos_mass")),
        "final_bos_weight": max_or_none(stats_values(samples, "final_bos_weight")),
        "fixed_public_final_weight_sum": max_or_none(stats_values(samples, "final_fixed_weight_sum")),
        "max_final_weight": max_or_none(stats_values(samples, "max")),
        "has_nan": has_nan_payload(weight_payload),
    }


def leakage_verification_passes(root: Path) -> bool:
    payload = load_json_object(root / "leakage_verification" / "leakage_verification.json")
    if not payload:
        return False
    api = payload.get("attack_api_check", {})
    return bool(payload.get("manual_server_forward_matches_full")) and bool(api.get("passes"))


def heldout_seed_stop_decision(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reasons: List[str] = []
    for row in rows:
        method = str(row.get("method"))
        if bool(row.get("has_nan")):
            reasons.append(f"{method}: NaN detected")
        if int(row.get("failed_sample_count") or 0) > 0:
            reasons.append(f"{method}: failed_sample_count > 0")
        fixed_sum = safe_number(row.get("fixed_public_final_weight_sum")) or 0.0
        if fixed_sum > 1e-6:
            reasons.append(f"{method}: fixed public final weight sum {fixed_sum:.6g} > 1e-6")
        max_weight = safe_number(row.get("max_final_weight"))
        configured = safe_number(row.get("configured_weight_max"))
        if max_weight is not None and configured is not None and max_weight > configured + 1e-6:
            reasons.append(f"{method}: final max weight {max_weight:.6g} exceeds configured weight_max {configured:.6g}")
        if not bool(row.get("attack_api_passes", False)):
            reasons.append(f"{method}: attack API check failed")
        if not bool(row.get("leakage_verification_passes", False)):
            reasons.append(f"{method}: leakage verification failed")
    if reasons:
        return {"stop": True, "rule": "B", "reasons": reasons}

    candidate_rows = [row for row in rows if str(row.get("method")) in {"P2", "P3"}]
    if len(candidate_rows) == 2 and all(
        (safe_number(row.get("mean_token_accuracy_delta")) or 0.0) < 0.0
        and (safe_number(row.get("mean_bleu_delta")) or 0.0) < 0.0
        and int(row.get("win_count") or 0) <= int(row.get("loss_count") or 0)
        for row in candidate_rows
    ):
        return {
            "stop": True,
            "rule": "A",
            "reasons": [
                "P2 and P3 both have negative mean token accuracy delta, negative mean BLEU delta, and win_count <= loss_count."
            ],
        }
    return {"stop": False, "rule": "", "reasons": []}


def build_heldout_tasks(args: argparse.Namespace, seed: int, labels: Sequence[str]) -> Tuple[List[AutoBatchTask], AutoBatchPlan]:
    gpu_memories = query_gpu_memory()
    plan = plan_auto_batch_jobs(
        gpu_memories,
        min_free_mb_per_job=args.batch_free_mb_per_job,
        reserve_free_mb=args.batch_reserve_free_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        max_total_jobs=args.max_parallel_jobs,
    )
    slots = gpu_slots_from_jobs(plan.jobs_by_gpu)
    if not slots:
        raise RuntimeError("no runnable GPU slots for held-out validation")
    tasks: List[AutoBatchTask] = []
    for index, label in enumerate(labels):
        if label in HELDOUT_BASELINES:
            task = AutoBatchTask(
                method=label,
                target_layer=args.target_layer,
                gpu_index=slots[index % len(slots)],
                run_name=heldout_run_name(label, seed, args.target_layer, args.epoch, args.k),
                output_subdir=f"heldout_seed{seed}",
            )
        else:
            spec = HELDOUT_FROZEN_CANDIDATES[label]
            task = AutoBatchTask(
                method=label,
                target_layer=args.target_layer,
                gpu_index=slots[index % len(slots)],
                run_name=heldout_run_name(label, seed, args.target_layer, args.epoch, args.k),
                output_subdir=f"heldout_seed{seed}",
                beta=float(spec["beta"]),
                weight_power=float(spec["weight_power"]),
                weight_min=float(spec["weight_min"]),
                weight_max=float(spec["weight_max"]),
                residual_rollout=bool(spec["residual_rollout"]),
                server_rollout_depth=str(spec["server_rollout_depth"]),
            )
        tasks.append(task)
    return tasks, plan


def write_heldout_validation_plan(root: Path) -> None:
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    dev_rows = read_csv_rows(root / "dev_ablation.csv")
    structure_rows = read_csv_rows(root / "structure_ablation.csv")
    leakage_passes = leakage_verification_passes(root)
    p2_frozen = next(
        (
            row
            for row in structure_rows
            if row.get("method") == "mts_mean_query"
            and row.get("residual_rollout") == "True"
            and row.get("server_rollout_depth") == "last2"
            and row.get("beta") == "0.75"
            and row.get("weight_power") == "0.5"
            and row.get("weight_max") == "4.0"
        ),
        None,
    )
    p3_frozen = next(
        (
            row
            for row in structure_rows
            if row.get("method") == "mts_last_window"
            and row.get("residual_rollout") == "True"
            and row.get("server_rollout_depth") == "all"
            and row.get("beta") == "0.25"
            and row.get("weight_power") == "0.5"
            and row.get("weight_max") == "2.0"
        ),
        None,
    )
    lines = [
        "# MTS-PIA Held-Out Validation Plan",
        "",
        "Seed42 is treated only as the development set used for parameter selection. Seeds 43 and 44 are held-out validation seeds.",
        "",
        "## Frozen Candidates",
        "",
        f"- P2: {HELDOUT_FROZEN_CANDIDATES['P2']}",
        f"- P3: {HELDOUT_FROZEN_CANDIDATES['P3']}",
        "",
        "## Pre-Run Checks",
        "",
        f"- dev_ablation.csv rows: {len(dev_rows)}",
        f"- structure_ablation.csv rows: {len(structure_rows)}",
        f"- P2 frozen structure row found: {p2_frozen is not None}",
        f"- P3 frozen structure row found: {p3_frozen is not None}",
        f"- leakage verification passes: {leakage_passes}",
        "- B0 remains all-valid uniform loss via `uniform_all_token_activation_loss`.",
        "- B0V remains variable-only uniform loss via `weighted_variable_activation_loss(..., weights=None)`.",
        "- P2/P3 keep server-side attention rollout, variable mask, tempering, clipping, beta mixing, and activation calibration.",
        "- The attack API does not receive prompt, original_ids, original_tokens, or token_ids.",
        "- Last valid position is recorded as a sequence boundary, not as EOS.",
        "",
        "No parameter search is allowed on seed43 or seed44.",
    ]
    (analysis_dir / "mts_heldout_validation_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_heldout_seed_decision(seed: int, rows: Sequence[Dict[str, Any]], decision: Dict[str, Any]) -> None:
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# MTS-PIA Held-Out Seed {seed} Decision",
        "",
        "This held-out seed was not used for parameter selection.",
        "",
        f"- stop: {decision['stop']}",
        f"- stop_rule: {decision['rule'] or 'none'}",
    ]
    for row in rows:
        lines.append(
            "- {method}: acc_delta_vs_B0V={acc}, BLEU_delta_vs_B0V={bleu}, win/tie/loss={w}/{t}/{l}, failed={fail}, fixed_sum={fixed}, max_weight={maxw}".format(
                method=row["method"],
                acc=fmt_md(row["mean_token_accuracy_delta_vs_B0V"]),
                bleu=fmt_md(row["mean_bleu_delta_vs_B0V"]),
                w=row["win_count"],
                t=row["tie_count"],
                l=row["loss_count"],
                fail=row["failed_sample_count"],
                fixed=fmt_md(row["fixed_public_final_weight_sum"]),
                maxw=fmt_md(row["max_final_weight"]),
            )
        )
    for reason in decision["reasons"]:
        lines.append(f"- reason: {reason}")
    (analysis_dir / f"mts_heldout_seed{seed}_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stop_report(seed: int, rows: Sequence[Dict[str, Any]], decision: Dict[str, Any]) -> None:
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MTS-PIA Stop Report",
        "",
        f"Stopped after held-out seed {seed}. Results are kept on disk.",
        "",
        "Conclusion: development-set gains could not be safely treated as reproduced on the held-out validation path. This is not evidence that MTS-PIA is effective.",
        "",
        f"- stop_rule: {decision['rule']}",
    ]
    for reason in decision["reasons"]:
        lines.append(f"- reason: {reason}")
    for row in rows:
        lines.append(
            "- {method}: acc_delta_vs_B0V={acc}, BLEU_delta_vs_B0V={bleu}, win/tie/loss={w}/{t}/{l}".format(
                method=row["method"],
                acc=fmt_md(row["mean_token_accuracy_delta_vs_B0V"]),
                bleu=fmt_md(row["mean_bleu_delta_vs_B0V"]),
                w=row["win_count"],
                t=row["tie_count"],
                l=row["loss_count"],
            )
        )
    (analysis_dir / "mts_stop_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seed_paired_summary(root: Path, seed: int, bootstrap_iters: int = 10000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seed_dir = root / f"heldout_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    b0_outputs = load_run_outputs(heldout_run_dir(root, seed, "B0"))
    b0v_outputs = load_run_outputs(heldout_run_dir(root, seed, "B0V"))
    b0_metrics = b0_outputs["metrics"]
    b0v_metrics = b0v_outputs["metrics"]
    attack_api_passes = bool(validate_attack_api().get("passes"))
    leakage_passes = leakage_verification_passes(root)
    rows: List[Dict[str, Any]] = []
    for label in ["P2", "P3"]:
        run_dir = heldout_run_dir(root, seed, label)
        outputs = load_run_outputs(run_dir)
        metrics = outputs["metrics"]
        weights = weight_summary(outputs["weights"])
        paired = compute_paired_delta_summary(
            method=label,
            baseline_rows=b0v_outputs["predictions"],
            candidate_rows=outputs["predictions"],
            bootstrap_iters=bootstrap_iters,
            bootstrap_seed=seed * 100 + (2 if label == "P2" else 3),
        )
        acc = metric_mean(metrics, "token_accuracy")
        bleu = metric_mean(metrics, "bleu")
        b0_acc = metric_mean(b0_metrics, "token_accuracy")
        b0_bleu = metric_mean(b0_metrics, "bleu")
        spec = HELDOUT_FROZEN_CANDIDATES[label]
        row = {
            **paired,
            "seed": seed,
            "run_name": run_dir.name,
            "token_accuracy_mean": acc,
            "bleu_mean": bleu,
            "mean_token_accuracy_delta_vs_B0": (acc - b0_acc) if acc is not None and b0_acc is not None else None,
            "mean_bleu_delta_vs_B0": (bleu - b0_bleu) if bleu is not None and b0_bleu is not None else None,
            "mean_token_accuracy_delta_vs_B0V": paired["mean_token_accuracy_delta"],
            "mean_bleu_delta_vs_B0V": paired["mean_bleu_delta"],
            "raw_bos_mass": weights["raw_bos_mass"],
            "final_bos_weight": weights["final_bos_weight"],
            "fixed_public_final_weight_sum": weights["fixed_public_final_weight_sum"],
            "max_final_weight": weights["max_final_weight"],
            "configured_weight_max": spec["weight_max"],
            "completed_sample_count": int(metrics.get("completed_sample_count", 0)) if metrics else 0,
            "failed_sample_count": int(metrics.get("failed_sample_count", 0)) if metrics else 0,
            "has_nan": bool(has_nan_payload(metrics) or weights["has_nan"]),
            "attack_api_passes": attack_api_passes,
            "leakage_verification_passes": leakage_passes,
            "output_dir": str(run_dir),
        }
        rows.append(row)

    write_csv(seed_dir / "paired_summary.csv", rows)
    decision = heldout_seed_stop_decision(rows)
    lines = [
        f"# Held-Out Seed {seed} Paired Summary",
        "",
        "| method | acc delta vs B0V | BLEU delta vs B0V | acc delta vs B0 | BLEU delta vs B0 | win/tie/loss | acc CI | BLEU CI | raw BOS | final BOS | fixed sum | max weight | done/fail |",
        "|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {acc_b0v} | {bleu_b0v} | {acc_b0} | {bleu_b0} | {w}/{t}/{l} | [{acc_l}, {acc_h}] | [{bleu_l}, {bleu_h}] | {raw} | {bos} | {fixed} | {maxw} | {done}/{fail} |".format(
                method=row["method"],
                acc_b0v=fmt_md(row["mean_token_accuracy_delta_vs_B0V"]),
                bleu_b0v=fmt_md(row["mean_bleu_delta_vs_B0V"]),
                acc_b0=fmt_md(row["mean_token_accuracy_delta_vs_B0"]),
                bleu_b0=fmt_md(row["mean_bleu_delta_vs_B0"]),
                w=row["win_count"],
                t=row["tie_count"],
                l=row["loss_count"],
                acc_l=fmt_md(row["token_accuracy_ci_low"]),
                acc_h=fmt_md(row["token_accuracy_ci_high"]),
                bleu_l=fmt_md(row["bleu_ci_low"]),
                bleu_h=fmt_md(row["bleu_ci_high"]),
                raw=fmt_md(row["raw_bos_mass"]),
                bos=fmt_md(row["final_bos_weight"]),
                fixed=fmt_md(row["fixed_public_final_weight_sum"]),
                maxw=fmt_md(row["max_final_weight"]),
                done=row["completed_sample_count"],
                fail=row["failed_sample_count"],
            )
        )
    lines += ["", "## Decision", "", f"- stop: {decision['stop']}", f"- rule: {decision['rule'] or 'none'}"]
    for reason in decision["reasons"]:
        lines.append(f"- reason: {reason}")
    (seed_dir / "paired_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_heldout_seed_decision(seed, rows, decision)
    if decision["stop"]:
        write_stop_report(seed, rows, decision)
    return rows, decision


def write_heldout_summary(root: Path, seeds: Sequence[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        rows.extend(read_csv_rows(root / f"heldout_seed{seed}" / "paired_summary.csv"))
    write_csv(root / "heldout_summary.csv", rows)

    by_method: Dict[str, List[Dict[str, Any]]] = {"P2": [], "P3": []}
    for row in rows:
        method = str(row.get("method"))
        if method in by_method:
            by_method[method].append(row)
    eligible: List[str] = []
    pooled: Dict[str, Dict[str, Any]] = {}
    for method, method_rows in by_method.items():
        seed_pass = all(
            (safe_number(row.get("mean_token_accuracy_delta_vs_B0V")) or 0.0) >= 0.0
            and (safe_number(row.get("mean_bleu_delta_vs_B0V")) or 0.0) >= 0.0
            and int(row.get("win_count") or 0) > int(row.get("loss_count") or 0)
            for row in method_rows
        ) and len(method_rows) == len(seeds)
        total_pairs = sum(int(row.get("pair_count") or 0) for row in method_rows)
        pooled_acc = (
            sum((safe_number(row.get("mean_token_accuracy_delta_vs_B0V")) or 0.0) * int(row.get("pair_count") or 0) for row in method_rows) / total_pairs
            if total_pairs
            else None
        )
        pooled_bleu = (
            sum((safe_number(row.get("mean_bleu_delta_vs_B0V")) or 0.0) * int(row.get("pair_count") or 0) for row in method_rows) / total_pairs
            if total_pairs
            else None
        )
        audit_ok = all(
            str(row.get("has_nan")).lower() != "true"
            and int(row.get("failed_sample_count") or 0) == 0
            and (safe_number(row.get("fixed_public_final_weight_sum")) or 0.0) <= 1e-6
            and (safe_number(row.get("max_final_weight")) or 0.0) <= (safe_number(row.get("configured_weight_max")) or 0.0) + 1e-6
            and str(row.get("attack_api_passes")).lower() == "true"
            and str(row.get("leakage_verification_passes")).lower() == "true"
            for row in method_rows
        )
        pooled[method] = {
            "seed_pass": seed_pass,
            "pooled_token_accuracy_delta_vs_B0V": pooled_acc,
            "pooled_bleu_delta_vs_B0V": pooled_bleu,
            "audit_ok": audit_ok,
            "total_pairs": total_pairs,
            "win_count": sum(int(row.get("win_count") or 0) for row in method_rows),
            "tie_count": sum(int(row.get("tie_count") or 0) for row in method_rows),
            "loss_count": sum(int(row.get("loss_count") or 0) for row in method_rows),
        }
        if seed_pass and pooled_acc is not None and pooled_bleu is not None and pooled_acc > 0.0 and pooled_bleu > 0.0 and audit_ok:
            eligible.append(method)
    decision = {
        "eligible_methods": eligible,
        "continue_layers": bool(eligible),
        "winner": max(eligible, key=lambda method: (pooled[method]["pooled_token_accuracy_delta_vs_B0V"], pooled[method]["pooled_bleu_delta_vs_B0V"])) if eligible else "",
        "pooled": pooled,
    }

    lines = ["# MTS-PIA Held-Out Summary", "", "Seed43 and seed44 are held-out validation seeds; no tuning was performed on them.", ""]
    lines.append("| method | seed pass | pooled acc delta vs B0V | pooled BLEU delta vs B0V | win/tie/loss | audit ok |")
    lines.append("|---|---|---:|---:|---|---|")
    for method, item in pooled.items():
        lines.append(
            f"| {method} | {item['seed_pass']} | {fmt_md(item['pooled_token_accuracy_delta_vs_B0V'])} | {fmt_md(item['pooled_bleu_delta_vs_B0V'])} | {item['win_count']}/{item['tie_count']}/{item['loss_count']} | {item['audit_ok']} |"
        )
    lines += ["", f"- continue_layers: {decision['continue_layers']}", f"- winner: {decision['winner'] or 'none'}"]
    (root / "heldout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "mts_heldout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, decision


def run_heldout_seed(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.output_root)
    write_design()
    write_heldout_validation_plan(root)
    tasks, plan = build_heldout_tasks(args, args.seed, ["B0", "B0V", "P2", "P3"])
    if args.dry_run_batch:
        for task in tasks:
            print(f"CUDA_VISIBLE_DEVICES={task.gpu_index}", " ".join(build_single_run_command(args, task)))
    else:
        run_auto_batch_task_queue(args, tasks, plan.jobs_by_gpu)
    rows, decision = write_seed_paired_summary(root, args.seed, bootstrap_iters=args.bootstrap_iters)
    return {"rows": rows, "decision": decision}


def run_dev_ablation(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    gpu_memories = query_gpu_memory()
    plan = plan_auto_batch_jobs(
        gpu_memories,
        min_free_mb_per_job=args.batch_free_mb_per_job,
        reserve_free_mb=args.batch_reserve_free_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        max_total_jobs=args.max_parallel_jobs,
    )
    gpu_slots = gpu_slots_from_jobs(plan.jobs_by_gpu)
    tasks = build_dev_ablation_tasks(
        gpu_slots=gpu_slots,
        seed=args.seed,
        target_layer=args.target_layer,
        epoch=args.epoch,
        top_k=args.k,
        weight_min=args.weight_min,
        betas=args.ablation_betas,
        powers=args.ablation_weight_powers,
        weight_maxes=args.ablation_weight_maxes,
    )
    baseline_tasks = [task for task in tasks if task.output_subdir == "dev_ablation_baselines"]
    grid_tasks = [task for task in tasks if task.output_subdir == "dev_ablation_runs"]
    print(
        json.dumps(
            {
                "dev_ablation_plan": {
                    "gpu_memories": [gpu.__dict__ for gpu in gpu_memories],
                    "jobs_by_gpu": plan.jobs_by_gpu,
                    "total_jobs": plan.total_jobs,
                    "baseline_tasks": len(baseline_tasks),
                    "grid_tasks": len(grid_tasks),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run_batch:
        for task in tasks:
            print(f"CUDA_VISIBLE_DEVICES={task.gpu_index}", " ".join(build_single_run_command(args, task)))
        return

    run_auto_batch_task_queue(args, baseline_tasks, plan.jobs_by_gpu)
    write_summary(root)
    run_auto_batch_task_queue(args, grid_tasks, plan.jobs_by_gpu)
    write_summary(root)
    _rows, best_p2, best_p3 = write_dev_ablation_outputs(root)

    structure_inputs = [row for row in [best_p2, best_p3] if row is not None]
    if structure_inputs:
        structure_tasks = build_structure_ablation_tasks(
            structure_inputs,
            gpu_slots=gpu_slots,
            seed=args.seed,
            target_layer=args.target_layer,
            epoch=args.epoch,
            top_k=args.k,
            weight_min=args.weight_min,
        )
        print(json.dumps({"structure_ablation_tasks": len(structure_tasks)}, ensure_ascii=False), flush=True)
        run_auto_batch_task_queue(args, structure_tasks, plan.jobs_by_gpu)
        write_summary(root)
        write_structure_ablation_outputs(root)
        write_dev_ablation_outputs(root)


def write_report(root: Path, rows: List[Dict[str, Any]]) -> None:
    scored_rows = [r for r in rows if r["token_accuracy_mean"] is not None]
    best = max(scored_rows, key=lambda r: float(r["token_accuracy_mean"]), default=None)
    baseline_by_layer = {
        int(r["target_layer"]): r
        for r in scored_rows
        if r.get("method") == "original_pia_baseline" and r.get("target_layer") is not None
    }

    def as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def fmt(value: Any, digits: int = 4) -> str:
        number = as_float(value)
        if number is None:
            return "NA"
        return f"{number:.{digits}f}"

    def unique_values(key: str) -> str:
        values = sorted({str(r.get(key)) for r in rows if r.get(key) is not None})
        return ", ".join(values) if values else "NA"

    def weight_stats_path(row: Dict[str, Any]) -> Optional[Path]:
        metrics_path = row.get("metrics_path")
        if not metrics_path:
            return None
        path = Path(str(metrics_path)).with_name("token_weight_stats.json")
        if path.exists():
            return path
        rooted = root / path
        if rooted.exists():
            return rooted
        return None

    def mean_key(samples: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = []
        for sample in samples:
            value = sample.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return statistics.mean(values)

    def min_key(samples: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [float(s[key]) for s in samples if s.get(key) is not None]
        return min(values) if values else None

    def max_key(samples: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [float(s[key]) for s in samples if s.get(key) is not None]
        return max(values) if values else None

    def query_span(sample: Dict[str, Any]) -> str:
        positions = sample.get("query_positions") or []
        if not positions:
            return "NA"
        if len(positions) <= 4:
            return ",".join(str(p) for p in positions)
        return f"{positions[0]}..{positions[-1]} (n={len(positions)})"

    result_lines = [
        "| method | layer | token_acc | delta_vs_B0 | BLEU | n | fail |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(scored_rows, key=lambda x: (int(x.get("target_layer") or -1), str(x.get("method")))):
        layer = int(r["target_layer"])
        baseline_acc = as_float(baseline_by_layer.get(layer, {}).get("token_accuracy_mean"))
        acc = as_float(r.get("token_accuracy_mean"))
        delta = None if acc is None or baseline_acc is None else acc - baseline_acc
        result_lines.append(
            f"| {r['method']} | {layer} | {fmt(acc)} | {fmt(delta, 4)} | "
            f"{fmt(r.get('bleu_mean'))} | {r.get('completed_sample_count')} | {r.get('failed_sample_count')} |"
        )

    weight_lines = [
        "| method | source | variable_n | mean_w | std_w | min_w | max_w | raw_bos | bos_w_max | last_valid_w | fixed_sum_max | query_positions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(scored_rows, key=lambda x: (int(x.get("target_layer") or -1), str(x.get("method")))):
        path = weight_stats_path(r)
        if not path:
            continue
        try:
            stats_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        samples = stats_payload.get("samples", [])
        if not samples:
            continue
        sources = sorted({str(s.get("source") or s.get("loss_mode") or "NA") for s in samples})
        weight_lines.append(
            f"| {r['method']} | {', '.join(sources)} | {fmt(mean_key(samples, 'variable_count'), 1)} | "
            f"{fmt(mean_key(samples, 'mean'))} | {fmt(mean_key(samples, 'std'))} | "
            f"{fmt(min_key(samples, 'min'))} | {fmt(max_key(samples, 'max'))} | "
            f"{fmt(mean_key(samples, 'raw_bos_mass'))} | {fmt(max_key(samples, 'final_bos_weight'))} | "
            f"{fmt(mean_key(samples, 'final_last_valid_weight'))} | {fmt(max_key(samples, 'final_fixed_weight_sum'))} | "
            f"{query_span(samples[0])} |"
        )

    mts_rows = [r for r in scored_rows if r.get("method") in {"mts_mean_query", "mts_last_window"}]
    improved_vs_b0 = []
    tied_vs_b0 = []
    for row in mts_rows:
        layer = int(row["target_layer"])
        base_acc = as_float(baseline_by_layer.get(layer, {}).get("token_accuracy_mean"))
        acc = as_float(row.get("token_accuracy_mean"))
        if base_acc is None or acc is None:
            continue
        if acc > base_acc:
            improved_vs_b0.append(str(row["method"]))
        elif math.isclose(acc, base_acc, rel_tol=0.0, abs_tol=1e-12):
            tied_vs_b0.append(str(row["method"]))

    failed_total = sum(int(r.get("failed_sample_count") or 0) for r in rows)
    completed_total = sum(int(r.get("completed_sample_count") or 0) for r in rows)
    main_observation = (
        "At least one MTS row is above the same-layer B0 row in this summary: "
        + ", ".join(sorted(set(improved_vs_b0)))
        if improved_vs_b0
        else "No MTS row is above the same-layer B0 row in this summary."
    )
    if tied_vs_b0:
        main_observation += " Tied MTS rows: " + ", ".join(sorted(set(tied_vs_b0))) + "."
    lines = [
        "# Masked Server Attention PIA Report",
        "",
        "The earlier `pia_masked_server_attn_pia.py` branch was effectively SAW-PIA plus resource-aware batching. This version completes the masked and tempered server-attention (MTS) body.",
        "",
        "## Layer Mapping",
        "",
        "`capture_prefix_activation(target_layer=L)` returns the output of 0-based TinyLlama block `L`. Therefore `target_layer=17` corresponds to `H^(17)`, and server-side layers start from block `18`.",
        "",
        "## Method Definitions",
        "",
        "- B0 / `original_pia_baseline`: original PIA activation matching over all valid tokens.",
        "- B0V / `variable_only_uniform`: variable-token-only uniform activation matching.",
        "- P1 / `server_attn_last_raw`: raw last-query server rollout negative control.",
        "- P2 / `mts_mean_query`: variable-mask MTS weights from mean query attention.",
        "- P3 / `mts_last_window`: variable-mask MTS weights from the last query window.",
        "",
        "P2/P3 exclude fixed public positions and any token-id special positions only when token ids are available. In the attack path token ids are not passed, so the audit reports `last_valid_position` rather than claiming EOS detection.",
        "",
        "## Experimental Setup",
        "",
        f"- Output root: `{root}`",
        f"- Dataset names: {unique_values('dataset_name')}",
        "- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`",
        f"- Layers: {unique_values('target_layer')}",
        f"- Epochs: {unique_values('epoch')}",
        f"- Embedding top-k: {unique_values('top_k_embedding')}",
        f"- Semantic top-y: {unique_values('top_y_semantic')}",
        f"- Max token length: {unique_values('max_token_len')}",
        f"- Completed prompt-runs: {completed_total}; failed prompt-runs: {failed_total}",
        "",
        "## Results",
        "",
        *result_lines,
        "",
        "## Weight Audit",
        "",
        *weight_lines,
        "",
        "## Main Observation",
        "",
        main_observation,
        "",
        "Interpret this as a pilot result only. Matching or exceeding B0 on a two-sample smoke run is useful for debugging, but it is not evidence of stable improvement.",
        "",
        "## Best Observed Row",
        "",
        json.dumps(best, indent=2, ensure_ascii=False) if best else "No completed rows.",
        "",
        "## Caution",
        "",
        "Do not claim stable improvement unless MTS beats both B0 and B0V under the planned multi-seed and multi-layer checks.",
    ]
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "masked_server_attn_report.md").write_text("\n".join(lines), encoding="utf-8")
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/masked_server_attn_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_design() -> None:
    Path("analysis").mkdir(exist_ok=True)
    design = """# Masked and Tempered Server-Attention Prompt Inversion Design

## Layer mapping

TinyLlama uses 0-based block indices. `capture_prefix_activation(model, target_layer=L, ...)` manually executes blocks `0..L` and returns the output of block `L`. Thus `target_layer=17` is `H^(17)`, and server-side layers are `18..21`.

## Starting point

The existing SAW-PIA result already feeds `H_obs` through public server-side layers, extracts each server layer attention, averages heads, applies `RowNorm(I + A)`, and computes rollout in forward order:

`R = A_tilde_last @ ... @ A_tilde_first`

The leakage verification shows that `H_obs` is the target block output, server layers start at `target_layer + 1`, and manual server forward matches full-model forward.

## Motivation

Full Skytrax-150 top-1 results show that raw last-query weighting degrades strongly at deeper layers. The rollout mass can concentrate on position 0, and the old weighted loss does not explicitly exclude fixed public/special token positions.

## Audit semantics

The attack function receives `H_obs`, sequence length, tokenizer/model/config, and public framing assumptions. It does not receive `original_ids`, `original_tokens`, or prompt text.

- `fixed_public` positions such as position 0 can be determined by the public protocol and are safe to mark as public framing positions.
- When `token_ids=None`, `token_id_based_special_mask_available=false`; the branch cannot claim it identified EOS from token id.
- `last_valid_position` is only the last position allowed by the attention mask. It is not called EOS in the audit.
- Only if a public protocol explicitly fixes the final token as EOS should a caller exclude that position as public framing.
- Weight statistics therefore use `raw_last_valid_mass` and `final_last_valid_weight`, not EOS names.

## Implemented methods

- B0 / `original_pia_baseline`: original PIA baseline with uniform activation loss over all valid tokens.
- B0V / `variable_only_uniform`: new variable-token-only uniform baseline. Fixed public and special positions have final weight 0.
- P1 / `server_attn_last_raw`: old raw last-query rollout, kept only as a negative control.
- P2 / `mts_mean_query`: MTS weights from mean query attention over variable query positions.
- P3 / `mts_last_window`: MTS weights from the final variable query window.

No dummy attention or alpha-initialization combo is used in this branch.

## MTS loss

The branch builds a `variable_mask` from the attention mask and public fixed positions. For P2/P3 it extracts server-side attention by forwarding only from `H_obs` through public server layers, computes attention rollout, and converts rollout mass into token weights:

1. choose the source query set: `mean_query` or `last_window_mean`;
2. zero all non-variable positions;
3. apply power tempering with `--weight-power`;
4. clip to `--weight-min` / `--weight-max`;
5. renormalize variable-token mean weight to 1.0;
6. mix with uniform weights using `--beta`;
7. enforce fixed public/special final weights as 0.

With `--beta 0`, P2/P3 reduce to B0V for the weighted loss and recovered ids in the unit tests.

## Development ablation

`--mode dev-ablation` runs the requested Skytrax-28 seed42 layer17 development grid in the same branch:

- baselines: B0, B0V, P1;
- P2/P3 grid: beta 0.25/0.50/0.75, power 0.25/0.50/1.00, weight_max 2/4;
- fixed settings: epoch 100, K=10, Y=10, residual rollout on, rollout depth all, weight_min 0.25;
- auto-batch is capacity-aware, so `--max-jobs-per-gpu 1` keeps one active task per GPU;
- outputs: `dev_ablation.csv`, `dev_ablation.md`, `dev_weight_summary.csv`, `dev_token_accuracy.png`, and `dev_weight_concentration.png`.

## Resource-aware batching

The new runner adds `--mode plan-batch` and `--auto-batch`. It reads current GPU free memory with `nvidia-smi`, reserves a safety buffer, and schedules concurrent single-prompt optimization workers only on GPUs with enough free memory.

Current conservative defaults:

- `--batch-free-mb-per-job 6144`
- `--batch-reserve-free-mb 2048`
- `--max-jobs-per-gpu 2`
- `--max-parallel-jobs 3`

Under the checked server state, this selects two workers: GPU0 and GPU3. This shortens wall-clock time while avoiding the tighter-memory GPU1/GPU2.

## Required checks

- Unit tests cover beta=0 equivalence, recovered ids under beta=0, fixed/special zero weights, max clipping, last-window variable-only query positions, and B0 all-valid loss.
- Leakage verification checks that the attack API does not receive prompt/original ids, manual server forward matches full-model forward, and no dummy attention is used.
- Current pilot smoke is Skytrax-28, dataset_len=2, seed=42, layer=17, epoch=100, K=10, Y=10, methods B0/B0V/P1/P2/P3.
"""
    Path("analysis/masked_server_attn_design.md").write_text(design, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "download-skytrax",
            "verify",
            "single",
            "multi-layer",
            "dev-ablation",
            "heldout-seed",
            "summarize",
            "summarize-dev",
            "summarize-heldout",
            "write-design",
            "plan-batch",
        ],
        default="single",
    )
    parser.add_argument("--method", choices=METHODS, default="P1")
    parser.add_argument("--methods", choices=METHODS, nargs="*", default=["B0", "B0V", "P1", "P2", "P3"])
    parser.add_argument("--output-root", default="runs/masked_server_attn_pia")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-name", default="Skytrax-150")
    parser.add_argument("--dataset-path", default="data/skytrax_150.json")
    parser.add_argument("--dataset-len", type=int, default=150)
    parser.add_argument("--dataset-out", default="data/skytrax_150.json")
    parser.add_argument("--dataset-size", type=int, default=150)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--target-layers", type=int, nargs="*", default=[11, 17, 19])
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--stage-a-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.1)
    parser.add_argument("--lambda-dummy", type=float, default=0.1)
    parser.add_argument("--lambda-context", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--y", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--weight-source", choices=["last_query_raw", "mean_query", "last_window_mean", "uniform"], default="mean_query")
    parser.add_argument("--weight-floor", type=float, default=0.05)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--weight-min", type=float, default=0.25)
    parser.add_argument("--weight-max", type=float, default=4.0)
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--attention-start-ratio", type=float, default=0.7)
    parser.add_argument("--attention-full-ratio", type=float, default=0.7)
    parser.add_argument("--uncertainty-fraction", type=float, default=0.3)
    parser.add_argument("--adaptive-min-beta", type=float, default=0.05)
    parser.add_argument("--refine-epoch", type=int, default=10)
    parser.add_argument("--lambda-projection", type=float, default=0.1)
    parser.add_argument("--refine-lr-scale", type=float, default=0.25)
    parser.add_argument("--residual-alpha-rho", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--last-window-size", type=int, default=8)
    parser.add_argument("--no-residual-rollout", action="store_true")
    parser.add_argument("--server-rollout-depth", choices=["all", "last2"], default="all")
    parser.add_argument("--max-token-len", type=int, default=896)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--naive-discretization", action="store_true")
    parser.add_argument("--disable-semantic-speculation", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--dry-run-batch", action="store_true")
    parser.add_argument("--batch-free-mb-per-job", type=int, default=6144)
    parser.add_argument("--batch-reserve-free-mb", type=int, default=2048)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=2)
    parser.add_argument("--max-parallel-jobs", type=int, default=3)
    parser.add_argument("--ablation-betas", type=float, nargs="*", default=[0.25, 0.50, 0.75])
    parser.add_argument("--ablation-weight-powers", type=float, nargs="*", default=[0.25, 0.50, 1.00])
    parser.add_argument("--ablation-weight-maxes", type=float, nargs="*", default=[2.0, 4.0])
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--transformers-cache", default=os.environ.get("TRANSFORMERS_CACHE", "/home/zhiqi/data/hf_cache/transformers"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.method = canonical_method(args.method)
    args.methods = [canonical_method(method) for method in args.methods]
    args.semantic_speculation = not args.disable_semantic_speculation
    validate_strict_top1_for_mode(args)
    if args.transformers_cache:
        os.environ.setdefault("TRANSFORMERS_CACHE", args.transformers_cache)
    if args.mode == "download-skytrax":
        download_skytrax_150(args)
    elif args.mode == "verify":
        write_design()
        verify_no_leakage(args)
    elif args.mode == "single":
        write_design()
        run_single(args)
        write_summary(Path(args.output_root))
    elif args.mode == "multi-layer":
        write_design()
        if args.auto_batch:
            run_multi_layer_auto_batch(args)
        else:
            run_multi_layer(args)
    elif args.mode == "dev-ablation":
        write_design()
        run_dev_ablation(args)
    elif args.mode == "heldout-seed":
        run_heldout_seed(args)
    elif args.mode == "summarize":
        write_design()
        write_summary(Path(args.output_root))
    elif args.mode == "summarize-dev":
        write_design()
        write_summary(Path(args.output_root))
        write_structure_ablation_outputs(Path(args.output_root))
        write_dev_ablation_outputs(Path(args.output_root))
    elif args.mode == "summarize-heldout":
        write_design()
        write_heldout_validation_plan(Path(args.output_root))
        for seed in args.seeds:
            seed_dir = Path(args.output_root) / f"heldout_seed{seed}"
            if seed_dir.exists():
                write_seed_paired_summary(Path(args.output_root), seed, bootstrap_iters=args.bootstrap_iters)
        write_heldout_summary(Path(args.output_root), args.seeds)
    elif args.mode == "write-design":
        write_design()
    elif args.mode == "plan-batch":
        print_auto_batch_plan(args)


if __name__ == "__main__":
    main()
