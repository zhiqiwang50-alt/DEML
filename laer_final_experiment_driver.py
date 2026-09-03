import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "paper_final"
SPLIT_PATH = ROOT / "runs" / "suffix_attention_edge_rerank_pia" / "splits" / "skytrax150_split_20260706.json"
DATASET_PATH = ROOT / "data" / "skytrax_150.json"
DEV_DATASET_PATH = ROOT / "data" / "airline.json"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
METHODS = ["B0", "B0_SPARSE", "B0_LAER", "B0_SHUFFLED_LAER"]
METHOD_DIR = {
    "B0": "b0",
    "B0_SPARSE": "b0_sparse",
    "B0_LAER": "b0_laer",
    "B0_SHUFFLED_LAER": "b0_shuffled_laer",
}
MAIN_SEEDS = [42, 43, 44]
MAIN_LAYER = 17
ROBUSTNESS_LAYERS = [11, 19]
ROBUSTNESS_SEEDS = [42]
PROMPT_SPLIT_SEED = 20260706
HOLDOUT_COUNT = 100
FULL_DATASET_LEN = 150
BOOTSTRAP_ROUNDS = 10000
GPU_FREE_MB_THRESHOLD = 22000

HYPERPARAMETERS = {
    "epoch": 100,
    "lr": 0.08,
    "lambda_vocab": 0.1,
    "grad_clip": 1.0,
    "top_k_embedding": 1,
    "top_y_semantic": 0,
    "semantic_speculation": False,
    "adaptive_discretization": False,
    "uncertainty_detector": "margin",
    "uncertainty_fraction": 0.20,
    "patch_size": 16,
    "max_repair_passes": 2,
    "eps_cut": 1e-6,
    "query_window": 64,
    "attn_row_fraction": 0.20,
    "max_attn_rows": 128,
    "entropy_middle_fraction": 0.80,
    "public_sink_threshold": 0.50,
}

RELEVANT_CODE_FILES = [
    "laer_final_experiment_driver.py",
    "pia_b0_laer.py",
    "pia_b0_attention_consistency_repair.py",
    "pia_attention_validation_audit.py",
    "pia_masked_server_attn_pia.py",
    "pia_tinyllama.py",
    "scripts/run_b0_laer.sh",
    "tests/test_b0_laer.py",
    "tests/test_b0_attention_consistency_repair.py",
    "tests/test_attention_validation_audit.py",
]


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(path: Path, default: Any) -> Any:
    try:
        return load_json(path)
    except Exception:
        return default


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def csv_write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def normalize_prompt(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").replace("\t", " ").split())


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def read_filtered_prompts(path: Path, seed: int, dataset_len: int) -> List[str]:
    prompts = load_json(path)
    filtered = []
    for text in prompts:
        norm = normalize_prompt(text)
        if len(norm) < 20:
            continue
        if not norm.isascii():
            continue
        filtered.append(norm)
    random.Random(seed).shuffle(filtered)
    return filtered[:dataset_len]


def split_payload() -> Dict[str, Any]:
    return load_json(SPLIT_PATH)


def holdout_ids() -> List[int]:
    ids = [int(x) for x in split_payload()["holdout_ids"]]
    if len(ids) != HOLDOUT_COUNT:
        raise RuntimeError(f"expected {HOLDOUT_COUNT} holdout ids, got {len(ids)}")
    return ids


def git_output(args: Sequence[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def run_matrix() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in MAIN_SEEDS:
        for method in METHODS:
            rows.append({"phase": "main", "layer": MAIN_LAYER, "seed": seed, "method": method, "prompt_count": HOLDOUT_COUNT})
    for layer in ROBUSTNESS_LAYERS:
        for seed in ROBUSTNESS_SEEDS:
            for method in METHODS:
                rows.append({"phase": "layer_robustness", "layer": layer, "seed": seed, "method": method, "prompt_count": HOLDOUT_COUNT})
    return rows


def run_subdir(row: Dict[str, Any]) -> str:
    return f"{row['phase']}/layer{row['layer']}/seed{row['seed']}"


def run_dir(row: Dict[str, Any]) -> Path:
    return RUN_ROOT / run_subdir(row) / METHOD_DIR[row["method"]]


def command_for(row: Dict[str, Any], *, prompt_limit: int = 0, epoch: Optional[int] = None, resume: bool = True) -> List[str]:
    cmd = [
        sys.executable,
        "pia_b0_laer.py",
        "--mode",
        "single",
        "--method",
        row["method"],
        "--output-root",
        str(RUN_ROOT.relative_to(ROOT)),
        "--run-subdir",
        run_subdir(row),
        "--dataset-name",
        "Skytrax",
        "--dataset-path",
        str(DATASET_PATH.relative_to(ROOT)),
        "--dataset-len",
        str(HOLDOUT_COUNT if prompt_limit <= 0 else prompt_limit),
        "--full-dataset-len",
        str(FULL_DATASET_LEN),
        "--split-path",
        str(SPLIT_PATH.relative_to(ROOT)),
        "--split-name",
        "holdout",
        "--prompt-split-seed",
        str(PROMPT_SPLIT_SEED),
        "--prompt-limit",
        str(prompt_limit),
        "--seed",
        str(row["seed"]),
        "--target-layer",
        str(row["layer"]),
        "--epoch",
        str(epoch if epoch is not None else HYPERPARAMETERS["epoch"]),
        "--lr",
        str(HYPERPARAMETERS["lr"]),
        "--lambda-vocab",
        str(HYPERPARAMETERS["lambda_vocab"]),
        "--max-token-len",
        "896",
        "--grad-clip",
        str(HYPERPARAMETERS["grad_clip"]),
        "--uncertainty-fraction",
        str(HYPERPARAMETERS["uncertainty_fraction"]),
        "--patch-size",
        str(HYPERPARAMETERS["patch_size"]),
        "--max-repair-passes",
        str(HYPERPARAMETERS["max_repair_passes"]),
        "--eps-cut",
        str(HYPERPARAMETERS["eps_cut"]),
        "--query-window",
        str(HYPERPARAMETERS["query_window"]),
        "--attn-row-fraction",
        str(HYPERPARAMETERS["attn_row_fraction"]),
        "--max-attn-rows",
        str(HYPERPARAMETERS["max_attn_rows"]),
        "--entropy-middle-fraction",
        str(HYPERPARAMETERS["entropy_middle_fraction"]),
        "--public-sink-threshold",
        str(HYPERPARAMETERS["public_sink_threshold"]),
        "--local-files-only",
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def run_command(cmd: Sequence[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {
        "cmd": list(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "runtime_seconds": time.time() - start,
    }


def create_provenance() -> Dict[str, Any]:
    split = split_payload()
    hids = holdout_ids()
    canonical = read_filtered_prompts(DATASET_PATH, PROMPT_SPLIT_SEED, FULL_DATASET_LEN)
    holdout_prompts = [canonical[idx] for idx in hids]
    holdout_hashes = {prompt_hash(text) for text in holdout_prompts}
    dev_prompts = read_filtered_prompts(DEV_DATASET_PATH, 42, 10000)
    dev_hashes = {prompt_hash(text) for text in dev_prompts}
    exposure_hits = []
    for cfg_path in (ROOT / "runs").glob("**/config.json"):
        if "paper_final" in cfg_path.parts:
            continue
        try:
            cfg = load_json(cfg_path)
        except Exception:
            continue
        method = str(cfg.get("method", "")).lower()
        if method not in {"b0_sparse", "b0_laer", "b0_shuffled_laer"}:
            continue
        meta_path = cfg_path.with_name("dataset_meta.json")
        selected_ids: List[int] = []
        if meta_path.exists():
            try:
                meta = load_json(meta_path)
                selected_ids = [int(x) for x in meta.get("selected_prompt_ids", [])]
            except Exception:
                selected_ids = []
        dataset_path = str(cfg.get("dataset_path", ""))
        if selected_ids:
            overlap = sorted(set(selected_ids) & set(hids))
            if overlap:
                exposure_hits.append({"config": str(cfg_path.relative_to(ROOT)), "method": method, "overlap_holdout_ids": overlap[:20], "overlap_count": len(overlap)})
        elif dataset_path.endswith("skytrax_150.json"):
            exposure_hits.append({"config": str(cfg_path.relative_to(ROOT)), "method": method, "overlap_count": "unknown_dataset_level", "reason": "previous run used skytrax_150 without explicit selected_prompt_ids"})
    status = "PREVIOUSLY_EXPOSED" if exposure_hits else "CLEAN"
    payload = {
        "timestamp": now_stamp(),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "split_sha256": sha256_file(SPLIT_PATH),
        "split_path": str(SPLIT_PATH.relative_to(ROOT)),
        "holdout_count": len(hids),
        "holdout_ids": hids,
        "dev_dataset_path": str(DEV_DATASET_PATH.relative_to(ROOT)),
        "dev_overlap_count": len(holdout_hashes & dev_hashes),
        "holdout_status": status,
        "exposure_hits": exposure_hits,
        "split_payload_summary": {k: split[k] for k in ["dataset_name", "dataset_path", "dataset_len", "prompt_split_seed", "note"] if k in split},
    }
    json_dump(RUN_ROOT / "paper_provenance_audit.json", payload)
    json_dump(ROOT / "paper_final" / "provenance_audit.json", payload)
    return payload


def create_preregistration(force: bool = False) -> Dict[str, Any]:
    path = RUN_ROOT / "PRE_REGISTRATION.json"
    if path.exists() and not force:
        return load_json(path)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    provenance = create_provenance()
    payload = {
        "experiment_timestamp": now_stamp(),
        "git_status_short": git_output(["status", "--short"]),
        "git_commit_hash": git_output(["rev-parse", "HEAD"]),
        "code_sha256": {name: sha256_file(ROOT / name) for name in RELEVANT_CODE_FILES if (ROOT / name).exists()},
        "dataset_sha256": provenance["dataset_sha256"],
        "split_sha256": provenance["split_sha256"],
        "model": MODEL_NAME,
        "methods": METHODS,
        "paper_style_reference": {
            "status": "UNAVAILABLE_REFERENCE",
            "reason": "The repository contains PIA-style Top-K/semantic/calibration code paths, but no frozen, audited paper-style Skytrax-150 holdout reference configuration is pre-registered here.",
        },
        "main_experiment": {"layer": MAIN_LAYER, "seeds": MAIN_SEEDS, "holdout_prompt_count": HOLDOUT_COUNT},
        "split_layer_robustness": {"layers": ROBUSTNESS_LAYERS, "seeds": ROBUSTNESS_SEEDS, "holdout_prompt_count": HOLDOUT_COUNT},
        "hyperparameters": HYPERPARAMETERS,
        "primary_metric": "Token Accuracy",
        "secondary_metrics": ["BLEU", "NED", "Exact Match"],
        "primary_hypothesis": "H1: B0-SPARSE improves Token Accuracy over B0 on Skytrax-150 holdout, layer17, seeds 42/43/44.",
        "secondary_hypotheses": {
            "H2": "B0-LAER > B0-SPARSE",
            "H3": "B0-LAER > B0-SHUFFLED-LAER",
            "H4": "B0-SPARSE remains positive over B0 on layer11 and layer19.",
        },
        "success_criteria": {
            "primary": "B0-SPARSE minus B0 Token Accuracy mean delta > 0 and prompt-cluster bootstrap 95% CI lower bound > 0.",
            "attention": "B0-LAER must exceed both B0-SPARSE and B0-SHUFFLED-LAER with positive statistical evidence before claiming true server attention adds reconstruction benefit.",
        },
        "run_matrix": run_matrix(),
        "holdout_status": provenance["holdout_status"],
    }
    json_dump(path, payload)
    json_dump(RUN_ROOT / "paper_experiment_manifest.json", payload)
    return payload


def integrity_checks() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    compile_files = ["pia_b0_laer.py", "pia_b0_attention_consistency_repair.py", "pia_attention_validation_audit.py", "pia_tinyllama.py", "laer_final_experiment_driver.py"]
    checks.append({"name": "py_compile", **run_command([sys.executable, "-m", "py_compile", *compile_files])})
    checks.append({"name": "unit_tests_b0_laer", **run_command([sys.executable, "-m", "unittest", "tests.test_b0_laer", "-v"])})
    checks.append({"name": "unit_tests_acdr", **run_command([sys.executable, "-m", "unittest", "tests.test_b0_attention_consistency_repair", "-v"])})
    checks.append({"name": "unit_tests_attention_validation", **run_command([sys.executable, "-m", "unittest", "tests.test_attention_validation_audit", "-v"])})
    verify_cmd = [
        sys.executable,
        "pia_b0_attention_consistency_repair.py",
        "--mode",
        "verify",
        "--dataset-path",
        str(DATASET_PATH.relative_to(ROOT)),
        "--dataset-len",
        "1",
        "--seed",
        str(PROMPT_SPLIT_SEED),
        "--target-layer",
        str(MAIN_LAYER),
        "--epoch",
        "2",
        "--k",
        "1",
        "--y",
        "0",
        "--disable-semantic-speculation",
        "--no-adaptive-discretization",
        "--local-files-only",
        "--output-root",
        str((RUN_ROOT / "integrity").relative_to(ROOT)),
    ]
    checks.append({"name": "attack_api_leakage_boundary_audit", **run_command(verify_cmd)})
    payload = {
        "timestamp": now_stamp(),
        "checks": checks,
        "passed": all(int(row["returncode"]) == 0 for row in checks),
        "strict_top1_expected": True,
        "q_gt_k_legality_covered_by": "tests.test_b0_laer.test_laer_edges_are_causal_and_no_fixed_keys",
        "fixed_public_token_audit_covered_by": "tests.test_b0_laer.test_laer_edges_are_causal_and_no_fixed_keys",
        "state_persistence_covered_by": "tests.test_b0_laer.test_state_persistence_and_rollback_are_explicit",
    }
    json_dump(RUN_ROOT / "paper_integrity_audit.json", payload)
    if not payload["passed"]:
        raise RuntimeError("integrity checks failed; see runs/paper_final/paper_integrity_audit.json")
    return payload


def smoke() -> Dict[str, Any]:
    rows = []
    for method in METHODS:
        row = {"phase": "smoke", "layer": MAIN_LAYER, "seed": 42, "method": method, "prompt_count": 1}
        d = RUN_ROOT / "smoke" / "layer17" / "seed42" / METHOD_DIR[method]
        d.mkdir(parents=True, exist_ok=True)
        cmd = command_for(row, prompt_limit=1, epoch=2, resume=True)
        d.joinpath("command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        with d.joinpath("stdout.log").open("a", encoding="utf-8") as out, d.joinpath("stderr.log").open("a", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=out, stderr=err)
        rows.append({"method": method, "returncode": proc.returncode, "dir": str(d.relative_to(ROOT))})
    payload = {
        "timestamp": now_stamp(),
        "purpose": "Pipeline smoke only; Acc/BLEU are not used for method or parameter decisions.",
        "rows": rows,
        "passed": all(row["returncode"] == 0 for row in rows),
    }
    json_dump(RUN_ROOT / "paper_smoke_audit.json", payload)
    if not payload["passed"]:
        raise RuntimeError("smoke failed; see runs/paper_final/paper_smoke_audit.json")
    return payload


def gpu_free_mb() -> Dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    free = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            free[int(parts[0])] = int(parts[1])
    return free


def complete_valid(row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    d = run_dir(row)
    pred = load_jsonl(d / "predictions.jsonl")
    ids = [int(r.get("prompt_id", -1)) for r in pred]
    metrics = try_load_json(d / "metrics.json", {}) if (d / "metrics.json").exists() else {}
    failures = load_jsonl(d / "failures.jsonl")
    audit_counts = {}
    for name in ["variable_mask_audit.json", "patch_repair_stats.json", "acceptance_gate_stats.json", "loss_breakdown.json"]:
        p = d / name
        audit_payload = try_load_json(p, {}) if p.exists() else {}
        audit_counts[name] = len(audit_payload.get("samples", [])) if audit_payload else None
    info = {
        "dir": str(d.relative_to(ROOT)),
        "complete_marker": (d / "COMPLETE").exists(),
        "predictions": len(pred),
        "unique_prompt_ids": len(set(ids)),
        "duplicates": len(ids) - len(set(ids)),
        "completed_sample_count": metrics.get("completed_sample_count"),
        "failed_sample_count": metrics.get("failed_sample_count"),
        "failure_lines": len(failures),
        "has_nan": metrics.get("has_nan"),
        "audit_counts": audit_counts,
    }
    ok = (
        info["complete_marker"]
        and info["predictions"] == HOLDOUT_COUNT
        and info["unique_prompt_ids"] == HOLDOUT_COUNT
        and info["duplicates"] == 0
        and info["completed_sample_count"] == HOLDOUT_COUNT
        and info["failed_sample_count"] == 0
        and info["failure_lines"] == 0
        and info["has_nan"] is False
        and all(v == HOLDOUT_COUNT for v in audit_counts.values())
    )
    return ok, info


def active_pia_commands() -> List[str]:
    try:
        out = subprocess.check_output(["pgrep", "-af", "pia_b0_laer.py"], text=True)
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if "laer_final_experiment_driver.py" not in line]


def row_running_external(row: Dict[str, Any]) -> bool:
    method_marker = f"--method {row['method']}"
    subdir_marker = f"--run-subdir {run_subdir(row)}"
    return any(method_marker in cmd and subdir_marker in cmd for cmd in active_pia_commands())


def launch_scheduler(max_parallel: int = 4, poll_seconds: int = 120) -> Dict[str, Any]:
    create_preregistration()
    tasks = run_matrix()
    active: Dict[int, Dict[str, Any]] = {}
    completed: List[Dict[str, Any]] = []
    while True:
        for pid, item in list(active.items()):
            proc = item["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            item["stdout"].close()
            item["stderr"].close()
            row = item["row"]
            ok, info = complete_valid(row)
            completed.append({**row, "returncode": rc, "valid_complete": ok, **info})
            del active[pid]
        remaining = []
        for row in tasks:
            ok, _info = complete_valid(row)
            if ok:
                continue
            if any(item["row"] == row for item in active.values()):
                continue
            if row_running_external(row):
                continue
            remaining.append(row)
        if not remaining and not active:
            break
        free = gpu_free_mb()
        used_gpus = {int(item["gpu"]) for item in active.values()}
        available = [gpu for gpu, mb in sorted(free.items()) if gpu not in used_gpus and mb >= GPU_FREE_MB_THRESHOLD]
        while remaining and available and len(active) < max_parallel:
            row = remaining.pop(0)
            gpu = available.pop(0)
            d = run_dir(row)
            d.mkdir(parents=True, exist_ok=True)
            cmd = command_for(row, resume=True)
            d.joinpath("command.txt").write_text("CUDA_VISIBLE_DEVICES={} {}\n".format(gpu, " ".join(cmd)), encoding="utf-8")
            stdout = d.joinpath("stdout.log").open("a", encoding="utf-8")
            stderr = d.joinpath("stderr.log").open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, text=True)
            active[proc.pid] = {"proc": proc, "row": row, "gpu": gpu, "stdout": stdout, "stderr": stderr, "started_at": now_stamp()}
        status = {"timestamp": now_stamp(), "active": [{"pid": pid, "gpu": item["gpu"], **item["row"]} for pid, item in active.items()], "completed_observed": completed}
        json_dump(RUN_ROOT / "paper_driver_status.json", status)
        if not active and remaining and not available:
            time.sleep(poll_seconds)
        elif active:
            time.sleep(poll_seconds)
    final_status = status_report()
    json_dump(RUN_ROOT / "paper_driver_status.json", final_status)
    return final_status


def status_report() -> Dict[str, Any]:
    rows = []
    for row in run_matrix():
        ok, info = complete_valid(row)
        rows.append({**row, "valid_complete": ok, **info})
    payload = {
        "timestamp": now_stamp(),
        "runs": rows,
        "complete_count": sum(1 for r in rows if r["valid_complete"]),
        "total_count": len(rows),
        "all_complete": all(r["valid_complete"] for r in rows),
    }
    json_dump(RUN_ROOT / "paper_completeness_audit.json", payload)
    return payload


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(vals) if vals else None


def load_rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return load_jsonl(run_dir(row) / "predictions.jsonl")


def summarize_run(row: Dict[str, Any]) -> Dict[str, Any]:
    d = run_dir(row)
    metrics = load_json(d / "metrics.json")
    return {
        **row,
        "method_dir": METHOD_DIR[row["method"]],
        "token_accuracy": metrics.get("token_accuracy", {}).get("mean"),
        "bleu": metrics.get("bleu", {}).get("mean"),
        "ned": metrics.get("ned", {}).get("mean"),
        "exact_match": metrics.get("exact_match", {}).get("mean"),
        "runtime_mean": metrics.get("runtime_seconds", {}).get("mean"),
        "runtime_std": metrics.get("runtime_seconds", {}).get("std"),
        "peak_gpu_memory_mean": metrics.get("peak_gpu_memory_mb", {}).get("mean"),
        "completed_sample_count": metrics.get("completed_sample_count"),
        "failed_sample_count": metrics.get("failed_sample_count"),
        "has_nan": metrics.get("has_nan"),
    }


def paired_delta(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    aa = {int(r["prompt_id"]): r for r in a}
    bb = {int(r["prompt_id"]): r for r in b}
    pids = sorted(set(aa) & set(bb))
    acc = [float(aa[i]["token_accuracy"]) - float(bb[i]["token_accuracy"]) for i in pids]
    bleu = [float(aa[i]["bleu"]) - float(bb[i]["bleu"]) for i in pids]
    fix = harm = 0
    for pid in pids:
        ar, br = aa[pid], bb[pid]
        gold = [int(x) for x in ar.get("original_token_ids", [])]
        ai = [int(x) for x in ar.get("recovered_token_ids", [])]
        bi = [int(x) for x in br.get("recovered_token_ids", [])]
        for idx, gid in enumerate(gold):
            aok = idx < len(ai) and ai[idx] == gid
            bok = idx < len(bi) and bi[idx] == gid
            if aok and not bok:
                fix += 1
            elif bok and not aok:
                harm += 1
    return {
        "paired_prompt_count": len(pids),
        "mean_acc_delta": mean(acc),
        "mean_bleu_delta": mean(bleu),
        "win_count": sum(1 for x in acc if x > 1e-12),
        "tie_count": sum(1 for x in acc if abs(x) <= 1e-12),
        "loss_count": sum(1 for x in acc if x < -1e-12),
        "fix_count": fix,
        "harm_count": harm,
        "net_fix_count": fix - harm,
        "acc_deltas_by_prompt": dict(zip(pids, acc)),
        "bleu_deltas_by_prompt": dict(zip(pids, bleu)),
    }


def bootstrap_ci(values: Sequence[float], seed: int) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values]
    if not vals:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_ROUNDS):
        draws.append(statistics.mean(vals[rng.randrange(len(vals))] for _i in vals))
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def prompt_cluster_bootstrap(delta_by_seed: Dict[int, Dict[int, float]], seed: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    prompts = sorted(set.intersection(*(set(v) for v in delta_by_seed.values()))) if delta_by_seed else []
    if not prompts:
        return None, None, None
    per_prompt = {pid: statistics.mean(delta_by_seed[s][pid] for s in sorted(delta_by_seed)) for pid in prompts}
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = [prompts[rng.randrange(len(prompts))] for _i in prompts]
        draws.append(statistics.mean(per_prompt[pid] for pid in sample))
    draws.sort()
    return statistics.mean(per_prompt.values()), draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def patch_mechanism_for(row: Dict[str, Any]) -> Dict[str, Any]:
    samples = load_json(run_dir(row) / "acceptance_gate_stats.json").get("samples", [])
    events = []
    for sample in samples:
        events.extend(sample.get("events", []))
    accepted = [e for e in events if e.get("accepted")]
    rejected = [e for e in events if not e.get("accepted")]
    fix = sum(int(e.get("gt_beneficial", 0)) for e in accepted)
    harm = sum(int(e.get("gt_harmful", 0)) for e in accepted)
    rejected_fix = sum(int(e.get("gt_beneficial", 0)) for e in rejected)
    rejected_harm = sum(int(e.get("gt_harmful", 0)) for e in rejected)
    return {
        **row,
        "patch_event_count": len(events),
        "accepted_patch_count": len(accepted),
        "rejected_patch_count": len(rejected),
        "accept_rate": len(accepted) / len(events) if events else None,
        "accepted_fix": fix,
        "accepted_harm": harm,
        "accepted_net_fix": fix - harm,
        "repair_precision": fix / (fix + harm) if (fix + harm) > 0 else None,
        "rejected_beneficial": rejected_fix,
        "rejected_harmful": rejected_harm,
        "laer_rejected_count": sum(1 for e in rejected if e.get("rejected_by_laer")),
        "harm_rejection_rate": rejected_harm / (rejected_harm + harm) if (rejected_harm + harm) > 0 else None,
        "beneficial_retention_rate": fix / (fix + rejected_fix) if (fix + rejected_fix) > 0 else None,
        "illegal_q_le_k_total": sum(int(e.get("laer_illegal_q_le_k_count", 0)) for e in events),
        "fixed_public_key_hit_total": sum(int(e.get("laer_fixed_public_key_hit_count", 0)) for e in events),
    }


def summarize() -> Dict[str, Any]:
    status = status_report()
    if not status["all_complete"]:
        raise RuntimeError("cannot summarize until all paper-final runs are valid complete")
    seed_rows = [summarize_run(r) for r in run_matrix()]
    csv_write(RUN_ROOT / "paper_seed_results.csv", seed_rows)

    main_rows = [r for r in seed_rows if r["phase"] == "main" and r["layer"] == MAIN_LAYER]
    main_summary = []
    for method in METHODS:
        rows = [r for r in main_rows if r["method"] == method]
        main_summary.append({
            "method": method,
            "layer": MAIN_LAYER,
            "seed_count": len(rows),
            "prompt_runs": len(rows) * HOLDOUT_COUNT,
            "token_accuracy": mean(r["token_accuracy"] for r in rows),
            "bleu": mean(r["bleu"] for r in rows),
            "ned": mean(r["ned"] for r in rows),
            "exact_match": mean(r["exact_match"] for r in rows),
        })
    csv_write(RUN_ROOT / "paper_main_results.csv", main_summary)

    layer_rows = []
    for layer in [11, 17, 19]:
        for method in METHODS:
            rows = [r for r in seed_rows if r["layer"] == layer and r["seed"] == 42 and r["method"] == method]
            if rows:
                layer_rows.append(rows[0])
    csv_write(RUN_ROOT / "paper_layer_results.csv", layer_rows)

    comparisons = [
        ("B0_SPARSE", "B0"),
        ("B0_LAER", "B0"),
        ("B0_LAER", "B0_SPARSE"),
        ("B0_LAER", "B0_SHUFFLED_LAER"),
    ]
    paired_rows = []
    cluster_rows = []
    for a, b in comparisons:
        delta_by_seed_acc: Dict[int, Dict[int, float]] = {}
        delta_by_seed_bleu: Dict[int, Dict[int, float]] = {}
        for seed in MAIN_SEEDS:
            ar = {"phase": "main", "layer": MAIN_LAYER, "seed": seed, "method": a, "prompt_count": HOLDOUT_COUNT}
            br = {"phase": "main", "layer": MAIN_LAYER, "seed": seed, "method": b, "prompt_count": HOLDOUT_COUNT}
            out = paired_delta(load_rows(ar), load_rows(br))
            acc_ci = bootstrap_ci(list(out["acc_deltas_by_prompt"].values()), 1000 + seed)
            bleu_ci = bootstrap_ci(list(out["bleu_deltas_by_prompt"].values()), 2000 + seed)
            paired_rows.append({
                "scope": f"layer17_seed{seed}",
                "comparison": f"{a}_vs_{b}",
                **{k: v for k, v in out.items() if not k.endswith("_by_prompt")},
                "acc_ci_low": acc_ci[0],
                "acc_ci_high": acc_ci[1],
                "bleu_ci_low": bleu_ci[0],
                "bleu_ci_high": bleu_ci[1],
            })
            delta_by_seed_acc[seed] = out["acc_deltas_by_prompt"]
            delta_by_seed_bleu[seed] = out["bleu_deltas_by_prompt"]
        acc_cluster = prompt_cluster_bootstrap(delta_by_seed_acc, 3000 + len(cluster_rows))
        bleu_cluster = prompt_cluster_bootstrap(delta_by_seed_bleu, 4000 + len(cluster_rows))
        cluster_rows.append({
            "scope": "layer17_seed42_43_44_prompt_cluster",
            "comparison": f"{a}_vs_{b}",
            "mean_acc_delta": acc_cluster[0],
            "acc_ci_low": acc_cluster[1],
            "acc_ci_high": acc_cluster[2],
            "mean_bleu_delta": bleu_cluster[0],
            "bleu_ci_low": bleu_cluster[1],
            "bleu_ci_high": bleu_cluster[2],
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        })
    csv_write(RUN_ROOT / "paper_paired_comparisons.csv", paired_rows)
    csv_write(RUN_ROOT / "paper_cluster_bootstrap.csv", cluster_rows)

    patch_rows = [patch_mechanism_for(r) for r in run_matrix()]
    csv_write(RUN_ROOT / "paper_patch_mechanism.csv", patch_rows)
    csv_write(RUN_ROOT / "paper_attention_control.csv", [r for r in paired_rows if "B0_LAER_vs_B0_SHUFFLED_LAER" in r["comparison"]] + [r for r in cluster_rows if "B0_LAER_vs_B0_SHUFFLED_LAER" in r["comparison"]])
    csv_write(RUN_ROOT / "paper_efficiency.csv", [{
        "method": method,
        "mean_runtime_per_prompt": mean(r["runtime_mean"] for r in seed_rows if r["method"] == method),
        "std_runtime_per_prompt": mean(r["runtime_std"] for r in seed_rows if r["method"] == method),
        "mean_peak_gpu_memory_mb": mean(r["peak_gpu_memory_mean"] for r in seed_rows if r["method"] == method),
    } for method in METHODS])
    csv_write(RUN_ROOT / "paper_failure_summary.csv", [{
        **r,
        "failure_lines": len(load_jsonl(run_dir(r) / "failures.jsonl")),
    } for r in run_matrix()])

    primary = next(r for r in cluster_rows if r["comparison"] == "B0_SPARSE_vs_B0")
    laer_sparse = next(r for r in cluster_rows if r["comparison"] == "B0_LAER_vs_B0_SPARSE")
    laer_shuf = next(r for r in cluster_rows if r["comparison"] == "B0_LAER_vs_B0_SHUFFLED_LAER")
    if primary["mean_acc_delta"] > 0 and primary["acc_ci_low"] > 0:
        if laer_sparse["mean_acc_delta"] > 0 and laer_sparse["acc_ci_low"] > 0 and laer_shuf["mean_acc_delta"] > 0 and laer_shuf["acc_ci_low"] > 0:
            conclusion = "B"
        else:
            conclusion = "A"
    else:
        conclusion = "C"

    md_lines = [
        "# Paper Final Report",
        "",
        "## Observation",
        "",
        f"Main layer17 pooled methods: {main_summary}.",
        "",
        "## Statistical Evidence",
        "",
    ]
    for row in cluster_rows:
        md_lines.append(f"- {row['comparison']}: Acc delta {row['mean_acc_delta']}, 95% CI [{row['acc_ci_low']}, {row['acc_ci_high']}]; BLEU delta {row['mean_bleu_delta']}, 95% CI [{row['bleu_ci_low']}, {row['bleu_ci_high']}].")
    md_lines += [
        "",
        "## Mechanism Evidence",
        "",
        "Patch-level statistics are saved in `paper_patch_mechanism.csv`. Ground truth labels are computed only after each attack run completes.",
        "",
        "## Robustness",
        "",
        "Layer-wise seed42 results for layers 11/17/19 are saved in `paper_layer_results.csv`.",
        "",
        "## Efficiency",
        "",
        "Runtime and peak GPU memory summaries are saved in `paper_efficiency.csv`.",
        "",
        "## Supported Claims",
        "",
        "- Claims are limited by the pre-registered success criteria and the completed heldout results.",
        "",
        "## Unsupported Claims",
        "",
        "- Do not claim stable improvement beyond the frozen matrix.",
        "- Do not claim true attention contribution unless B0-LAER exceeds both B0-SPARSE and shuffled-LAER under the pre-registered criteria.",
        "",
        "## Final Paper Recommendation",
        "",
        f"Conclusion code: {conclusion}.",
    ]
    (RUN_ROOT / "paper_main_results.md").write_text("\n".join(md_lines[:10]) + "\n", encoding="utf-8")
    (RUN_ROOT / "paper_final_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (RUN_ROOT / "paper_final_tables.tex").write_text("% See CSV files for exact generated values.\n", encoding="utf-8")
    payload = {"conclusion": conclusion, "status": status, "main_results": main_summary, "cluster_bootstrap": cluster_rows}
    json_dump(RUN_ROOT / "paper_summary_payload.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-registered final paper experiment driver")
    parser.add_argument("--mode", choices=["preregister", "integrity", "smoke", "launch", "status", "summarize", "all"], default="status")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    if args.mode == "preregister":
        print(json.dumps(create_preregistration(), ensure_ascii=False, indent=2))
    elif args.mode == "integrity":
        print(json.dumps(integrity_checks(), ensure_ascii=False, indent=2))
    elif args.mode == "smoke":
        create_preregistration()
        print(json.dumps(smoke(), ensure_ascii=False, indent=2))
    elif args.mode == "launch":
        print(json.dumps(launch_scheduler(args.max_parallel, args.poll_seconds), ensure_ascii=False, indent=2))
    elif args.mode == "status":
        print(json.dumps(status_report(), ensure_ascii=False, indent=2))
    elif args.mode == "summarize":
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
    elif args.mode == "all":
        create_preregistration()
        integrity_checks()
        smoke()
        launch_scheduler(args.max_parallel, args.poll_seconds)
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
