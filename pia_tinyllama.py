import argparse
import csv
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask


EXPERIMENT_TITLE = "TinyLlama-adapted reproduction of Prompt Inversion Attack experiments"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_LAMBDA = 0.1
TOTAL_BLOCKS = 22

ATTACKER_MAP = {
    3: {2: (8, 7), 3: (15, 14)},
    4: {2: (6, 5), 3: (12, 11), 4: (18, 17)},
    5: {2: (5, 4), 3: (10, 9), 4: (15, 14), 5: (20, 19)},
}


@dataclass
class RunConfig:
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
    lr: float
    lambda_coeff: float
    top_k_embedding: int
    top_y_semantic: int
    constrained_optimization: bool
    adaptive_discretization: bool
    semantic_speculation: bool
    gaussian_noise_sigma: float
    quantization_bits: Optional[int]
    max_token_len: int
    compute_perplexity: bool
    mode: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def jsonl_append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_json_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            for key in ("content", "text", "prompt", "CONCLUSION"):
                if isinstance(item.get(key), str):
                    prompts.append(item[key])
                    break
    return prompts


def normalize_prompt(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\t", " ").split())


def load_dataset_prompts(dataset_name: str, dataset_path: str, dataset_len: int, seed: int) -> Tuple[List[str], Dict[str, Any]]:
    prompts = read_json_list(dataset_path)
    filtered = []
    for text in prompts:
        norm = normalize_prompt(text)
        if len(norm) < 20:
            continue
        if not norm.isascii():
            continue
        filtered.append(norm)
    rng = random.Random(seed)
    rng.shuffle(filtered)
    selected = filtered[:dataset_len]
    meta = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "available_valid_prompts": len(filtered),
        "requested_prompts": dataset_len,
        "selected_prompts": len(selected),
        "selection_seed": seed,
        "filtering": "ASCII English prompts, non-empty, normalized whitespace, length >= 20 characters",
    }
    return selected, meta


def first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"forward hook output did not contain a tensor, got {type(output)!r}")


def capture_activation(model: torch.nn.Module, block: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    captured: List[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        captured.append(first_tensor(output))

    handle = block.register_forward_hook(hook)
    try:
        _ = model(**inputs)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(
            "TinyLlama target block forward hook captured no activation. "
            "This usually means the wrong block object was selected or the model forward did not execute that block."
        )
    return captured[0]


def capture_prefix_activation(
    model: torch.nn.Module,
    target_layer: int,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input_ids is None and inputs_embeds is None:
        raise ValueError("input_ids or inputs_embeds is required")
    if inputs_embeds is None:
        inputs_embeds = model.get_input_embeddings()(input_ids)
    batch_size, seq_length, _ = inputs_embeds.shape
    device = inputs_embeds.device
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length), device=device, dtype=torch.long)
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length=0,
    )
    captured: List[torch.Tensor] = []
    block = model.model.layers[target_layer]

    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        captured.append(first_tensor(output))

    handle = block.register_forward_hook(hook)
    hidden_states = inputs_embeds
    try:
        for layer_index in range(target_layer + 1):
            layer_outputs = model.model.layers[layer_index](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(
            f"TinyLlama target block forward hook captured no activation for layer {target_layer}. "
            "The prefix forward did not execute the selected block."
        )
    return captured[0]


def load_tinyllama(dtype: torch.dtype, local_files_only: bool = True) -> Tuple[Any, torch.nn.Module]:
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=local_files_only)
    if config.model_type != "llama":
        raise ValueError(f"Expected TinyLlama model_type='llama', got {config.model_type!r}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


def args_local_files_only() -> bool:
    return os.environ.get("PIA_ALLOW_HF_DOWNLOAD") != "1"


def token_texts(tokenizer: Any, ids: Sequence[int]) -> List[str]:
    return tokenizer.convert_ids_to_tokens([int(x) for x in ids])


def text_from_ids(tokenizer: Any, ids: Sequence[int]) -> str:
    return tokenizer.decode([int(x) for x in ids], skip_special_tokens=True)


def non_text_positions(tokenizer: Any, ids: Sequence[int]) -> List[int]:
    special = set(tokenizer.all_special_ids or [])
    return [i for i, token_id in enumerate(ids) if int(token_id) in special]


def token_accuracy(tokenizer: Any, original: Sequence[int], recovered: Sequence[int]) -> float:
    excluded = set(non_text_positions(tokenizer, original))
    total = 0
    correct = 0
    for i, token_id in enumerate(original):
        if i in excluded:
            continue
        total += 1
        if i < len(recovered) and int(token_id) == int(recovered[i]):
            correct += 1
    return correct / total if total else 0.0


def bleu_score(tokenizer: Any, original: Sequence[int], recovered: Sequence[int]) -> float:
    excluded = set(non_text_positions(tokenizer, original))
    ref = [str(original[i]) for i in range(len(original)) if i not in excluded]
    hyp = [str(recovered[i]) for i in range(min(len(original), len(recovered))) if i not in excluded]
    if not ref or not hyp:
        return 0.0
    return float(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method1))


def optional_nerr(original_text: str, recovered_text: str) -> Optional[float]:
    if os.environ.get("PIA_ENABLE_NERR") != "1":
        return None
    try:
        from flair.data import Sentence
        from flair.models import SequenceTagger
    except Exception:
        return None
    try:
        if not hasattr(optional_nerr, "_tagger"):
            optional_nerr._tagger = SequenceTagger.load("flair/ner-english-large")
        tagger = optional_nerr._tagger
        sentence = Sentence(original_text)
        tagger.predict(sentence)
        entities = [entity.text for entity in sentence.get_spans("ner")]
        if not entities:
            return 1.0
        matched = sum(1 for entity in entities if entity in recovered_text)
        return matched / len(entities)
    except Exception:
        return None


def nearest_embedding_loss(learnable: torch.Tensor, embed_weight: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    if learnable.numel() == 0:
        return torch.tensor(0.0, device=learnable.device)
    weight = embed_weight.detach().float()
    vectors = learnable.detach().float()
    best_ids = torch.zeros(vectors.shape[0], device=learnable.device, dtype=torch.long)
    best_dists = torch.full((vectors.shape[0],), float("inf"), device=learnable.device)
    with torch.no_grad():
        vector_norm = torch.sum(vectors * vectors, dim=1, keepdim=True)
        for start in range(0, weight.shape[0], chunk_size):
            chunk = weight[start : start + chunk_size]
            dists = vector_norm + torch.sum(chunk * chunk, dim=1).unsqueeze(0) - 2 * vectors @ chunk.t()
            values, indices = torch.min(dists, dim=1)
            update = values < best_dists
            best_dists = torch.where(update, values, best_dists)
            best_ids = torch.where(update, indices + start, best_ids)
    nearest = embed_weight[best_ids].detach().float()
    return F.mse_loss(learnable.float(), nearest, reduction="mean")


def build_embeds(base: torch.Tensor, learnable: torch.Tensor, variable_positions: Sequence[int]) -> torch.Tensor:
    embeds = base.clone()
    if variable_positions:
        index = torch.tensor(variable_positions, device=base.device, dtype=torch.long)
        embeds[:, index, :] = learnable.to(dtype=base.dtype).unsqueeze(0)
    return embeds


def embedding_bounds(embed_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = embed_weight.detach().float()
    return weight.min(dim=0).values, weight.max(dim=0).values


def inferred_boundary_special_tokens(tokenizer: Any, seq_len: int) -> Dict[int, int]:
    fixed: Dict[int, int] = {}
    if seq_len <= 0:
        return fixed
    if getattr(tokenizer, "add_bos_token", True) and tokenizer.bos_token_id is not None:
        fixed[0] = int(tokenizer.bos_token_id)
    if seq_len > 1 and getattr(tokenizer, "add_eos_token", False) and tokenizer.eos_token_id is not None:
        fixed[seq_len - 1] = int(tokenizer.eos_token_id)
    return fixed


def random_initial_token_ids(tokenizer: Any, seq_len: int, device: torch.device, fixed_token_ids: Dict[int, int]) -> torch.Tensor:
    special = set(int(x) for x in (tokenizer.all_special_ids or []))
    vocab_size = len(tokenizer)
    candidates = [idx for idx in range(vocab_size) if idx not in special]
    if not candidates:
        raise RuntimeError("Tokenizer has no non-special token ids for random recovery initialization")
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    sampled = candidate_tensor[torch.randint(0, candidate_tensor.numel(), (seq_len,), device=device)]
    for pos, token_id in fixed_token_ids.items():
        if 0 <= pos < seq_len:
            sampled[pos] = int(token_id)
    return sampled.unsqueeze(0)


def optimize_prompt_embeddings(
    model: torch.nn.Module,
    target_layer: int,
    embed_layer: torch.nn.Module,
    tokenizer: Any,
    seq_len: int,
    device: torch.device,
    attention_mask: torch.Tensor,
    target_activation: torch.Tensor,
    fixed_token_ids: Dict[int, int],
    cfg: RunConfig,
) -> Tuple[torch.Tensor, Dict[str, float], List[Dict[str, float]]]:
    initial_ids = random_initial_token_ids(tokenizer, seq_len, device, fixed_token_ids)
    base_embeds = embed_layer(initial_ids).detach()
    fixed_positions = sorted(fixed_token_ids)
    variable_positions = [i for i in range(seq_len) if i not in set(fixed_positions)]
    left, right = embedding_bounds(embed_layer.weight)
    left = left.to(device)
    right = right.to(device)
    if variable_positions:
        init = torch.empty(
            len(variable_positions),
            base_embeds.shape[-1],
            device=device,
            dtype=torch.float32,
        ).uniform_(float(left.min().detach().cpu()), float(right.max().detach().cpu()))
        noise = torch.empty_like(init).uniform_(-0.1, 0.1)
        learnable = (init + noise).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([learnable], lr=cfg.lr)
    else:
        learnable = torch.empty(0, base_embeds.shape[-1], device=device, dtype=torch.float32)
        optimizer = None
    history: List[Dict[str, float]] = []
    target = target_activation.detach()
    final = {"activation_loss": math.nan, "constraint_loss": math.nan, "total_loss": math.nan, "cosine_similarity": math.nan}

    for step in range(cfg.epoch):
        if cfg.constrained_optimization and variable_positions:
            with torch.no_grad():
                learnable.clamp_(left.unsqueeze(0), right.unsqueeze(0))
        embeds = build_embeds(base_embeds, learnable, variable_positions)
        recovered_activation = capture_prefix_activation(
            model,
            target_layer,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )
        activation_loss = F.mse_loss(recovered_activation.float(), target.to(recovered_activation.device).float())
        if cfg.constrained_optimization:
            constraint_loss = nearest_embedding_loss(learnable, embed_layer.weight)
        else:
            constraint_loss = torch.tensor(0.0, device=device)
        total_loss = activation_loss + cfg.lambda_coeff * constraint_loss
        cosine = F.cosine_similarity(recovered_activation.float(), target.to(recovered_activation.device).float(), dim=-1).mean()

        if torch.isnan(total_loss) or torch.isnan(cosine):
            raise RuntimeError(f"NaN detected at optimization step {step + 1}")
        if optimizer is not None:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        final = {
            "activation_loss": float(activation_loss.detach().cpu()),
            "constraint_loss": float(constraint_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu()),
        }
        if (step + 1) % max(1, min(100, cfg.epoch)) == 0 or step == cfg.epoch - 1:
            history.append({"step": step + 1, **final})
            print(
                f"step={step + 1} activation_loss={final['activation_loss']:.6f} "
                f"constraint_loss={final['constraint_loss']:.6f} total_loss={final['total_loss']:.6f} "
                f"cosine={final['cosine_similarity']:.6f}",
                flush=True,
            )
    if cfg.constrained_optimization and variable_positions:
        with torch.no_grad():
            learnable.clamp_(left.unsqueeze(0), right.unsqueeze(0))
    return build_embeds(base_embeds, learnable.detach(), variable_positions), final, history


def embedding_candidates(embeds: torch.Tensor, embed_weight: torch.Tensor, top_k: int) -> List[List[int]]:
    weight = embed_weight.detach().float()
    top_k = max(1, min(top_k, weight.shape[0]))
    out = []
    for vector in embeds.squeeze(0):
        sims = F.cosine_similarity(vector.float(), weight, dim=-1)
        out.append([int(x) for x in torch.topk(sims.detach().cpu(), top_k).indices.tolist()])
    return out


def semantic_candidates(model: torch.nn.Module, prefix: Sequence[int], top_y: int) -> List[int]:
    if top_y <= 0 or not prefix:
        return []
    device = next(model.parameters()).device
    ids = torch.tensor([list(prefix)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, -1, :]
        filtered = top_k_top_p_filtering(logits, top_k=min(top_y, logits.shape[-1]), top_p=1.0)
        probs = F.softmax(filtered, dim=-1)
    return [int(x) for x in torch.topk(probs[0], min(top_y, probs.shape[-1])).indices.detach().cpu().tolist()]


def activation_calibrated_discretization(
    model: torch.nn.Module,
    target_layer: int,
    attention_mask: torch.Tensor,
    target_activation: torch.Tensor,
    initial_ids: List[int],
    embed_candidate_sets: List[List[int]],
    fixed_positions: Sequence[int],
    cfg: RunConfig,
) -> Tuple[List[int], float]:
    fixed = set(fixed_positions)
    recovered = list(initial_ids)
    per_pos_scores = []
    for pos in range(len(recovered)):
        if pos in fixed:
            per_pos_scores.append(1.0)
            continue
        candidates = list(embed_candidate_sets[pos]) if cfg.top_k_embedding > 0 else []
        if cfg.semantic_speculation and cfg.top_y_semantic > 0 and pos > 0:
            candidates.extend(semantic_candidates(model, recovered[:pos], cfg.top_y_semantic))
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            continue
        proposal_ids = []
        for cand in candidates:
            proposal = list(recovered)
            proposal[pos] = int(cand)
            proposal_ids.append(proposal)
        ids = torch.tensor(proposal_ids, dtype=torch.long, device=attention_mask.device)
        masks = attention_mask.expand(len(proposal_ids), -1).contiguous()
        with torch.no_grad():
            activation = capture_prefix_activation(
                model,
                target_layer,
                input_ids=ids,
                attention_mask=masks,
            )
        target_pos = target_activation[:, pos : pos + 1, :].expand(len(proposal_ids), -1, -1)
        dists = torch.mean((activation[:, pos : pos + 1, :].float() - target_pos.float()) ** 2, dim=(1, 2))
        best = int(torch.argmin(dists).detach().cpu())
        recovered[pos] = int(candidates[best])
        per_pos_scores.append(float((-dists[best]).detach().cpu()))
    return recovered, float(sum(per_pos_scores) / max(1, len(per_pos_scores)))


def naive_discretization(embeds: torch.Tensor, embed_weight: torch.Tensor) -> List[int]:
    return [items[0] for items in embedding_candidates(embeds, embed_weight, 1)]


def apply_defense(activation: torch.Tensor, sigma: float, bits: Optional[int]) -> torch.Tensor:
    out = activation.detach().clone()
    if sigma > 0:
        out = out + torch.randn_like(out) * sigma
    if bits is not None:
        max_abs = out.abs().max()
        if max_abs > 0:
            levels = (2 ** (bits - 1)) - 1
            scale = max_abs / levels
            out = torch.clamp(torch.round(out / scale), -levels, levels) * scale
    return out


def prompt_perplexity(model: torch.nn.Module, ids: Sequence[int]) -> Optional[float]:
    if len(ids) < 2:
        return None
    device = next(model.parameters()).device
    input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=input_ids).loss
    return float(torch.exp(loss.detach()).cpu())


def peak_memory_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {}
    for key in ("token_accuracy", "bleu", "runtime_seconds", "peak_gpu_memory_mb", "perplexity"):
        vals = [row[key] for row in rows if row.get(key) is not None]
        metrics[key] = {
            "mean": float(statistics.mean(vals)) if vals else None,
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None,
        }
    nerr_vals = [row["nerr"] for row in rows if row.get("nerr") is not None]
    metrics["nerr"] = {
        "mean": float(statistics.mean(nerr_vals)) if nerr_vals else None,
        "std": float(statistics.pstdev(nerr_vals)) if len(nerr_vals) > 1 else 0.0 if nerr_vals else None,
        "status": "available" if nerr_vals else "NERR unavailable",
    }
    metrics["num_samples"] = len(rows)
    return metrics


def write_tables(output_dir: Path, rows: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    if not rows:
        return
    csv_path = output_dir / "predictions.csv"
    keys = sorted(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        f"# {EXPERIMENT_TITLE}",
        "",
        "| metric | mean | std |",
        "| --- | ---: | ---: |",
    ]
    for key, value in metrics.items():
        if isinstance(value, dict) and "mean" in value:
            mean = "NERR unavailable" if value["mean"] is None else f"{value['mean']:.6f}"
            std = "" if value["std"] is None else f"{value['std']:.6f}"
            md.append(f"| {key} | {mean} | {std} |")
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_plot(output_dir: Path, rows: List[Dict[str, Any]], title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    if not rows:
        return
    labels = [str(row["prompt_id"]) for row in rows]
    acc = [row["token_accuracy"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, acc, color="#3b82f6")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("prompt_id")
    ax.set_ylabel("Token Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "token_accuracy.png", dpi=200)
    fig.savefig(output_dir / "token_accuracy.pdf")
    plt.close(fig)


def run_one_config(cfg: RunConfig, resume: bool) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "COMPLETE"
    if resume and done.exists():
        print(f"resume: skip complete config {cfg.output_dir}", flush=True)
        with (out / "metrics.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    json_dump(out / "config.json", {"title": EXPERIMENT_TITLE, **cfg.__dict__})
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=args_local_files_only())
    model.to(device)
    blocks = model.model.layers
    if len(blocks) != TOTAL_BLOCKS:
        raise RuntimeError(f"Expected TinyLlama 22 transformer blocks, got {len(blocks)}")
    if cfg.target_layer < 0 or cfg.target_layer >= len(blocks):
        raise ValueError(f"target_layer must be in 0..21, got {cfg.target_layer}")
    embed_layer = model.get_input_embeddings()
    prompts, dataset_meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.seed)
    json_dump(out / "dataset_meta.json", dataset_meta)
    if len(prompts) < cfg.dataset_len:
        print(
            f"warning: requested {cfg.dataset_len} prompts but only {len(prompts)} valid prompts are available",
            flush=True,
        )
    predictions_path = out / "predictions.jsonl"
    if predictions_path.exists() and not resume:
        predictions_path.unlink()
    completed_ids = set()
    if resume and predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed_ids.add(json.loads(line)["prompt_id"])
    rows: List[Dict[str, Any]] = []
    if resume and predictions_path.exists():
        row_by_prompt_id = {}
        with predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row_by_prompt_id[row["prompt_id"]] = row
        rows = [row_by_prompt_id[key] for key in sorted(row_by_prompt_id)]

    failed_rows: List[Dict[str, Any]] = []

    for prompt_id, prompt in enumerate(prompts):
        if prompt_id in completed_ids:
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
            target_activation = apply_defense(target_activation, cfg.gaussian_noise_sigma, cfg.quantization_bits)
            preflight = {
                "model_name": MODEL_NAME,
                "block_count": len(blocks),
                "target_layer_index": cfg.target_layer,
                "inverted_block_count": cfg.inverted_block_count,
                "hook_module_name": f"model.model.layers.{cfg.target_layer}",
                "activation_shape": list(target_activation.shape),
                "prompt_token_count": len(original_ids),
                "gpu": gpu_id,
                "manual_prefix_forward": True,
                "recovery_uses_ground_truth_tokens": False,
            }
            print(f"preflight={json.dumps(preflight, ensure_ascii=True)}", flush=True)
            fixed_token_ids = inferred_boundary_special_tokens(tokenizer, input_ids.shape[1])
            fixed_positions = sorted(fixed_token_ids)
            optimized_embeds, losses, history = optimize_prompt_embeddings(
                model,
                cfg.target_layer,
                embed_layer,
                tokenizer,
                input_ids.shape[1],
                device,
                attention_mask,
                target_activation,
                fixed_token_ids,
                cfg,
            )
            embed_sets = embedding_candidates(optimized_embeds, embed_layer.weight, max(1, cfg.top_k_embedding))
            recovered_ids = naive_discretization(optimized_embeds, embed_layer.weight)
            for pos, token_id in fixed_token_ids.items():
                if 0 <= pos < len(recovered_ids):
                    recovered_ids[pos] = int(token_id)
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
            recovered_text = text_from_ids(tokenizer, recovered_ids)
            original_text = text_from_ids(tokenizer, original_ids)
            nerr = optional_nerr(original_text, recovered_text)
            elapsed = time.time() - start
            row = {
                "title": EXPERIMENT_TITLE,
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
                "nerr": nerr,
                "optimization_loss": losses["total_loss"],
                "activation_loss": losses["activation_loss"],
                "constraint_loss": losses["constraint_loss"],
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
                "perplexity": prompt_perplexity(model, recovered_ids) if cfg.compute_perplexity else None,
                "gpu": gpu_id,
                "semantic_oracle": "TinyLlama itself; adapted difference from paper Llama-7B oracle",
                "recovery_uses_ground_truth_tokens": False,
                "fixed_token_source": "tokenizer boundary conventions only; original token ids are used only for target activation and final evaluation",
                "preflight": preflight,
                "optimization_history": history,
            }
            jsonl_append(predictions_path, row)
            rows.append(row)
            print(
                f"prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} nerr={row['nerr']} elapsed={elapsed:.2f}s recovered={recovered_text!r}",
                flush=True,
            )
        except Exception as exc:
            failure = {
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
            jsonl_append(out / "failures.jsonl", failure)
            print(f"prompt_id={prompt_id} failed error={repr(exc)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics = summarize_rows(rows)
    metrics.update({
        "title": EXPERIMENT_TITLE,
        "config": cfg.__dict__,
        "dataset_meta": dataset_meta,
        "completed_sample_count": len(rows),
        "failed_sample_count": len(failed_rows),
        "nerr_note": "NERR unavailable unless Flair NER model is locally available and loads successfully.",
    })
    json_dump(out / "metrics.json", metrics)
    write_tables(out, rows, metrics)
    write_plot(out, rows, EXPERIMENT_TITLE)
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    return metrics


def config_from_args(args: argparse.Namespace, run_name: str, output_dir: str, seed: int, participants: int, position: int) -> RunConfig:
    inverted, target = ATTACKER_MAP[participants][position]
    return RunConfig(
        run_name=run_name,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=args.dataset_len,
        seed=seed,
        participant_number=participants,
        attacker_position=position,
        inverted_block_count=inverted,
        target_layer=args.target_layer if args.target_layer is not None else target,
        epoch=args.epoch,
        lr=args.lr,
        lambda_coeff=args.lambda_coeff,
        top_k_embedding=args.k,
        top_y_semantic=args.y,
        constrained_optimization=not args.naive_optimization,
        adaptive_discretization=not args.naive_discretization,
        semantic_speculation=not args.disable_semantic_speculation,
        gaussian_noise_sigma=args.gaussian_noise_sigma,
        quantization_bits=args.quantization_bits,
        max_token_len=args.max_token_len,
        compute_perplexity=args.compute_perplexity,
        mode=args.mode,
    )


def run_smoke(args: argparse.Namespace) -> None:
    cfg = config_from_args(args, "smoke", args.output_dir or "runs/tinyllama_full/smoke", args.seed, 3, 2)
    cfg.dataset_len = 1
    cfg.target_layer = args.target_layer if args.target_layer is not None else 3
    cfg.inverted_block_count = cfg.target_layer + 1
    run_one_config(cfg, args.resume)


def run_main(args: argparse.Namespace) -> None:
    results = []
    for seed in args.seeds:
        for participants, positions in ATTACKER_MAP.items():
            for position in positions:
                cfg = config_from_args(
                    args,
                    f"main_p{participants}_a{position}_seed{seed}",
                    f"runs/tinyllama_full/main_results/p{participants}_a{position}_seed{seed}",
                    seed,
                    participants,
                    position,
                )
                results.append(run_one_config(cfg, args.resume))
    aggregate("runs/tinyllama_full/main_results", "main_results", results)


def run_single(args: argparse.Namespace) -> None:
    if args.participant_number not in ATTACKER_MAP:
        raise ValueError(f"unsupported participant_number={args.participant_number}")
    if args.attacker_position not in ATTACKER_MAP[args.participant_number]:
        raise ValueError(
            f"unsupported attacker_position={args.attacker_position} for participant_number={args.participant_number}"
        )
    run_name = args.run_name or f"p{args.participant_number}_a{args.attacker_position}_seed{args.seed}"
    output_dir = args.output_dir or f"runs/tinyllama_full/main_results/{run_name}"
    cfg = config_from_args(
        args,
        run_name,
        output_dir,
        args.seed,
        args.participant_number,
        args.attacker_position,
    )
    run_one_config(cfg, args.resume)


def run_ablation(args: argparse.Namespace) -> None:
    variants = [
        ("naive_opt_naive_disc", True, True, True),
        ("constrained_opt_naive_disc", False, True, True),
        ("naive_opt_adaptive_disc", True, False, False),
        ("constrained_opt_adaptive_disc", False, False, False),
    ]
    results = []
    for name, naive_opt, naive_disc, disable_sem in variants:
        cfg = config_from_args(args, name, f"runs/tinyllama_full/ablations/{name}", args.seed, 4, 4)
        cfg.naive_name = name if hasattr(cfg, "naive_name") else name
        cfg.constrained_optimization = not naive_opt
        cfg.adaptive_discretization = not naive_disc
        cfg.semantic_speculation = not disable_sem
        results.append(run_one_config(cfg, args.resume))
    aggregate("runs/tinyllama_full/ablations", "ablations", results)


def run_defense(args: argparse.Namespace) -> None:
    settings = []
    for sigma in [0, 0.1, 0.5, 1.0]:
        settings.append((f"gaussian_sigma_{sigma}", sigma, None))
    for bits in [None, 8, 4]:
        settings.append((f"quantization_{bits or 'none'}", 0.0, bits))
    results = []
    for name, sigma, bits in settings:
        cfg = config_from_args(args, name, f"runs/tinyllama_full/defenses/{name}", args.seed, 4, 4)
        cfg.gaussian_noise_sigma = sigma
        cfg.quantization_bits = bits
        cfg.compute_perplexity = True
        results.append(run_one_config(cfg, args.resume))
    aggregate("runs/tinyllama_full/defenses", "defenses", results)


def run_baselines(args: argparse.Namespace) -> None:
    variants = [
        ("song_style_softmax_embedding_baseline", True, True, True),
        ("naive_optimization_baseline", True, False, False),
        ("naive_discretization_baseline", False, True, True),
    ]
    results = []
    for name, naive_opt, naive_disc, disable_sem in variants:
        cfg = config_from_args(args, name, f"runs/tinyllama_full/baselines/{name}", args.seed, 4, 4)
        cfg.constrained_optimization = not naive_opt
        cfg.adaptive_discretization = not naive_disc
        cfg.semantic_speculation = not disable_sem
        results.append(run_one_config(cfg, args.resume))
    aggregate("runs/tinyllama_full/baselines", "baselines", results)


def aggregate(base_dir: str, name: str, results: List[Dict[str, Any]]) -> None:
    out = Path(base_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        cfg = result["config"]
        rows.append({
            "title": EXPERIMENT_TITLE,
            "run_name": cfg["run_name"],
            "participant_number": cfg["participant_number"],
            "attacker_position": cfg["attacker_position"],
            "inverted_block_count": cfg["inverted_block_count"],
            "target_layer_index": cfg["target_layer"],
            "seed": cfg["seed"],
            "token_accuracy": result["token_accuracy"]["mean"],
            "token_accuracy_std": result["token_accuracy"]["std"],
            "bleu": result["bleu"]["mean"],
            "bleu_std": result["bleu"]["std"],
            "nerr": result["nerr"]["mean"],
            "nerr_status": result["nerr"]["status"],
            "mean_runtime": result["runtime_seconds"]["mean"],
            "peak_gpu_memory": result["peak_gpu_memory_mb"]["mean"],
            "num_samples": result["num_samples"],
        })
    if rows:
        with (out / f"{name}_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        md = [f"# {EXPERIMENT_TITLE}: {name}", "", "| run | p | pos | blocks | layer | acc | BLEU | NERR | runtime |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
        for row in rows:
            nerr = "NERR unavailable" if row["nerr"] is None else f"{row['nerr']:.4f}"
            md.append(
                f"| {row['run_name']} | {row['participant_number']} | {row['attacker_position']} | "
                f"{row['inverted_block_count']} | {row['target_layer_index']} | {row['token_accuracy']:.4f} | "
                f"{row['bleu']:.4f} | {nerr} | {row['mean_runtime']:.2f} |"
            )
        (out / f"{name}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def generate_table2() -> None:
    base = Path("runs/tinyllama_full/main_results")
    base.mkdir(parents=True, exist_ok=True)
    run_dir_re = re.compile(r"^p[345]_a[2-5]_seed(?:42|43|44)$")
    metrics_paths = sorted(
        path for path in base.glob("p*_a*_seed*/metrics.json")
        if run_dir_re.match(path.parent.name)
    )
    rows = []
    for path in metrics_paths:
        with path.open("r", encoding="utf-8") as f:
            metric = json.load(f)
        cfg = metric["config"]
        data_meta = metric.get("dataset_meta", {})
        pred_path = path.parent / "predictions.jsonl"
        fail_path = path.parent / "failures.jsonl"
        row_by_prompt_id = {}
        if pred_path.exists():
            with pred_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        pred_row = json.loads(line)
                        row_by_prompt_id[pred_row["prompt_id"]] = pred_row
        dedup_rows = [row_by_prompt_id[key] for key in sorted(row_by_prompt_id)]
        failed_prompt_ids = set()
        if fail_path.exists():
            with fail_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        failed_prompt_ids.add(json.loads(line)["prompt_id"])
        dedup_metrics = summarize_rows(dedup_rows) if dedup_rows else metric
        rows.append({
            "dataset_label": "Skytrax-28 pilot" if data_meta.get("available_valid_prompts", 0) < 150 else "Skytrax",
            "dataset_size": data_meta.get("selected_prompts"),
            "participant_number": cfg["participant_number"],
            "attacker_position": cfg["attacker_position"],
            "inverted_blocks": cfg["inverted_block_count"],
            "target_layer": cfg["target_layer"],
            "seed": cfg["seed"],
            "token_accuracy_mean": dedup_metrics["token_accuracy"]["mean"],
            "token_accuracy_std": dedup_metrics["token_accuracy"]["std"],
            "bleu_mean": dedup_metrics["bleu"]["mean"],
            "bleu_std": dedup_metrics["bleu"]["std"],
            "nerr": dedup_metrics["nerr"]["mean"],
            "nerr_status": dedup_metrics["nerr"]["status"],
            "mean_runtime": dedup_metrics["runtime_seconds"]["mean"],
            "peak_gpu_memory": dedup_metrics["peak_gpu_memory_mb"]["mean"],
            "completed_sample_count": len(dedup_rows),
            "failed_sample_count": len(failed_prompt_ids),
            "metrics_path": str(path),
        })
    rows.sort(key=lambda r: (r["participant_number"], r["attacker_position"], r["seed"]))
    csv_path = base / "table2_tinyllama.csv"
    fieldnames = [
        "dataset_label",
        "dataset_size",
        "participant_number",
        "attacker_position",
        "inverted_blocks",
        "target_layer",
        "seed",
        "token_accuracy_mean",
        "token_accuracy_std",
        "bleu_mean",
        "bleu_std",
        "nerr",
        "nerr_status",
        "mean_runtime",
        "peak_gpu_memory",
        "completed_sample_count",
        "failed_sample_count",
        "metrics_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        f"# {EXPERIMENT_TITLE}: Table 2 TinyLlama Adaptation",
        "",
        "| dataset | size | participants | attacker position | inverted blocks | target layer | seed | acc mean | acc std | BLEU mean | BLEU std | NERR | runtime | peak GPU MB | completed | failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        nerr = "NERR unavailable" if row["nerr"] is None else f"{row['nerr']:.6f}"
        md.append(
            f"| {row['dataset_label']} | {row['dataset_size']} | {row['participant_number']} | "
            f"{row['attacker_position']} | {row['inverted_blocks']} | {row['target_layer']} | {row['seed']} | "
            f"{row['token_accuracy_mean'] if row['token_accuracy_mean'] is not None else 'NA'} | "
            f"{row['token_accuracy_std'] if row['token_accuracy_std'] is not None else 'NA'} | "
            f"{row['bleu_mean'] if row['bleu_mean'] is not None else 'NA'} | "
            f"{row['bleu_std'] if row['bleu_std'] is not None else 'NA'} | {nerr} | "
            f"{row['mean_runtime'] if row['mean_runtime'] is not None else 'NA'} | "
            f"{row['peak_gpu_memory'] if row['peak_gpu_memory'] is not None else 'NA'} | "
            f"{row['completed_sample_count']} | {row['failed_sample_count']} |"
        )
    md_path = base / "table2_tinyllama.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [f"p{r['participant_number']}-a{r['attacker_position']}-s{r['seed']}" for r in rows]
        values = [0 if r["token_accuracy_mean"] is None else r["token_accuracy_mean"] for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 4))
        ax.bar(labels, values, color="#2563eb")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Token Accuracy")
        ax.set_title(EXPERIMENT_TITLE)
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        fig.savefig(base / "table2_tinyllama.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        (base / "table2_plot_error.txt").write_text(repr(exc), encoding="utf-8")
    analysis = [
        f"# {EXPERIMENT_TITLE}: main results",
        "",
        "These results are a TinyLlama-adapted reproduction, not a strict reproduction of the paper's Llama-65B results.",
        "",
        "Dataset label is `Skytrax-28 pilot` whenever only the local 28 valid prompts are available. This is not the paper's 150-prompt Skytrax main result.",
        "",
        "Differences from the paper: TinyLlama-1.1B instead of Llama-65B/Llama-2-70B/OPT-66B, 22 transformer blocks instead of the paper's large-model layer counts, local available Skytrax sample count, and TinyLlama itself as semantic oracle instead of Llama-7B.",
        "",
        "Do not compare these numbers directly against the paper's Llama-65B Table 2, and do not use short-text 1.0 accuracy results as evidence of exceeding the paper.",
        "",
        "The completed pilot configurations test whether intermediate activations can leak prompt information in the TinyLlama-adapted setting.",
        "",
        md_path.read_text(encoding="utf-8"),
    ]
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/tinyllama_main_results.md").write_text("\n".join(analysis), encoding="utf-8")


def verify_prefix_forward_and_recovery(args: argparse.Namespace) -> None:
    out = Path(args.output_dir or "runs/tinyllama_full/verification")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=args_local_files_only())
    model.to(device)
    prompts, dataset_meta = load_dataset_prompts(args.dataset_name, args.dataset_path, max(1, args.dataset_len), args.seed)
    if not prompts:
        raise RuntimeError("No prompt available for verification")
    prompt = prompts[0]
    tokenized = tokenizer(prompt, add_special_tokens=True, truncation=False, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    target_layer = args.target_layer if args.target_layer is not None else 3
    block = model.model.layers[target_layer]
    with torch.no_grad():
        full_activation = capture_activation(
            model,
            block,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        ).detach()
        prefix_activation = capture_prefix_activation(
            model,
            target_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).detach()
    max_abs_diff = float((full_activation.float() - prefix_activation.float()).abs().max().detach().cpu())
    mean_abs_diff = float((full_activation.float() - prefix_activation.float()).abs().mean().detach().cpu())
    verification = {
        "title": EXPERIMENT_TITLE,
        "model_name": MODEL_NAME,
        "target_layer": target_layer,
        "dtype": str(dtype),
        "activation_shape_full": list(full_activation.shape),
        "activation_shape_prefix": list(prefix_activation.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "passes_strict_1e_5": max_abs_diff < 1e-5,
        "passes_fp16_bf16_tolerance_5e_3": max_abs_diff < 5e-3,
        "dataset_meta": dataset_meta,
        "recovery_uses_ground_truth_tokens": False,
        "ground_truth_token_policy": "Original input_ids are used for target activation and final evaluation only; recovery initialization uses random non-special token ids plus tokenizer boundary special-token conventions.",
    }
    json_dump(out / "prefix_forward_verification.json", verification)

    cfg = config_from_args(
        args,
        run_name="verification_no_gt_recovery",
        output_dir=str(out / "no_gt_recovery"),
        seed=args.seed,
        participants=args.participant_number,
        position=args.attacker_position,
    )
    cfg.dataset_len = 1
    cfg.target_layer = target_layer
    cfg.inverted_block_count = target_layer + 1
    cfg.epoch = min(args.epoch, 5)
    run_one_config(cfg, resume=False)
    print(json.dumps(verification, ensure_ascii=True), flush=True)


def recompute_token_accuracy_report() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=args_local_files_only(),
    )
    rows = []
    run_dir_re = re.compile(r"^p[345]_a[2-5]_seed(?:42|43|44)$")
    for path in sorted(Path("runs/tinyllama_full/main_results").glob("p*_a*_seed*/predictions.jsonl")):
        if not run_dir_re.match(path.parent.name):
            continue
        values = []
        with path.open("r", encoding="utf-8") as f:
            by_prompt: Dict[int, Dict[str, Any]] = {}
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    by_prompt[int(row["prompt_id"])] = row
        for row in by_prompt.values():
            values.append(token_accuracy(tokenizer, row["original_token_ids"], row["recovered_token_ids"]))
        rows.append({
            "predictions_path": str(path),
            "sample_count": len(values),
            "token_accuracy_mean_excluding_special": statistics.mean(values) if values else None,
            "token_accuracy_std_excluding_special": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "excluded_special_token_ids": [int(x) for x in (tokenizer.all_special_ids or [])],
        })
    Path("analysis").mkdir(exist_ok=True)
    json_dump(Path("analysis/token_accuracy_recheck.json"), {"rows": rows})


def write_status_files() -> None:
    Path("analysis").mkdir(exist_ok=True)
    status = f"""# {EXPERIMENT_TITLE}: reproduction status

## Dataset status

- Skytrax: local `data/airline.json` is available, but it currently contains only 29 raw prompts / 28 valid prompts, so it is insufficient for the requested 150-prompt protocol. A direct `git ls-remote` to the README-referenced public Skytrax repository succeeded, but `git clone` into `runs/tinyllama_full/data_sources/` disconnected while reading the sideband packet. No proxy or system configuration was changed. Until a complete legal Skytrax CSV is available, main runs are labeled `Skytrax-28 pilot`.
- CMS: no verifiable local CMS dataset was found. `data/medical.json` exists, but it is not labeled or documented as CMS, so it was not treated as a completed CMS reproduction.
- ECHR: no verifiable local ECHR dataset was found.

## Baseline status

- Song et al.-style softmax embedding optimization baseline: implemented as a local baseline label for engineering comparison only.
- Li et al. and Morris et al.: not faithfully reproduced. No official implementation and matching experimental setup were found in this repository.

## Grey-box LoRA status

No verifiable TinyLlama LoRA adapter, matching training data, and held-out evaluation setup were found locally. White-box results are not reported as grey-box.
"""
    Path("analysis/reproduction_status.md").write_text(status, encoding="utf-8")
    grey = f"""# TinyLlama LoRA grey-box extension plan

This is a plan for a future TinyLlama LoRA grey-box extension, not a completed result.

Minimum requirements:

1. A verifiable TinyLlama LoRA adapter with training metadata.
2. Public or locally approved training data and held-out evaluation data.
3. A layer-to-participant LoRA partition matching the collaborative inference setup.
4. An alternating optimization implementation that fixes unknown LoRA while optimizing prompt embeddings, then fixes recovered prompt embeddings while optimizing unknown preceding LoRA parameters.
5. Separate reporting under `runs/tinyllama_full/greybox/` as TinyLlama LoRA grey-box extension.
"""
    Path("runs/tinyllama_full/greybox").mkdir(parents=True, exist_ok=True)
    Path("runs/tinyllama_full/greybox/greybox_plan.md").write_text(grey, encoding="utf-8")


def write_report() -> None:
    write_status_files()
    commit = os.popen("git rev-parse --short HEAD").read().strip()
    report = f"""# {EXPERIMENT_TITLE}

## Scope

This report is a TinyLlama-adapted reproduction of Prompt Inversion Attack experiments. It is not a strict numerical reproduction of the paper's Llama-65B / Llama-2-70B / OPT-66B experiments, and it must not be interpreted as exceeding or fully reproducing the original paper results.

## Environment

- Model: `{MODEL_NAME}`
- TinyLlama transformer blocks: 22, layer indices 0..21
- Commit: `{commit}`
- Conda env: `deml`
- Output root: `runs/tinyllama_full/`

## Adaptation Differences

- Original paper models: Llama-65B, Llama-2-70B, OPT-66B. This run uses TinyLlama-1.1B-Chat.
- Original semantic oracle: Llama-7B. This implementation uses TinyLlama itself as the frozen semantic oracle.
- Data availability: local Skytrax has fewer than 150 valid prompts; CMS and ECHR were not locally available.
- Metrics are TinyLlama-adapted and not directly comparable with paper tables.

## Status

See `analysis/reproduction_status.md` for dataset, baseline, and grey-box status.

## Results

Run-specific `metrics.json`, `predictions.jsonl`, `predictions.csv`, `summary.md`, and PNG/PDF charts are written under `runs/tinyllama_full/`.

## Conclusion

Completed TinyLlama-adapted runs can test whether intermediate activations leak prompt information in this smaller model. The resulting values should be interpreted as evidence for the adapted setting only, not as direct paper-number comparisons.
"""
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/tinyllama_full_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "single",
            "main",
            "ablation",
            "defense",
            "baselines",
            "table2",
            "report",
            "verify",
            "recompute-accuracy",
        ],
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-name", default="Skytrax")
    parser.add_argument("--dataset-path", default="data/airline.json")
    parser.add_argument("--dataset-len", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-coeff", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--y", type=int, default=10)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--max-token-len", type=int, default=256)
    parser.add_argument("--naive-optimization", action="store_true")
    parser.add_argument("--naive-discretization", action="store_true")
    parser.add_argument("--disable-semantic-speculation", action="store_true")
    parser.add_argument("--gaussian-noise-sigma", type=float, default=0.0)
    parser.add_argument("--quantization-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--compute-perplexity", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "smoke":
            run_smoke(args)
        elif args.mode == "single":
            run_single(args)
        elif args.mode == "main":
            run_main(args)
        elif args.mode == "ablation":
            run_ablation(args)
        elif args.mode == "defense":
            run_defense(args)
        elif args.mode == "baselines":
            run_baselines(args)
        elif args.mode == "report":
            write_report()
        elif args.mode == "table2":
            generate_table2()
        elif args.mode == "verify":
            verify_prefix_forward_and_recovery(args)
        elif args.mode == "recompute-accuracy":
            recompute_token_accuracy_report()
        else:
            raise NotImplementedError(args.mode)
    except Exception:
        failure_dir = Path("runs/tinyllama_full/failures")
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"failure_{int(time.time())}.log"
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"failure logged to {failure_path}", flush=True)
        raise


if __name__ == "__main__":
    main()
