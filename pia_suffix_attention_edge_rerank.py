import argparse
import ast
import inspect
import json
import math
import os
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_CACHE", "/data/zhiqi/hf_cache/transformers")

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask

from pia_masked_server_attn_pia import build_variable_mask
from pia_tinyllama import (
    bleu_score,
    capture_prefix_activation,
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
    semantic_candidates,
    set_seed,
    text_from_ids,
    token_accuracy,
    token_texts,
)

OUTPUT_ROOT_DEFAULT = "runs/suffix_attention_edge_rerank_pia"
METHOD_ALIASES = {"B0V": "E0", "calibration_only": "E0", "tail_only": "E1", "edge_only": "E2", "tail_edge": "E3"}


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


@dataclass(frozen=True)
class AttentionEdge:
    layer: int
    head: int
    query: int
    key: int
    ref_weight: float


@dataclass
class EdgeGraph:
    edges: List[AttentionEdge]
    stats: Dict[str, Any]

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def bos_or_fixed_edge_count(self) -> int:
        return int(self.stats.get("bos_or_fixed_edge_count", 0))

    @property
    def is_empty(self) -> bool:
        return not self.edges

    def to_json(self) -> Dict[str, Any]:
        return {"edges": [asdict(edge) for edge in self.edges], "stats": self.stats}


@dataclass
class CandidateScore:
    index: int
    ids: List[int]
    calibration_score: float
    tail_loss: Optional[float] = None
    edge_loss: Optional[float] = None


@dataclass
class RerankConfig:
    method: str = "E0"
    lambda_tail_rank: float = 0.0
    lambda_edge_rank: float = 0.0
    margin_threshold: float = 0.01
    calibration_tolerance: float = 0.02
    edge_improvement_threshold: float = 0.0
    tail_improvement_threshold: float = 0.0


@dataclass
class RerankResult:
    selected_index: int
    selected_ids: List[int]
    baseline_index: int
    baseline_ids: List[int]
    override: bool
    reason: str
    final_scores: List[Dict[str, Any]]


@dataclass
class SAERConfig:
    method: str
    run_name: str
    output_dir: str
    dataset_name: str
    dataset_path: str
    dataset_len: int
    prompt_split_seed: int
    split: str
    prompt_limit: int
    init_seed: int
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
    top_r: int
    sink_threshold: float
    entropy_min: float
    min_head_support: int
    min_layer_support: int
    margin_threshold: float
    lambda_tail_rank: float
    lambda_edge_rank: float
    calibration_tolerance: float
    edge_improvement_threshold: float
    tail_improvement_threshold: float
    local_files_only: bool = True
    fix_boundary_specials: bool = True


def _attention_heads(attn: torch.Tensor) -> torch.Tensor:
    if attn.dim() == 4:
        return attn[0]
    if attn.dim() == 3:
        return attn
    if attn.dim() == 2:
        return attn.unsqueeze(0)
    raise RuntimeError(f"unsupported attention rank {attn.dim()}")


def _entropy(row: torch.Tensor, eps: float = 1e-12) -> float:
    values = row.float().clamp_min(eps)
    values = values / values.sum().clamp_min(eps)
    return float((-(values * values.log()).sum()).detach().cpu())


def build_attention_edge_graph(
    ref_attentions: Sequence[Tuple[int, torch.Tensor]],
    variable_mask: torch.Tensor,
    top_r: int = 3,
    sink_threshold: float = 0.70,
    entropy_min: float = 0.30,
    min_head_support: int = 2,
    min_layer_support: int = 2,
    eps: float = 1e-12,
) -> EdgeGraph:
    variable = variable_mask.detach().bool().cpu()
    fixed_positions = {idx for idx, is_variable in enumerate(variable.tolist()) if not is_variable}
    votes: Dict[Tuple[int, int], Dict[str, Any]] = {}
    per_head_counts: Dict[str, int] = {}
    bos_mass: List[float] = []
    skipped_sink = 0
    skipped_low_entropy = 0
    skipped_few_keys = 0
    usable_rows = 0
    candidate_edge_count = 0

    for layer, attn in ref_attentions:
        heads = _attention_heads(attn.detach().float()).cpu()
        for head in range(heads.shape[0]):
            for query in range(heads.shape[1]):
                row = heads[head, query]
                if row.numel():
                    bos_mass.append(float(row[0].detach().cpu()))
                if query >= len(variable) or not bool(variable[query]):
                    continue
                usable_keys = [key for key in range(min(query + 1, len(variable))) if bool(variable[key])]
                if len(usable_keys) < 2:
                    skipped_few_keys += 1
                    continue
                values = row[usable_keys].clamp_min(0.0)
                mass = values.sum()
                if float(mass.detach().cpu()) <= eps:
                    skipped_few_keys += 1
                    continue
                values = values / mass.clamp_min(eps)
                if float(values.max().detach().cpu()) > float(sink_threshold):
                    skipped_sink += 1
                    continue
                if _entropy(values, eps) < float(entropy_min):
                    skipped_low_entropy += 1
                    continue
                usable_rows += 1
                for local_key in torch.topk(values, min(max(1, int(top_r)), len(usable_keys))).indices.tolist():
                    key = int(usable_keys[int(local_key)])
                    if key in fixed_positions or query in fixed_positions:
                        continue
                    vote = votes.setdefault((query, key), {"heads": set(), "layers": set(), "instances": []})
                    vote["heads"].add((int(layer), int(head)))
                    vote["layers"].add(int(layer))
                    vote["instances"].append((int(layer), int(head), int(query), int(key), float(row[key].detach().cpu())))
                    per_head_counts[f"{int(layer)}:{int(head)}"] = per_head_counts.get(f"{int(layer)}:{int(head)}", 0) + 1
                    candidate_edge_count += 1

    edges: List[AttentionEdge] = []
    for vote in votes.values():
        if len(vote["heads"]) >= int(min_head_support) or len(vote["layers"]) >= int(min_layer_support):
            for instance in vote["instances"]:
                layer, head, query, key, weight = instance
                if query not in fixed_positions and key not in fixed_positions and key <= query:
                    edges.append(AttentionEdge(layer, head, query, key, weight))

    fixed_edge_count = sum(1 for edge in edges if edge.query == 0 or edge.key == 0 or edge.query in fixed_positions or edge.key in fixed_positions)
    stats = {
        "raw_bos_attention_mass_mean": float(statistics.mean(bos_mass)) if bos_mass else 0.0,
        "bos_or_fixed_edge_count": int(fixed_edge_count),
        "edge_count": len(edges),
        "candidate_edge_count": candidate_edge_count,
        "per_layer_head_edge_count": per_head_counts,
        "skipped_sink_rows": skipped_sink,
        "skipped_low_entropy_rows": skipped_low_entropy,
        "skipped_few_key_rows": skipped_few_keys,
        "usable_rows": usable_rows,
        "graph_empty": not edges,
    }
    return EdgeGraph(edges=edges, stats=stats)


def _attention_lookup(attentions: Sequence[Tuple[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    return {int(layer): _attention_heads(attn).float() for layer, attn in attentions}


def centered_log_score(attn: torch.Tensor, query: int, key: int, variable_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    variable = variable_mask.to(attn.device).bool()
    usable = torch.zeros_like(variable, dtype=torch.bool)
    usable[: query + 1] = variable[: query + 1]
    row = attn[query].float().clamp_min(eps)
    usable_logs = torch.log(row[usable] + eps)
    return torch.log(row[key] + eps) - usable_logs.mean()


def edge_consistency_loss(
    ref_attentions: Sequence[Tuple[int, torch.Tensor]],
    pred_attentions: Sequence[Tuple[int, torch.Tensor]],
    graph: EdgeGraph,
    variable_mask: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    ref_lookup = _attention_lookup(ref_attentions)
    pred_lookup = _attention_lookup(pred_attentions)
    terms: List[torch.Tensor] = []
    skipped = 0
    for edge in graph.edges:
        if edge.layer not in ref_lookup or edge.layer not in pred_lookup:
            skipped += 1
            continue
        pred_heads = pred_lookup[edge.layer]
        ref_heads = ref_lookup[edge.layer].to(pred_heads.device)
        if edge.head >= pred_heads.shape[0]:
            skipped += 1
            continue
        pred_delta = centered_log_score(pred_heads[edge.head], edge.query, edge.key, variable_mask, eps)
        ref_delta = centered_log_score(ref_heads[edge.head], edge.query, edge.key, variable_mask.to(ref_heads.device), eps).detach()
        terms.append(F.huber_loss(pred_delta, ref_delta, reduction="mean"))
    anchor = pred_attentions[0][1] if pred_attentions else torch.tensor(0.0)
    loss = torch.stack(terms).mean() if terms else anchor.float().sum() * 0.0
    return loss, {"used_edges": len(terms), "skipped_edges": skipped, "edge_count": graph.edge_count}


def rank_norm(values: Sequence[Optional[float]]) -> List[float]:
    finite = [float(v) if v is not None and math.isfinite(float(v)) else float("inf") for v in values]
    order = sorted(range(len(finite)), key=lambda idx: (finite[idx], idx))
    ranks = [0.0] * len(finite)
    denom = max(1, len(finite) - 1)
    for rank, idx in enumerate(order):
        ranks[idx] = rank / denom
    return ranks


def gated_rerank_candidates(candidates: Sequence[CandidateScore], cfg: RerankConfig, graph_is_empty: bool) -> RerankResult:
    if not candidates:
        raise RuntimeError("candidate set is empty")
    method = canonical_method(cfg.method)
    ordered = sorted(candidates, key=lambda c: (float(c.calibration_score), int(c.index)))
    baseline = ordered[0]
    if method == "E0":
        return RerankResult(baseline.index, list(baseline.ids), baseline.index, list(baseline.ids), False, "calibration_only", [])
    uses_edge = method in {"E2", "E3"} and float(cfg.lambda_edge_rank) > 0.0
    uses_tail = method in {"E1", "E3"} and float(cfg.lambda_tail_rank) > 0.0
    if method in {"E2", "E3"} and graph_is_empty:
        return RerankResult(baseline.index, list(baseline.ids), baseline.index, list(baseline.ids), False, "empty_edge_graph_degrade_to_E0", [])
    if method == "E2" and not uses_edge:
        return RerankResult(baseline.index, list(baseline.ids), baseline.index, list(baseline.ids), False, "lambda_edge_rank_zero", [])
    if method == "E3" and not uses_edge:
        cfg = RerankConfig("E1", cfg.lambda_tail_rank, 0.0, cfg.margin_threshold, cfg.calibration_tolerance, cfg.edge_improvement_threshold, cfg.tail_improvement_threshold)
        method, uses_tail = "E1", cfg.lambda_tail_rank > 0.0
    if len(ordered) < 2:
        return RerankResult(baseline.index, list(baseline.ids), baseline.index, list(baseline.ids), False, "single_candidate", [])
    if float(ordered[1].calibration_score) - float(baseline.calibration_score) > float(cfg.margin_threshold):
        return RerankResult(baseline.index, list(baseline.ids), baseline.index, list(baseline.ids), False, "calibration_margin_too_large", [])

    cal_rank = rank_norm([c.calibration_score for c in candidates])
    tail_rank = rank_norm([c.tail_loss for c in candidates])
    edge_rank = rank_norm([c.edge_loss for c in candidates])
    base_tail = baseline.tail_loss if baseline.tail_loss is not None else float("inf")
    base_edge = baseline.edge_loss if baseline.edge_loss is not None else float("inf")
    rows: List[Dict[str, Any]] = []
    best = baseline
    best_score = float("inf")
    for idx, cand in enumerate(candidates):
        allowed = True
        reasons: List[str] = []
        if float(cand.calibration_score) - float(baseline.calibration_score) > float(cfg.calibration_tolerance):
            allowed = False
            reasons.append("outside_calibration_tolerance")
        if uses_edge and (cand.edge_loss is None or float(base_edge) - float(cand.edge_loss) < float(cfg.edge_improvement_threshold)):
            allowed = False
            reasons.append("edge_improvement_too_small")
        if uses_tail and (cand.tail_loss is None or float(base_tail) - float(cand.tail_loss) < float(cfg.tail_improvement_threshold)):
            allowed = False
            reasons.append("tail_improvement_too_small")
        score = cal_rank[idx]
        if uses_tail:
            score += float(cfg.lambda_tail_rank) * tail_rank[idx]
        if uses_edge:
            score += float(cfg.lambda_edge_rank) * edge_rank[idx]
        rows.append({"index": cand.index, "calibration_score": cand.calibration_score, "tail_loss": cand.tail_loss, "edge_loss": cand.edge_loss, "final_score": score, "allowed": allowed, "reasons": reasons})
        if allowed and score < best_score:
            best_score = score
            best = cand
    return RerankResult(best.index, list(best.ids), baseline.index, list(baseline.ids), best.index != baseline.index, "gated_rerank_override" if best.index != baseline.index else "baseline_retained", rows)


def server_suffix_forward(model, start_layer: int, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = True, detach: bool = True):
    batch, seq_len, _ = hidden_states.shape
    pos = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
    causal = _prepare_4d_causal_attention_mask(attention_mask, (batch, seq_len), hidden_states, past_key_values_length=0)
    x = hidden_states
    attentions = []
    for layer_idx in range(start_layer, len(model.model.layers)):
        out = model.model.layers[layer_idx](x, attention_mask=causal, position_ids=pos, past_key_value=None, output_attentions=output_attentions, use_cache=False)
        x = out[0]
        if output_attentions:
            attn = out[1]
            if attn is None:
                raise RuntimeError(f"server layer {layer_idx} did not return attention")
            attentions.append((int(layer_idx), attn.float()[0].detach() if detach else attn.float()[0]))
    final = model.model.norm(x)
    return final, model.lm_head(final), attentions


def _tail_loss(pred_tail: torch.Tensor, ref_tail: torch.Tensor, variable_mask: torch.Tensor) -> torch.Tensor:
    variable = variable_mask.to(pred_tail.device).bool()
    per_token = torch.mean((pred_tail.float() - ref_tail.to(pred_tail.device).float()) ** 2, dim=-1)
    return per_token[:, variable].mean()


def _b0v_optimize(model, tokenizer, cfg: SAERConfig, observed_activation: torch.Tensor, seq_len: int, device: torch.device, fixed_public: Dict[int, int], mask_audit: Any):
    import pia_server_attention_consistency_pia as sacr
    from pia_attention_guided import random_public_embeddings

    sacr_cfg = sacr.SACRConfig(
        method="variable_only_uniform",
        run_name=cfg.run_name,
        output_dir=cfg.output_dir,
        dataset_name=cfg.dataset_name,
        dataset_path=cfg.dataset_path,
        dataset_len=cfg.dataset_len,
        prompt_split_seed=cfg.prompt_split_seed,
        split=cfg.split,
        prompt_limit=cfg.prompt_limit,
        init_seed=cfg.init_seed,
        participant_number=cfg.participant_number,
        attacker_position=cfg.attacker_position,
        inverted_block_count=cfg.target_layer + 1,
        target_layer=cfg.target_layer,
        epoch=cfg.epoch,
        lr=cfg.lr,
        lambda_vocab=cfg.lambda_vocab,
        lambda_tail=0.0,
        lambda_attn=0.0,
        rho_tail=0.10,
        rho_attn=0.10,
        top_k_embedding=cfg.top_k_embedding,
        top_y_semantic=cfg.top_y_semantic,
        max_token_len=cfg.max_token_len,
        grad_clip=cfg.grad_clip,
        server_attention_layers="all",
        adaptive_discretization=True,
        semantic_speculation=True,
        local_files_only=cfg.local_files_only,
    )
    init_embeds = random_public_embeddings(tokenizer, model.get_input_embeddings(), seq_len, device, fixed_public)
    z, losses, history, _consistency, _grads = sacr.stage_b_optimize(
        model, tokenizer, sacr_cfg, observed_activation, seq_len, device, fixed_public, mask_audit, init_embeds
    )
    return z, losses, history


def _candidate_losses(
    model,
    cfg: SAERConfig,
    candidate_ids: Sequence[int],
    attention_mask: torch.Tensor,
    observed_activation: torch.Tensor,
    ref_tail: torch.Tensor,
    ref_attentions,
    graph: EdgeGraph,
    variable_mask: torch.Tensor,
    device: torch.device,
    score_auxiliary: bool = True,
):
    ids_tensor = torch.tensor([list(candidate_ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        h_pred = capture_prefix_activation(model, cfg.target_layer, input_ids=ids_tensor, attention_mask=attention_mask)
    cal = float(torch.mean((h_pred.float() - observed_activation.to(device).float()) ** 2).detach().cpu())
    method = canonical_method(cfg.method)
    tail_value = None
    edge_value = None
    if score_auxiliary and method in {"E1", "E2", "E3"}:
        tail, _logits, pred_attn = server_suffix_forward(
            model,
            cfg.target_layer + 1,
            h_pred.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=method in {"E2", "E3"},
            detach=True,
        )
        if method in {"E1", "E3"}:
            tail_value = float(_tail_loss(tail, ref_tail, variable_mask).detach().cpu())
        if method in {"E2", "E3"} and not graph.is_empty:
            edge_value = float(edge_consistency_loss(ref_attentions, pred_attn, graph, variable_mask)[0].detach().cpu())
    return cal, tail_value, edge_value


def _calibrate_and_rerank(model, tokenizer, cfg: SAERConfig, observed_activation: torch.Tensor, z: torch.Tensor, fixed_public: Dict[int, int], mask_audit: Any, ref_tail: torch.Tensor, ref_attentions, graph: EdgeGraph, device: torch.device):
    embed_layer = model.get_input_embeddings()
    seq_len = int(z.shape[1])
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    fixed_positions = set(int(x) for x in fixed_public)
    candidate_sets = embedding_candidates(z, embed_layer.weight, max(1, cfg.top_k_embedding))
    recovered = naive_discretization(z, embed_layer.weight)
    for pos, token_id in fixed_public.items():
        if 0 <= pos < len(recovered):
            recovered[pos] = int(token_id)
    audits: List[Dict[str, Any]] = []
    edge_values: List[float] = []
    tail_values: List[float] = []
    overrides = 0
    fixed_overrides = 0
    for pos in range(seq_len):
        if pos in fixed_positions:
            continue
        candidates = list(candidate_sets[pos])
        if cfg.top_y_semantic > 0 and pos > 0:
            candidates.extend(semantic_candidates(model, recovered[:pos], cfg.top_y_semantic))
        candidates = list(dict.fromkeys(int(x) for x in candidates))
        scored: List[CandidateScore] = []
        for idx, cand in enumerate(candidates):
            proposal = list(recovered)
            proposal[pos] = int(cand)
            cal, tail, edge = _candidate_losses(
                model,
                cfg,
                proposal,
                attention_mask,
                observed_activation,
                ref_tail,
                ref_attentions,
                graph,
                mask_audit.variable_mask,
                device,
                score_auxiliary=False,
            )
            scored.append(CandidateScore(index=idx, ids=proposal, calibration_score=cal, tail_loss=tail, edge_loss=edge))
        ordered = sorted(scored, key=lambda c: (float(c.calibration_score), int(c.index)))
        margin = float("inf") if len(ordered) < 2 else float(ordered[1].calibration_score) - float(ordered[0].calibration_score)
        needs_aux = (
            canonical_method(cfg.method) in {"E1", "E2", "E3"}
            and len(ordered) >= 2
            and margin <= float(cfg.margin_threshold)
            and not (canonical_method(cfg.method) in {"E2", "E3"} and graph.is_empty)
        )
        if needs_aux:
            rescored: List[CandidateScore] = []
            for item in scored:
                if float(item.calibration_score) - float(ordered[0].calibration_score) > float(cfg.calibration_tolerance):
                    rescored.append(item)
                    continue
                cal, tail, edge = _candidate_losses(
                    model,
                    cfg,
                    item.ids,
                    attention_mask,
                    observed_activation,
                    ref_tail,
                    ref_attentions,
                    graph,
                    mask_audit.variable_mask,
                    device,
                    score_auxiliary=True,
                )
                item = CandidateScore(index=item.index, ids=item.ids, calibration_score=cal, tail_loss=tail, edge_loss=edge)
                rescored.append(item)
                if edge is not None:
                    edge_values.append(edge)
                if tail is not None:
                    tail_values.append(tail)
            scored = rescored
        result = gated_rerank_candidates(
            scored,
            RerankConfig(
                method=cfg.method,
                lambda_tail_rank=cfg.lambda_tail_rank,
                lambda_edge_rank=cfg.lambda_edge_rank,
                margin_threshold=cfg.margin_threshold,
                calibration_tolerance=cfg.calibration_tolerance,
                edge_improvement_threshold=cfg.edge_improvement_threshold,
                tail_improvement_threshold=cfg.tail_improvement_threshold,
            ),
            graph.is_empty,
        )
        if result.override:
            overrides += 1
            fixed_overrides += int(pos in fixed_positions)
        recovered = list(result.selected_ids)
        audits.append(
            {
                "position": int(pos),
                "candidate_count": len(scored),
                "baseline_index": result.baseline_index,
                "selected_index": result.selected_index,
                "override": result.override,
                "reason": result.reason,
                "baseline_token": int(result.baseline_ids[pos]),
                "selected_token": int(result.selected_ids[pos]),
                "final_scores": result.final_scores,
            }
        )
    for pos, token_id in fixed_public.items():
        if 0 <= pos < len(recovered):
            recovered[pos] = int(token_id)
    summary = {
        "override_count": overrides,
        "fixed_token_override_count": fixed_overrides,
        "override_rate": overrides / max(1, len(audits)),
        "mean_edge_score": float(statistics.mean(edge_values)) if edge_values else None,
        "mean_tail_score": float(statistics.mean(tail_values)) if tail_values else None,
        "edge_score_distinguishes_candidates": bool((max(edge_values) - min(edge_values)) > 1e-9) if len(edge_values) > 1 else False,
    }
    return recovered, audits, summary


def invert_observed(model, tokenizer, cfg, observed_activation, seq_len, device):
    cfg.method = canonical_method(cfg.method)
    set_seed(cfg.init_seed)
    fixed_public = inferred_boundary_special_tokens(tokenizer, seq_len) if cfg.fix_boundary_specials else {}
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    mask_audit = build_variable_mask(
        attention_mask=attention_mask,
        fixed_public=fixed_public,
        special_token_ids=getattr(tokenizer, "all_special_ids", None),
        token_ids=None,
    )
    z, losses, history = _b0v_optimize(model, tokenizer, cfg, observed_activation, seq_len, device, fixed_public, mask_audit)
    with torch.no_grad():
        ref_tail, _logits, ref_attentions = server_suffix_forward(
            model,
            cfg.target_layer + 1,
            observed_activation.to(dtype=next(model.parameters()).dtype),
            attention_mask,
            output_attentions=True,
            detach=True,
        )
    graph = build_attention_edge_graph(
        ref_attentions,
        mask_audit.variable_mask,
        top_r=cfg.top_r,
        sink_threshold=cfg.sink_threshold,
        entropy_min=cfg.entropy_min,
        min_head_support=cfg.min_head_support,
        min_layer_support=cfg.min_layer_support,
    )
    recovered, audits, summary = _calibrate_and_rerank(
        model, tokenizer, cfg, observed_activation, z, fixed_public, mask_audit, ref_tail, ref_attentions, graph, device
    )
    return recovered, losses, graph, audits, summary, history


def validate_attack_api() -> Dict[str, Any]:
    checked = [invert_observed, build_attention_edge_graph, edge_consistency_loss, gated_rerank_candidates]
    sig = inspect.signature(invert_observed)
    banned = ["prompt", "original", "input_ids", "token_ids", "ground", "dummy"]
    signature_hits = [name for name in sig.parameters if any(fragment in name for fragment in banned)]
    global_hits = []
    for fn in checked:
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in {"prompt", "original_ids", "original_tokens", "ground_truth_embedding", "dummy_x", "dummy_attention", "gradient_token_reranking"}:
                global_hits.append({"function": fn.__name__, "name": node.id})
    return {"signature": str(sig), "parameter_names": list(sig.parameters), "banned_signature_hits": signature_hits, "banned_global_reference_hits": global_hits, "passes": not signature_hits and not global_hits}


def split_path(output_root: str, seed: int) -> Path:
    return Path(output_root) / "splits" / f"skytrax150_split_{seed}.json"


def ensure_split(args) -> Dict[str, Any]:
    path = split_path(args.output_root, args.prompt_split_seed)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    prompts, meta = load_dataset_prompts(args.dataset_name, args.dataset_path, args.dataset_len, args.prompt_split_seed)
    ids = list(range(len(prompts)))
    random.Random(args.prompt_split_seed).shuffle(ids)
    split = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "dataset_len": args.dataset_len,
        "prompt_split_seed": args.prompt_split_seed,
        "dataset_meta": meta,
        "all_ids": ids,
        "tune_ids": ids[: args.tune_count],
        "holdout_ids": ids[args.tune_count : args.tune_count + args.holdout_count],
        "note": "Skytrax-150 fixed split; init_seed is independent.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    json_dump(path, split)
    return split


def run_sacr_gradient_audit(args) -> Dict[str, Any]:
    import pia_server_attention_consistency_pia as sacr

    cfg = sacr.SACRConfig(
        method="attention_consistency",
        run_name="gradient_audit",
        output_dir="runs/server_attention_consistency_pia/reports/gradient_audit_tmp",
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=3,
        prompt_split_seed=args.prompt_split_seed,
        split="all",
        prompt_limit=1,
        init_seed=42,
        participant_number=4,
        attacker_position=4,
        inverted_block_count=args.target_layer + 1,
        target_layer=args.target_layer,
        epoch=1,
        lr=0.1,
        lambda_vocab=0.001,
        lambda_tail=0.0,
        lambda_attn=0.05,
        rho_tail=0.1,
        rho_attn=0.1,
        top_k_embedding=10,
        top_y_semantic=10,
        max_token_len=args.max_token_len,
        grad_clip=1.0,
        server_attention_layers="all",
        adaptive_discretization=True,
        semantic_speculation=True,
        local_files_only=True,
    )
    tokenizer, model, device = sacr.load_model(cfg)
    prompts, _meta = load_dataset_prompts(args.dataset_name, args.dataset_path, 3, args.prompt_split_seed)
    encoded = tokenizer(prompts[0], add_special_tokens=True, truncation=False, return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        h_obs = capture_prefix_activation(model, args.target_layer, input_ids=ids, attention_mask=mask).detach()
    fixed = inferred_boundary_special_tokens(tokenizer, int(ids.shape[1]))
    audit = build_variable_mask(attention_mask=mask, fixed_public=fixed, special_token_ids=getattr(tokenizer, "all_special_ids", None), token_ids=None)
    embed_layer = model.get_input_embeddings()
    z = embed_layer(torch.randint(0, len(tokenizer), (1, int(ids.shape[1])), device=device)).detach().clone().float().requires_grad_(True)
    hidden = capture_prefix_activation(model, args.target_layer, inputs_embeds=z.to(dtype=embed_layer.weight.dtype), attention_mask=mask)
    with torch.no_grad():
        _tail, _logits, ref_attn = sacr.server_forward_consistency(model, args.target_layer + 1, h_obs.to(dtype=next(model.parameters()).dtype), mask, True, "all", detach_attentions=True)
    _ptail, _plogits, pred_attn = sacr.server_forward_consistency(model, args.target_layer + 1, hidden.to(dtype=next(model.parameters()).dtype), mask, True, "all", detach_attentions=False)
    loss, stats = sacr.attention_js_consistency_loss(ref_attn, pred_attn, audit.variable_mask)
    grad = torch.autograd.grad(loss, z, retain_graph=True, allow_unused=True)[0]
    norm = float(torch.linalg.vector_norm(grad.detach().float()).cpu()) if grad is not None else 0.0
    perturbed = z.detach().clone()
    perturbed[:, audit.variable_mask.to(device).bool(), :] += torch.randn_like(perturbed[:, audit.variable_mask.to(device).bool(), :]) * 1e-4
    perturbed.requires_grad_(True)
    hidden_p = capture_prefix_activation(model, args.target_layer, inputs_embeds=perturbed.to(dtype=embed_layer.weight.dtype), attention_mask=mask)
    _tail_p, _logits_p, pred_attn_p = sacr.server_forward_consistency(model, args.target_layer + 1, hidden_p.to(dtype=next(model.parameters()).dtype), mask, True, "all", detach_attentions=False)
    loss_p, _stats_p = sacr.attention_js_consistency_loss(ref_attn, pred_attn_p, audit.variable_mask)
    payload = {
        "methods_audited": ["C2 attention_consistency", "C3 tail_attention_consistency"],
        "attention_used_rows": stats.get("used_rows"),
        "attention_skipped_rows": stats.get("skipped_rows"),
        "L_attn": float(loss.detach().cpu()),
        "grad_z_L_attn_norm": norm,
        "attention_grad_norm": norm,
        "difference_when_attention_loss_closed": norm,
        "perturbed_L_attn": float(loss_p.detach().cpu()),
        "perturbation_changes_L_attn": abs(float(loss_p.detach().cpu()) - float(loss.detach().cpu())) > 1e-10,
        "detach_or_no_grad_issue_detected": norm <= 1e-12,
        "fixed_public_invalid_positions_excluded": True,
        "conclusion_code": "A" if norm > 1e-12 and int(stats.get("used_rows", 0)) > 0 else "B",
    }
    out_json = Path("runs/server_attention_consistency_pia/reports/attention_gradient_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json_dump(out_json, payload)
    Path("analysis").mkdir(exist_ok=True)
    conclusion = "A. attention loss 梯度链路正确，但方法性能为负。" if payload["conclusion_code"] == "A" else "B. attention loss 没有实际梯度，需要修复后才能评价。"
    Path("analysis/sacr_attention_gradient_audit.md").write_text(
        "# SACR Attention Gradient Audit\n\n"
        f"结论：{conclusion}\n\n"
        f"- attention_used_rows: {payload['attention_used_rows']}\n"
        f"- attention_skipped_rows: {payload['attention_skipped_rows']}\n"
        f"- L_attn: {payload['L_attn']}\n"
        f"- ||grad_z(L_attn)||: {payload['grad_z_L_attn_norm']}\n"
        f"- 关闭 attention loss 后的差异: {payload['difference_when_attention_loss_closed']}\n"
        f"- 微小扰动后 L_attn 是否变化: {payload['perturbation_changes_L_attn']}\n"
        f"- fixed public / invalid positions 是否排除: {payload['fixed_public_invalid_positions_excluded']}\n\n"
        "本审计没有重跑 C2/C3 大规模实验，也没有修改 SACR 方法。\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def write_docs() -> None:
    Path("analysis").mkdir(exist_ok=True)
    Path("analysis/suffix_attention_edge_rerank_design.md").write_text(
        "# SAER-PIA Design\n\n"
        "SAER keeps B0V variable-only activation matching as the continuous optimization backbone. "
        "Server-side suffix attention from H_obs is frozen into an edge graph and used only for gated discrete candidate reranking.\n",
        encoding="utf-8",
    )
    Path("analysis/suffix_attention_edge_rerank_report.md").write_text(
        "# SAER-PIA Report\n\n"
        "The implementation currently contains the audited edge-graph and gated-rerank core plus SACR gradient audit entrypoint. "
        "No held-out effectiveness claim is made.\n",
        encoding="utf-8",
    )


def _load_model_for_cfg(cfg: SAERConfig):
    set_seed(cfg.init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model = load_tinyllama(dtype, local_files_only=cfg.local_files_only)
    model.to(device)
    return tokenizer, model, device


def _prompts_for_cfg(cfg: SAERConfig, output_root: str):
    split = json.loads(split_path(output_root, cfg.prompt_split_seed).read_text(encoding="utf-8"))
    prompts, meta = load_dataset_prompts(cfg.dataset_name, cfg.dataset_path, cfg.dataset_len, cfg.prompt_split_seed)
    ids = split[f"{cfg.split}_ids"] if cfg.split in {"tune", "holdout"} else split["all_ids"]
    if cfg.prompt_limit > 0:
        ids = ids[: cfg.prompt_limit]
    return [(int(idx), prompts[int(idx)]) for idx in ids], {"canonical_dataset_meta": meta, "split": split, "selected_ids": ids}


def _cfg_from_args(args, method: str, split: str, prompt_limit: int, init_seed: int, subdir: str, tag: str = "") -> SAERConfig:
    run_name = "_".join(
        part
        for part in [canonical_method(method), split, f"init{init_seed}", f"layer{args.target_layer}", f"epoch{args.epoch}", tag]
        if part
    )
    return SAERConfig(
        method=canonical_method(method),
        run_name=run_name,
        output_dir=str(Path(args.output_root) / subdir / run_name),
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_len=args.dataset_len,
        prompt_split_seed=args.prompt_split_seed,
        split=split,
        prompt_limit=prompt_limit,
        init_seed=init_seed,
        participant_number=args.participant_number,
        attacker_position=args.attacker_position,
        target_layer=args.target_layer,
        epoch=args.epoch,
        lr=args.lr,
        lambda_vocab=args.lambda_vocab,
        top_k_embedding=args.k,
        top_y_semantic=args.y,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        top_r=args.top_r,
        sink_threshold=args.sink_threshold,
        entropy_min=args.entropy_min,
        min_head_support=args.min_head_support,
        min_layer_support=args.min_layer_support,
        margin_threshold=args.margin_threshold,
        lambda_tail_rank=args.lambda_tail_rank,
        lambda_edge_rank=args.lambda_edge_rank,
        calibration_tolerance=args.calibration_tolerance,
        edge_improvement_threshold=args.edge_improvement_threshold,
        tail_improvement_threshold=args.tail_improvement_threshold,
        local_files_only=True,
    )


def _mean(rows: List[Dict[str, Any]], key: str):
    vals = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return {"mean": float(statistics.mean(vals)) if vals else None, "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None}


def run_one_config(cfg: SAERConfig, output_root: str, resume: bool = False) -> Dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "COMPLETE"
    if resume and done.exists():
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    json_dump(out / "config.json", {"title": "SAER-PIA", **asdict(cfg)})
    tokenizer, model, device = _load_model_for_cfg(cfg)
    prompt_items, dataset_meta = _prompts_for_cfg(cfg, output_root)
    json_dump(out / "dataset_meta.json", dataset_meta)
    predictions = out / "predictions.jsonl"
    failures = out / "failures.jsonl"
    audits_path = out / "candidate_rerank_audit.jsonl"
    for path in [predictions, failures, audits_path]:
        if path.exists() and not resume:
            path.unlink()
        path.touch(exist_ok=True)
    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    edge_stats: List[Dict[str, Any]] = []
    for prompt_id, prompt_text in prompt_items:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = __import__("time").time()
            encoded = tokenizer(prompt_text, add_special_tokens=True, truncation=False, return_tensors="pt")
            ids_tensor = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            if ids_tensor.shape[1] > cfg.max_token_len:
                raise RuntimeError(f"prompt_id={prompt_id} exceeds max_token_len")
            original_ids = [int(x) for x in ids_tensor[0].detach().cpu().tolist()]
            with torch.no_grad():
                observed = capture_prefix_activation(model, cfg.target_layer, input_ids=ids_tensor, attention_mask=attention_mask).detach()
            recovered_ids, losses, graph, audits, rerank_summary, _history = invert_observed(
                model, tokenizer, cfg, observed, int(ids_tensor.shape[1]), device
            )
            override_correct = 0
            override_incorrect = 0
            for item in audits:
                if item.get("override"):
                    pos = int(item["position"])
                    correct = pos < len(original_ids) and int(item["selected_token"]) == int(original_ids[pos])
                    override_correct += int(correct)
                    override_incorrect += int(not correct)
            elapsed = __import__("time").time() - start
            row = {
                "method": cfg.method,
                "prompt_id": int(prompt_id),
                "prompt": prompt_text,
                "original_text": text_from_ids(tokenizer, original_ids),
                "recovered_text": text_from_ids(tokenizer, recovered_ids),
                "original_token_ids": original_ids,
                "recovered_token_ids": recovered_ids,
                "original_tokens": token_texts(tokenizer, original_ids),
                "recovered_tokens": token_texts(tokenizer, recovered_ids),
                "token_accuracy": token_accuracy(tokenizer, original_ids, recovered_ids),
                "bleu": bleu_score(tokenizer, original_ids, recovered_ids),
                "nerr": optional_nerr(text_from_ids(tokenizer, original_ids), text_from_ids(tokenizer, recovered_ids)),
                "runtime_seconds": elapsed,
                "peak_gpu_memory_mb": peak_memory_mb(),
                **losses,
                "edge_count": graph.edge_count,
                "graph_empty": graph.is_empty,
                "override_count": rerank_summary["override_count"],
                "override_rate": rerank_summary["override_rate"],
                "override_correct": override_correct,
                "override_incorrect": override_incorrect,
                "fixed_token_override_count": rerank_summary["fixed_token_override_count"],
                "mean_edge_score": rerank_summary["mean_edge_score"],
                "mean_tail_score": rerank_summary["mean_tail_score"],
                "edge_score_distinguishes_candidates": rerank_summary["edge_score_distinguishes_candidates"],
                "recovery_uses_ground_truth_tokens": False,
            }
            jsonl_append(predictions, row)
            jsonl_append(audits_path, {"prompt_id": int(prompt_id), "audit": audits})
            rows.append(row)
            edge_stats.append({"prompt_id": int(prompt_id), **graph.stats})
            json_dump(out / "edge_graph.json", graph.to_json())
            print(
                f"method={cfg.method} prompt_id={prompt_id} token_accuracy={row['token_accuracy']:.6f} "
                f"bleu={row['bleu']:.6f} edge_count={graph.edge_count} override={row['override_count']} elapsed={elapsed:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {"method": cfg.method, "prompt_id": int(prompt_id), "error": repr(exc), "traceback": __import__("traceback").format_exc()}
            jsonl_append(failures, failure)
            failed.append(failure)
            print(json.dumps({"failure": failure}, ensure_ascii=True), flush=True)
    metrics = {
        "method": cfg.method,
        "run_name": cfg.run_name,
        "split": cfg.split,
        "init_seed": cfg.init_seed,
        "token_accuracy": _mean(rows, "token_accuracy"),
        "bleu": _mean(rows, "bleu"),
        "runtime_seconds": _mean(rows, "runtime_seconds"),
        "completed_sample_count": len(rows),
        "failed_sample_count": len(failed),
        "dataset_meta": dataset_meta,
        "output_dir": cfg.output_dir,
        "override_count": sum(int(row.get("override_count", 0)) for row in rows),
        "override_correct": sum(int(row.get("override_correct", 0)) for row in rows),
        "override_incorrect": sum(int(row.get("override_incorrect", 0)) for row in rows),
        "mean_override_rate": float(statistics.mean([float(row.get("override_rate", 0.0)) for row in rows])) if rows else None,
        "graph_empty_count": sum(1 for row in rows if row.get("graph_empty")),
        "mean_edge_count": float(statistics.mean([float(row.get("edge_count", 0.0)) for row in rows])) if rows else None,
        "edge_score_distinguishing_sample_count": sum(1 for row in rows if row.get("edge_score_distinguishes_candidates")),
    }
    json_dump(out / "metrics.json", metrics)
    json_dump(out / "edge_graph_stats.json", {"samples": edge_stats})
    json_dump(out / "leakage_verification.json", {"attack_api_check": validate_attack_api(), "attack_api_passes": validate_attack_api()["passes"]})
    done.write_text("complete\n", encoding="utf-8")
    return metrics


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = __import__("csv").DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": metrics["method"],
        "token_accuracy": metrics["token_accuracy"]["mean"],
        "bleu": metrics["bleu"]["mean"],
        "completed_sample_count": metrics["completed_sample_count"],
        "failed_sample_count": metrics["failed_sample_count"],
        "override_count": metrics["override_count"],
        "override_correct": metrics["override_correct"],
        "override_incorrect": metrics["override_incorrect"],
        "mean_override_rate": metrics["mean_override_rate"],
        "graph_empty_count": metrics["graph_empty_count"],
        "mean_edge_count": metrics["mean_edge_count"],
        "edge_score_distinguishing_sample_count": metrics["edge_score_distinguishing_sample_count"],
        "output_dir": metrics["output_dir"],
    }


def run_smoke(args) -> None:
    ensure_split(args)
    rows = []
    for method in ["E0", "E1", "E2", "E3"]:
        cfg = _cfg_from_args(args, method, "tune", 3, 42, "smoke", "smoke3")
        rows.append(_summary_row(run_one_config(cfg, args.output_root, args.resume)))
    report_dir = Path(args.output_root) / "reports"
    _write_csv(report_dir / "smoke_summary.csv", rows)
    lines = ["# SAER Smoke Summary", "", "| method | Acc | BLEU | override | correct/incorrect | graph empty | mean edges |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['token_accuracy']} | {row['bleu']} | {row['override_count']} | "
            f"{row['override_correct']}/{row['override_incorrect']} | {row['graph_empty_count']} | {row['mean_edge_count']} |"
        )
    (report_dir / "smoke_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["verify", "sacr-gradient-audit", "single", "smoke"], default="verify")
    parser.add_argument("--method", default="E0")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--dataset-name", default="Skytrax")
    parser.add_argument("--dataset-path", default="data/skytrax_150.json")
    parser.add_argument("--dataset-len", type=int, default=150)
    parser.add_argument("--prompt-split-seed", type=int, default=20260706)
    parser.add_argument("--tune-count", type=int, default=50)
    parser.add_argument("--holdout-count", type=int, default=100)
    parser.add_argument("--target-layer", type=int, default=17)
    parser.add_argument("--max-token-len", type=int, default=896)
    parser.add_argument("--split", choices=["tune", "holdout", "all"], default="tune")
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--init-seed", type=int, default=42)
    parser.add_argument("--participant-number", type=int, default=4)
    parser.add_argument("--attacker-position", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lambda-vocab", type=float, default=0.001)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--y", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--top-r", type=int, default=3)
    parser.add_argument("--sink-threshold", type=float, default=0.70)
    parser.add_argument("--entropy-min", type=float, default=0.30)
    parser.add_argument("--min-head-support", type=int, default=2)
    parser.add_argument("--min-layer-support", type=int, default=2)
    parser.add_argument("--margin-threshold", type=float, default=0.01)
    parser.add_argument("--lambda-tail-rank", type=float, default=0.10)
    parser.add_argument("--lambda-edge-rank", type=float, default=0.10)
    parser.add_argument("--calibration-tolerance", type=float, default=0.02)
    parser.add_argument("--edge-improvement-threshold", type=float, default=0.0)
    parser.add_argument("--tail-improvement-threshold", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "verify":
        split = ensure_split(args)
        write_docs()
        print(json.dumps({"attack_api": validate_attack_api(), "split_path": str(split_path(args.output_root, args.prompt_split_seed)), "tune_count": len(split["tune_ids"]), "holdout_count": len(split["holdout_ids"])}, ensure_ascii=False))
    elif args.mode == "sacr-gradient-audit":
        run_sacr_gradient_audit(args)
    elif args.mode == "single":
        ensure_split(args)
        cfg = _cfg_from_args(args, args.method, args.split, args.prompt_limit, args.init_seed, args.split, "single")
        run_one_config(cfg, args.output_root, args.resume)
    elif args.mode == "smoke":
        run_smoke(args)


if __name__ == "__main__":
    main()
