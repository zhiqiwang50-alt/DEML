import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    top_k_top_p_filtering,
)


DEFAULT_PROMPTS = [
    "The airline staff were friendly and the seat was comfortable.",
    "The flight was delayed for three hours and the service was poor.",
    "The hotel room was clean and the breakfast was excellent.",
    "Customer support solved my problem quickly.",
    "The food was expensive but the quality was good.",
]


@dataclass
class ModelAdapter:
    model_type: str
    task: str
    blocks: torch.nn.ModuleList
    block_prefix: str

    def block_name(self, layer_id: int) -> str:
        return f"{self.block_prefix}.{layer_id}"


def ensure_prompt_file(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_PROMPTS, f, indent=2)


def load_prompts(path: str, dataset_len: int) -> List[str]:
    ensure_prompt_file(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{path} must be a JSON string list")
    return data[:dataset_len]


def extract_first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return extract_first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Hook output did not contain a tensor: {type(output)!r}")


def capture_block_activation(
    model: torch.nn.Module,
    block: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    captured: List[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        captured.append(extract_first_tensor(output))

    handle = block.register_forward_hook(hook)
    try:
        _ = model(**inputs)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(
            "Forward hook did not capture an activation. "
            "Check model adapter, target layer, and model forward inputs."
        )
    return captured[0]


def build_adapter(model: torch.nn.Module, model_type: str) -> ModelAdapter:
    if model_type in {"llama", "mistral", "qwen2"} and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
        if layers is not None:
            return ModelAdapter(model_type=model_type, task="causal", blocks=layers, block_prefix="model.layers")

    if model_type == "gpt2" and hasattr(model, "transformer"):
        blocks = getattr(model.transformer, "h", None)
        if blocks is not None:
            return ModelAdapter(model_type=model_type, task="causal", blocks=blocks, block_prefix="transformer.h")

    if model_type == "bert" and hasattr(model, "bert"):
        encoder = getattr(model.bert, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            return ModelAdapter(
                model_type=model_type,
                task="masked",
                blocks=encoder.layer,
                block_prefix="bert.encoder.layer",
            )

    raise ValueError(f"Unsupported model_type={model_type!r} for small-model inversion")


def local_files_only() -> bool:
    return os.environ.get("PIA_ALLOW_HF_DOWNLOAD") != "1"


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype) -> Tuple[Any, torch.nn.Module, ModelAdapter]:
    use_local_files = local_files_only()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=use_local_files)
    if config.model_type == "bert":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=use_local_files,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=use_local_files,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=use_local_files,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model.eval()
    adapter = build_adapter(model, config.model_type)
    return tokenizer, model, adapter


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tokenize_prompt(tokenizer: Any, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer(
        prompt,
        padding=False,
        truncation=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in tokenized.items()}


def token_texts(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    return tokenizer.convert_ids_to_tokens(list(token_ids))


def ids_to_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def fixed_token_positions(tokenizer: Any, input_ids: torch.Tensor) -> List[int]:
    special_ids = set(tokenizer.all_special_ids or [])
    if not special_ids:
        return []
    return [idx for idx, token_id in enumerate(input_ids[0].tolist()) if token_id in special_ids]


def variable_positions(seq_len: int, fixed_positions: Iterable[int]) -> List[int]:
    fixed = set(fixed_positions)
    return [idx for idx in range(seq_len) if idx not in fixed]


def build_inputs_embeds(
    base_embeds: torch.Tensor,
    learnable: torch.Tensor,
    var_positions: Sequence[int],
) -> torch.Tensor:
    embeds = base_embeds.clone()
    if var_positions:
        index = torch.tensor(var_positions, device=base_embeds.device, dtype=torch.long)
        embeds[:, index, :] = learnable.to(dtype=base_embeds.dtype).unsqueeze(0)
    return embeds


def optimize_embeddings(
    model: torch.nn.Module,
    block: torch.nn.Module,
    embed_layer: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_activation: torch.Tensor,
    fixed_positions: Sequence[int],
    epoch: int,
    lr: float,
    init_param: float,
    optim_method: str,
    log_every: int,
) -> Tuple[torch.Tensor, float, float]:
    device = input_ids.device
    seq_len = input_ids.shape[1]
    var_positions = variable_positions(seq_len, fixed_positions)
    base_embeds = embed_layer(input_ids).detach()
    if var_positions:
        learnable = torch.empty(
            len(var_positions),
            base_embeds.shape[-1],
            device=device,
            dtype=torch.float32,
        ).uniform_(-init_param, init_param)
        learnable.requires_grad_(True)
        optimizer = torch.optim.Adam([learnable], lr=lr)
    else:
        learnable = torch.empty(0, base_embeds.shape[-1], device=device, dtype=dtype)
        optimizer = None

    final_loss = 0.0
    final_cosine = 0.0
    loss_func = torch.nn.MSELoss(reduction="mean")
    target_activation = target_activation.detach()

    for step in range(epoch):
        embeds = build_inputs_embeds(base_embeds, learnable, var_positions)
        relaxed_activation = capture_block_activation(
            model,
            block,
            {"inputs_embeds": embeds, "attention_mask": attention_mask},
        )
        target = target_activation.to(relaxed_activation.device)
        cosine = F.cosine_similarity(relaxed_activation.float(), target.float(), dim=-1).mean()
        mse = loss_func(relaxed_activation.float(), target.float())
        loss = mse if optim_method == "MSELoss" else -cosine

        if optimizer is None:
            final_loss = float(loss.detach().cpu())
            final_cosine = float(cosine.detach().cpu())
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        final_cosine = float(cosine.detach().cpu())
        if log_every > 0 and ((step + 1) % log_every == 0 or step == epoch - 1):
            print(
                f"epoch={step + 1} loss={final_loss:.6f} "
                f"activation_cosine={final_cosine:.6f}",
                flush=True,
            )

    return build_inputs_embeds(base_embeds, learnable.detach(), var_positions), final_loss, final_cosine


def topk_embedding_candidates(
    embeds: torch.Tensor,
    embed_layer: torch.nn.Module,
    tokenizer: Any,
    top_k: int,
    filter_nonascii: bool,
) -> Tuple[List[List[int]], List[List[float]]]:
    squeezed = embeds.squeeze(0)
    weight = embed_layer.weight.detach()
    candidates: List[List[int]] = []
    scores: List[List[float]] = []
    max_k = min(max(top_k, 1), weight.shape[0])
    for embed in squeezed:
        sims = F.cosine_similarity(embed.float(), weight.float(), dim=-1)
        values, indices = torch.topk(sims.detach().cpu(), max_k)
        ids: List[int] = []
        vals: List[float] = []
        for value, token_id in zip(values.tolist(), indices.tolist()):
            if filter_nonascii and not tokenizer.decode([token_id]).isascii():
                continue
            ids.append(int(token_id))
            vals.append(float(value))
        if not ids:
            ids = [int(indices[0])]
            vals = [float(values[0])]
        candidates.append(ids)
        scores.append(vals)
    return candidates, scores


def causal_next_token_candidates(
    model: torch.nn.Module,
    prefix_ids: Sequence[int],
    top_k: int,
) -> List[int]:
    if not prefix_ids or top_k <= 0:
        return []
    device = next(model.parameters()).device
    input_ids = torch.tensor([list(prefix_ids)], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, -1, :]
        filtered = top_k_top_p_filtering(logits, top_k=min(top_k, logits.shape[-1]), top_p=1.0)
        probs = F.softmax(filtered, dim=-1)
    return [int(idx) for idx in torch.topk(probs[0], min(top_k, probs.shape[-1])).indices.detach().cpu()]


def refine_with_activation_matching(
    model: torch.nn.Module,
    block: torch.nn.Module,
    attention_mask: torch.Tensor,
    target_activation: torch.Tensor,
    current_ids: List[int],
    candidates: List[List[int]],
    fixed_positions: Sequence[int],
    use_perplexity: bool,
    top_k_ppl: int,
) -> Tuple[List[int], float]:
    fixed = set(fixed_positions)
    device = attention_mask.device
    target_activation = target_activation.detach()
    best_position_cosines: List[float] = []

    for pos, candidate_ids in enumerate(candidates):
        if pos in fixed:
            best_position_cosines.append(1.0)
            continue
        merged = list(candidate_ids)
        if use_perplexity and pos > 0:
            merged.extend(causal_next_token_candidates(model, current_ids[:pos], top_k_ppl))
        merged = list(dict.fromkeys(int(item) for item in merged))
        if not merged:
            continue

        batch_ids = []
        for token_id in merged:
            proposal = list(current_ids)
            proposal[pos] = token_id
            batch_ids.append(proposal)
        proposal_input_ids = torch.tensor(batch_ids, device=device, dtype=torch.long)
        proposal_attention = attention_mask.expand(len(batch_ids), -1).contiguous()
        with torch.no_grad():
            activation = capture_block_activation(
                model,
                block,
                {"input_ids": proposal_input_ids, "attention_mask": proposal_attention},
            )
        target_pos = target_activation[:, pos : pos + 1, :].expand(len(batch_ids), -1, -1)
        sims = F.cosine_similarity(
            activation[:, pos : pos + 1, :].float().squeeze(1),
            target_pos.float().squeeze(1),
            dim=-1,
        )
        best_idx = int(torch.argmax(sims).detach().cpu())
        current_ids[pos] = int(merged[best_idx])
        best_position_cosines.append(float(sims[best_idx].detach().cpu()))

    mean_position_cosine = sum(best_position_cosines) / max(len(best_position_cosines), 1)
    return current_ids, mean_position_cosine


def token_accuracy(original_ids: Sequence[int], recovered_ids: Sequence[int]) -> float:
    if not original_ids:
        return 0.0
    matched = sum(int(a == b) for a, b in zip(original_ids, recovered_ids))
    return matched / len(original_ids)


def jsonl_append(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def run(args: argparse.Namespace) -> None:
    if args.disable_perplexity:
        use_perplexity = False
    else:
        use_perplexity = True

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompts(args.dataset_path, args.dataset_len)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer, model, adapter = load_model_and_tokenizer(args.base_model_name, dtype=dtype)
    device = select_device()
    model.to(device)

    if adapter.task == "masked":
        use_perplexity = False

    if args.num_invert_layers < 0 or args.num_invert_layers >= len(adapter.blocks):
        raise ValueError(
            f"Layer index {args.num_invert_layers} is out of range for "
            f"{args.base_model_name}: valid range is 0..{len(adapter.blocks) - 1}"
        )

    for param in model.parameters():
        param.requires_grad_(False)

    target_block = adapter.blocks[args.num_invert_layers]
    embed_layer = model.get_input_embeddings()
    output_jsonl = os.path.join(args.output_dir, "results.jsonl")
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    meta = {
        "model_name": args.base_model_name,
        "model_type": adapter.model_type,
        "task": adapter.task,
        "target_layer": args.num_invert_layers,
        "target_layer_module": adapter.block_name(args.num_invert_layers),
        "num_transformer_blocks": len(adapter.blocks),
        "dataset_path": args.dataset_path,
        "dataset_len": len(prompts),
        "epoch": args.epoch,
        "top_k_cos": args.top_k_cos,
        "top_k_ppl": args.top_k_ppl,
        "perplexity_enabled": use_perplexity,
        "note": (
            "BERT is a middle-activation inversion extension comparison experiment, "
            "not an autoregressive reproduction identical to the original paper."
            if adapter.task == "masked"
            else "Decoder-only causal LM prompt inversion experiment."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    print(
        "model_type={model_type} task={task} blocks={blocks} target_layer={layer} "
        "target_module={module}".format(
            model_type=adapter.model_type,
            task=adapter.task,
            blocks=len(adapter.blocks),
            layer=args.num_invert_layers,
            module=adapter.block_name(args.num_invert_layers),
        ),
        flush=True,
    )

    for sample_idx, prompt in enumerate(prompts):
        sample_start = time.time()
        tokenized = tokenize_prompt(tokenizer, prompt, device)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids))
        original_ids = [int(item) for item in input_ids[0].detach().cpu().tolist()]

        with torch.no_grad():
            target_activation = capture_block_activation(
                model,
                target_block,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            ).detach()
        print(
            f"sample={sample_idx} hook_activation_shape={tuple(target_activation.shape)}",
            flush=True,
        )

        fixed_positions = fixed_token_positions(tokenizer, input_ids)
        if adapter.task == "causal":
            fixed_positions = [pos for pos in fixed_positions if pos == 0]

        optimized_embeds, final_loss, final_activation_cosine = optimize_embeddings(
            model=model,
            block=target_block,
            embed_layer=embed_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_activation=target_activation,
            fixed_positions=fixed_positions,
            epoch=args.epoch,
            lr=args.lr,
            init_param=args.init_param,
            optim_method=args.optim_method,
            log_every=args.log_every,
        )

        candidates, candidate_scores = topk_embedding_candidates(
            optimized_embeds,
            embed_layer,
            tokenizer,
            top_k=args.top_k_cos,
            filter_nonascii=args.filter_nonascii,
        )
        recovered_ids = [items[0] for items in candidates]
        for pos in fixed_positions:
            recovered_ids[pos] = original_ids[pos]

        recovered_ids, refine_cosine = refine_with_activation_matching(
            model=model,
            block=target_block,
            attention_mask=attention_mask,
            target_activation=target_activation,
            current_ids=recovered_ids,
            candidates=candidates,
            fixed_positions=fixed_positions,
            use_perplexity=use_perplexity,
            top_k_ppl=args.top_k_ppl,
        )

        elapsed = time.time() - sample_start
        accuracy = token_accuracy(original_ids, recovered_ids)
        row = {
            "model_name": args.base_model_name,
            "model_type": adapter.model_type,
            "task": adapter.task,
            "target_layer": args.num_invert_layers,
            "target_layer_module": adapter.block_name(args.num_invert_layers),
            "sample_index": sample_idx,
            "prompt": prompt,
            "original_token_ids": original_ids,
            "original_tokens": token_texts(tokenizer, original_ids),
            "recovered_token_ids": recovered_ids,
            "recovered_tokens": token_texts(tokenizer, recovered_ids),
            "recovered_text": ids_to_text(tokenizer, recovered_ids),
            "embedding_candidate_cosine": [scores[0] for scores in candidate_scores],
            "final_optimization_loss": final_loss,
            "final_activation_cosine": final_activation_cosine,
            "refine_position_cosine": refine_cosine,
            "token_accuracy": accuracy,
            "runtime_seconds": elapsed,
            "perplexity_enabled": use_perplexity,
            "fixed_positions": list(fixed_positions),
        }
        jsonl_append(output_jsonl, row)
        print(
            f"sample={sample_idx} token_accuracy={accuracy:.4f} "
            f"loss={final_loss:.6f} activation_cosine={final_activation_cosine:.6f} "
            f"refine_cosine={refine_cosine:.6f} recovered_text={row['recovered_text']!r}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-name", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default="experiments/prompts_en.json")
    parser.add_argument("--dataset-len", type=int, default=5)
    parser.add_argument("--num-invert-layers", "--target-layer", dest="num_invert_layers", type=int, default=3)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--init-param", type=float, default=0.1)
    parser.add_argument("--optim-method", type=str, default="cosine", choices=["cosine", "MSELoss"])
    parser.add_argument("--top-k-cos", type=int, default=10)
    parser.add_argument("--top-k-ppl", type=int, default=10)
    parser.add_argument("--disable-perplexity", action="store_true")
    parser.add_argument("--filter-nonascii", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    return args


if __name__ == "__main__":
    run(parse_args())
