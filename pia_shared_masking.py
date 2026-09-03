import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
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
    uniform = torch.where(
        variable,
        torch.ones(seq_len, dtype=torch.float32, device=device),
        torch.zeros(seq_len, dtype=torch.float32, device=device),
    )
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
        rows.append({"position": pos,"valid": bool(valid[pos].detach().cpu()),"fixed_public": bool(fixed[pos].detach().cpu()),"special": bool(special[pos].detach().cpu()),"token_id_special": bool(token_id_special[pos].detach().cpu()),"known_public_special": bool(fixed[pos].detach().cpu()),"variable_mask": bool(variable[pos].detach().cpu()),"exclusion_reason": reason})
    return VariableMaskAudit(valid.detach(), fixed.detach(), special.detach(), variable.detach(), uniform.detach(), variable_count, known_public_special_positions, bool(token_id_special_available), last_valid_position, rows)


def variable_mask_audit_payload(audit: VariableMaskAudit) -> Dict[str, Any]:
    return {"variable_count": audit.variable_count,"known_public_special_positions": audit.known_public_special_positions,"token_id_based_special_mask_available": audit.token_id_based_special_mask_available,"last_valid_position": audit.last_valid_position,"semantic_note": ("fixed_public positions such as position 0 are public framing positions available to the attacker. When attack API token ids are unavailable, this audit cannot claim EOS detection by token id. The last valid position is only a sequence boundary; exclude it as EOS only if the public protocol explicitly fixes the final token as EOS."),"rows": audit.rows}


def uniform_all_token_activation_loss(pred: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    per_token = torch.mean((pred.float() - target.to(pred.device).float()) ** 2, dim=-1)
    valid = attention_mask.to(pred.device)[0].bool()
    if int(valid.sum().detach().cpu()) == 0:
        raise RuntimeError("attention_mask has no valid positions")
    return per_token[:, valid].mean()


def weighted_variable_activation_loss(pred: torch.Tensor, target: torch.Tensor, variable_mask: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    per_token = torch.mean((pred.float() - target.to(pred.device).float()) ** 2, dim=-1)
    variable = variable_mask.to(pred.device).bool()
    if int(variable.sum().detach().cpu()) == 0:
        raise RuntimeError("variable_mask has no variable positions")
    w = variable.float() if weights is None else weights.to(pred.device).float() * variable.float()
    return (per_token * w.unsqueeze(0)).sum() / w.sum().clamp_min(1e-12)
