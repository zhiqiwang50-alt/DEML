# CAR-PTR Strict Top-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate CAR-PTR, a white-box strict Top-1 prompt inversion branch using confidence-aware server attention rollout residual weighting plus conservative projected Top-1 residual correction.

**Architecture:** Extend the existing `pia_masked_server_attn_pia.py` path without redefining B0 or B0V. Add CAR as a scheduled attention-weighted residual loss and PTR as a post-Stage-B single-token correction pass that accepts only unweighted B0V residual improvements. Add tests, strict Top-1 validation, audit outputs, and a staged runner for Skytrax-28, multi-layer development, and Skytrax-150 validation.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers local model cache, TinyLlama-1.1B, existing DEML scripts, PowerShell local shell, SSH to rack-server-codex, `unittest` tests.

---

## Safety and Execution Boundaries

This implementation is limited to the authorized academic environment:

- Remote project: `/data/zhiqi/Collaborative_Inference/DEML` on `rack-server-codex`.
- Local model/data only: TinyLlama and Skytrax.
- Offline, isolated evaluation only.
- No third-party testing, network scanning, credential collection, privilege escalation, persistence, malware, destructive behavior, or external attack traffic.

Do not automatically run `git commit` or `git push`. Replace commit steps with `git status --short` and `git diff --stat` checkpoints because the user explicitly prohibited automatic commit/push.

## File Map

- Modify `pia_masked_server_attn_pia.py`: method registration, strict Top-1 validation, CAR attention/alpha construction, scheduled residual loss, PTR correction, audit output, CLI args.
- Modify `tests/test_masked_server_attn_mts.py`: focused unit tests for CAR, PTR, strict Top-1, API boundary, and audit invariants.
- Create `runs/car_ptr_strict_top1/runner.py`: staged experiment launcher with one job per GPU and frozen settings.
- Create `runs/car_ptr_strict_top1/analyze.py`: paired deltas, win/tie/loss, bootstrap confidence intervals, audit summaries, stage decisions.
- Create `analysis/car_ptr_strict_top1_report.md`: generated or manually updated report target summarizing results and stop/continue decisions.
- Keep unchanged by definition: B0 all-valid uniform baseline and B0V variable-only uniform baseline.

## Task 1: Add Failing Tests for Method Registration and Strict Top-1 Validation

**Files:**
- Modify: `tests/test_masked_server_attn_mts.py`
- Later modify: `pia_masked_server_attn_pia.py`

- [ ] **Step 1: Add tests for new aliases and strict Top-1 rejection**

Append these tests to `tests/test_masked_server_attn_mts.py`:

```python
class TestCarPtrStrictTop1(unittest.TestCase):
    def test_new_method_aliases_resolve(self):
        import pia_masked_server_attn_pia as pia

        self.assertEqual(
            pia.METHOD_ALIASES["P4CAR"],
            "confidence_aware_rollout_residual",
        )
        self.assertEqual(
            pia.METHOD_ALIASES["P4DPTR"],
            "attn_weighted_residual_ptr",
        )
        self.assertEqual(
            pia.METHOD_ALIASES["P4CARPTR"],
            "confidence_aware_rollout_residual_ptr",
        )
        self.assertIn(
            "confidence_aware_rollout_residual",
            pia.SERVER_ATTENTION_METHODS,
        )
        self.assertIn(
            "confidence_aware_rollout_residual_ptr",
            pia.PTR_METHODS,
        )

    def test_strict_top1_config_accepts_only_k1_y0_no_semantic(self):
        import argparse
        import pia_masked_server_attn_pia as pia

        ok = argparse.Namespace(
            method="confidence_aware_rollout_residual_ptr",
            k=1,
            y=0,
            semantic_speculation=False,
        )
        pia.validate_strict_top1_config(ok)

        bad_k = argparse.Namespace(
            method="confidence_aware_rollout_residual_ptr",
            k=10,
            y=0,
            semantic_speculation=False,
        )
        with self.assertRaises(ValueError):
            pia.validate_strict_top1_config(bad_k)

        bad_y = argparse.Namespace(
            method="confidence_aware_rollout_residual_ptr",
            k=1,
            y=1,
            semantic_speculation=False,
        )
        with self.assertRaises(ValueError):
            pia.validate_strict_top1_config(bad_y)

        bad_semantic = argparse.Namespace(
            method="confidence_aware_rollout_residual_ptr",
            k=1,
            y=0,
            semantic_speculation=True,
        )
        with self.assertRaises(ValueError):
            pia.validate_strict_top1_config(bad_semantic)
```

- [ ] **Step 2: Run the focused failing test**

Run on rack-server-codex:

```bash
cd /data/zhiqi/Collaborative_Inference/DEML
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deml
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_new_method_aliases_resolve
```

Expected: FAIL because `P4CAR`, `P4DPTR`, `P4CARPTR`, `PTR_METHODS`, or `validate_strict_top1_config` is not defined yet.

- [ ] **Step 3: Add method aliases and sets**

In `pia_masked_server_attn_pia.py`, near the existing `METHOD_ALIASES` and method sets, add:

```python
METHOD_ALIASES.update({
    "P4CAR": "confidence_aware_rollout_residual",
    "CAR": "confidence_aware_rollout_residual",
    "P4DPTR": "attn_weighted_residual_ptr",
    "P4CARPTR": "confidence_aware_rollout_residual_ptr",
    "CARPTR": "confidence_aware_rollout_residual_ptr",
})

CAR_METHODS = {
    "confidence_aware_rollout_residual",
    "confidence_aware_rollout_residual_ptr",
}

PTR_METHODS = {
    "attn_weighted_residual_ptr",
    "confidence_aware_rollout_residual_ptr",
}
```

Also include the CAR and PTR methods in the existing server-attention and attention-weighted residual method groups:

```python
SERVER_ATTENTION_METHODS.update(CAR_METHODS)
SERVER_ATTENTION_METHODS.update(PTR_METHODS)
ATTENTION_WEIGHTED_RESIDUAL_METHODS.update({
    "confidence_aware_rollout_residual",
    "attn_weighted_residual_ptr",
    "confidence_aware_rollout_residual_ptr",
})
```

- [ ] **Step 4: Add strict Top-1 validator**

In `pia_masked_server_attn_pia.py`, near existing argument/config validation helpers, add:

```python
def validate_strict_top1_config(args) -> None:
    method = canonical_method_name(getattr(args, "method", ""))
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
```

Call it after argument parsing and alias canonicalization, before model loading:

```python
validate_strict_top1_config(args)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1
```

Expected: PASS for method alias and strict Top-1 tests.

- [ ] **Step 6: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: modified `pia_masked_server_attn_pia.py` and `tests/test_masked_server_attn_mts.py`; no commit or push.

## Task 2: Implement CAR Confidence-Aware Attention Weights

**Files:**
- Modify: `tests/test_masked_server_attn_mts.py`
- Modify: `pia_masked_server_attn_pia.py`

- [ ] **Step 1: Add tests for consensus, confidence, bounds, and masks**

Append to `TestCarPtrStrictTop1`:

```python
    def test_car_attention_identical_rollouts_has_confidence_one(self):
        import torch
        import pia_masked_server_attn_pia as pia

        r_all = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
        r_last2 = r_all.clone()
        variable_mask = torch.tensor([False, True, True, True])
        alpha, confidence, stats = pia.build_confidence_aware_residual_alpha(
            r_all=r_all,
            r_last2=r_last2,
            variable_mask=variable_mask,
            weight_min=0.5,
            weight_max=1.5,
            strength=0.25,
            power=0.5,
            eps=1e-8,
        )

        self.assertEqual(float(alpha[0]), 0.0)
        self.assertTrue(torch.allclose(confidence[variable_mask], torch.ones(3), atol=1e-6))
        w = alpha[variable_mask].pow(2)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertGreaterEqual(float(w.min()), 0.5 - 1e-6)
        self.assertLessEqual(float(w.max()), 1.5 + 1e-6)
        self.assertEqual(stats["fixed_public_final_weight_sum"], 0.0)

    def test_car_attention_disagreement_moves_weights_toward_one(self):
        import torch
        import pia_masked_server_attn_pia as pia

        variable_mask = torch.tensor([False, True, True, True])
        r_all = torch.tensor([0.0, 0.8, 0.1, 0.1], dtype=torch.float32)
        r_last2 = torch.tensor([0.0, 0.1, 0.8, 0.1], dtype=torch.float32)
        alpha, confidence, _ = pia.build_confidence_aware_residual_alpha(
            r_all=r_all,
            r_last2=r_last2,
            variable_mask=variable_mask,
            weight_min=0.5,
            weight_max=1.5,
            strength=0.25,
            power=0.5,
            eps=1e-8,
        )

        self.assertLess(float(confidence[1]), 1.0)
        self.assertLess(float(abs(alpha[1].pow(2) - 1.0)), 0.5)
        self.assertAlmostEqual(float(alpha[variable_mask].pow(2).mean()), 1.0, places=6)
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_car_attention_identical_rollouts_has_confidence_one
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_car_attention_disagreement_moves_weights_toward_one
```

Expected: FAIL because `build_confidence_aware_residual_alpha` is not implemented.

- [ ] **Step 3: Add CAR helper**

In `pia_masked_server_attn_pia.py`, near existing attention weight helpers such as `bounded_mean_one_weights`, add:

```python
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
    variable_mask = variable_mask.bool()
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
            "final_max_weight": 0.0,
            "confidence_mean": 0.0,
            "rollout_disagreement_mean": 0.0,
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
    raw_w = raw_w.clamp(min=float(weight_min), max=float(weight_max))
    mean_w = raw_w.mean().clamp_min(eps)
    w = (raw_w / mean_w).clamp(min=float(weight_min), max=float(weight_max))
    w = w / w.mean().clamp_min(eps)

    alpha[variable_mask] = torch.sqrt(w)
    confidence[variable_mask] = variable_confidence
    fixed_weight_sum = float(alpha[~variable_mask].pow(2).sum().item())
    stats = {
        "variable_final_weight_mean": float(w.mean().item()),
        "variable_final_weight_std": float(w.std(unbiased=False).item()) if w.numel() > 1 else 0.0,
        "fixed_public_final_weight_sum": fixed_weight_sum,
        "final_max_weight": float(w.max().item()),
        "final_min_weight": float(w.min().item()),
        "confidence_mean": float(variable_confidence.mean().item()),
        "confidence_min": float(variable_confidence.min().item()),
        "confidence_max": float(variable_confidence.max().item()),
        "rollout_disagreement_mean": float(variable_disagreement.mean().item()),
        "rollout_disagreement_median": float(median_disagreement.item()),
    }
    return alpha, confidence, stats
```

- [ ] **Step 4: Run CAR helper tests**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_car_attention_identical_rollouts_has_confidence_one
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_car_attention_disagreement_moves_weights_toward_one
```

Expected: PASS.

- [ ] **Step 5: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: only planned files changed; no commit or push.

## Task 3: Add CAR Scheduling and Integrate CAR Loss

**Files:**
- Modify: `tests/test_masked_server_attn_mts.py`
- Modify: `pia_masked_server_attn_pia.py`

- [ ] **Step 1: Add tests for CAR schedule**

Append to `TestCarPtrStrictTop1`:

```python
    def test_car_schedule_uses_uniform_then_ramp_then_full_weight(self):
        import torch
        import pia_masked_server_attn_pia as pia

        alpha = torch.tensor([0.0, 0.8, 1.2], dtype=torch.float32)
        variable_mask = torch.tensor([False, True, True])

        warmup = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=50,
            warmup_epochs=50,
            ramp_end_epoch=70,
            weight_min=0.5,
            weight_max=1.5,
        )
        mid = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=60,
            warmup_epochs=50,
            ramp_end_epoch=70,
            weight_min=0.5,
            weight_max=1.5,
        )
        full = pia.schedule_residual_alpha(
            alpha=alpha,
            variable_mask=variable_mask,
            epoch_index=70,
            warmup_epochs=50,
            ramp_end_epoch=70,
            weight_min=0.5,
            weight_max=1.5,
        )

        self.assertEqual(float(warmup[0]), 0.0)
        self.assertTrue(torch.allclose(warmup[variable_mask].pow(2), torch.ones(2), atol=1e-6))
        self.assertGreater(float(mid[2].pow(2)), 1.0)
        self.assertLess(float(mid[2].pow(2)), float(full[2].pow(2)))
        self.assertAlmostEqual(float(full[variable_mask].pow(2).mean()), 1.0, places=6)
```

- [ ] **Step 2: Run the failing schedule test**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_car_schedule_uses_uniform_then_ramp_then_full_weight
```

Expected: FAIL because `schedule_residual_alpha` does not exist.

- [ ] **Step 3: Add schedule helper**

In `pia_masked_server_attn_pia.py`, near schedule helpers, add:

```python
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
    variable_mask = variable_mask.bool()
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
    scheduled_w = scheduled_w.clamp(min=float(weight_min), max=float(weight_max))
    scheduled_w = scheduled_w / scheduled_w.mean().clamp_min(eps)
    scheduled_w = scheduled_w.clamp(min=float(weight_min), max=float(weight_max))
    scheduled_w = scheduled_w / scheduled_w.mean().clamp_min(eps)
    out[variable_mask] = torch.sqrt(scheduled_w)
    return out
```

- [ ] **Step 4: Integrate CAR loss into Stage B**

In `stage_b_optimize` or the existing loss-selection block, branch CAR methods to use scheduled residual alpha:

```python
if method in CAR_METHODS:
    scheduled_alpha = schedule_residual_alpha(
        alpha=attention_alpha,
        variable_mask=variable_mask,
        epoch_index=epoch + 1,
        warmup_epochs=int(getattr(config, "car_warmup_epochs", 50)),
        ramp_end_epoch=int(getattr(config, "car_ramp_end_epoch", 70)),
        weight_min=float(getattr(config, "car_weight_min", 0.5)),
        weight_max=float(getattr(config, "car_weight_max", 1.5)),
    ).to(pred_hidden.device)
    act_loss = attention_weighted_residual_loss(
        pred_hidden,
        observed_hidden,
        scheduled_alpha,
        variable_mask=variable_mask,
    )
```

Use the existing vocabulary auxiliary loss unchanged:

```python
loss = act_loss + float(getattr(config, "vocab_weight", 0.1)) * vocab_loss
```

- [ ] **Step 5: Run CAR tests and existing MTS tests**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1
python -m unittest tests.test_masked_server_attn_mts
```

Expected: PASS.

- [ ] **Step 6: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: planned code/test changes; no commit or push.

## Task 4: Extend Server Attention Bundle for Full and Last2 Rollouts

**Files:**
- Modify: `tests/test_masked_server_attn_mts.py`
- Modify: `pia_masked_server_attn_pia.py`

- [ ] **Step 1: Add a shape and stats test for CAR bundle output**

Append to `TestCarPtrStrictTop1`:

```python
    def test_car_bundle_stats_require_both_rollout_views(self):
        import torch
        import pia_masked_server_attn_pia as pia

        r_all = torch.tensor([0.0, 0.2, 0.5, 0.3], dtype=torch.float32)
        r_last2 = torch.tensor([0.0, 0.3, 0.4, 0.3], dtype=torch.float32)
        variable_mask = torch.tensor([False, True, True, True])
        alpha, confidence, stats = pia.build_car_attention_bundle_from_rollouts(
            r_all=r_all,
            r_last2=r_last2,
            variable_mask=variable_mask,
            weight_min=0.5,
            weight_max=1.5,
            strength=0.25,
            power=0.5,
        )

        self.assertEqual(alpha.shape, r_all.shape)
        self.assertEqual(confidence.shape, r_all.shape)
        self.assertIn("rollout_all_variable_mass", stats)
        self.assertIn("rollout_last2_variable_mass", stats)
        self.assertEqual(stats["fixed_public_final_weight_sum"], 0.0)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_car_bundle_stats_require_both_rollout_views
```

Expected: FAIL because `build_car_attention_bundle_from_rollouts` is not defined.

- [ ] **Step 3: Add bundle helper**

In `pia_masked_server_attn_pia.py`, add:

```python
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
    stats.update({
        "rollout_all_variable_mass": float(r_all[variable_mask].sum().item()) if int(variable_mask.sum().item()) else 0.0,
        "rollout_last2_variable_mass": float(r_last2[variable_mask].sum().item()) if int(variable_mask.sum().item()) else 0.0,
        "car_strength": float(strength),
        "car_power": float(power),
        "car_weight_min": float(weight_min),
        "car_weight_max": float(weight_max),
    })
    return alpha, confidence, stats
```

- [ ] **Step 4: Connect helper to `server_attention_bundle` for CAR methods**

Inside `server_attention_bundle`, when `method in CAR_METHODS`, compute both rollouts from the collected suffix attentions:

```python
if method in CAR_METHODS:
    r_all = rollout_from_attentions(
        suffix_attentions,
        residual_rollout=True,
        depth="all",
        query_mode="last_window",
    )
    r_last2 = rollout_from_attentions(
        suffix_attentions,
        residual_rollout=True,
        depth="last2",
        query_mode="last_window",
    )
    attention_alpha, attention_confidence, attention_stats = build_car_attention_bundle_from_rollouts(
        r_all=r_all,
        r_last2=r_last2,
        variable_mask=variable_mask,
        weight_min=float(getattr(config, "car_weight_min", 0.5)),
        weight_max=float(getattr(config, "car_weight_max", 1.5)),
        strength=float(getattr(config, "car_strength", 0.25)),
        power=float(getattr(config, "car_power", 0.5)),
    )
```

If the existing bundle returns a dict, add keys:

```python
"attention_alpha": attention_alpha,
"attention_confidence": attention_confidence,
"attention_stats": attention_stats,
"rollout_depth": "all+last2",
"attention_formula": "confidence_aware_rollout_residual",
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1
python -m unittest tests.test_masked_server_attn_mts
```

Expected: PASS.

- [ ] **Step 6: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: planned changes only; no commit or push.

## Task 5: Implement PTR Position Selection and Acceptance Helpers

**Files:**
- Modify: `tests/test_masked_server_attn_mts.py`
- Modify: `pia_masked_server_attn_pia.py`

- [ ] **Step 1: Add PTR helper tests**

Append to `TestCarPtrStrictTop1`:

```python
    def test_ptr_selects_max_confidence_weighted_gradient_variable_only(self):
        import torch
        import pia_masked_server_attn_pia as pia

        grad_norms = torch.tensor([100.0, 1.0, 4.0, 3.0], dtype=torch.float32)
        confidence = torch.tensor([1.0, 1.0, 0.2, 2.0], dtype=torch.float32)
        variable_mask = torch.tensor([False, True, True, True])
        attempted = torch.tensor([False, False, False, False])

        selected = pia.select_ptr_position(
            grad_norms=grad_norms,
            confidence=confidence,
            variable_mask=variable_mask,
            attempted_mask=attempted,
        )
        self.assertEqual(selected, 3)

        attempted[3] = True
        selected = pia.select_ptr_position(
            grad_norms=grad_norms,
            confidence=confidence,
            variable_mask=variable_mask,
            attempted_mask=attempted,
        )
        self.assertEqual(selected, 2)

    def test_ptr_acceptance_requires_strict_b0v_improvement(self):
        import pia_masked_server_attn_pia as pia

        self.assertTrue(pia.ptr_should_accept(1.0, 0.999998, 1e-6))
        self.assertFalse(pia.ptr_should_accept(1.0, 0.9999995, 1e-6))
        self.assertFalse(pia.ptr_should_accept(1.0, 1.0, 1e-6))
```

- [ ] **Step 2: Run failing PTR tests**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_ptr_selects_max_confidence_weighted_gradient_variable_only
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_ptr_acceptance_requires_strict_b0v_improvement
```

Expected: FAIL because PTR helpers do not exist.

- [ ] **Step 3: Add PTR helper functions**

In `pia_masked_server_attn_pia.py`, near discretization/refinement helpers, add:

```python
def select_ptr_position(
    *,
    grad_norms: torch.Tensor,
    confidence: torch.Tensor,
    variable_mask: torch.Tensor,
    attempted_mask: torch.Tensor,
) -> int | None:
    grad_norms = grad_norms.detach().float()
    confidence = confidence.detach().float()
    selectable = variable_mask.bool() & (~attempted_mask.bool())
    if int(selectable.sum().item()) == 0:
        return None
    score = grad_norms * confidence.clamp_min(0.0)
    score = score.masked_fill(~selectable, float("-inf"))
    idx = int(torch.argmax(score).item())
    if not torch.isfinite(score[idx]):
        return None
    return idx


def ptr_should_accept(current_loss: float, candidate_loss: float, min_improvement: float) -> bool:
    return float(candidate_loss) < float(current_loss) - float(min_improvement)
```

- [ ] **Step 4: Run PTR helper tests**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_ptr_selects_max_confidence_weighted_gradient_variable_only
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_ptr_acceptance_requires_strict_b0v_improvement
```

Expected: PASS.

- [ ] **Step 5: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: planned changes only; no commit or push.

## Task 6: Implement Projected Top-1 Residual Correction

**Files:**
- Modify: `tests/test_masked_server_attn_mts.py`
- Modify: `pia_masked_server_attn_pia.py`

- [ ] **Step 1: Add a PTR audit-shape test**

Append to `TestCarPtrStrictTop1`:

```python
    def test_ptr_stats_schema_records_single_path_decisions(self):
        import pia_masked_server_attn_pia as pia

        stats = pia.empty_ptr_stats()
        self.assertEqual(stats["enabled"], True)
        self.assertEqual(stats["max_candidates_per_proposal"], 1)
        self.assertEqual(stats["semantic_speculation"], False)
        self.assertIn("proposal_count", stats)
        self.assertIn("accept_count", stats)
        self.assertIn("reject_count", stats)
        self.assertIn("token_change_count", stats)
        self.assertIn("history", stats)
```

- [ ] **Step 2: Run failing schema test**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_ptr_stats_schema_records_single_path_decisions
```

Expected: FAIL because `empty_ptr_stats` does not exist.

- [ ] **Step 3: Add PTR stats helper**

In `pia_masked_server_attn_pia.py`, add:

```python
def empty_ptr_stats() -> dict:
    return {
        "enabled": True,
        "strict_top1": True,
        "max_candidates_per_proposal": 1,
        "semantic_speculation": False,
        "proposal_count": 0,
        "accept_count": 0,
        "reject_count": 0,
        "token_change_count": 0,
        "consecutive_reject_count": 0,
        "pre_b0v_loss": None,
        "post_b0v_loss": None,
        "total_b0v_delta": 0.0,
        "history": [],
    }
```

- [ ] **Step 4: Implement PTR correction function**

Add this function near existing projection/refinement code. Reuse the existing model prefix/suffix forward utilities already used by `stage_b_optimize` and final activation calibration:

```python
def projected_top1_residual_correction(
    *,
    input_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    observed_hidden: torch.Tensor,
    variable_mask: torch.Tensor,
    attention_alpha: torch.Tensor,
    confidence: torch.Tensor | None,
    forward_hidden_fn,
    max_proposals: int = 10,
    max_accepts: int = 5,
    inner_steps: int = 5,
    lr: float = 0.025,
    min_improvement: float = 1e-6,
    patience: int = 3,
) -> tuple[torch.Tensor, dict]:
    stats = empty_ptr_stats()
    current_ids = input_ids.detach().clone()
    variable_mask = variable_mask.bool()
    if confidence is None:
        confidence = torch.ones_like(variable_mask, dtype=torch.float32, device=observed_hidden.device)
    confidence = confidence.to(observed_hidden.device).float()
    alpha = attention_alpha.to(observed_hidden.device).float()
    alpha = torch.where(variable_mask.to(alpha.device), alpha, torch.zeros_like(alpha))

    def b0v_loss(ids: torch.Tensor) -> torch.Tensor:
        hidden = forward_hidden_fn(ids)
        diff = hidden - observed_hidden
        return diff[:, variable_mask, :].pow(2).mean()

    current_loss = b0v_loss(current_ids)
    stats["pre_b0v_loss"] = float(current_loss.detach().item())
    attempted = torch.zeros_like(variable_mask, dtype=torch.bool)

    while stats["proposal_count"] < max_proposals and stats["accept_count"] < max_accepts:
        if stats["consecutive_reject_count"] >= patience:
            break
        embeds = torch.nn.functional.embedding(current_ids, embedding_weight).detach()
        embeds = embeds.clone().requires_grad_(True)
        hidden = forward_hidden_fn(None, inputs_embeds=embeds)
        residual = hidden - observed_hidden
        prop_loss = (residual.pow(2) * alpha.view(1, -1, 1).pow(2)).mean()
        grad = torch.autograd.grad(prop_loss, embeds, retain_graph=False, create_graph=False)[0]
        grad_norms = grad.detach().pow(2).sum(dim=-1).sqrt().squeeze(0)
        selected = select_ptr_position(
            grad_norms=grad_norms,
            confidence=confidence,
            variable_mask=variable_mask,
            attempted_mask=attempted,
        )
        if selected is None:
            break
        attempted[selected] = True
        stats["proposal_count"] += 1

        candidate_embeds = embeds.detach().clone()
        token_param = torch.nn.Parameter(candidate_embeds[:, selected, :].clone())
        optimizer = torch.optim.Adam([token_param], lr=lr)
        for _ in range(inner_steps):
            optimizer.zero_grad(set_to_none=True)
            proposal_embeds = candidate_embeds.clone()
            proposal_embeds[:, selected, :] = token_param
            proposal_hidden = forward_hidden_fn(None, inputs_embeds=proposal_embeds)
            proposal_residual = proposal_hidden - observed_hidden
            loss = (proposal_residual.pow(2) * alpha.view(1, -1, 1).pow(2)).mean()
            loss.backward()
            optimizer.step()

        distances = torch.cdist(token_param.detach(), embedding_weight.detach().unsqueeze(0)).squeeze(0)
        candidate_token = int(torch.argmin(distances, dim=-1).item())
        candidate_ids = current_ids.detach().clone()
        old_token = int(candidate_ids[0, selected].item())
        candidate_ids[0, selected] = candidate_token
        candidate_loss = b0v_loss(candidate_ids)

        accepted = ptr_should_accept(
            float(current_loss.detach().item()),
            float(candidate_loss.detach().item()),
            min_improvement,
        )
        stats["history"].append({
            "proposal_index": int(stats["proposal_count"]),
            "position": int(selected),
            "old_token": old_token,
            "candidate_token": candidate_token,
            "b0v_loss_before": float(current_loss.detach().item()),
            "b0v_loss_after": float(candidate_loss.detach().item()),
            "accepted": bool(accepted),
        })
        if accepted:
            current_ids = candidate_ids
            current_loss = candidate_loss.detach()
            stats["accept_count"] += 1
            stats["consecutive_reject_count"] = 0
            if candidate_token != old_token:
                stats["token_change_count"] += 1
        else:
            stats["reject_count"] += 1
            stats["consecutive_reject_count"] += 1

    stats["post_b0v_loss"] = float(current_loss.detach().item())
    stats["total_b0v_delta"] = float(stats["pre_b0v_loss"] - stats["post_b0v_loss"])
    return current_ids, stats
```

If the existing forward helper does not accept both `ids` and `inputs_embeds`, wrap it in `invert_observed` with this interface:

```python
def forward_hidden_fn(ids=None, inputs_embeds=None):
    return run_manual_prefix_forward(..., input_ids=ids, inputs_embeds=inputs_embeds)
```

- [ ] **Step 5: Run schema test and full unit test file**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_ptr_stats_schema_records_single_path_decisions
python -m unittest tests.test_masked_server_attn_mts
```

Expected: PASS. If the PTR function exposes a signature mismatch with existing forward helpers, adapt only the wrapper, not the algorithm.

- [ ] **Step 6: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: planned changes only; no commit or push.

## Task 7: Integrate PTR After Stage B and Before Final Output

**Files:**
- Modify: `pia_masked_server_attn_pia.py`
- Modify: `tests/test_masked_server_attn_mts.py`

- [ ] **Step 1: Add an API-boundary test for PTR methods**

Append to `TestCarPtrStrictTop1`:

```python
    def test_attack_api_for_car_ptr_does_not_allow_ground_truth_fields(self):
        import pia_masked_server_attn_pia as pia

        banned = {"prompt", "original_ids", "original_tokens"}
        api_fields = set(pia.attack_api_allowed_fields_for_method("confidence_aware_rollout_residual_ptr"))
        self.assertTrue(banned.isdisjoint(api_fields))
        self.assertIn("observed_hidden", api_fields)
        self.assertIn("model", api_fields)
```

- [ ] **Step 2: Run failing API-boundary test**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_attack_api_for_car_ptr_does_not_allow_ground_truth_fields
```

Expected: FAIL if `attack_api_allowed_fields_for_method` does not exist or includes banned fields.

- [ ] **Step 3: Add or extend API allowed-fields helper**

In `pia_masked_server_attn_pia.py`, near `validate_attack_api`, add:

```python
def attack_api_allowed_fields_for_method(method: str) -> set[str]:
    canonical = canonical_method_name(method)
    fields = {
        "model",
        "tokenizer",
        "observed_hidden",
        "server_suffix",
        "participant_number",
        "attacker_position",
        "target_layer",
        "max_token_len",
        "public_fixed_positions",
        "config",
    }
    if canonical in SERVER_ATTENTION_METHODS:
        fields.add("server_attention_rollout")
    return fields
```

Update `validate_attack_api` to include the new methods and assert banned names are absent from the function signature or invocation payload.

- [ ] **Step 4: Call PTR for PTR methods**

In `invert_observed`, after Stage B continuous optimization and strict nearest-neighbor projection, add:

```python
ptr_stats = None
if method in PTR_METHODS:
    strict_args = SimpleNamespace(
        method=method,
        k=int(config.k),
        y=int(config.y),
        semantic_speculation=bool(getattr(config, "semantic_speculation", False)),
    )
    validate_strict_top1_config(strict_args)
    final_ids, ptr_stats = projected_top1_residual_correction(
        input_ids=final_ids,
        embedding_weight=model.get_input_embeddings().weight,
        observed_hidden=observed_hidden,
        variable_mask=variable_mask,
        attention_alpha=attention_alpha,
        confidence=attention_confidence,
        forward_hidden_fn=forward_hidden_fn,
        max_proposals=int(getattr(config, "ptr_max_proposals", 10)),
        max_accepts=int(getattr(config, "ptr_max_accepts", 5)),
        inner_steps=int(getattr(config, "ptr_inner_steps", 5)),
        lr=float(getattr(config, "ptr_lr", 0.025)),
        min_improvement=float(getattr(config, "ptr_min_improvement", 1e-6)),
        patience=int(getattr(config, "ptr_patience", 3)),
    )
```

Write `ptr_stats` into the per-sample prediction record and aggregate metrics. If the existing code writes per-run JSON artifacts, also write:

```text
ptr_correction_stats.json
```

- [ ] **Step 5: Run API test and existing leakage verification mode**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_attack_api_for_car_ptr_does_not_allow_ground_truth_fields
python -m unittest tests.test_masked_server_attn_mts
CUDA_VISIBLE_DEVICES=0 python pia_masked_server_attn_pia.py --mode verify --method P4CARPTR --dataset-path data/airline.json --dataset-len 2 --target-layer 17 --epoch 5 --k 1 --y 0 --disable-semantic-speculation --local-files-only
```

Expected: tests PASS; verify writes leakage audit with no prompt/original ids/tokens and no failures.

- [ ] **Step 6: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: planned changes only; no commit or push.

## Task 8: Add CLI Arguments and Audit Fields

**Files:**
- Modify: `pia_masked_server_attn_pia.py`
- Modify: `tests/test_masked_server_attn_mts.py`

- [ ] **Step 1: Add audit schema test**

Append to `TestCarPtrStrictTop1`:

```python
    def test_car_ptr_audit_schema_names_last_valid_not_eos(self):
        import pia_masked_server_attn_pia as pia

        audit = pia.build_car_ptr_audit_record(
            method="confidence_aware_rollout_residual_ptr",
            strict_top1=True,
            known_public_special_positions=[0],
            token_id_based_special_mask_available=False,
            last_valid_position=12,
            fixed_public_final_weight_sum=0.0,
            final_last_valid_weight=0.0,
            final_max_weight=1.2,
            ptr_stats=pia.empty_ptr_stats(),
        )
        self.assertIn("final_last_valid_weight", audit)
        self.assertNotIn("final_eos_weight", audit)
        self.assertIn("raw_last_valid_mass", audit)
        self.assertNotIn("raw_eos_mass", audit)
        self.assertFalse(audit["token_id_based_special_mask_available"])
```

- [ ] **Step 2: Run failing audit test**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_car_ptr_audit_schema_names_last_valid_not_eos
```

Expected: FAIL because `build_car_ptr_audit_record` is not implemented.

- [ ] **Step 3: Add CLI arguments**

In `pia_masked_server_attn_pia.py`, near parser setup, add:

```python
parser.add_argument("--car-strength", type=float, default=0.25)
parser.add_argument("--car-power", type=float, default=0.5)
parser.add_argument("--car-weight-min", type=float, default=0.5)
parser.add_argument("--car-weight-max", type=float, default=1.5)
parser.add_argument("--car-warmup-epochs", type=int, default=50)
parser.add_argument("--car-ramp-end-epoch", type=int, default=70)
parser.add_argument("--ptr-max-proposals", type=int, default=10)
parser.add_argument("--ptr-max-accepts", type=int, default=5)
parser.add_argument("--ptr-inner-steps", type=int, default=5)
parser.add_argument("--ptr-lr", type=float, default=0.025)
parser.add_argument("--ptr-min-improvement", type=float, default=1e-6)
parser.add_argument("--ptr-patience", type=int, default=3)
```

- [ ] **Step 4: Add audit builder**

In `pia_masked_server_attn_pia.py`, add:

```python
def build_car_ptr_audit_record(
    *,
    method: str,
    strict_top1: bool,
    known_public_special_positions: list[int],
    token_id_based_special_mask_available: bool,
    last_valid_position: int,
    fixed_public_final_weight_sum: float,
    final_last_valid_weight: float,
    final_max_weight: float,
    ptr_stats: dict | None,
    raw_last_valid_mass: float = 0.0,
) -> dict:
    return {
        "method": canonical_method_name(method),
        "strict_top1": bool(strict_top1),
        "known_public_special_positions": list(known_public_special_positions),
        "token_id_based_special_mask_available": bool(token_id_based_special_mask_available),
        "last_valid_position": int(last_valid_position),
        "fixed_public_final_weight_sum": float(fixed_public_final_weight_sum),
        "final_last_valid_weight": float(final_last_valid_weight),
        "raw_last_valid_mass": float(raw_last_valid_mass),
        "final_max_weight": float(final_max_weight),
        "ptr_stats": ptr_stats,
        "special_token_note": (
            "Position 0 and other fixed_public positions are publicly determined by the protocol. "
            "Without token ids, the attack does not claim token-id-based EOS detection. "
            "The final valid position may be excluded only when public framing rules define it."
        ),
    }
```

- [ ] **Step 5: Wire audit fields into JSON outputs**

Where `variable_mask_audit.json`, `token_weight_stats.json`, or method audit JSONs are written, include:

```python
"known_public_special_positions": known_public_special_positions,
"token_id_based_special_mask_available": False,
"last_valid_position": int(last_valid_position),
"final_last_valid_weight": final_last_valid_weight,
"raw_last_valid_mass": raw_last_valid_mass,
"strict_top1": method in CAR_METHODS or method in PTR_METHODS,
"ptr_stats": ptr_stats,
```

Ensure these old field names do not appear in new CAR-PTR output:

```text
final_eos_weight
raw_eos_mass
```

- [ ] **Step 6: Run audit tests**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts.TestCarPtrStrictTop1.test_car_ptr_audit_schema_names_last_valid_not_eos
python -m unittest tests.test_masked_server_attn_mts
```

Expected: PASS.

- [ ] **Step 7: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: planned changes only; no commit or push.

## Task 9: Stage 0 Smoke Test and Verification

**Files:**
- No new code files unless tests expose a bug.
- Verify output files under `runs/car_ptr_strict_top1/smoke/`.

- [ ] **Step 1: Run all unit tests**

Run:

```bash
cd /data/zhiqi/Collaborative_Inference/DEML
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deml
python -m unittest tests.test_masked_server_attn_mts
```

Expected: PASS with all existing and new tests.

- [ ] **Step 2: Run CAR-PTR len2 epoch5 smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python pia_masked_server_attn_pia.py \
  --mode single \
  --method P4CARPTR \
  --dataset-path data/airline.json \
  --dataset-len 2 \
  --seed 42 \
  --participant-number 4 \
  --attacker-position 4 \
  --target-layer 17 \
  --epoch 5 \
  --max-token-len 896 \
  --k 1 \
  --y 0 \
  --disable-semantic-speculation \
  --local-files-only \
  --output-dir runs/car_ptr_strict_top1/smoke/P4CARPTR_seed42_layer17_len2
```

Expected: run completes and writes `metrics.json`, `predictions.jsonl`, `token_weight_stats.json`, `variable_mask_audit.json`, `ptr_correction_stats.json`, and `COMPLETE`.

- [ ] **Step 3: Inspect smoke audit**

Run:

```bash
python runs/car_ptr_strict_top1/analyze.py \
  --mode audit-smoke \
  --input runs/car_ptr_strict_top1/smoke/P4CARPTR_seed42_layer17_len2
```

Expected:

```text
strict_top1=true
fixed_public_final_weight_sum=0
variable_final_weight_mean=1
final_max_weight<=1.5
semantic_speculation=false
failed_sample_count=0
nan_count=0
ptr_acceptance_violations=0
```

- [ ] **Step 4: Stop on any smoke failure**

If any expected smoke condition fails, do not start development experiments. Use `superpowers:systematic-debugging` to isolate the first failing invariant.

- [ ] **Step 5: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: no unplanned changes; no commit or push.

## Task 10: Create Staged Runner

**Files:**
- Create: `runs/car_ptr_strict_top1/runner.py`

- [ ] **Step 1: Create runner skeleton**

Create `runs/car_ptr_strict_top1/runner.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


METHODS = ["B0", "B0V", "P4S-T1", "P4D-T1", "CAR", "P4D-PTR", "CAR-PTR"]
METHOD_TO_CLI = {
    "B0": "original_pia_baseline",
    "B0V": "variable_only_uniform",
    "P4S-T1": "attention_scaled_activation",
    "P4D-T1": "attention_weighted_residual",
    "CAR": "P4CAR",
    "P4D-PTR": "P4DPTR",
    "CAR-PTR": "P4CARPTR",
}


def build_command(method_name, seed, layer, dataset_len, output_dir):
    cli_method = METHOD_TO_CLI[method_name]
    return [
        "python", "pia_masked_server_attn_pia.py",
        "--mode", "single",
        "--method", cli_method,
        "--dataset-path", "data/airline.json",
        "--dataset-len", str(dataset_len),
        "--seed", str(seed),
        "--participant-number", "4",
        "--attacker-position", "4",
        "--target-layer", str(layer),
        "--epoch", "100",
        "--max-token-len", "896",
        "--k", "1",
        "--y", "0",
        "--disable-semantic-speculation",
        "--local-files-only",
        "--output-dir", str(output_dir),
    ]


def run_grid(root, methods, seeds, layers, dataset_len, gpus):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    queue = []
    for seed in seeds:
        for layer in layers:
            for method in methods:
                out = root / f"{method}_seed{seed}_layer{layer}_len{dataset_len}"
                queue.append((method, seed, layer, out))

    active = {}
    failures = []
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            method, seed, layer, out = queue.pop(0)
            complete = out / "COMPLETE"
            if complete.exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = build_command(method, seed, layer, dataset_len, out)
            log = open(out / "run.log", "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            active[gpu] = (proc, log, method, seed, layer, out)
            print(json.dumps({"event": "started", "gpu": gpu, "method": method, "seed": seed, "layer": layer, "output": str(out)}), flush=True)
        time.sleep(10)
        for gpu, item in list(active.items()):
            proc, log, method, seed, layer, out = item
            ret = proc.poll()
            if ret is None:
                continue
            log.close()
            del active[gpu]
            event = {"event": "finished", "gpu": gpu, "method": method, "seed": seed, "layer": layer, "returncode": ret, "output": str(out)}
            print(json.dumps(event), flush=True)
            if ret != 0:
                failures.append(event)
    if failures:
        raise SystemExit(json.dumps({"failures": failures}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "layer-dev", "full"], required=True)
    parser.add_argument("--root", default="runs/car_ptr_strict_top1")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if args.stage == "dev":
        run_grid(Path(args.root) / "dev_seed42_layer17", METHODS, [42], [17], 28, gpus)
    elif args.stage == "layer-dev":
        run_grid(Path(args.root) / "layer_dev_seed42", METHODS, [42], [11, 17, 19], 28, gpus)
    else:
        run_grid(Path(args.root) / "skytrax150_full", METHODS, [42, 43, 44], [11, 17, 19], 150, gpus)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run syntax check**

Run:

```bash
python -m py_compile runs/car_ptr_strict_top1/runner.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: new runner file and planned changes; no commit or push.

## Task 11: Create Analysis Script

**Files:**
- Create: `runs/car_ptr_strict_top1/analyze.py`

- [ ] **Step 1: Create analysis script**

Create `runs/car_ptr_strict_top1/analyze.py`:

```python
#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[int(row["prompt_id"])] = row
    return rows


def metric(row, name):
    if name in row:
        return float(row[name])
    if name == "token_accuracy":
        return float(row.get("token_acc", 0.0))
    return float(row.get(name, 0.0))


def paired(method_dir, baseline_dir, boot=10000):
    m = load_predictions(Path(method_dir) / "predictions.jsonl")
    b = load_predictions(Path(baseline_dir) / "predictions.jsonl")
    prompt_ids = sorted(set(m) & set(b))
    acc_delta = [metric(m[i], "token_accuracy") - metric(b[i], "token_accuracy") for i in prompt_ids]
    bleu_delta = [metric(m[i], "bleu") - metric(b[i], "bleu") for i in prompt_ids]
    wins = sum(1 for x in acc_delta if x > 0)
    ties = sum(1 for x in acc_delta if x == 0)
    losses = sum(1 for x in acc_delta if x < 0)
    rng = random.Random(1234)

    def ci(values):
        if not values:
            return (0.0, 0.0)
        means = []
        n = len(values)
        for _ in range(boot):
            sample = [values[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        return (means[int(0.025 * boot)], means[int(0.975 * boot)])

    return {
        "paired_prompt_count": len(prompt_ids),
        "mean_token_accuracy_delta": sum(acc_delta) / len(acc_delta) if acc_delta else 0.0,
        "mean_bleu_delta": sum(bleu_delta) / len(bleu_delta) if bleu_delta else 0.0,
        "win_count": wins,
        "tie_count": ties,
        "loss_count": losses,
        "token_accuracy_ci_low": ci(acc_delta)[0],
        "token_accuracy_ci_high": ci(acc_delta)[1],
        "bleu_ci_low": ci(bleu_delta)[0],
        "bleu_ci_high": ci(bleu_delta)[1],
    }


def audit_smoke(path):
    path = Path(path)
    metrics = load_json(path / "metrics.json")
    weights = load_json(path / "token_weight_stats.json")
    audit = load_json(path / "variable_mask_audit.json")
    ptr_path = path / "ptr_correction_stats.json"
    ptr = load_json(ptr_path) if ptr_path.exists() else {}
    checks = {
        "strict_top1": bool(audit.get("strict_top1", False)),
        "fixed_public_final_weight_sum": float(weights.get("fixed_public_final_weight_sum", audit.get("fixed_public_final_weight_sum", 0.0))),
        "variable_final_weight_mean": float(weights.get("variable_final_weight_mean", 1.0)),
        "final_max_weight": float(weights.get("final_max_weight", 0.0)),
        "semantic_speculation": bool(audit.get("semantic_speculation", False)),
        "failed_sample_count": int(metrics.get("failed_sample_count", 0)),
        "nan_count": int(metrics.get("nan_count", 0)),
        "ptr_acceptance_violations": int(ptr.get("acceptance_violation_count", 0)),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    bad = []
    if not checks["strict_top1"]:
        bad.append("strict_top1")
    if abs(checks["fixed_public_final_weight_sum"]) > 1e-6:
        bad.append("fixed_public_final_weight_sum")
    if abs(checks["variable_final_weight_mean"] - 1.0) > 1e-4:
        bad.append("variable_final_weight_mean")
    if checks["final_max_weight"] > 1.5 + 1e-6:
        bad.append("final_max_weight")
    if checks["semantic_speculation"]:
        bad.append("semantic_speculation")
    if checks["failed_sample_count"] != 0:
        bad.append("failed_sample_count")
    if checks["nan_count"] != 0:
        bad.append("nan_count")
    if checks["ptr_acceptance_violations"] != 0:
        bad.append("ptr_acceptance_violations")
    if bad:
        raise SystemExit("audit failed: " + ",".join(bad))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit-smoke", "paired"], required=True)
    parser.add_argument("--input")
    parser.add_argument("--method-dir")
    parser.add_argument("--baseline-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "audit-smoke":
        audit_smoke(args.input)
    else:
        result = paired(args.method_dir, args.baseline_dir)
        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                writer.writeheader()
                writer.writerow(result)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run:

```bash
python -m py_compile runs/car_ptr_strict_top1/analyze.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: new analysis script and planned changes; no commit or push.

## Task 12: Stage 1 Development Experiment

**Files:**
- Generated outputs under `runs/car_ptr_strict_top1/dev_seed42_layer17/`
- Update: `analysis/car_ptr_strict_top1_report.md`

- [ ] **Step 1: Run Stage 1 dev grid**

Run:

```bash
cd /data/zhiqi/Collaborative_Inference/DEML
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deml
python runs/car_ptr_strict_top1/runner.py --stage dev --gpus 0,1,2,3
```

Expected: all seven methods complete for Skytrax-28 seed42 layer17 and each output directory contains `COMPLETE`.

- [ ] **Step 2: Generate paired summaries against B0, B0V, and P4S-T1**

Run paired analysis for CAR-PTR:

```bash
python runs/car_ptr_strict_top1/analyze.py \
  --mode paired \
  --method-dir runs/car_ptr_strict_top1/dev_seed42_layer17/CAR-PTR_seed42_layer17_len28 \
  --baseline-dir runs/car_ptr_strict_top1/dev_seed42_layer17/P4S-T1_seed42_layer17_len28 \
  --output runs/car_ptr_strict_top1/dev_seed42_layer17/car_ptr_vs_p4s_paired.csv
```

Repeat the same command with B0 and B0V baseline directories, changing the output file names to `car_ptr_vs_b0_paired.csv` and `car_ptr_vs_b0v_paired.csv`.

- [ ] **Step 3: Write Stage 1 report section**

Update `analysis/car_ptr_strict_top1_report.md` with:

```markdown
## Stage 1: Skytrax-28 seed42 layer17

Strict Top-1 settings: K=1, Y=0, semantic speculation disabled.

| Method | Token Accuracy | BLEU | Completed | Failed | NaN |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | ... | ... | ... | ... | ... |
| B0V | ... | ... | ... | ... | ... |
| P4S-T1 | ... | ... | ... | ... | ... |
| P4D-T1 | ... | ... | ... | ... | ... |
| CAR | ... | ... | ... | ... | ... |
| P4D-PTR | ... | ... | ... | ... | ... |
| CAR-PTR | ... | ... | ... | ... | ... |

CAR-PTR continuation check:
- Token Accuracy > P4S-T1: true/false
- BLEU >= P4S-T1: true/false
- Token Accuracy and BLEU > B0: true/false
- Token Accuracy and BLEU > B0V: true/false
- Paired wins > losses against P4S-T1: true/false
- Audit violations: none/list

Decision: continue/stop.
```

Fill the table from actual `metrics.json` files only.

- [ ] **Step 4: Stop if Stage 1 criteria fail**

If any required Stage 1 continuation check fails, stop experiments and write:

```markdown
## Stop Decision

CAR-PTR did not pass the Skytrax-28 seed42 layer17 development gate. This is not evidence of stable improvement over the paper baseline. No multi-layer or Skytrax-150 expansion was run.
```

- [ ] **Step 5: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: generated results and report changes; no commit or push.

## Task 13: Stage 2 Layer Development Experiment

**Files:**
- Generated outputs under `runs/car_ptr_strict_top1/layer_dev_seed42/`
- Update: `analysis/car_ptr_strict_top1_report.md`

- [ ] **Step 1: Run only if Stage 1 passed**

Confirm `analysis/car_ptr_strict_top1_report.md` says `Decision: continue` for Stage 1.

- [ ] **Step 2: Run layer development grid**

Run:

```bash
python runs/car_ptr_strict_top1/runner.py --stage layer-dev --gpus 0,1,2,3
```

Expected: all seven methods complete for layers 11, 17, and 19 on Skytrax-28 seed42.

- [ ] **Step 3: Summarize layer results**

Update `analysis/car_ptr_strict_top1_report.md`:

```markdown
## Stage 2: Skytrax-28 seed42 layers 11/17/19

| Layer | Method | Token Accuracy | BLEU | Delta Acc vs P4S-T1 | Delta BLEU vs P4S-T1 | Delta Acc vs B0V | Delta BLEU vs B0V |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | CAR-PTR | ... | ... | ... | ... | ... | ... |
| 17 | CAR-PTR | ... | ... | ... | ... | ... | ... |
| 19 | CAR-PTR | ... | ... | ... | ... | ... | ... |

Layer gate:
- Pooled Token Accuracy and BLEU > P4S-T1: true/false
- At least 2/3 layers nonnegative vs P4S-T1: true/false
- CAR-PTR > B0V on every layer: true/false
- Audit violations: none/list

Decision: continue/stop.
```

- [ ] **Step 4: Stop if Stage 2 criteria fail**

If any Stage 2 continuation condition fails, stop and do not run Skytrax-150.

- [ ] **Step 5: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: generated layer results and report changes; no commit or push.

## Task 14: Stage 3 Skytrax-150 Full Validation

**Files:**
- Generated outputs under `runs/car_ptr_strict_top1/skytrax150_full/`
- Update: `analysis/car_ptr_strict_top1_report.md`

- [ ] **Step 1: Run only if Stage 2 passed**

Confirm `analysis/car_ptr_strict_top1_report.md` says `Decision: continue` for Stage 2.

- [ ] **Step 2: Run full validation grid**

Run:

```bash
python runs/car_ptr_strict_top1/runner.py --stage full --gpus 0,1,2,3
```

Expected: 63 method/seed/layer cells complete, each with `COMPLETE`, metrics, predictions, audit, and PTR stats where applicable.

- [ ] **Step 3: Generate pooled paired analysis**

For CAR-PTR against P4S-T1, B0, and B0V, compute:

- Pooled Token Accuracy delta.
- Pooled BLEU delta.
- Per-seed pooled deltas.
- Per-layer pooled deltas.
- Nine seed-layer cell deltas.
- Win/tie/loss counts.
- 10,000-sample paired bootstrap confidence intervals.
- Audit violations.

Store results as:

```text
runs/car_ptr_strict_top1/skytrax150_full/heldout_summary.csv
runs/car_ptr_strict_top1/skytrax150_full/heldout_summary.md
```

- [ ] **Step 4: Update final report**

Update `analysis/car_ptr_strict_top1_report.md`:

```markdown
## Stage 3: Skytrax-150 full validation

| Comparator | Pooled Delta Acc | Acc 95% CI | Pooled Delta BLEU | BLEU 95% CI | Wins | Ties | Losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAR-PTR vs P4S-T1 | ... | [..., ...] | ... | [..., ...] | ... | ... | ... |
| CAR-PTR vs B0 | ... | [..., ...] | ... | [..., ...] | ... | ... | ... |
| CAR-PTR vs B0V | ... | [..., ...] | ... | [..., ...] | ... | ... | ... |

Stability criteria:
- Pooled Token Accuracy > P4S-T1: true/false
- Pooled BLEU > P4S-T1: true/false
- Every seed nonnegative vs P4S-T1: true/false
- At least 6/9 seed-layer cells nonnegative on both metrics: true/false
- Pooled wins > losses: true/false
- Bootstrap lower bounds > 0 for both metrics: true/false
- No audit violations: true/false

Final wording allowed:
- If all true: stable improvement over P4S-T1 under strict Top-1.
- Otherwise: observed improvement/mixed result/negative result, not stable improvement.
```

- [ ] **Step 5: Checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: generated full validation results and report changes; no commit or push.

## Task 15: Final Review and Handoff

**Files:**
- Review: `pia_masked_server_attn_pia.py`
- Review: `tests/test_masked_server_attn_mts.py`
- Review: `runs/car_ptr_strict_top1/runner.py`
- Review: `runs/car_ptr_strict_top1/analyze.py`
- Review: `analysis/car_ptr_strict_top1_report.md`

- [ ] **Step 1: Run final verification**

Run:

```bash
python -m unittest tests.test_masked_server_attn_mts
python -m py_compile pia_masked_server_attn_pia.py runs/car_ptr_strict_top1/runner.py runs/car_ptr_strict_top1/analyze.py
```

Expected: tests PASS and compile command exits 0.

- [ ] **Step 2: Confirm no banned API usage**

Run:

```bash
rg -n "original_ids|original_tokens|prompt" pia_masked_server_attn_pia.py tests/test_masked_server_attn_mts.py runs/car_ptr_strict_top1 analysis/car_ptr_strict_top1_report.md
```

Expected: any hits are in audit text, test banned-field assertions, or reporting notes only. No attack function receives or reads these fields.

- [ ] **Step 3: Confirm strict Top-1 artifacts**

Run:

```bash
rg -n '"k": 1|"y": 0|semantic_speculation|strict_top1' runs/car_ptr_strict_top1
```

Expected: all CAR/PTR run configs have `k=1`, `y=0`, semantic speculation disabled, and `strict_top1=true`.

- [ ] **Step 4: Confirm report wording**

Run:

```bash
rg -n "stable improvement|稳定提升|significant|显著" analysis/car_ptr_strict_top1_report.md
```

Expected: any stable-improvement wording is conditional on all Stage 3 success criteria passing. Do not use "显著提升" unless statistical claims are explicitly supported.

- [ ] **Step 5: Final git checkpoint without commit**

Run:

```bash
git status --short
git diff --stat
```

Expected: all intended files visible; no commit or push.

## Self-Review

Spec coverage:

- Safety and authorization scope are included in the spec and this plan.
- Strict Top-1 is enforced by tests, config validation, runner flags, and audits.
- B0 and B0V definitions are preserved.
- CAR formula uses `alpha_i * (h_pred_i - h_obs_i)` and not the P4S `alpha_i h_pred_i - h_obs_i` form.
- PTR is deterministic single-path Top-1 and accepts only B0V residual improvements.
- Stage 0, Stage 1, Stage 2, Stage 3, and stop/continue gates are represented.
- Reporting includes deltas against B0, B0V, and P4S-T1, paired analysis, bootstrap CIs, and audit fields.
- Automatic commit/push is omitted per user instruction.

Placeholder scan:

- This plan intentionally contains no unresolved placeholder markers.
- Metric tables include ellipses only inside report templates that must be filled from generated metric files during execution, not as implementation placeholders.

Type consistency:

- New aliases map to canonical method names used in method sets.
- CAR helper returns `(alpha, confidence, stats)`.
- PTR helper accepts `attention_alpha` and optional `confidence`.
- Audit fields consistently use `last_valid`, not EOS.
