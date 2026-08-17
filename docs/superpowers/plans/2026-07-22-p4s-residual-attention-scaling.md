# P4S Residual Attention Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rho-controlled residual version of P4S attention-scaled activation loss and run the agreed seed42 global-vs-layer-adaptive experiment.

**Architecture:** Keep strict Top-1 and the existing P4S pipeline. Add one helper that maps alpha to 1 + rho * (alpha - 1) on variable positions, preserves fixed/public positions at zero, and re-normalizes variable positions to mean one. Register a new P4S-Res method that reuses existing server attention rollout, schedule, clipping, and calibration code.

**Tech Stack:** Python, PyTorch, TinyLlama DEML prompt inversion scripts, unittest.

---

### Task 1: Residual Alpha Tests

**Files:**
- Modify: `tests/test_masked_server_attn_mts.py`

- [ ] **Step 1: Add tests before implementation**

Add tests asserting that rho=0 returns variable alpha=1, rho=1 recovers the current alpha, fixed positions remain zero, and P4S-Res is registered without dummy/alpha initialization.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python3 -m unittest tests.test_masked_server_attn_mts.AttentionScalePureFunctionTests`
Expected: failure because `residualize_attention_alpha` and P4S-Res registration do not exist yet.

### Task 2: Residual Alpha Implementation

**Files:**
- Modify: `pia_masked_server_attn_pia.py`

- [ ] **Step 1: Add method registration**

Register `P4SRES -> attn_scale_mean_query_schedule_residual` and include it in `ATTN_SCALE_METHODS`.

- [ ] **Step 2: Add config and CLI**

Add `residual_alpha_rho: float` to `SAWConfig`, `AutoBatchTask`, `build_single_run_command`, and argparse via `--residual-alpha-rho`.

- [ ] **Step 3: Add residual helper**

Implement `residualize_attention_alpha(alpha, variable_mask, rho, alpha_min, alpha_max)` using 1 + rho * (alpha - 1), then bounded mean-one normalization on variable positions.

- [ ] **Step 4: Apply helper only to P4S-Res schedule**

Inside Stage B, after computing scheduled/interpolated alpha, apply residualization for `attn_scale_mean_query_schedule_residual`. Keep P4S unchanged.

### Task 3: Verification and Experiment

**Files:**
- Read: `runs/attention_scaled_activation_pia_top1_skytrax150/summary.csv`
- Create: `runs/p4s_residual_top1/` outputs

- [ ] **Step 1: Run unit tests GREEN**

Run: `python3 -m unittest tests.test_masked_server_attn_mts`
Expected: all tests pass.

- [ ] **Step 2: Run smoke**

Run a 2-sample layer17 P4S-Res smoke with K/Y=1/1, epoch=20, rho=0.5. Expected: COMPLETE, failed_sample_count=0.

- [ ] **Step 3: Run seed42 comparison**

Run Skytrax-150 seed42 layer 11/17/19 for two frozen residual settings:
- Global: rho=0.5 for all layers.
- Layer-adaptive: layer11 rho=1.0, layer17 rho=0.6, layer19 rho=0.4.

- [ ] **Step 4: Summarize against existing B0/B0V/P4S**

Generate `runs/p4s_residual_top1/seed42_residual_summary.csv` and `.md` with Acc/BLEU deltas vs B0, B0V, and P4S.

