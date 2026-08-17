# P4 Top-1 Skytrax-150 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current P4 Top-1 attention-scaled activation branch with two controlled components and run Skytrax-150 multi-layer, multi-seed comparisons against B0/B0V.

**Architecture:** Keep `pia_masked_server_attn_pia.py` as the experiment entry point. Add P4 variants as methods that reuse the existing attention-scaled activation loss and server attention rollout. Add runner/summarizer artifacts under `runs/attention_scaled_activation_pia_top1_skytrax150/` and reports under `analysis/`.

**Tech Stack:** Python, PyTorch, Hugging Face local TinyLlama cache, existing DEML prompt inversion utilities, remote multi-GPU scheduling through `nvidia-smi`.

---

### Task 1: Add Tests for P4 Variants

**Files:**
- Modify: `tests/test_masked_server_attn_mts.py`

- [ ] **Step 1: Write failing tests**

Add pure-function tests for a linear attention schedule and adaptive beta. The tests should require:

```python
def test_linear_schedule_ramps_between_start_and_full(self):
    self.assertAlmostEqual(pia.attention_scale_linear_schedule_beta(1, 100, 0.5, 0.7, 0.25), 0.0)
    self.assertAlmostEqual(pia.attention_scale_linear_schedule_beta(50, 100, 0.5, 0.7, 0.25), 0.0)
    self.assertAlmostEqual(pia.attention_scale_linear_schedule_beta(60, 100, 0.5, 0.7, 0.25), 0.125)
    self.assertAlmostEqual(pia.attention_scale_linear_schedule_beta(70, 100, 0.5, 0.7, 0.25), 0.25)

def test_adaptive_beta_keeps_certain_tokens_near_uniform(self):
    alpha = torch.tensor([0.0, 0.8, 1.2, 1.5])
    variable = torch.tensor([False, True, True, True])
    uncertainty = torch.tensor([0.0, 0.0, 0.5, 1.0])
    out, stats = pia.apply_adaptive_attention_beta(alpha, variable, uncertainty, base_beta=0.25, min_beta=0.05, alpha_min=0.5, alpha_max=1.5)
    self.assertEqual(float(out[0]), 0.0)
    self.assertLess(abs(float(out[1]) - 1.0), abs(float(alpha[1]) - 1.0))
    self.assertGreater(abs(float(out[3]) - 1.0), abs(float(out[1]) - 1.0))
    self.assertAlmostEqual(float(out[variable].mean()), 1.0, places=6)
    self.assertGreater(stats["adaptive_beta_mean"], 0.05)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/zhiqi/miniconda3/envs/deml/bin/python -m unittest tests.test_masked_server_attn_mts.AttentionScalePureFunctionTests
```

Expected: fail because the new schedule/adaptive functions do not exist.

### Task 2: Implement P4-Schedule and P4-Adaptive

**Files:**
- Modify: `pia_masked_server_attn_pia.py`

- [ ] **Step 1: Add method aliases**

Register:

```python
"P4S": "attn_scale_mean_query_schedule"
"P4A": "attn_scale_mean_query_adaptive_beta"
```

Both belong to `ATTN_SCALE_METHODS`.

- [ ] **Step 2: Add config/CLI fields**

Add:

```python
attention_full_ratio: float = 0.70
adaptive_min_beta: float = 0.05
```

Expose as `--attention-full-ratio` and `--adaptive-min-beta`.

- [ ] **Step 3: Add schedule/adaptive helpers**

Implement:

```python
def attention_scale_linear_schedule_beta(step_index, epoch, start_ratio, full_ratio, max_beta): ...
def embedding_uncertainty_scores(z, embed_weight, variable_mask): ...
def apply_adaptive_attention_beta(alpha, variable_mask, uncertainty_scores, base_beta, min_beta, alpha_min, alpha_max): ...
```

The adaptive helper should keep fixed positions zero, set low-uncertainty variable tokens closer to 1, and renormalize variable alpha mean to 1.

- [ ] **Step 4: Integrate variants**

For `attn_scale_mean_query_schedule`, build raw attention alpha with beta 1 internally, then apply scheduled beta during Stage B.

For `attn_scale_mean_query_adaptive_beta`, use embedding-margin uncertainty to compute token-level beta before Stage B, with scheduled activation still controlled by `attention_start_ratio`.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
/home/zhiqi/miniconda3/envs/deml/bin/python -m py_compile pia_masked_server_attn_pia.py tests/test_masked_server_attn_mts.py
/home/zhiqi/miniconda3/envs/deml/bin/python -m unittest tests.test_masked_server_attn_mts.AttentionScalePureFunctionTests
/home/zhiqi/miniconda3/envs/deml/bin/python -m unittest tests.test_masked_server_attn_mts
```

Expected: all tests pass.

### Task 3: Smoke Test New Variants

**Files:**
- Output: `runs/attention_scaled_activation_pia_top1_skytrax150/smoke/`

- [ ] **Step 1: Run tiny smoke**

Run B0V, P4-frozen, P4-schedule, and P4-adaptive on `dataset_len=2`, `layer17`, `epoch=20`, `k=1`, `y=1`.

- [ ] **Step 2: Audit smoke outputs**

Confirm each output has `metrics.json`, `config.json`, `alpha_scale_stats.json` for P4 variants, `COMPLETE`, no NaN, no failed samples.

### Task 4: Build Reusable Skytrax-150 Runner

**Files:**
- Create: `runs/attention_scaled_activation_pia_top1_skytrax150/runner.py`

- [ ] **Step 1: Implement config audit**

Before scheduling, scan existing runs and reuse only if method, seed, layer, dataset_len=150, epoch=100, k=1, y=1, and max_token_len=896 all match.

- [ ] **Step 2: Implement GPU scheduler**

Read free memory from `nvidia-smi`, run at most one task per GPU, prefer highest free memory GPUs, and keep updating `runner_status.json`.

- [ ] **Step 3: Define method matrix**

Methods: B0, B0V, P4-frozen, P4-schedule, P4-adaptive.

Layers: 11, 17, 19.

Seeds: 42, 43, 44.

### Task 5: Run Skytrax-150 and Summarize

**Files:**
- Output: `runs/attention_scaled_activation_pia_top1_skytrax150/summary.csv`
- Output: `runs/attention_scaled_activation_pia_top1_skytrax150/summary.md`
- Output: `analysis/attention_scaled_activation_top1_skytrax150_report.md`

- [ ] **Step 1: Launch runner**

Run the reusable runner in the background, using available GPUs.

- [ ] **Step 2: Periodically audit progress**

Report completed/failed/running counts, active GPUs, and latest per-method metrics.

- [ ] **Step 3: Generate final summary**

For each method/layer/seed, compute Token Accuracy, BLEU, deltas vs B0 and B0V, completed/failed count, NaN, and alpha audit. Then compute pooled summaries across seeds/layers.
