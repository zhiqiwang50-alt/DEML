# DEML Prompt Inversion Attack Experiments

This repository is based on Prompt Inversion Attack against Collaborative Inference of Large Language Models and adds white-box TinyLlama experiments for attention/gradient-inspired prompt inversion.

This README is a handoff for a fresh Codex session on another computer. Do not rely on prior chat context: this file defines the project structure, current conclusions, and first commands to run.

## 1. Project Goal

The original paper studies prompt inversion in collaborative inference. A malicious participant receives an intermediate activation H_obs from the previous participant and tries to recover the input prompt.

This fork keeps the paper-style PIA baseline and adds experimental branches asking whether white-box server-side attention or gradient signals can improve reconstruction without leaking ground-truth prompt tokens into the attack API.

Current high-level conclusion:

- Do not claim stable improvement yet.
- MTS and SACR gave development-set or weak positive signals but did not satisfy robust expansion criteria.
- The latest Attention Weighted PIA branch, B0VAG, showed positive signs on Skytrax-28, but frozen Skytrax-150 validation did not beat the paper-style B0 baseline.

## 2. White-Box Threat Model Used Here

The current experiments use the stronger white-box setting:

- The attacker knows the full model parameters.
- The attacker observes the transmitted activation H_obs at the collaborative split point.
- The attacker can run the server-side suffix locally/offline.
- The attack function must not receive or read prompt, original_ids, original_tokens, or ground-truth token ids.

If you change this threat model, document it before running new experiments.

## 3. Repository Layout

Core files:

| Path | Purpose |
|---|---|
| invert.py | Original inversion entry point for larger LLM settings. |
| pia_tinyllama.py | TinyLlama reproduction of the paper-style PIA baseline. |
| dataset.py, utils.py, misc.py | Shared helpers from the base project. |
| data/airline.json | Skytrax-28 development set used in TinyLlama pilots. |
| data/skytrax_150.json | Skytrax-150 validation set used for frozen candidate checks. |

Added experiment branches:

| File | Short name | Meaning |
|---|---|---|
| pia_cagdm.py | CA-GDM | Calibration-aware gradient/dummy matching exploration. Earlier gradient override was negative. |
| pia_masked_server_attn_pia.py | MTS-PIA | Server attention rollout directly weights activation matching over variable tokens. Held-out validation was negative/inconclusive. |
| pia_server_attention_consistency_pia.py | SACR-PIA | Keeps B0V objective and adds capped server-side consistency regularization. Weak holdout signal only. |
| pia_attention_weighted_pia.py | AWP-PIA | Latest branch: B0/B0V plus server-side attention-weighted activation loss and optional uncertainty gate. |
| pia_suffix_attention_edge_rerank.py | SAER | Suffix attention edge reranking exploration. |

Useful scripts:

| Script | Purpose |
|---|---|
| scripts/run_attention_weighted_pia.sh | Wrapper for latest AWP branch. |
| scripts/run_masked_server_attn_pia.sh | Wrapper for MTS-PIA. |
| scripts/run_server_attention_consistency_pia.sh | Wrapper for SACR-PIA. |
| scripts/run_cagdm.sh | Wrapper for CA-GDM. |
| scripts/run_tinyllama_smoke.sh | TinyLlama baseline smoke run from the original reproduction work. |

Reports to read first:

| Report | What it tells you |
|---|---|
| analysis/attention_weighted_pia_report.md | Latest AWP Skytrax-28 and Skytrax-150 results. |
| analysis/attention_weighted_pia_skytrax150_report.md | Frozen Skytrax-150 validation: B0VAG is negative vs B0. |
| analysis/masked_server_attn_report.md | MTS development, held-out, and layer override summary. |
| analysis/server_attention_consistency_report.md | SACR tune/holdout summary and selection audit. |
| analysis/suffix_attention_edge_rerank_report.md | SAER experiment notes. |

## 4. Environment Setup on a New Machine

Recommended environment:

    conda create -n deml python=3.10 -y
    conda activate deml
    pip install -r requirements.txt

TinyLlama experiments require PyTorch, Transformers, NLTK, and GPU memory. A single 24GB GPU can run smoke tests. Multi-layer or Skytrax-150 runs are much faster with multiple GPUs.

Model cache guidance:

    export HF_HOME=/path/to/hf_cache
    export TRANSFORMERS_CACHE=/path/to/hf_cache/transformers

If the model is already cached, use --local-files-only. If not cached, omit --local-files-only for the first run so Hugging Face can download TinyLlama/TinyLlama-1.1B-Chat-v1.0.

The wrapper scripts/run_attention_weighted_pia.sh defaults to the original server Python path. On a new machine, pass your Python explicitly:

    PYTHON_BIN=/path/to/conda/env/bin/python scripts/run_attention_weighted_pia.sh --mode verify

## 5. First Commands for a Fresh Codex Session

Run these before starting new experiments:

    git status -sb
    python -m py_compile pia_attention_weighted_pia.py pia_masked_server_attn_pia.py pia_server_attention_consistency_pia.py
    python -m unittest tests.test_attention_weighted_pia
    PYTHON_BIN=/path/to/conda/env/bin/python scripts/run_attention_weighted_pia.sh --mode verify

Expected meaning:

- py_compile checks syntax.
- tests.test_attention_weighted_pia checks schedule/gate/weighting behavior and attack API signature.
- --mode verify checks that the AWP attack API does not accept ground-truth prompt inputs.

## 6. Latest AWP-PIA Methods

The latest branch is implemented in pia_attention_weighted_pia.py.

| Method | Definition | Role |
|---|---|---|
| B0 | Paper-style original PIA baseline with all-valid uniform activation loss. | Main paper baseline. |
| B0V | Variable-only uniform activation loss. | Strong baseline constraint. |
| B0A | B0 plus server-side attention weighting. | Direct formula version. |
| B0VA | B0V plus server-side attention weighting. | Variable-token weighting. |
| B0VAG | B0V plus attention weighting plus uncertainty gate. | Frozen candidate tested on Skytrax-150. |

AWP default idea:

    H_obs -> server suffix attention -> server rollout weights w_i
    L_attn-act = sum_i w_i * || h_i(X_tilde) - h_i,target ||^2

Stabilization components:

- last-window query attention, last_window_size=16
- rollout depth all
- residual rollout on
- power tempering, weight_power=0.5
- clipping, weight_min=0.5 and weight_max=2.0
- late schedule, attention_start_ratio=0.7
- uncertainty gate for B0VAG, uncertainty_fraction=0.3

## 7. Reproducing the Latest Experiments

### 7.1 Verify AWP API

    PYTHON_BIN=/path/to/conda/env/bin/python scripts/run_attention_weighted_pia.sh --mode verify

### 7.2 One-sample smoke test

    PYTHON_BIN=/path/to/conda/env/bin/python scripts/run_attention_weighted_pia.sh \
      --mode single \
      --method B0VAG \
      --dataset-name Skytrax-28 \
      --dataset-path data/airline.json \
      --dataset-len 1 \
      --target-layer 17 \
      --epoch 2 \
      --topk-label topk10 \
      --output-root runs/attention_weighted_pia_smoke

### 7.3 Skytrax-28 development stage

This runs B0/B0V/B0A/B0VA/B0VAG across layers 11/17/19 and both Top-K=10 and Top-1 settings.

    PYTHON_BIN=/path/to/conda/env/bin/python scripts/run_attention_weighted_pia.sh \
      --mode stage1 \
      --output-root runs/attention_weighted_pia \
      --dataset-name Skytrax-28 \
      --dataset-path data/airline.json \
      --dataset-len 28 \
      --seed 42 \
      --target-layers 11 17 19 \
      --epoch 100 \
      --local-files-only \
      --resume

Summarize:

    PYTHON_BIN=/path/to/conda/env/bin/python scripts/run_attention_weighted_pia.sh \
      --mode summarize \
      --output-root runs/attention_weighted_pia

### 7.4 Skytrax-150 frozen validation

Historical validation used only Top-K=10 and methods B0/B0V/B0A/B0VAG on layers 11/17/19.

Run each method-layer explicitly, or use your own job scheduler while keeping parameters frozen. Replace METHOD with B0, B0V, B0A, or B0VAG; replace LAYER with 11, 17, or 19:

    PYTHON_BIN=/path/to/conda/env/bin/python scripts/run_attention_weighted_pia.sh \
      --mode single \
      --method METHOD \
      --target-layer LAYER \
      --topk-label topk10 \
      --output-root runs/attention_weighted_pia_skytrax150 \
      --dataset-name Skytrax-150 \
      --dataset-path data/skytrax_150.json \
      --dataset-len 150 \
      --seed 42 \
      --participant-number 4 \
      --attacker-position 4 \
      --epoch 100 \
      --lr 0.1 \
      --lambda-vocab 0.1 \
      --max-token-len 896 \
      --grad-clip 1.0 \
      --beta 0.25 \
      --weight-power 0.5 \
      --weight-min 0.5 \
      --weight-max 2.0 \
      --last-window-size 16 \
      --attention-start-ratio 0.7 \
      --uncertainty-fraction 0.3 \
      --server-rollout-depth all \
      --local-files-only \
      --resume

Then summarize:

    PYTHON_BIN=/path/to/conda/env/bin/python scripts/run_attention_weighted_pia.sh \
      --mode summarize \
      --output-root runs/attention_weighted_pia_skytrax150

Historical result: B0VAG did not beat B0 on any of layers 11/17/19. Treat it as a negative validation result.

## 8. Current Experimental Results to Preserve

### AWP Skytrax-28 Top-K=10

| layer | method | Acc | BLEU | dAcc vs B0 | dAcc vs B0V |
|---:|---|---:|---:|---:|---:|
| 11 | B0VAG | 0.9536 | 0.9440 | -0.0011 | +0.0259 |
| 17 | B0VAG | 0.9888 | 0.9773 | +0.0040 | -0.0070 |
| 19 | B0VAG | 0.9993 | 0.9987 | +0.0033 | +0.0003 |

Interpretation: positive signal vs B0 on 2/3 layers, but not robust vs B0V.

### AWP Skytrax-150 Frozen Candidate

| layer | B0 Acc | B0V Acc | B0A Acc | B0VAG Acc | B0VAG dAcc vs B0 | B0VAG dAcc vs B0V |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.948995 | 0.936689 | 0.937927 | 0.938231 | -0.010764 | +0.001541 |
| 17 | 0.993061 | 0.996142 | 0.991912 | 0.991761 | -0.001300 | -0.004381 |
| 19 | 0.994753 | 0.996112 | 0.996482 | 0.994565 | -0.000188 | -0.001547 |

Interpretation: frozen B0VAG does not reproduce stable improvement on Skytrax-150.

## 9. What Not to Do Next Without a Reason

Do not immediately expand AWP/B0VAG to more datasets or claim a stable method. The latest validation is negative vs B0.

A sensible next Codex task would be one of:

1. Analyze why attention weights hurt on Skytrax-150.
2. Inspect whether weights concentrate on fixed/BOS/low-information tokens.
3. Try a more conservative continuous auxiliary loss instead of direct loss weighting.
4. Run held-out seeds only after a method passes a clear development criterion.

## 10. Git and Artifact Policy

Commit source, scripts, tests, small JSON datasets, and Markdown reports.

Do not commit:

- runs/ experiment outputs
- Hugging Face/model caches
- __pycache__/
- large PDFs or PPT decks
- local logs or temporary notebooks

The .gitignore is configured for these rules. If a small dataset is intentionally needed despite data/ being generally ignored, add it explicitly with git add -f data/<file>.json.

## 11. Citation

If you use the base paper or code, cite:

    @article{qu2025prompt,
      title={Prompt Inversion Attack against Collaborative Inference of Large Language Models},
      author={Qu, Wenjie and Zhou, Yuguang and Wu, Yongji and Xiao, Tingsong and Yuan, Binhang and Li, Yiming and Zhang, Jiaheng},
      journal={arXiv preprint arXiv:2503.09022},
      year={2025}
    }
