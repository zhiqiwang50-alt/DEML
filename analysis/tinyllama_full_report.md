# TinyLlama-adapted reproduction of Prompt Inversion Attack experiments

## Scope

This report is a TinyLlama-adapted reproduction of Prompt Inversion Attack experiments. It is not a strict numerical reproduction of the paper's Llama-65B / Llama-2-70B / OPT-66B experiments, and it must not be interpreted as exceeding or fully reproducing the original paper results.

## Environment

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- TinyLlama transformer blocks: 22, layer indices 0..21
- Commit: `b6e321e`
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
