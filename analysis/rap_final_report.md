# RAP-FINAL Dev Report

Seed42/43/44 are development/diagnostic only, not final held-out evidence.

## Pooled Metrics

| method | Acc | BLEU | failed | NaN |
| --- | ---: | ---: | ---: | --- |
| original_pia_baseline | 0.834913 | 0.640501 | 0 | False |
| variable_only_uniform | 0.836043 | 0.647207 | 0 | False |
| sparse_fixed | 0.839434 | 0.649482 | 0 | False |
| rel_order | 0.838074 | 0.646660 | 0 | False |
| rap_cont_gate | 0.839246 | 0.648814 | 0 | False |
| rap_final | 0.836871 | 0.647993 | 0 | False |
| shuffled_rel | 0.837552 | 0.649648 | 0 | False |

## Key Attribution

- Margin uncertainty audit kept margin-only detector: enrichment ratio 3.211586.
- SPARSE_FIXED vs B0V is the most plausible useful component on development, but it is still seed-dependent.
- RAP_FINAL does not beat SHUFFLED_REL on development; relational structure cannot be claimed as the core contributor.
- RAP_FINAL is often equal to B0V because the discrete Top-1 acceptance gate rejects candidates that do not improve the discrete path.
- The archived GPU0 OOM partial result is kept under `dev_seed44/sparse_fixed_oom_gpu0_*` and is excluded from clean summaries.

Detailed CSV outputs: `runs/rap_final/dev_component_summary_clean.csv`, `runs/rap_final/dev_pooled_summary.csv`, `runs/rap_final/dev_paired_summary.csv`.
