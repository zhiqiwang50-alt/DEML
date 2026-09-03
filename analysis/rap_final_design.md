# RAP-FINAL Design

## White-box Setting

The attacker receives the target intermediate activation, public model parameters, public embedding matrix, sequence length, and public framing rules. The attack path does not receive prompt text, original token ids, original tokens, or ground-truth references.

## Frozen Strict Top-1 Configuration

- K = 1, Y = 0.
- semantic speculation = false.
- activation calibration = false.
- patch_size = 16.
- uncertainty_fraction = 0.20.
- top_r = 4.
- max_repair_passes = 2.
- lr = 0.08.
- lambda_vocab = 0.1.

## Method Family

- B0: original all-valid uniform activation matching baseline.
- B0V: variable-only uniform activation matching baseline.
- SPARSE_FIXED: B0V coarse optimization, uncertainty detection, fixed sparse patches, cut-loss acceptance.
- REL_ORDER: SPARSE_FIXED with true relational ordering, but no relational gate.
- RAP_CONT_GATE: fixed patches with continuous relational gate.
- RAP_FINAL: fixed patches, relational ordering, sparse candidate generation, discrete Top-1 cut/relational acceptance, rollback.
- SHUFFLED_REL: same as RAP_FINAL, but legal relational edge weights are deterministically shuffled.

## Relational Score

For each server layer:

`S_l[q,k] = mean_h A_lh[q,k] * ||V_lh[k]||_2`

Only legal variable causal edges `q > k` are used for layer scale normalization:

`S_norm_l = S_l / (mean_legal_variable_causal_edges(S_l) + eps)`

The final relation graph is:

`S = mean_l S_norm_l`

Attention is used only for relational routing and repair validation, not as scalar attention-weighted activation loss.

## Discrete Top-1 Acceptance Gate

After a sparse candidate is optimized, RAP_FINAL first discretizes it into Top-1 token ids, restores public fixed ids, and forwards public embeddings from those ids. A patch is accepted only when:

- continuous cut loss improves;
- at least one variable token changes after Top-1 discretization;
- discrete cut loss improves;
- discrete relational discrepancy does not worsen.

Rejected candidates are rolled back.

