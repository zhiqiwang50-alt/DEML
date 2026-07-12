# Calibration-Aware Gradient-Directed Matching Design

This is a TinyLlama white-box pilot, not a full reproduction of the original large-model PIA paper.

## Checked Facts

1. The original PIA-style baseline is continuous embedding optimization followed by activation calibration.
2. Alpha-NN initialization is a separable initialization module and is not gradient matching.
3. Naive global gradient fusion is retained only as a failure control because prior runs show it can destroy token recovery.
4. Existing CG-GDPI gate-open rate can be zero, so those runs cannot prove gradient reranking is effective.
5. CA-GDM keeps calibration first and lets gradients assist only when calibration is uncertain and the attack-side gradient is reliable.

## Method

CA-GDM computes activation calibration for the full candidate set, keeps calibration top-M, and computes a gradient direction score only inside that top-M. The gate opens only when consensus, gradient margin, calibration uncertainty, candidate entropy, and token diversity pass. When the gate closes, the method selects calibration top-1.

The supported gradient scores are cosine direction matching and dot-product first-order descent scoring. Top-M equal to 1 and force-gate-closed both degenerate to pure calibration.

## Reporting Rule

A result with gate open rate equal to zero must not be described as evidence that gradient matching helped. If P1 or P2 does not exceed B1, the report must say so directly.
