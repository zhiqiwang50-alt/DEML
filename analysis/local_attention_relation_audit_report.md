# Local Attention Relation Audit Report

## Candidate Label Summary

| Label | Count | JS true mean/median/std | mass true mean/median/std | LAER true mean/median/std | LAER shuffled mean/median/std | mean GT gain |
|---|---:|---:|---:|---:|---:|---:|
| positive | 125 | -0.020729736484587192 / -0.009399265050888062 / 0.026631236149903176 | -0.0032997279170554825 / -0.0022672041013720445 / 0.005359639061724526 | -0.0021876688610875028 / -0.0012675663814521753 / 0.003295994446240973 | -0.0008591668077717828 / -0.00046531493334393825 / 0.002474337855705827 | 1.2 |
| neutral | 70 | -0.028047353508216993 / -0.017104700207710266 / 0.035321398465906886 | -0.0017817355356811344 / -0.0007033336991702933 / 0.004120699761101316 | -0.0010572832941820247 / -0.0005584120454538305 / 0.0025359735508073173 | -0.000469101037679994 / -2.041907930712362e-05 / 0.0026179907869386503 | 0.0 |
| harmful | 21 | -0.025314606903564362 / -0.012766346335411072 / 0.03434204564741051 | 0.0015271852862257232 / 0.0008359642597497441 / 0.0033290871834452474 | 0.0007963526204208336 / 0.0004146430867666244 / 0.0016831688689387708 | 0.0007015832421755538 / 0.0003170638239749198 / 0.0011453295913628305 | -1.2857142857142858 |

## Overall

- frozen candidate count: 216
- candidate hash agreement: 1.0
- before hash agreement: 1.0
- selected rows same true/shuffled: 1.0
- changed positions match candidate trace: 1.0
- local attention query selection rate: 0.0
- LAER illegal q<=k count: 0
- LAER fixed/public key hit count: 0
- P(harmful | delta_JS > 0): 0.12
- P(harmful | delta_JS <= 0): 0.09424083769633508
- Spearman(-delta_JS_true, gt_gain): -0.027564258727245144
- Spearman(-delta_JS_shuffled, gt_gain): 0.0338219368773543
- Spearman(-delta_mass_true, gt_gain): 0.3803553916536785
- Spearman(-delta_mass_shuffled, gt_gain): 0.2707246410828743
- Spearman(-delta_LAER_true, gt_gain): 0.3881529544909534
- Spearman(-delta_LAER_shuffled, gt_gain): 0.26316145063194796
- AUC JS true/shuffled: 0.4458901098901099 / 0.5199120879120879
- AUC mass true/shuffled: 0.6846593406593406 / 0.6333186813186813
- AUC LAER true/shuffled: 0.7021538461538461 / 0.6371868131868131
- AUC harmful LAER true/shuffled: 0.8273504273504273 / 0.7494505494505495

## Gate Replay

| Gate | Accepted | Positive retained | Harmful retained | Harmful rejected | Beneficial rejected | Precision | Harm rejection | Beneficial retention | Mean GT gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUT_ONLY | 216 | 125 | 21 | 0 | 0 | 0.5787037037037037 | 0.0 | 1.0 | 0.5694444444444444 |
| MASS_GATE | 167 | 115 | 7 | 14 | 10 | 0.688622754491018 | 0.6666666666666666 | 0.92 | 0.7604790419161677 |
| LAER_GATE | 166 | 114 | 6 | 15 | 11 | 0.6867469879518072 | 0.7142857142857143 | 0.912 | 0.7771084337349398 |
| SHUFFLED_LAER_GATE | 135 | 95 | 4 | 17 | 30 | 0.7037037037037037 | 0.8095238095238095 | 0.76 | 0.8074074074074075 |

## Final Judgment

This is a diagnostic replay, not a formal new attack. The candidate path remains B0-SPARSE, and local attention scores are computed only after the frozen candidate is selected.

- TRUE LAER positive-candidate AUC: 0.7021538461538461
- SHUFFLED LAER positive-candidate AUC: 0.6371868131868131
- TRUE/SHUFFLED LAER harmful-candidate AUC: 0.8273504273504273 / 0.7494505494505495

TRUE LAER is directionally stronger than the shuffled control on this dev diagnostic. The LAER gate rejects most harmful candidates while retaining most beneficial candidates, so it is worth considering a formal `B0 + Sparse + Local Attention Validation` method. This is not yet a heldout result and should not be described as stable improvement.

## Required Answers

1. Sequential RNG fixed candidate count: 216.
2. Alignment with formal B0-SPARSE: the replay uses run-level RNG and matches the formal B0-SPARSE dev Token Accuracy/BLEU when evaluated as CUT_ONLY.
3. Positive/neutral/harmful counts: 125 / 70 / 21.
4. delta_mass distributions are listed in the Candidate Label Summary table.
5. delta_LAER distributions are listed in the Candidate Label Summary table.
6. true vs shuffled Spearman/AUC: mass Spearman 0.3803553916536785 / 0.2707246410828743; LAER Spearman 0.3881529544909534 / 0.26316145063194796; LAER AUC 0.7021538461538461 / 0.6371868131868131.
7. LAER gate harmful rejection is reported in the Gate Replay table.
8. LAER gate beneficial rejection is reported in the Gate Replay table.
9. Evidence supports local attention relation as a verifier diagnostic on this dev split.
10. Recommendation: implement a formal method only as the next controlled experiment, still using B0 as the main baseline and requiring heldout validation before any strong claim.
