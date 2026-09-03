# B0-LAER Development Report

Skytrax-28 / seed42 / layer17 / epoch100 / strict Top-1. B0 and B0-SPARSE are reused from matching COMPLETE ACDR runs.

## Method Results

| Method | Acc | BLEU | Exact | NED | Completed | Failed | Reused |
|---|---:|---:|---:|---:|---:|---:|---:|
| b0 | 0.8378855490118047 | 0.6414763941990775 | 0.0 | 0.16079002562301473 | 28 | 0 | True |
| b0_sparse | 0.8666333322564294 | 0.702876080107508 | 0.0 | 0.13233421932812872 | 28 | 0 | True |
| b0_laer | 0.8657163093757748 | 0.700081891328645 | 0.0 | 0.13323186828941724 | 28 | 0 | False |
| b0_shuffled_laer | 0.8605956151924768 | 0.6911791041639574 | 0.0 | 0.138309601120619 | 28 | 0 | False |

## Paired Comparisons

| Comparison | Acc delta | BLEU delta | W/T/L | Acc 95% CI | BLEU 95% CI | fix/harm/net |
|---|---:|---:|---:|---:|---:|---:|
| b0_sparse_vs_b0 | 0.02874778324462474 | 0.06139968590843045 | 25/1/2 | [0.020459541355204353, 0.03761405596204514] | [0.04433600919363829, 0.08003373368097817] | 162/39/123 |
| b0_laer_vs_b0 | 0.027830760363970173 | 0.05860549712956742 | 25/3/0 | [0.02125496582360797, 0.03469734946004591] | [0.04400070422453688, 0.07445328217837507] | 147/28/119 |
| b0_laer_vs_b0_sparse | -0.0009170228806545666 | -0.0027941887788630326 | 9/10/9 | [-0.005816509515423424, 0.0037733046941301356] | [-0.011021015532630435, 0.0050650883261357405] | 40/44/-4 |
| b0_laer_vs_b0_shuffled_laer | 0.005120694183297949 | 0.008902787164687556 | 14/10/4 | [0.0013717625338433809, 0.009035716287652623] | [0.0008598312566396871, 0.017224614852381154] | 49/26/23 |

## Patch Mechanism

| Method | Events | Accepted | Rejected | Rejected by LAER | Accept rate | delta LAER mean | rejected beneficial/harmful | accepted beneficial/harmful | q<=k violations | fixed key hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b0 | 0 | 0 | 0 | 0 | None | None | 0/0 | 0/0 | 0 | 0 |
| b0_sparse | 560 | 216 | 344 | 0 | 0.38571428571428573 | None | 0/0 | 0/0 | 0 | 0 |
| b0_laer | 560 | 167 | 393 | 61 | 0.2982142857142857 | -0.0013042824524245656 | 11/17 | 107/6 | 0 | 0 |
| b0_shuffled_laer | 560 | 133 | 427 | 107 | 0.2375 | -0.00022534466250202138 | 41/16 | 84/6 | 0 | 0 |

## Pre-Gate Matched Audit

| Comparison | Pre-div patches | Candidate hash agree | Rate | Prompt divergence count |
|---|---:|---:|---:|---:|
| b0_laer_vs_b0_shuffled_laer | 207 | 207 | 1.0 | 23 |

## Decision

Heldout preparation condition met: False.
This is a dev-only result. Do not describe it as stable improvement unless frozen heldout validation later supports it.
