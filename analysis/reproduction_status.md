# TinyLlama-adapted reproduction of Prompt Inversion Attack experiments: reproduction status

## Dataset status

- Skytrax: local `data/airline.json` is available, but it currently contains only 29 raw prompts / 28 valid prompts, so it is insufficient for the requested 150-prompt protocol. A direct `git ls-remote` to the README-referenced public Skytrax repository succeeded, but `git clone` into `runs/tinyllama_full/data_sources/` disconnected while reading the sideband packet. No proxy or system configuration was changed. Until a complete legal Skytrax CSV is available, main runs are labeled `Skytrax-28 pilot`.
- CMS: no verifiable local CMS dataset was found. `data/medical.json` exists, but it is not labeled or documented as CMS, so it was not treated as a completed CMS reproduction.
- ECHR: no verifiable local ECHR dataset was found.

## Baseline status

- Song et al.-style softmax embedding optimization baseline: implemented as a local baseline label for engineering comparison only.
- Li et al. and Morris et al.: not faithfully reproduced. No official implementation and matching experimental setup were found in this repository.

## Grey-box LoRA status

No verifiable TinyLlama LoRA adapter, matching training data, and held-out evaluation setup were found locally. White-box results are not reported as grey-box.
