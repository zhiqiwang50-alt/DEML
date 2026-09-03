# LAER and B0-SPARSE Prompt Inversion Method

Clean package for the LAER / B0-SPARSE strict Top-1 white-box prompt inversion method.

Included: main LAER/B0-SPARSE code, direct repair/audit helpers, runner scripts, tests, design/report/audit notes, lightweight summary files, and small data/split files needed for reproduction.

`paper_final_driver.py` has been renamed to `laer_final_experiment_driver.py` in this branch because it is the LAER final-experiment scheduler, not a separate paper-final method.

Excluded: AIR, P4, MTS, CAGDM, SACR, RAP method variants, raw prediction logs, and large generated outputs.

Main entry points:

```bash
bash scripts/run_b0_laer.sh --help
python laer_final_experiment_driver.py --help
```
