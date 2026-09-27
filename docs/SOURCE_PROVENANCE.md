# Source provenance

This repository packages the final code path used for the HSI-FT manuscript.

## Shared upstream HSIwheat preparation

The fixed 40-band cubes, subplot geometry, yield targets and spike-and-leaf masks are shared with the preceding HSIwheat evaluation study. The companion repository is:

`ZJiangsan/HSIRecon-Transfer`

`preprocessing/prepare_40band_data.py` is copied from that established preparation workflow so the 190-to-40-band conversion is visible here as well.

## New HSI-FT experiment

The final HSI-FT analysis uses:

- a deterministic plot-disjoint split generated with seed 2026;
- a decomposition fitted only on the 875 reconstruction-training plots;
- one fixed three-band input condition;
- five reconstruction seeds for each of Direct MLP, adjusted u²-MDN and Gram;
- five genuine-HSI yield-model seeds;
- a frozen measured-HSI feature scaler and frozen genuine-HSI yield models for transfer;
- held-out representation diagnostics computed only after all fitted components are frozen.

Development-only scripts (pilot Gram runs, temporary seed-0 rerun scripts, storage-recovery scripts and abandoned multi-triplet runners) are intentionally excluded from the public execution path.

## Reported results

The CSV files under `results/` are copied from the final five-reconstruction-seed analysis used for the manuscript tables and figures. Large reconstructed cubes and model checkpoints are not committed.
