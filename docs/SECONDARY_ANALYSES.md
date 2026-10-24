# HSI-FT additional analyses (24 October 2026)

This package supplements the [HSI-FT repository](https://github.com/ZJiangsan/HSI-FT) with two post-hoc, frozen-model diagnostics for the fixed **460.669/550.628/801.674 nm** input and the 40-band wheat dataset.

## Scope and interpretation

- **Paired plot-level bootstrap** (`n=2,000`, seed `20261024`) resamples the same 50 held-out plots for all methods and estimates percentile 95% intervals for mean seed-averaged transfer R² and Gram-versus-baseline R² differences. The results describe held-out plot sampling uncertainty conditional on the trained reconstruction and yield models. They do **not** represent 2,000 independent model fits, new sites, new seasons, or alternative sensors.
- **Feature-group substitution** applies the five unchanged, measured-HSI-trained yield predictors to features with either reconstructed band means and measured band standard deviations, or measured means and reconstructed standard deviations. The subplot pixel-count feature remains unchanged. These are oracle diagnostics relying on held-out measured HSI; they are **not** usable inference-time reconstruction methods or causal isolation of individual loss terms.

## Reproduce on an existing experiment installation

Place `HSI_FT_additional_analyses.py` in the original repository root (beside `config.py`, `common.py` and `yield_transfer.py`). Provide the prepared dataset, cached 15 `yield_test_features.npy` arrays, 5 `seed_*_bestval.pth` yield-model checkpoints, 40-band measured-feature/scaler caches, and `test_predictions_all.csv` from `final_transfer_diagnostics.py`.

The upstream configuration uses `HSIWHEAT40_ROOT` and expects a subdirectory named `hsi_ft_outputs`, containing `reconstruction`, `yield_transfer`, and `cache`.

```bash
export HSIWHEAT40_ROOT=/path/to/HSIwheat_40
python HSI_FT_additional_analyses.py --repo . --bootstrap 2000 --seed 20261024
```

On the TrinityX copy documented for this study, the output directory was named `new_manuscript_method_transfer_5seeds` rather than `hsi_ft_outputs`; an optional **symbolic link** can provide the expected name without copying data:

```bash
ln -s /path/to/HSIwheat_40/new_manuscript_method_transfer_5seeds /path/to/HSIwheat_40/hsi_ft_outputs
```

Requires the original repository's Python environment, including NumPy, pandas, scikit-learn and PyTorch. This script reads the **existing** frozen checkpoints, does not retrain or select models, and writes new CSVs to `hsi_ft_outputs/yield_transfer/final_current_only_gram5/`.

## Outputs included in this package

| File | Meaning |
|---|---|
| `results/additional_plot_bootstrap_summary.csv` | Point estimates and percentile CIs |
| `results/additional_plot_bootstrap_iterations.csv` | All bootstrap replicates including iteration 0 (original full sample) |
| `results/additional_feature_group_swap_summary.csv` | Seed-averaged performance for each substitution condition |
| `results/additional_feature_group_swap_all.csv` | Every reconstruction-seed × yield-seed × feature-condition result |

### Principal verified results

- Gram versus direct MLP: mean R² difference **+0.07979**; 95% paired plot-bootstrap interval **[+0.02204, +0.13059]**.
- Gram versus adjusted u²-MDN: mean R² difference **+0.06483**; 95% paired plot-bootstrap interval **[+0.01048, +0.11308]**.
- Full reconstruction R²: direct MLP **0.6562**, adjusted u²-MDN **0.6712**, Gram **0.7360**.

Do not interpret the intervals as validation of external transfer. The study has one held-out plot split and one three-band input condition.

## Suggested GitHub integration

Add the Python script to the repository root; add this file as `docs/SECONDARY_ANALYSES.md`; place the four CSVs under `results/` without overwriting other result files. Link `docs/SECONDARY_ANALYSES.md` from `README.md` and `docs/REPRODUCIBILITY.md`. Do not commit raw data, full-resolution predictions, or trained checkpoints unless release rights and file-size limits have been verified.