# Reproducibility notes

## Study scope

This repository reproduces the **single fixed-input mechanistic experiment** in HSI-FT. It intentionally does not repeat the 11-triplet wavelength sweep from the preceding HSIwheat evaluation study.

## Fixed sparse input

Nominal triplet:

```text
460.7 / 550.6 / 801.7 nm
```

Nearest wavelengths in the prepared 40-band array during the reported experiment:

```text
460.6694560669456
550.6276150627615
801.673640167364
```

Indices:

```text
(3, 12, 36)
```

## Frozen split

The plot split is generated before decomposition training using NumPy RNG seed 2026:

```text
875 train
96 validation
50 test
```

All pixels and downstream subplots from a plot remain in the same split.

## Reconstruction methods

### Direct MLP

```text
3 -> 128 -> 128 -> 40
```

- ReLU hidden layers
- standardized input/output
- MSE objective
- AdamW, lr 1e-3, weight decay 1e-5
- batch size 16,384
- 1,000,000 sampled training pixels/epoch with replacement
- validation sample <= 300,000 pixels
- max 500 epochs
- patience 40
- minimum-validation-MSE checkpoint

### Adjusted u²-MDN

Same-resolution adaptation retaining the dense encoder, stick-breaking/Dirichlet-style latent representation and decoder.

- Adam, lr 1e-3
- batch size 16,384
- 1,000,000 sampled training pixels/epoch
- same validation sample budget as Direct MLP
- max 500 epochs
- patience 40
- minimum-validation-loss checkpoint

### Gram mapper

- frozen train-only 40-band decomposition
- centered three-band input
- bias-free 3-to-40 mapper plus residual channel repetition
- frozen abundance encoder
- Gram matching + reconstruction consistency objective
- Adam, lr 1e-4, weight decay 1e-4
- 5522 epochs for every reconstruction seed
- validation-best checkpoint within the fixed budget

Absolute spectral level is restored with a train/validation-only ridge regression from three observed-band plot means to the 40-band plot mean.

## Downstream model

Features:

```text
40 band means + 40 band SDs + SL pixel count = 81
```

MLP:

```text
81 -> 10 -> 10 -> 10 -> 10 -> 1
```

- ReLU
- Xavier initialization
- Adam, lr 1e-3
- MSE
- batch size 32
- max 100 epochs
- checkpoint selected only by genuine-HSI validation RMSE

The genuine-HSI training scaler is frozen and reused for reconstructed HSI.

## Replication units

For every reconstruction method:

```text
5 reconstruction seeds x 5 yield-model seeds
```

Method-level uncertainty in the paper is summarized across reconstruction-seed means, because the same five yield-model seeds are repeatedly applied to every reconstruction seed.

## Primary transfer metrics

```text
ΔR²   = R²_measured - R²_reconstructed
ΔRMSE = RMSE_reconstructed - RMSE_measured
ΔMAE  = MAE_reconstructed - MAE_measured
```

Prediction-shift RMSE compares the outputs of the same frozen yield model on reconstructed versus measured HSI for identical held-out subplots.

## Mechanism diagnostics

The final analysis reports:

- standardized error in the 40 mean features;
- standardized error in the 40 SD features;
- covariance and correlation mismatch;
- measured-HSI-training PCA displacement;
- model-sensitivity-weighted feature error;
- prediction shift.

These are held-out diagnostics only and are never used to select checkpoints or hyperparameters.
