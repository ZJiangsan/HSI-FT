#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final HSI-FT downstream analysis.

Uses the fixed Current triplet and reconstruction seeds 0-4 for Direct MLP,
adjusted u²-MDN and Gram. Yield models are trained only on measured HSI; the
measured-HSI scaler and model weights are then frozen for transfer.

Outputs are written to:
    <HSIWHEAT40_ROOT>/hsi_ft_outputs/yield_transfer/final_current_only_gram5/
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import config as C
from common import load_subplot_meta_with_new_split, resolve_triplets, safe_name
from yield_transfer import (
    _device,
    apply_scaler,
    build_measured_features,
    build_recon_test_features,
    metrics,
    predict,
    train_or_load_yield_models,
)

TRIPLET = "Current"
METHODS = ("SimpleMLP", "AdjustedU2MDN", "Gram")
RECON_SEEDS = (0, 1, 2, 3, 4)
YIELD_SEEDS = (0, 1, 2, 3, 4)
C.YIELD_SEEDS = YIELD_SEEDS

OUT = C.YIELD_OUT / "final_current_only_gram5"
OUT.mkdir(parents=True, exist_ok=True)

DISPLAY = {
    "SimpleMLP": "Direct MLP",
    "AdjustedU2MDN": "Adjusted u²-MDN",
    "Gram": "Gram",
}


def run_dir(method, seed):
    return C.RECON_OUT / safe_name(method) / safe_name(TRIPLET) / f"seed_{seed}"


def load_recon_metrics(method, seed):
    p = run_dir(method, seed) / "reconstruction_test_metrics.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def load_recon_features(meta, method, seed):
    p = run_dir(method, seed) / "yield_test_features.npy"
    if p.exists():
        return np.load(p).astype(np.float32, copy=False)
    x = build_recon_test_features(meta, method, TRIPLET, seed)
    np.save(p, np.asarray(x, dtype=np.float32))
    return np.asarray(x, dtype=np.float32)


def rel_fro(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.linalg.norm(a, ord="fro")
    return np.nan if d < 1e-12 else float(np.linalg.norm(b - a, ord="fro") / d)


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mean_abs_input_gradient(models, Ztest, device):
    acc = np.zeros_like(Ztest, dtype=np.float64)
    for seed in YIELD_SEEDS:
        model = models[seed]
        model.eval()
        x = torch.tensor(Ztest, dtype=torch.float32, device=device, requires_grad=True)
        out = model(x)
        grad = torch.autograd.grad(out.sum(), x)[0]
        acc += np.abs(grad.detach().cpu().numpy())
    return acc / len(YIELD_SEEDS)


lookup = {name: (idx, wl) for name, idx, wl in resolve_triplets()}
if TRIPLET not in lookup:
    raise KeyError(f"{TRIPLET!r} is not active in config.py")
selected_idx, wavelengths = lookup[TRIPLET]

for method in METHODS:
    for seed in RECON_SEEDS:
        rd = run_dir(method, seed)
        if not (rd / "reconstruction_test_metrics.json").exists():
            raise RuntimeError(f"Missing reconstruction metrics: {method} seed {seed}")
        if not ((rd / "yield_test_features.npy").exists() or (rd / "reconstructed40").exists()):
            raise RuntimeError(f"Missing reconstruction features/cubes: {method} seed {seed}")

device = _device()
meta = load_subplot_meta_with_new_split()

split_counts = (
    meta["split"].astype(str).str.lower().value_counts()
    .rename_axis("split").reset_index(name="n_subplots")
)
split_counts.to_csv(OUT / "subplot_split_counts.csv", index=False)

X_measured = build_measured_features(meta)
(
    models, scaler_mean, scaler_std, measured_preds, measured_baseline,
    train_idx, val_idx, test_idx, y, Z_measured
) = train_or_load_yield_models(meta, X_measured, device)

measured_baseline.to_csv(OUT / "measured_hsi_baseline_5seeds.csv", index=False)
np.save(OUT / "measured_feature_mean.npy", scaler_mean)
np.save(OUT / "measured_feature_std.npy", scaler_std)

X_measured = np.asarray(X_measured, dtype=np.float32)
Z_measured = np.asarray(Z_measured, dtype=np.float32)
y = np.asarray(y)
Xtest = X_measured[test_idx]
Ztrain = Z_measured[train_idx]
Ztest = Z_measured[test_idx]
ytest = y[test_idx]

spec = slice(0, 80)
means = slice(0, 40)
sds = slice(40, 80)

pca = PCA(n_components=0.95, svd_solver="full")
pca.fit(Ztrain[:, spec])
Pmeas = pca.transform(Ztest[:, spec])

abs_grad = mean_abs_input_gradient(models, Ztest, device)
grad80 = abs_grad[:, spec]
grad_weight = grad80 / (grad80.sum(axis=1, keepdims=True) + 1e-12)
mean_sensitivity = grad80.mean(axis=0)

pd.DataFrame({
    "feature_index": np.arange(80),
    "feature_type": ["mean"] * 40 + ["sd"] * 40,
    "mean_abs_gradient": mean_sensitivity,
}).to_csv(OUT / "measured_hsi_yield_model_feature_sensitivity.csv", index=False)

baseline_lookup = measured_baseline.set_index("yield_seed").to_dict("index")

transfer_rows = []
diag_rows = []
feature_rows = []
prediction_rows = []

for method in METHODS:
    for rseed in RECON_SEEDS:
        rmet = load_recon_metrics(method, rseed)
        Xrec = load_recon_features(meta, method, rseed)
        if Xrec.shape != Xtest.shape:
            raise ValueError(f"{method} seed {rseed}: {Xrec.shape} != {Xtest.shape}")

        Zrec = apply_scaler(Xrec, scaler_mean, scaler_std)
        dz = np.asarray(Zrec - Ztest, dtype=np.float64)
        dx = np.asarray(Xrec - Xtest, dtype=np.float64)

        A = Ztest[:, spec].astype(np.float64)
        B = Zrec[:, spec].astype(np.float64)
        cov_a, cov_b = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
        corr_a, corr_b = np.corrcoef(A, rowvar=False), np.corrcoef(B, rowvar=False)
        corr_a = np.nan_to_num(corr_a)
        corr_b = np.nan_to_num(corr_b)
        iu = np.triu_indices(80, k=1)

        Prec = pca.transform(B)
        pca_cov_a = np.atleast_2d(np.cov(Pmeas, rowvar=False))
        pca_cov_b = np.atleast_2d(np.cov(Prec, rowvar=False))

        diag = {
            "method": method,
            "triplet": TRIPLET,
            "recon_seed": rseed,
            "spectral_RMSE": rmet["rmse"],
            "spectral_SAM_deg": rmet["sam_deg"],
            "feature_mean_standardized_RMSE": float(np.sqrt(np.mean(dz[:, means] ** 2))),
            "feature_mean_standardized_MAE": float(np.mean(np.abs(dz[:, means]))),
            "feature_sd_standardized_RMSE": float(np.sqrt(np.mean(dz[:, sds] ** 2))),
            "feature_sd_standardized_MAE": float(np.mean(np.abs(dz[:, sds]))),
            "feature_all80_standardized_RMSE": float(np.sqrt(np.mean(dz[:, spec] ** 2))),
            "feature_all80_standardized_MAE": float(np.mean(np.abs(dz[:, spec]))),
            "feature_mean_raw_RMSE": float(np.sqrt(np.mean(dx[:, means] ** 2))),
            "feature_sd_raw_RMSE": float(np.sqrt(np.mean(dx[:, sds] ** 2))),
            "covariance_relative_Frobenius": rel_fro(cov_a, cov_b),
            "correlation_relative_Frobenius": rel_fro(corr_a, corr_b),
            "correlation_upper_triangle_agreement": safe_corr(corr_a[iu], corr_b[iu]),
            "pca_paired_shift_RMSE": float(np.sqrt(np.mean(np.sum((Prec - Pmeas) ** 2, axis=1)))),
            "pca_centroid_shift": float(np.linalg.norm(Prec.mean(0) - Pmeas.mean(0))),
            "pca_covariance_relative_Frobenius": rel_fro(pca_cov_a, pca_cov_b),
            "sensitivity_weighted_abs_error": float(
                np.mean(np.sum(grad_weight * np.abs(dz[:, spec]), axis=1))
            ),
        }
        diag_rows.append(diag)

        for j in range(80):
            feature_rows.append({
                "method": method,
                "recon_seed": rseed,
                "feature_index": j,
                "feature_type": "mean" if j < 40 else "sd",
                "band_index": j if j < 40 else j - 40,
                "standardized_RMSE": float(np.sqrt(np.mean(dz[:, j] ** 2))),
                "standardized_MAE": float(np.mean(np.abs(dz[:, j]))),
                "mean_abs_yield_gradient": float(mean_sensitivity[j]),
            })

        for yseed in YIELD_SEEDS:
            pred_meas = np.asarray(measured_preds[yseed])
            pred_rec = np.asarray(predict(models[yseed], Zrec, device))
            m = metrics(ytest, pred_rec)
            base = baseline_lookup[yseed]
            shift = float(math.sqrt(mean_squared_error(pred_meas, pred_rec)))
            transfer_rows.append({
                "method": method,
                "triplet": TRIPLET,
                "recon_seed": rseed,
                "yield_seed": yseed,
                "wl1_nm": wavelengths[0],
                "wl2_nm": wavelengths[1],
                "wl3_nm": wavelengths[2],
                "spectral_RMSE": rmet["rmse"],
                "spectral_SAM_deg": rmet["sam_deg"],
                "measured_R2": base["subplot_R2"],
                "measured_RMSE": base["subplot_RMSE"],
                "measured_MAE": base["subplot_MAE"],
                "recon_R2": m["R2"],
                "recon_RMSE": m["RMSE"],
                "recon_MAE": m["MAE"],
                "delta_R2": base["subplot_R2"] - m["R2"],
                "delta_RMSE": m["RMSE"] - base["subplot_RMSE"],
                "delta_MAE": m["MAE"] - base["subplot_MAE"],
                "prediction_shift_RMSE": shift,
            })
            for i in range(len(ytest)):
                prediction_rows.append({
                    "method": method,
                    "recon_seed": rseed,
                    "yield_seed": yseed,
                    "test_row": i,
                    "y_true": float(ytest[i]),
                    "pred_measured": float(pred_meas[i]),
                    "pred_reconstructed": float(pred_rec[i]),
                    "prediction_shift": float(pred_rec[i] - pred_meas[i]),
                })
        print("DONE", method, "seed", rseed)

transfer = pd.DataFrame(transfer_rows)
diag = pd.DataFrame(diag_rows)
features = pd.DataFrame(feature_rows)
predictions = pd.DataFrame(prediction_rows)

transfer.to_csv(OUT / "frozen_transfer_all.csv", index=False)
diag.to_csv(OUT / "representation_diagnostics_by_recon_seed.csv", index=False)
features.to_csv(OUT / "feature_errors_by_recon_seed.csv", index=False)
predictions.to_csv(OUT / "test_predictions_all.csv", index=False)

by_recon = (
    transfer.groupby(["method", "recon_seed"], as_index=False)
    .agg(
        n_yield_seeds=("yield_seed", "size"),
        R2_mean5yield=("recon_R2", "mean"),
        R2_sd5yield=("recon_R2", "std"),
        RMSE_mean5yield=("recon_RMSE", "mean"),
        RMSE_sd5yield=("recon_RMSE", "std"),
        MAE_mean5yield=("recon_MAE", "mean"),
        delta_R2_mean5yield=("delta_R2", "mean"),
        delta_R2_sd5yield=("delta_R2", "std"),
        delta_RMSE_mean5yield=("delta_RMSE", "mean"),
        delta_MAE_mean5yield=("delta_MAE", "mean"),
        prediction_shift_RMSE_mean5yield=("prediction_shift_RMSE", "mean"),
    )
    .merge(diag, on=["method", "recon_seed"], how="left")
)
by_recon.to_csv(OUT / "primary_transfer_by_reconstruction_seed.csv", index=False)

seed_ranges = (
    by_recon.groupby("method", as_index=False)
    .agg(
        R2_min_across_recon_seeds=("R2_mean5yield", "min"),
        R2_max_across_recon_seeds=("R2_mean5yield", "max"),
        delta_R2_min_across_recon_seeds=("delta_R2_mean5yield", "min"),
        delta_R2_max_across_recon_seeds=("delta_R2_mean5yield", "max"),
    )
)
seed_ranges.to_csv(OUT / "primary_seed_ranges.csv", index=False)

method_summary = (
    by_recon.groupby("method", as_index=False)
    .agg(
        n_reconstruction_seeds=("recon_seed", "size"),
        R2_mean=("R2_mean5yield", "mean"),
        R2_sd_across_recon_seeds=("R2_mean5yield", "std"),
        RMSE_mean=("RMSE_mean5yield", "mean"),
        RMSE_sd_across_recon_seeds=("RMSE_mean5yield", "std"),
        MAE_mean=("MAE_mean5yield", "mean"),
        delta_R2_mean=("delta_R2_mean5yield", "mean"),
        delta_R2_sd_across_recon_seeds=("delta_R2_mean5yield", "std"),
        delta_RMSE_mean=("delta_RMSE_mean5yield", "mean"),
        delta_MAE_mean=("delta_MAE_mean5yield", "mean"),
        prediction_shift_RMSE_mean=("prediction_shift_RMSE_mean5yield", "mean"),
        spectral_RMSE_mean=("spectral_RMSE", "mean"),
        spectral_RMSE_sd=("spectral_RMSE", "std"),
        spectral_SAM_deg_mean=("spectral_SAM_deg", "mean"),
        spectral_SAM_deg_sd=("spectral_SAM_deg", "std"),
        feature_mean_standardized_RMSE_mean=("feature_mean_standardized_RMSE", "mean"),
        feature_sd_standardized_RMSE_mean=("feature_sd_standardized_RMSE", "mean"),
        covariance_relative_Frobenius_mean=("covariance_relative_Frobenius", "mean"),
        correlation_relative_Frobenius_mean=("correlation_relative_Frobenius", "mean"),
        correlation_upper_triangle_agreement_mean=("correlation_upper_triangle_agreement", "mean"),
        pca_paired_shift_RMSE_mean=("pca_paired_shift_RMSE", "mean"),
        pca_centroid_shift_mean=("pca_centroid_shift", "mean"),
        pca_covariance_relative_Frobenius_mean=("pca_covariance_relative_Frobenius", "mean"),
        sensitivity_weighted_abs_error_mean=("sensitivity_weighted_abs_error", "mean"),
    )
)
method_summary.to_csv(OUT / "primary_method_summary.csv", index=False)

measured_summary = pd.DataFrame([{
    "n_yield_seeds": len(measured_baseline),
    "R2_mean": measured_baseline["subplot_R2"].mean(),
    "R2_sd": measured_baseline["subplot_R2"].std(),
    "RMSE_mean": measured_baseline["subplot_RMSE"].mean(),
    "RMSE_sd": measured_baseline["subplot_RMSE"].std(),
    "MAE_mean": measured_baseline["subplot_MAE"].mean(),
    "MAE_sd": measured_baseline["subplot_MAE"].std(),
}])
measured_summary.to_csv(OUT / "measured_hsi_summary.csv", index=False)

baseline_r2 = float(measured_summary.loc[0, "R2_mean"])
for metric, ylabel, fname, baseline in [
    ("R2_mean5yield", "Frozen-transfer R²", "figure_frozen_transfer_R2.png", baseline_r2),
    ("delta_R2_mean5yield", "ΔR²", "figure_delta_R2.png", 0.0),
]:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for i, method in enumerate(METHODS):
        d = by_recon[by_recon.method == method]
        ax.scatter(np.full(len(d), i), d[metric], s=60, alpha=0.85)
    ax.axhline(baseline, linestyle="--", linewidth=1.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([DISPLAY[m] for m in METHODS])
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

lookup = method_summary.set_index("method").to_dict("index")
lines = [
    "FINAL FIXED-TRIPLET HEADLINE SUMMARY — FIVE RECONSTRUCTION SEEDS",
    "=" * 70,
    f"Measured HSI: R2={measured_summary.loc[0,'R2_mean']:.4f} +/- {measured_summary.loc[0,'R2_sd']:.4f}; "
    f"RMSE={measured_summary.loc[0,'RMSE_mean']:.4f}",
]
for method in METHODS:
    s = lookup[method]
    lines.append(
        f"{method}: R2={s['R2_mean']:.4f} +/- {s['R2_sd_across_recon_seeds']:.4f}; "
        f"deltaR2={s['delta_R2_mean']:.4f} +/- {s['delta_R2_sd_across_recon_seeds']:.4f}; "
        f"deltaRMSE={s['delta_RMSE_mean']:.4f}; prediction-shift RMSE={s['prediction_shift_RMSE_mean']:.4f}; "
        f"spectral RMSE={s['spectral_RMSE_mean']:.6f}; SAM={s['spectral_SAM_deg_mean']:.4f}"
    )
gram_delta = lookup["Gram"]["delta_R2_mean"]
for baseline in ("SimpleMLP", "AdjustedU2MDN"):
    b = lookup[baseline]["delta_R2_mean"]
    lines.append(f"Gram reduction in mean deltaR2 vs {baseline}: {100*(b-gram_delta)/b:.1f}%")

text = "\n".join(lines)
(OUT / "headline_summary.txt").write_text(text + "\n", encoding="utf-8")
print("\n" + text)
print("\nOutputs:", OUT)
