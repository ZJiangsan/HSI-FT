#!/usr/bin/env python3
"""Measured-HSI prediction on an alternate yield-stratified plot partition.

Run hsi_ft_original_style_split.py first. Only downstream yield models are fit;
no decomposition or HSI reconstruction model is trained or changed. Feature
order is 40 means, 40 population SDs, and spike-and-leaf pixel count.
"""
import argparse
import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("HSIWHEAT40_ROOT", "/home/nibio/HSIwheat_40"))
REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "original_style_yield_stratified_seed2026")
    ap.add_argument("--features", type=int, nargs="+", choices=[80, 81], default=[80, 81])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    os.environ["HSIWHEAT40_ROOT"] = str(ROOT)
    sys.path.insert(0, str(REPO))
    import config as C
    import common
    import yield_transfer as Y

    split_path = args.base / "reconstruction_plot_split.csv"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    C.PLOT_SPLIT_CSV = split_path
    # Isolate this experiment's cache from the original reconstruction split.
    C.CACHE = args.base / "measured_prediction_cache"
    out = args.base / "prediction_only_5seeds"
    out.mkdir(parents=True, exist_ok=True)

    meta = common.load_subplot_meta_with_new_split().reset_index(drop=True)
    Xall = Y.build_measured_features(meta)
    if Xall.shape != (len(meta), 81):
        raise ValueError("Expected 81 measured features per subplot")
    yall = meta.subplot_yield.to_numpy(np.float32)
    idx = {split: np.flatnonzero(meta.split.to_numpy() == split)
           for split in ("train", "val", "test")}
    if any(not len(v) for v in idx.values()):
        raise ValueError("An empty downstream partition was found")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    rows = []

    for nf in args.features:
        X = np.asarray(Xall[:, :nf], dtype=np.float32)
        mean, std = Y.fit_scaler(X, idx["train"])
        Z = Y.apply_scaler(X, mean, std)
        d = out / ("features" + str(nf))
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "train_feature_mean.npy", mean)
        np.save(d / "train_feature_std.npy", std)

        for seed in range(5):
            common.seed_everything(seed)
            model = Y.YieldMLP(nf).to(device)
            checkpoint = d / ("yield_seed_" + str(seed) + "_bestval.pth")
            if checkpoint.exists():
                state = common.torch_load(checkpoint, map_location=device)
                if state.get("features") != nf or state.get("seed") != seed:
                    raise ValueError("Existing checkpoint does not match requested run")
                model.load_state_dict(state["model_state_dict"], strict=True)
                best_epoch = int(state["best_epoch"])
                best_val = float(state["best_val_RMSE"])
            else:
                optim = torch.optim.Adam(model.parameters(), lr=C.YIELD_LR)
                train = Y.loader(Z[idx["train"]], yall[idx["train"]], True)
                val = Y.loader(Z[idx["val"]], yall[idx["val"]], False)
                best, best_epoch, best_state = float("inf"), -1, None
                for epoch in range(1, C.YIELD_EPOCHS + 1):
                    model.train()
                    for xb, yb in train:
                        xb, yb = xb.to(device), yb.to(device)
                        optim.zero_grad(set_to_none=True)
                        torch.nn.functional.mse_loss(model(xb), yb).backward()
                        optim.step()
                    vm = Y.loader_mse(model, val, device)
                    if vm < best:
                        best, best_epoch = vm, epoch
                        best_state = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_state)
                best_val = math.sqrt(best)
                torch.save({"features": nf, "seed": seed, "best_epoch": best_epoch,
                            "best_val_RMSE": best_val, "model_state_dict": best_state},
                           checkpoint)

            ytest = yall[idx["test"]]
            prediction = Y.predict(model, Z[idx["test"]], device)
            subplot = Y.metrics(ytest, prediction)
            plot = Y.plot_metrics(meta.iloc[idx["test"]].reset_index(drop=True),
                                  ytest, prediction)
            row = {"features": nf, "seed": seed, "best_epoch": best_epoch,
                   "best_val_RMSE": best_val,
                   "subplot_R2": subplot["R2"], "subplot_RMSE": subplot["RMSE"],
                   "plot_R2": plot["R2"], "plot_RMSE": plot["RMSE"],
                   "plot_NRMSE": plot["NRMSE"]}
            rows.append(row)
            print(row, flush=True)
        pd.DataFrame([r for r in rows if r["features"] == nf]).to_csv(
            d / "results.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(out / "all_model_results.csv", index=False)
    print(results.groupby("features")[["subplot_R2", "plot_R2", "plot_RMSE"]]
          .agg(["mean", "std"]))


if __name__ == "__main__":
    main()
