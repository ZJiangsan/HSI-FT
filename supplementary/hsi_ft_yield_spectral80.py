#!/usr/bin/env python3
"""HSI-FT: reconstruction-specific yield prediction with 80 spectral features.

Uses ALREADY-SAVED train/val/test feature arrays from the completed 81-feature
experiment. Excludes column 80 (SL pixel count) from *predictor inputs* but
keeps the original subplot-yield target for an exact feature-ablation control.

Does not re-run hyperspectral reconstruction and does not modify any 81-feature
experiment outputs. One reconstruction seed per invocation, five yield seeds.
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
from torch import nn

ROOT = Path(os.environ.get('HSIWHEAT40_ROOT', '/home/nibio/HSIwheat_40'))
BASE = Path(os.environ.get('HSIFT_SUPPLEMENTARY_BASE', str(ROOT / 'new_manuscript_method_transfer_5seeds')))
REPO = Path(__file__).resolve().parents[1]
INPUT = BASE / 'yield_retrained_on_reconstruction'
OUTPUT = BASE / 'yield_retrained_spectral80'
METHODS = ('simplemlp', 'adjustedu2mdn', 'gram')
SPLIT_COUNTS = {'train': 44966, 'val': 4973, 'test': 2565}


def main(args):
    os.environ['HSIWHEAT40_ROOT'] = str(ROOT)
    sys.path.insert(0, str(REPO))
    import config as C
    import common
    import yield_transfer as Y

    method, rseed = args.method, args.recon_seed
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    inp = INPUT / method / f'seed_{rseed}'
    dest = OUTPUT / method / f'seed_{rseed}'
    dest.mkdir(parents=True, exist_ok=True)

    meta = common.load_subplot_meta_with_new_split().reset_index(drop=True)
    X, y, meta_by_split = {}, {}, {}
    for split, n in SPLIT_COUNTS.items():
        path = inp / f'{split}_features.npy'
        if not path.is_file():
            raise FileNotFoundError(f'Missing 81-feature matrix: {path}. Run the completed 81-feature pipeline first.')
        full = np.load(path, mmap_mode='r')
        if full.shape != (n, 81):
            raise ValueError(f'{path}: expected {(n, 81)}, got {full.shape}')
        if not np.isfinite(full).all():
            raise ValueError(f'Non-finite feature values: {path}')
        # First 40: band means; next 40: within-subplot band standard deviations.
        # Deliberately DROP the last, 81st, SL pixel-count feature.
        X[split] = np.array(full[:, :80], dtype=np.float32, copy=True)
        m = meta.loc[meta.split.eq(split)].copy().reset_index(drop=True)
        if len(m) != n:
            raise ValueError(f'Metadata count mismatch for {split}: {len(m)} vs {n}')
        meta_by_split[split] = m
        y[split] = m.subplot_yield.to_numpy(np.float32)
        print(f'{method} recon seed {rseed}: loaded {split} X={X[split].shape}, y={y[split].shape}', flush=True)

    mean, std = Y.fit_scaler(X['train'], np.arange(len(X['train'])))
    Z = {s: Y.apply_scaler(X[s], mean, std) for s in X}
    np.save(dest / 'spectral80_feature_mean_train_only.npy', mean)
    np.save(dest / 'spectral80_feature_std_train_only.npy', std)

    results, prediction_frames = [], []
    for yseed in range(5):
        common.seed_everything(yseed)
        model = Y.YieldMLP(80).to(device)
        ckpath = dest / f'yield_seed_{yseed}_bestval.pth'
        if ckpath.exists():
            ck = common.torch_load(ckpath, map_location=device)
            if ck.get('input_dim') != 80 or ck.get('method') != method or ck.get('recon_seed') != rseed:
                raise ValueError(f'Checkpoint metadata mismatch: {ckpath}')
            model.load_state_dict(ck['model_state_dict'], strict=True)
            best_epoch = int(ck['best_epoch'])
            best_val = float(ck['best_validation_RMSE'])
            print(f'Reusing trained yield seed {yseed} checkpoint: {ckpath}', flush=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=C.YIELD_LR)
            criterion = nn.MSELoss()
            train_loader = Y.loader(Z['train'], y['train'], True)
            val_loader = Y.loader(Z['val'], y['val'], False)
            best_mse = float('inf')
            best_epoch = -1
            best_state = None
            history = []
            for epoch in range(1, C.YIELD_EPOCHS + 1):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                val_mse = float(Y.loader_mse(model, val_loader, device))
                if val_mse < best_mse:
                    best_mse = val_mse
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                history.append({'epoch': epoch,
                                'validation_RMSE': math.sqrt(val_mse),
                                'best_validation_RMSE': math.sqrt(best_mse),
                                'best_epoch': best_epoch})
                if epoch == 1 or epoch % 20 == 0 or epoch == C.YIELD_EPOCHS:
                    print(f'{method} rseed={rseed} yseed={yseed} epoch={epoch} '
                          f'val_RMSE={math.sqrt(val_mse):.4f} best={math.sqrt(best_mse):.4f}@{best_epoch}', flush=True)
            model.load_state_dict(best_state)
            best_val = math.sqrt(best_mse)
            pd.DataFrame(history).to_csv(dest / f'yield_seed_{yseed}_history.csv', index=False)
            torch.save({'method': method, 'recon_seed': rseed, 'yield_seed': yseed,
                        'input_dim': 80, 'best_epoch': best_epoch,
                        'best_validation_RMSE': best_val,
                        'feature_mean': mean, 'feature_std': std,
                        'model_state_dict': best_state}, ckpath)

        prediction = Y.predict(model, Z['test'], device)
        subplot = Y.metrics(y['test'], prediction)
        plot = Y.plot_metrics(meta_by_split['test'], y['test'], prediction)
        results.append({'method': method, 'recon_seed': rseed, 'yield_seed': yseed,
                        'n_features': 80, 'best_epoch': best_epoch, 'best_val_RMSE': best_val,
                        'subplot_R2': subplot['R2'], 'subplot_RMSE': subplot['RMSE'],
                        'subplot_MAE': subplot['MAE'],
                        'plot_R2': plot['R2'], 'plot_RMSE': plot['RMSE'], 'plot_MAE': plot['MAE']})
        pf = meta_by_split['test'][['field', 'plot', 'subplot_index']].copy()
        pf['method'] = method
        pf['recon_seed'] = rseed
        pf['yield_seed'] = yseed
        pf['true_yield'] = y['test']
        pf['predicted_yield'] = prediction
        prediction_frames.append(pf)
        print(f'FINISHED {method} rseed={rseed} yseed={yseed} '
              f'R2={subplot["R2"]:.4f} RMSE={subplot["RMSE"]:.4f} '
              f'plot_R2={plot["R2"]:.4f}', flush=True)

    pd.DataFrame(results).to_csv(dest / 'yield_spectral80_results.csv', index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(dest / 'yield_spectral80_test_predictions.csv', index=False)
    print(f'SAVED {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--recon-seed', required=True, type=int, choices=range(5))
    parser.add_argument('--cpu', action='store_true')
    main(parser.parse_args())