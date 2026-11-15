#!/usr/bin/env python3
"""Matched reconstructed-HSI-specific yield evaluation using existing HSI-FT models.

Run one (method, reconstruction seed) per Slurm array task. This script DOES NOT
train reconstructions, overwrite the existing 50 test cubes, or use the
Gram/reference_mean40 diagnostic.

Requires the original HSI-FT-repo on Trinity. Its existing reconstruction,
subplot-feature extraction, split, and yield-predictor implementations are
imported, rather than reimplemented.
"""
import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get('HSIWHEAT40_ROOT', '/home/nibio/HSIwheat_40'))
BASE = Path(os.environ.get('HSIFT_SUPPLEMENTARY_BASE', str(ROOT / 'new_manuscript_method_transfer_5seeds')))
REPO = Path(__file__).resolve().parents[1]
OUT = BASE / 'yield_retrained_on_reconstruction'
METHODS = ('simplemlp', 'adjustedu2mdn', 'gram')


def init_project():
    if not REPO.is_dir():
        raise FileNotFoundError(f'Original HSI-FT code missing: {REPO}')
    os.environ['HSIWHEAT40_ROOT'] = str(ROOT)
    sys.path.insert(0, str(REPO))
    import config as C
    # Repoint only the output and cache paths; never overwrite the prior study.
    C.OUT = OUT
    C.CACHE = BASE / 'cache'   # original measured-feature cache is reused
    C.RECON_OUT = BASE / 'reconstruction'
    C.YIELD_OUT = OUT
    import common
    import reconstruct as R
    import yield_transfer as Y
    return C, common, R, Y


def load_reconstruction(method, seed, ck, device, C, common, R):
    data = common.torch_load(ck, map_location='cpu')
    actual = str(data.get('method', '')).lower()
    expected = {'simplemlp':'simplemlp', 'adjustedu2mdn':'adjustedu2mdn', 'gram':'gram'}[method]
    if actual and actual != expected:
        raise ValueError(f'Mismatched checkpoint method {actual} for {method}')
    if int(data['seed']) != seed:
        raise ValueError('Checkpoint seed mismatch')
    band_idx = tuple(int(i) for i in data['selected_band_indices_40'])
    if band_idx != (3,12,36):
        raise ValueError(f'Unexpected wavelength indices: {band_idx}')
    if method == 'simplemlp':
        model = R.SimpleMLP().to(device)
    elif method == 'adjustedu2mdn':
        model = R.AdjustedU2MDN().to(device)
    else:
        model = R.GramMapper().to(device)
    model.load_state_dict(data['model_state_dict'], strict=True)
    model.eval()
    return model, band_idx, data


def get_gram_context(C, common, R, band_idx, device):
    # Refit exactly the existing train/validation-only ridge mean restoration,
    # reusing the original implementation. Test HSI means are never used.
    pm, cube_table, _, _ = common.build_or_load_pixel_cache()
    tr = common.split_rows(cube_table, 'train')
    va = common.split_rows(cube_table, 'val')
    enc, dec, _, _ = common.load_frozen_decomposition(device)
    plot_mean40, _, _ = R.prepare_gram_targets(pm, cube_table, enc, device)
    _, mean3 = R.build_sparse_and_mean3(pm, cube_table, band_idx)
    ridge = R.fit_plot_mean_ridge(mean3, plot_mean40, cube_table, tr, va)
    return enc, dec, ridge


def infer_one_cube(method, model, ckdata, cube, band_idx, device, R,
                   gram_context=None):
    if cube.ndim != 3 or cube.shape[-1] != 40:
        raise ValueError(f'Expected H,W,40 measured cube, got {cube.shape}')
    flat = cube.reshape(-1,40)
    valid = np.any(flat != 0, axis=1)
    px = flat[valid].astype(np.float32, copy=False)
    if not len(px):
        raise ValueError('No nonzero spectra in measured cube')
    if method in ('simplemlp', 'adjustedu2mdn'):
        mu40 = np.asarray(ckdata['output_mean40'], dtype=np.float32)
        sd40 = np.asarray(ckdata['output_std40'], dtype=np.float32)
        input_mean = np.asarray(ckdata['input_mean'], dtype=np.float32)
        input_std = np.asarray(ckdata['input_std'], dtype=np.float32)
        # The original inference routine expects mean40/std40 and uses the
        # band indices to standardize its three input channels.
        if not (np.allclose(mu40[list(band_idx)], input_mean, rtol=1e-5, atol=1e-6)
                and np.allclose(sd40[list(band_idx)], input_std, rtol=1e-5, atol=1e-6)):
            raise ValueError('Input and output standardization differ: inspect original predictor before proceeding')
        if method == 'simplemlp':
            pred = R.simple_predict_pixels(model, px, band_idx, mu40, sd40, device)
        else:
            pred = R.u2_predict_pixels(model, px, band_idx, mu40, sd40, device)
    else:
        enc, dec, ridge = gram_context
        sparse = px[:, list(band_idx)]
        mean3 = sparse.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = R.gram_predict_centered(model, enc, dec, sparse, mean3, device)
        restored_mean = (mean3.astype(np.float64) @ ridge['W'] + ridge['b']).astype(np.float32)
        pred = centered + restored_mean.reshape(1,40)
    pred = np.asarray(pred, dtype=np.float32)
    if pred.shape != px.shape or not np.isfinite(pred).all():
        raise ValueError(f'Unexpected reconstruction output: {pred.shape} (expected {px.shape})')
    out = np.zeros_like(flat, dtype=np.float32)
    out[valid] = pred
    return out.reshape(cube.shape)


def feature_matrix(split, method, seed, model, ckdata, band_idx, device,
                   C, common, R, Y, gram_context):
    d = OUT / method / f'seed_{seed}'
    d.mkdir(parents=True, exist_ok=True)
    fpath = d / f'{split}_features.npy'
    meta = common.load_subplot_meta_with_new_split().reset_index(drop=True)
    sub = meta.loc[meta.split.eq(split)].copy().reset_index(drop=True)
    if fpath.exists():
        X = np.load(fpath)
        if X.shape != (len(sub), 81):
            raise ValueError(f'Cached feature shape mismatch: {fpath} {X.shape}')
        print(f'Loaded cached {split} features: {X.shape}', flush=True)
        return X, sub
    if split == 'test':
        existing = BASE / 'reconstruction' / method / 'current' / f'seed_{seed}' / 'yield_test_features.npy'
        if not existing.exists():
            raise FileNotFoundError(existing)
        X = np.load(existing)
        if X.shape != (len(sub),81):
            raise ValueError(f'Original test feature shape mismatch: {X.shape}')
        np.save(fpath, X)
        print(f'Reused original test features {existing}', flush=True)
        return X, sub
    maps = common.measured_maps()
    X = np.empty((len(sub),81),dtype=np.float32)
    groups = sub.groupby(['field','plot'], sort=False)
    print(f'{method} seed={seed}: reconstructing {groups.ngroups} {split} plots', flush=True)
    for j, ((field, plot), group) in enumerate(groups, 1):
        field = str(field)
        plot = common.normalize_plot_id(plot)
        measured = np.load(maps[field][plot]).astype(np.float32, copy=False)
        reconstruction = infer_one_cube(method, model, ckdata, measured,
                                        band_idx, device, R, gram_context)
        mask = np.load(common.frozen_mask_path(field, plot)).astype(bool,copy=False)
        X[group.index.to_numpy()] = Y._features_for_group(group, reconstruction, mask)
        if j == 1 or j % 50 == 0 or j == groups.ngroups:
            print(f'  {split}: {j}/{groups.ngroups} plots', flush=True)
    if not np.isfinite(X).all():
        raise RuntimeError(f'Nonfinite {split} features')
    # Atomic write prevents half-complete cached arrays being mistaken for complete.
    tmp = fpath.with_suffix('.tmp.npy')
    np.save(tmp, X)
    tmp.replace(fpath)
    return X, sub


def run(args):
    C, common, R, Y = init_project()
    method, seed = args.method, args.recon_seed
    rd = BASE / 'reconstruction' / method / 'current' / f'seed_{seed}'
    ck = rd / 'best_model.pth'
    if not ck.exists():
        raise FileNotFoundError(ck)
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Method={method}, recon seed={seed}, device={device}', flush=True)
    model, bands, ckdata = load_reconstruction(method,seed,ck,device,C,common,R)
    gram_context = get_gram_context(C, common, R, bands, device) if method == 'gram' else None
    Xs = {}
    metas = {}
    for split in ('train','val','test'):
        Xs[split], metas[split] = feature_matrix(split,method,seed,model,ckdata,bands,device,C,common,R,Y,gram_context)
    if [len(metas[x]) for x in ('train','val','test')] != [44966,4973,2565]:
        raise RuntimeError('Unexpected split counts; stop rather than train on a different split')
    y = {s: metas[s].subplot_yield.to_numpy(np.float32) for s in Xs}
    mean,std = Y.fit_scaler(Xs['train'],np.arange(len(Xs['train'])))
    Z = {s: Y.apply_scaler(Xs[s],mean,std) for s in Xs}
    dest = OUT / method / f'seed_{seed}'
    np.save(dest/'feature_mean_train_only.npy',mean)
    np.save(dest/'feature_std_train_only.npy',std)
    import torch.nn as nn
    rows=[]
    pred_frames=[]
    for yseed in range(5):
        outck = dest / f'yield_seed_{yseed}_bestval.pth'
        common.seed_everything(yseed)
        net = Y.YieldMLP(81).to(device)
        if outck.exists():
            prior = common.torch_load(outck,map_location=device)
            net.load_state_dict(prior['model_state_dict'])
            best_epoch = int(prior['best_epoch'])
            best_val = float(prior['best_validation_RMSE'])
        else:
            opt = torch.optim.Adam(net.parameters(), lr=C.YIELD_LR)
            criterion = nn.MSELoss()
            tr = Y.loader(Z['train'],y['train'],True)
            va = Y.loader(Z['val'],y['val'],False)
            best=float('inf'); best_epoch=-1; best_state=None
            history=[]
            for ep in range(1,C.YIELD_EPOCHS+1):
                net.train()
                for xb,yb in tr:
                    xb=xb.to(device);yb=yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    loss=criterion(net(xb),yb)
                    loss.backward();opt.step()
                vm=Y.loader_mse(net,va,device)
                if vm<best:
                    best=vm;best_epoch=ep;best_state=copy.deepcopy(net.state_dict())
                history.append({'epoch':ep,'validation_RMSE':math.sqrt(vm),
                                'best_validation_RMSE':math.sqrt(best),'best_epoch':best_epoch})
                if ep==1 or ep%20==0 or ep==C.YIELD_EPOCHS:
                    print(f'{method} rseed={seed} yseed={yseed} epoch={ep} val_RMSE={math.sqrt(vm):.4f} best={math.sqrt(best):.4f}@{best_epoch}',flush=True)
            net.load_state_dict(best_state)
            best_val=math.sqrt(best)
            pd.DataFrame(history).to_csv(dest/f'yield_seed_{yseed}_history.csv',index=False)
            torch.save({'method':method,'recon_seed':seed,'yield_seed':yseed,
                        'best_epoch':best_epoch,'best_validation_RMSE':best_val,
                        'feature_mean':mean,'feature_std':std,
                        'model_state_dict':best_state},outck)
        predictions=Y.predict(net,Z['test'],device)
        m=Y.metrics(y['test'],predictions)
        pm=Y.plot_metrics(metas['test'],y['test'],predictions)
        rows.append({'method':method,'recon_seed':seed,'yield_seed':yseed,
                     'best_epoch':best_epoch,'best_val_RMSE':best_val,
                     'subplot_R2':m['R2'],'subplot_RMSE':m['RMSE'],
                     'subplot_MAE':m['MAE'],'plot_R2':pm['R2'],
                     'plot_RMSE':pm['RMSE'],'plot_MAE':pm['MAE']})
        pframe=metas['test'][['field','plot','subplot_index']].copy()
        pframe['method']=method;pframe['recon_seed']=seed;pframe['yield_seed']=yseed
        pframe['true_yield']=y['test'];pframe['predicted_yield']=predictions
        pred_frames.append(pframe)
        print(f'FINISHED {method} rseed={seed} yseed={yseed} R2={m["R2"]:.4f} RMSE={m["RMSE"]:.4f}',flush=True)
    pd.DataFrame(rows).to_csv(dest/'yield_specific_results.csv',index=False)
    pd.concat(pred_frames,ignore_index=True).to_csv(dest/'yield_specific_test_predictions.csv',index=False)
    print(f'SAVED: {dest}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS,required=True)
    parser.add_argument('--recon-seed',type=int,choices=range(5),required=True)
    parser.add_argument('--cpu',action='store_true')
    run(parser.parse_args())