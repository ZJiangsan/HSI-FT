#!/usr/bin/env python3
"""Plot-cluster paired bootstrap for HSI-FT's 81/80 feature experiments.

Reads predictions saved by hsi_ft_reconstruction_specific_yield.py and
hsi_ft_yield_spectral80.py. Resamples the same 50 held-out plots in all
methods; calculates subplot R2 and RMSE averaged across 25 model fits.
The five reconstruction seeds (and five yield seeds per reconstruction)
are NOT treated as 25 independent biological replicates.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get('HSIFT_SUPPLEMENTARY_BASE', str(Path(os.environ.get('HSIWHEAT40_ROOT', '/home/nibio/HSIwheat_40')) / 'new_manuscript_method_transfer_5seeds')))
METHODS = ('simplemlp', 'adjustedu2mdn', 'gram')
REPO = Path(__file__).resolve().parents[1]


def load_predictions(root, fname):
    all_dfs = []
    for method in METHODS:
        for rs in range(5):
            p = root / method / f'seed_{rs}' / fname
            if not p.is_file():
                raise FileNotFoundError(p)
            df = pd.read_csv(p)
            required = {'field','plot','subplot_index','yield_seed','true_yield','predicted_yield'}
            if not required.issubset(df.columns):
                raise ValueError(f'{p}: missing columns {required-set(df.columns)}')
            df['method'] = method
            df['recon_seed'] = rs
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def matrices(df):
    # Explicit identifier-based alignment; never rely on accidental file order.
    df = df.copy()
    df['key'] = df['field'].astype(str) + '|' + df['plot'].astype(str) + '|' + df['subplot_index'].astype(str)
    df['plotkey'] = df['field'].astype(str) + '|' + df['plot'].astype(str)
    ref = df.loc[(df.method == METHODS[0]) & (df.recon_seed == 0) & (df.yield_seed == 0),
                 ['key','plotkey','true_yield']].sort_values('key').reset_index(drop=True)
    if len(ref) != 2565 or ref.key.duplicated().any() or ref.plotkey.nunique() != 50:
        raise ValueError(f'Expected 2565 unique subplots in 50 plots, found {len(ref)} and {ref.plotkey.nunique()}')
    y = ref.true_yield.to_numpy(float)
    matrix = {}
    for method in METHODS:
        vals = []
        for rs in range(5):
            for ys in range(5):
                sub = df.loc[(df.method == method) & (df.recon_seed == rs) & (df.yield_seed == ys)]
                sub = sub.set_index('key').reindex(ref.key)
                if len(sub) != len(ref) or sub.predicted_yield.isna().any():
                    raise ValueError(f'Missing or unaligned predictions: {method}/{rs}/{ys}')
                if not np.allclose(sub.true_yield.to_numpy(float), y, atol=1e-3, rtol=1e-5):
                    raise ValueError(f'Inconsistent test targets: {method}/{rs}/{ys}')
                vals.append(sub.predicted_yield.to_numpy(float))
        matrix[method] = np.stack(vals)
    plots = ref.plotkey.to_numpy()
    return y, plots, matrix


def measured81_predictions(plots, y):
    """Recreate measured baseline predictions from the SAME saved five frozen fits."""
    os.environ['HSIWHEAT40_ROOT'] = str(Path(os.environ.get('HSIWHEAT40_ROOT', '/home/nibio/HSIwheat_40')))
    sys.path.insert(0, str(REPO))
    import torch
    import common
    import yield_transfer as Y
    meta=common.load_subplot_meta_with_new_split().reset_index(drop=True)
    sel=meta.loc[meta.split.eq('test')].reset_index(drop=True)
    cache=BASE/'cache'/'measured40_subplot_features_newsplit.npy'
    X=np.load(cache)
    if X.shape != (len(meta),81):raise ValueError('Measured feature cache shape mismatch')
    x=X[meta.split.to_numpy() == 'test']
    keys=(sel.field.astype(str)+'|'+sel['plot'].astype(str)+'|'+sel.subplot_index.astype(str))
    # Align measured predictions with the same sorted test keys used in matrices().
    ix=np.argsort(keys.to_numpy())
    assert np.array_equal((sel.field.astype(str)+'|'+sel['plot'].astype(str)).to_numpy()[ix], plots)
    if not np.allclose(sel.subplot_yield.to_numpy(float)[ix], y,atol=1e-3,rtol=1e-5):
        raise ValueError('Measured model test targets not aligned to reconstruction results')
    all_preds=[]
    for seed in range(5):
        ck=common.torch_load(BASE/'yield_transfer'/'checkpoints_measured_hsi'/f'seed_{seed}_bestval.pth',map_location='cpu')
        net=Y.YieldMLP(81)
        net.load_state_dict(ck['model_state_dict'])
        mean=np.asarray(ck['feature_mean'],dtype=np.float32)
        std=np.asarray(ck['feature_std'],dtype=np.float32)
        z=Y.apply_scaler(x,mean,std)
        all_preds.append(Y.predict(net,z,torch.device('cpu'))[ix])
    return np.asarray(all_preds)


def scores(y, predictions):
    denom = np.sum((y - y.mean())**2)
    if denom < 1e-12:
        return float('nan'), float('nan')
    e = predictions-y[None,:]
    r2 = 1.0 - np.sum(e*e, axis=1)/denom
    rmse = np.sqrt(np.mean(e*e, axis=1))
    return float(r2.mean()), float(rmse.mean())


def main(args):
    if args.reps < 100:
        raise ValueError('Use at least 100 bootstrap resamples')
    data = {}
    for typ, folder, name in [
        ('81', 'yield_retrained_on_reconstruction', 'yield_specific_test_predictions.csv'),
        ('80', 'yield_retrained_spectral80', 'yield_spectral80_test_predictions.csv')]:
        d = load_predictions(BASE/folder, name)
        data[typ] = matrices(d)
    y81, plots81, _ = data['81']
    y80, plots80, _ = data['80']
    if not np.array_equal(plots81, plots80) or not np.allclose(y81, y80):
        raise ValueError('80-feature and 81-feature test examples do not align')
    measured = measured81_predictions(plots81,y81)
    groups = [np.where(plots81 == p)[0] for p in np.unique(plots81)]
    assert len(groups) == 50
    rng = np.random.default_rng(args.seed)
    records = []
    for b in range(args.reps+1):
        if b == 0:
            ix = np.arange(len(y81))
        else:
            ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        row = {'bootstrap': b}
        m_r2, m_rmse = scores(y81[ix],measured[:,ix])
        row['81_measured_R2'] = m_r2
        row['81_measured_RMSE'] = m_rmse
        for typ in ('81','80'):
            y, _, mat = data[typ]
            for method in METHODS:
                r2, rmse = scores(y[ix], mat[method][:,ix])
                row[f'{typ}_{method}_R2'] = r2
                row[f'{typ}_{method}_RMSE'] = rmse
            row[f'{typ}_gram_minus_simplemlp_R2'] = row[f'{typ}_gram_R2'] - row[f'{typ}_simplemlp_R2']
            row[f'{typ}_gram_minus_adjustedu2mdn_R2'] = row[f'{typ}_gram_R2'] - row[f'{typ}_adjustedu2mdn_R2']
            if typ == '81':
                row['81_gram_minus_measured_R2'] = row['81_gram_R2'] - row['81_measured_R2']
        for method in METHODS:
            row[f'{method}_81_minus_80_R2'] = row[f'81_{method}_R2'] - row[f'80_{method}_R2']
        records.append(row)
    df = pd.DataFrame(records)
    point = df.iloc[0]
    rep = df.iloc[1:]
    rows = []
    for col in df.columns.drop('bootstrap'):
        v = rep[col].dropna().to_numpy()
        if not len(v):
            continue
        lo, hi = np.quantile(v,[0.025,0.975])
        # Two-sided sign test approximation for a zero contrast: useful descriptively,
        # but no multiplicity adjustment and no extrapolation beyond this domain.
        p = 2*min(np.mean(v <= 0), np.mean(v >= 0)) if ('minus' in col) else np.nan
        rows.append({'metric':col,'point_estimate':point[col], 'CI_low':lo,'CI_high':hi,
                     'bootstrap_sign_p_unadjusted':min(1,p) if np.isfinite(p) else np.nan})
    out = BASE/'yield_retrained_on_reconstruction'/'paired_bootstrap_new'
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'paired_bootstrap_summary.csv',index=False)
    df.to_csv(out/'paired_bootstrap_replicates.csv',index=False)
    print('Saved',out)
    print(pd.DataFrame(rows).round(5).to_string(index=False))

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--reps',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=2026)
    main(ap.parse_args())