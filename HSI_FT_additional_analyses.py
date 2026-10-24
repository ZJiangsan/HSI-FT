#!/usr/bin/env python3
"""Post-hoc HSI-FT analyses from existing frozen checkpoints and caches.

Run from the HSI-FT repository directory:
    python /path/to/HSI_FT_additional_analyses.py --repo . --bootstrap 2000

No reconstruction or yield model is trained. The script requires existing
outputs from final_transfer_diagnostics.py and all five yield checkpoints.
Plot bootstrap intervals quantify held-out test-plot sampling uncertainty,
conditional on the fitted model/reconstruction seeds; they are not independent
replications of model training.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',default='.',help='Path to cloned HSI-FT repo')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=20261024)
    args=ap.parse_args()
    sys.path.insert(0,str(Path(args.repo).resolve()))
    import torch
    import config as C
    from common import load_subplot_meta_with_new_split, safe_name
    from yield_transfer import YieldMLP, apply_scaler, predict, _device
    from sklearn.metrics import r2_score

    root=C.YIELD_OUT/'final_current_only_gram5'
    pred_path=root/'test_predictions_all.csv'
    if not pred_path.exists():
        raise FileNotFoundError(f'Missing {pred_path}; run original final_transfer_diagnostics.py first')
    meta=load_subplot_meta_with_new_split().reset_index(drop=True)
    test=meta.loc[meta.split.astype(str).str.lower()=='test'].reset_index(drop=True)
    required=['field','plot']
    if any(k not in test for k in required):raise ValueError('Test metadata must contain field and plot')
    grp=pd.factorize(test.field.astype(str)+'::'+test['plot'].astype(str))[0]
    nplot=len(np.unique(grp))
    if nplot!=50: print(f'WARNING: expected 50 plots, found {nplot}')
    p=pd.read_csv(pred_path)
    needed={'method','recon_seed','yield_seed','test_row','y_true','pred_measured','pred_reconstructed'}
    if not needed.issubset(p):raise ValueError(f'Prediction CSV lacks {needed-set(p.columns)}')
    if p.test_row.max()+1 != len(test):raise ValueError('Prediction rows do not match test metadata order')
    methods=['SimpleMLP','AdjustedU2MDN','Gram']
    rseeds=range(5);yseeds=range(5)
    yy=p[(p.method=='SimpleMLP')&(p.recon_seed==0)&(p.yield_seed==0)].sort_values('test_row').y_true.to_numpy()
    if len(yy)!=len(test):raise ValueError('Test length mismatch')
    arrays={}
    baseline={}
    for ys in yseeds:
        d=p[(p.method=='SimpleMLP')&(p.recon_seed==0)&(p.yield_seed==ys)].sort_values('test_row')
        baseline[ys]=d.pred_measured.to_numpy()
        if len(d)!=len(yy):raise ValueError('Missing measured predictions')
        for m in methods:
            for rs in rseeds:
                sub=p[(p.method==m)&(p.recon_seed==rs)&(p.yield_seed==ys)].sort_values('test_row')
                if len(sub)!=len(yy):raise ValueError(f'Missing prediction: {m}, {rs}, {ys}')
                if not np.allclose(sub.y_true.to_numpy(),yy,atol=1e-4):raise ValueError('Inconsistent test labels')
                arrays[(m,rs,ys)]=sub.pred_reconstructed.to_numpy()

    # Reuse identical plot resamples across all methods for paired contrasts.
    rng=np.random.default_rng(args.seed)
    plot_rows=[np.flatnonzero(grp==g) for g in np.unique(grp)]
    # Bootstrap CIs for method differences in transfer R² and ΔR².
    records=[]
    for b in range(args.bootstrap+1):
        if b==0: ind=np.arange(len(yy))
        else:
            sampled=rng.integers(0,len(plot_rows),len(plot_rows))
            ind=np.concatenate([plot_rows[j] for j in sampled])
        if np.var(yy[ind])<1e-12:continue
        base=np.mean([r2_score(yy[ind],baseline[ys][ind]) for ys in yseeds])
        row={'iteration':b,'measured_R2':base}
        for m in methods:
            r2=np.mean([r2_score(yy[ind],arrays[(m,rs,ys)][ind]) for rs in rseeds for ys in yseeds])
            row[m+'_R2']=r2
            row[m+'_delta_R2']=base-r2
        row['Gram_minus_Direct_R2']=row['Gram_R2']-row['SimpleMLP_R2']
        row['Gram_minus_MDN_R2']=row['Gram_R2']-row['AdjustedU2MDN_R2']
        records.append(row)
    boot=pd.DataFrame(records)
    boot.to_csv(root/'additional_plot_bootstrap_iterations.csv',index=False)
    sums=[]
    for col in boot.columns.drop('iteration'):
        vals=boot.loc[boot.iteration>0,col].to_numpy()
        sums.append({'metric':col,'point_estimate':float(boot.loc[boot.iteration==0,col].iloc[0]),
                     'CI95_low':np.percentile(vals,2.5),'CI95_high':np.percentile(vals,97.5),
                     'bootstrap_unit':'held-out plot','n_plots':nplot,'n_bootstrap':len(vals)})
    pd.DataFrame(sums).to_csv(root/'additional_plot_bootstrap_summary.csv',index=False)
    print('Saved paired plot-bootstrap estimates')

    # Mean/SD feature-group swaps. No checkpoint retraining is allowed.
    checks=[C.YIELD_OUT/'checkpoints_measured_hsi'/f'seed_{ys}_bestval.pth' for ys in yseeds]
    if not all(x.exists() for x in checks):
        raise FileNotFoundError('A yield checkpoint is missing; refusing to train. '+str([str(x) for x in checks if not x.exists()]))
    meas_path=C.CACHE/'measured40_subplot_features_newsplit.npy'
    mu_path=C.YIELD_OUT/'measured_feature_mean.npy';sd_path=C.YIELD_OUT/'measured_feature_std.npy'
    if not all(x.exists() for x in [meas_path,mu_path,sd_path]):
        raise FileNotFoundError('Original measured feature/scaler caches missing; refusing to rebuild')
    measured=np.load(meas_path)[meta.split.astype(str).str.lower().to_numpy()=='test']
    mu=np.load(mu_path);sd=np.load(sd_path)
    device=_device()
    models={}
    for ys,path in zip(yseeds,checks):
        try:ck=torch.load(path,map_location=device,weights_only=False)
        except TypeError:ck=torch.load(path,map_location=device)
        mdl=YieldMLP(81).to(device);mdl.load_state_dict(ck['model_state_dict']);mdl.eval();models[ys]=mdl
    swaps=[]
    for m in methods:
        for rs in rseeds:
            path=C.RECON_OUT/safe_name(m)/safe_name('Current')/f'seed_{rs}'/'yield_test_features.npy'
            if not path.exists():raise FileNotFoundError(f'Missing cached features {path}; refusing to reconstruct')
            recon=np.load(path)
            if recon.shape!=measured.shape:raise ValueError(f'Feature dimension mismatch: {path}')
            inputs={
                'all_reconstructed':recon,
                'reconstructed_means_only':np.column_stack((recon[:,:40],measured[:,40:])),
                'reconstructed_SDs_only':np.column_stack((measured[:,:40],recon[:,40:80],measured[:,80:])),
                'measured_reference':measured,
            }
            for variant,x in inputs.items():
                z=apply_scaler(x,mu,sd)
                for ys,model in models.items():
                    pr=predict(model,z,device)
                    swaps.append({'method':m,'recon_seed':rs,'yield_seed':ys,'variant':variant,
                                  'R2':r2_score(yy,pr),
                                  'RMSE':float(np.sqrt(np.mean((yy-pr)**2)))})
    sw=pd.DataFrame(swaps)
    sw.to_csv(root/'additional_feature_group_swap_all.csv',index=False)
    grouped=sw.groupby(['method','variant','recon_seed'],as_index=False).agg(R2=('R2','mean'),RMSE=('RMSE','mean'))
    grouped.groupby(['method','variant'],as_index=False).agg(R2_mean=('R2','mean'),R2_sd_recon=('R2','std'),RMSE_mean=('RMSE','mean')).to_csv(root/'additional_feature_group_swap_summary.csv',index=False)
    print('Saved mean/SD feature-group substitution analysis')
    print('All outputs:',root)

if __name__=='__main__':main()