#!/usr/bin/env python3
"""HSI-FT measured-HSI denoising controls; does not modify existing data.

Train five fresh subplot yield MLPs for each (denoiser, input dimension).
Measured 40-band cubes are transformed in spectral space BEFORE subplot
mean/SD extraction, with the original SL mask and subplot definition.
Denoisers: raw, Savitzky-Golay 7/2, PCA trained only on training plots
with ranks 10 or 20. Data split: original 875/96/50 plots.

This tests whether simple smoothing/low-rank constraints can account for
Gram's apparent predictive gain. No ground-truth clean spectral cube exists;
results alone cannot establish the physical denoising mechanism.
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
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

ROOT=Path(os.environ.get('HSIWHEAT40_ROOT', '/home/nibio/HSIwheat_40'))
BASE=Path(os.environ.get('HSIFT_SUPPLEMENTARY_BASE',str(ROOT/'new_manuscript_method_transfer_5seeds')))
REPO=Path(__file__).resolve().parents[1]
OUT=BASE/'denoising_controls'
SPLITS=('train','val','test')
EXPECTED={'train':44966,'val':4973,'test':2565}


def select_cleaner(name, common, meta, maps):
    if name in ('raw','sg7'):
        return None
    rank=int(name.replace('pca',''))
    path=OUT/f'{name}_pca_fitted_train_only.npz'
    # Deterministic fit; reuse an already-created artifact when available.
    if path.exists():
        v=np.load(path)
        obj=PCA(n_components=rank)
        obj.components_=v['components']
        obj.mean_=v['mean']
        return obj
    train=meta.loc[meta.split.eq('train'), ['field','plot']].drop_duplicates()
    rng=np.random.default_rng(2026)
    samples=[]
    for j, row in enumerate(train.itertuples(index=False),1):
        field=str(row.field); plot=common.normalize_plot_id(row.plot)
        cube=np.load(maps[field][plot],mmap_mode='r')
        flat=np.asarray(cube.reshape(-1,40),dtype=np.float32)
        valid=np.flatnonzero(np.any(flat!=0,axis=1))
        if len(valid):
            idx=rng.choice(valid,size=min(128,len(valid)),replace=False)
            samples.append(flat[idx])
        if j%200==0: print('PCA training sample plots:',j,flush=True)
    X=np.concatenate(samples).astype(np.float32)
    model=PCA(n_components=rank, svd_solver='randomized', random_state=2026)
    model.fit(X)
    OUT.mkdir(parents=True,exist_ok=True)
    # write with an atomic rename: overlapping tasks may fit the same PCA.
    tmp=OUT/f'{name}_pca_{os.getpid()}.npz'
    np.savez(tmp,components=model.components_,mean=model.mean_)
    if not path.exists(): tmp.replace(path)
    else: tmp.unlink()
    print(f'PCA {rank}: training samples={len(X)} retained variance={model.explained_variance_ratio_.sum():.5f}',flush=True)
    return model


def transform(cube,name,model):
    if name=='raw':return np.asarray(cube,dtype=np.float32)
    flat=np.asarray(cube.reshape(-1,40),dtype=np.float32)
    mask=np.any(flat!=0,axis=1)
    out=np.zeros_like(flat,dtype=np.float32)
    X=flat[mask]
    if name=='sg7':
        out[mask]=savgol_filter(X,window_length=7,polyorder=2,axis=1,mode='interp').astype(np.float32)
    else:
        # Chunked inverse projection limits memory use for very large field cubes.
        valid_indices=np.flatnonzero(mask)
        for i in range(0,len(X),131072):
            z=model.transform(X[i:i+131072])
            out[valid_indices[i:i+131072]]=model.inverse_transform(z).astype(np.float32)
    return out.reshape(cube.shape)


def features_for_split(name, split, model, meta, maps, common, Y):
    d=OUT/name
    d.mkdir(parents=True,exist_ok=True)
    path=d/f'{split}_features81.npy'
    sub=meta.loc[meta.split.eq(split)].reset_index(drop=True)
    if len(sub)!=EXPECTED[split]:raise ValueError(f'Bad split {split}: {len(sub)}')
    if path.exists():
        X=np.load(path)
        if X.shape != (len(sub),81):raise ValueError(f'Bad cached features {path}: {X.shape}')
        print(f'Reusing {path}',flush=True)
        return X,sub
    X=np.empty((len(sub),81),dtype=np.float32)
    groups=sub.groupby(['field','plot'],sort=False)
    for j, ((field, plot), g) in enumerate(groups,1):
        field=str(field);plot=common.normalize_plot_id(plot)
        cube=np.load(maps[field][plot],mmap_mode='r')
        mask=np.load(common.frozen_mask_path(field,plot)).astype(bool,copy=False)
        X[g.index.to_numpy()]=Y._features_for_group(g,transform(cube,name,model),mask)
        if j==1 or j%100==0 or j==groups.ngroups:
            print(f'{name} {split} {j}/{groups.ngroups} plots',flush=True)
    if not np.isfinite(X).all():raise ValueError('Nonfinite denoising features')
    temp=path.with_name(path.stem+f'.{os.getpid()}.tmp.npy')
    np.save(temp,X);temp.replace(path)
    return X,sub


def run(args):
    os.environ['HSIWHEAT40_ROOT']=str(ROOT)
    sys.path.insert(0,str(REPO))
    import config as C
    import common
    import yield_transfer as Y
    # Ensure original metadata paths use user's current root.
    meta=common.load_subplot_meta_with_new_split().reset_index(drop=True)
    maps=common.measured_maps()
    model=select_cleaner(args.denoiser,common,meta,maps)
    X={}; y={}; metas={}
    for split in SPLITS:
        X[split],metas[split]=features_for_split(args.denoiser,split,model,meta,maps,common,Y)
        y[split]=metas[split].subplot_yield.to_numpy(np.float32)
    n=args.features
    Zraw={s:np.asarray(X[s][:,:n],dtype=np.float32) for s in SPLITS}
    mean,std=Y.fit_scaler(Zraw['train'],np.arange(len(Zraw['train'])))
    Z={s:Y.apply_scaler(Zraw[s],mean,std) for s in SPLITS}
    dest=OUT/args.denoiser/f'features{n}'
    dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'mean_train_only.npy',mean);np.save(dest/'std_train_only.npy',std)
    device=torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('Model input',n,'denoiser',args.denoiser,'device',device,flush=True)
    rows=[]; pfs=[]
    for seed in range(5):
        common.seed_everything(seed)
        net=Y.YieldMLP(n).to(device)
        ck=dest/f'yield_seed_{seed}_bestval.pth'
        if ck.exists():
            state=common.torch_load(ck,map_location=device)
            if int(state.get('input_dim',-1))!=n or state.get('denoiser')!=args.denoiser:
                raise ValueError('Existing checkpoint inconsistent with experiment')
            net.load_state_dict(state['model_state_dict'])
            best_epoch=int(state['best_epoch']);best_val=float(state['best_validation_RMSE'])
        else:
            opt=torch.optim.Adam(net.parameters(),lr=C.YIELD_LR)
            tr=Y.loader(Z['train'],y['train'],True)
            va=Y.loader(Z['val'],y['val'],False)
            best=float('inf'); best_state=None;best_epoch=-1
            for epoch in range(1,C.YIELD_EPOCHS+1):
                net.train()
                for xb,yb in tr:
                    xb=xb.to(device);yb=yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.mse_loss(net(xb),yb)
                    loss.backward();opt.step()
                vm=float(Y.loader_mse(net,va,device))
                if vm<best:
                    best=vm;best_epoch=epoch;best_state=copy.deepcopy(net.state_dict())
                if epoch%20==0:
                    print(f'{args.denoiser} features={n} seed={seed} ep={epoch} val_RMSE={math.sqrt(vm):.4f} best={math.sqrt(best):.4f}',flush=True)
            net.load_state_dict(best_state);best_val=math.sqrt(best)
            torch.save({'denoiser':args.denoiser,'input_dim':n,'yield_seed':seed,
                        'best_epoch':best_epoch,'best_validation_RMSE':best_val,
                        'feature_mean':mean,'feature_std':std,'model_state_dict':best_state},ck)
        pr=Y.predict(net,Z['test'],device)
        sub=Y.metrics(y['test'],pr);plot=Y.plot_metrics(metas['test'],y['test'],pr)
        rows.append({'denoiser':args.denoiser,'n_features':n,'yield_seed':seed,
                     'best_epoch':best_epoch,'subplot_R2':sub['R2'],'subplot_RMSE':sub['RMSE'],
                     'plot_R2':plot['R2'],'plot_RMSE':plot['RMSE']})
        pf=metas['test'][['field','plot','subplot_index']].copy()
        pf['yield_seed']=seed;pf['true_yield']=y['test'];pf['predicted_yield']=pr
        pfs.append(pf)
        print('RESULT',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(dest/'yield_results.csv',index=False)
    pd.concat(pfs,ignore_index=True).to_csv(dest/'test_predictions.csv',index=False)
    print('SAVED',dest,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--denoiser',choices=('raw','sg7','pca10','pca20'),required=True)
    p.add_argument('--features',choices=(80,81),type=int,required=True)
    p.add_argument('--cpu',action='store_true')
    run(p.parse_args())