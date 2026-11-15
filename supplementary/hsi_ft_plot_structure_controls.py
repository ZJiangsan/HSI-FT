#!/usr/bin/env python3
"""HSI-FT matched measured-HSI plot-structure controls (diagnostic).

This is NOT a replacement for Gram. Test whether its plot-mean processing and
within-plot compression, applied to genuine HSI, change downstream prediction.
Modes:
  ridge_mean : original spectra centered by genuine 40-band plot mean,
               restored by original training/validation-fitted Gram ridge
               estimate from three observed bands (3,12,36). Note: uses
               genuine plot mean for centering, so diagnostic only.
  shrink05   : retain genuine plot mean but halve pixel spectral deviations;
               descriptive within-plot smoothing, not physically calibrated.

For either condition, transform measured cubes consistently for all splits,
extract the same 81 subplot features, train 5 identical yield MLP seeds on
80 or 81 dimensions, and save predictions. No prior data overwritten.
"""
import argparse,copy,math,os,sys
from pathlib import Path
import numpy as np,pandas as pd,torch

ROOT=Path(os.environ.get('HSIWHEAT40_ROOT', '/home/nibio/HSIwheat_40'))
BASE=Path(os.environ.get('HSIFT_SUPPLEMENTARY_BASE',str(ROOT/'new_manuscript_method_transfer_5seeds')))
REPO=Path(__file__).resolve().parents[1]
OUT=BASE/'plot_structure_controls'
EXPECTED={'train':44966,'val':4973,'test':2565}


def cube_transform(cube,mode,ridge):
    if cube.ndim != 3 or cube.shape[-1]!=40:raise ValueError(f'Bad cube shape {cube.shape}')
    flat=np.asarray(cube.reshape(-1,40),dtype=np.float32)
    valid=np.any(flat!=0,axis=1)
    X=flat[valid]
    if len(X)==0:raise ValueError('Empty cube')
    mu=X.mean(0,dtype=np.float64)
    if mode=='ridge_mean':
        if ridge is None:raise RuntimeError('No fitted ridge restoration')
        m3=X[:,[3,12,36]].mean(0,dtype=np.float64)
        replacement=(m3@ridge['W']+ridge['b']).reshape(1,40)
        clean=X.astype(np.float64)-mu.reshape(1,40)+replacement
    elif mode=='shrink05':
        clean=0.5*X.astype(np.float64)+0.5*mu.reshape(1,40)
    else:raise ValueError(mode)
    out=np.zeros_like(flat,dtype=np.float32)
    out[valid]=clean.astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('Nonfinite transformed cube')
    return out.reshape(cube.shape)


def load_split_features(mode,split,ridge,meta,common,Y,maps):
    target=OUT/mode/f'{split}_features81.npy'
    sub=meta.loc[meta.split.eq(split)].reset_index(drop=True)
    if len(sub)!=EXPECTED[split]:raise ValueError(f'Wrong {split} size: {len(sub)}')
    if target.exists():
        X=np.load(target)
        if X.shape!=(len(sub),81):raise ValueError('Bad cached feature matrix')
        print('Reused',target,flush=True)
        return X,sub
    X=np.zeros((len(sub),81),dtype=np.float32)
    groups=sub.groupby(['field','plot'],sort=False)
    for count,((field,plot),g) in enumerate(groups,1):
        field=str(field);plot=common.normalize_plot_id(plot)
        cube=np.load(maps[field][plot],mmap_mode='r')
        mask=np.load(common.frozen_mask_path(field,plot)).astype(bool,copy=False)
        out=cube_transform(cube,mode,ridge)
        X[g.index.to_numpy()]=Y._features_for_group(g,out,mask)
        if count==1 or count%100==0 or count==groups.ngroups:
            print(f'{mode} {split} {count}/{groups.ngroups}',flush=True)
    if not np.isfinite(X).all():raise ValueError('Nonfinite features')
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(f'{target.stem}.{os.getpid()}.tmp.npy')
    np.save(tmp,X);tmp.replace(target)
    return X,sub


def run(args):
    os.environ['HSIWHEAT40_ROOT']=str(ROOT)
    sys.path.insert(0,str(REPO))
    import config as C,common,yield_transfer as Y
    C.OUT=BASE
    C.CACHE=BASE/'cache'
    C.RECON_OUT=BASE/'reconstruction'
    device=torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    meta=common.load_subplot_meta_with_new_split().reset_index(drop=True)
    maps=common.measured_maps()
    ridge=None
    if args.condition=='ridge_mean' and any(not (OUT/args.condition/f'{split}_features81.npy').exists() for split in EXPECTED):
        # Reuse the EXACT train/validation-only restoration fitted in Gram's
        # original code rather than inventing a different ridge implementation.
        # This intentionally loads original preprocessing and decomposition.
        from reconstruct import (prepare_gram_targets,build_sparse_and_mean3,fit_plot_mean_ridge)
        pixel,cubetable,_,_=common.build_or_load_pixel_cache()
        train_rows=common.split_rows(cubetable,'train')
        val_rows=common.split_rows(cubetable,'val')
        enc,dec,_,_=common.load_frozen_decomposition(device)
        plotmeans,_,_=prepare_gram_targets(pixel,cubetable,enc,device)
        _,mean3=build_sparse_and_mean3(pixel,cubetable,(3,12,36))
        ridge=fit_plot_mean_ridge(mean3,plotmeans,cubetable,train_rows,val_rows)
        print('Original Gram ridge alpha:',ridge['alpha'],flush=True)
    X={}; sub={}; y={}
    for s in ('train','val','test'):
        X[s],sub[s]=load_split_features(args.condition,s,ridge,meta,common,Y,maps)
        y[s]=sub[s].subplot_yield.to_numpy(np.float32)
    n=args.features
    mu,sd=Y.fit_scaler(X['train'][:,:n],np.arange(len(X['train'])))
    Z={s:Y.apply_scaler(X[s][:,:n],mu,sd) for s in X}
    dest=OUT/args.condition/f'features{n}'
    dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'feature_mean_train_only.npy',mu);np.save(dest/'feature_std_train_only.npy',sd)
    rows=[];pframes=[]
    for seed in range(5):
        common.seed_everything(seed)
        net=Y.YieldMLP(n).to(device)
        ck=dest/f'yield_seed_{seed}_bestval.pth'
        if ck.exists():
            data=common.torch_load(ck,map_location=device)
            if data.get('condition')!=args.condition or data.get('n_features')!=n:
                raise ValueError(f'Checkpoint metadata mismatch: {ck}')
            net.load_state_dict(data['model_state_dict'])
            best_epoch=int(data['best_epoch']);best_val=float(data['best_validation_RMSE'])
        else:
            opt=torch.optim.Adam(net.parameters(),lr=C.YIELD_LR)
            tr=Y.loader(Z['train'],y['train'],True)
            va=Y.loader(Z['val'],y['val'],False)
            best=float('inf');best_state=None;best_epoch=-1
            for epoch in range(1,C.YIELD_EPOCHS+1):
                net.train()
                for xb,yb in tr:
                    xb=xb.to(device);yb=yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.mse_loss(net(xb),yb)
                    loss.backward();opt.step()
                vm=Y.loader_mse(net,va,device)
                if vm<best:
                    best=vm;best_epoch=epoch;best_state=copy.deepcopy(net.state_dict())
                if epoch==1 or epoch%20==0 or epoch==C.YIELD_EPOCHS:
                    print(f'{args.condition} {n} features seed={seed} epoch={epoch} '
                        f'val_RMSE={math.sqrt(vm):.4f} best={math.sqrt(best):.4f}',flush=True)
            net.load_state_dict(best_state);best_val=math.sqrt(best)
            torch.save({'condition':args.condition,'n_features':n,'yield_seed':seed,
                'best_epoch':best_epoch,'best_validation_RMSE':best_val,
                'model_state_dict':best_state},ck)
        p=Y.predict(net,Z['test'],device)
        ms=Y.metrics(y['test'],p);mp=Y.plot_metrics(sub['test'],y['test'],p)
        rows.append({'condition':args.condition,'n_features':n,'yield_seed':seed,
            'best_epoch':best_epoch,'best_val_RMSE':best_val,
            'subplot_R2':ms['R2'],'subplot_RMSE':ms['RMSE'],
            'plot_R2':mp['R2'],'plot_RMSE':mp['RMSE']})
        pf=sub['test'][['field','plot','subplot_index']].copy()
        pf['yield_seed']=seed;pf['true_yield']=y['test'];pf['predicted_yield']=p
        pframes.append(pf)
        print('FINISHED:',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(dest/'yield_results.csv',index=False)
    pd.concat(pframes,ignore_index=True).to_csv(dest/'test_predictions.csv',index=False)
    print('SAVED',dest,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--condition',choices=('ridge_mean','shrink05'),required=True)
    p.add_argument('--features',type=int,choices=(80,81),required=True)
    p.add_argument('--cpu',action='store_true')
    run(p.parse_args())