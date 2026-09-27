from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

import config as C
from common import (
    frozen_mask_path,
    load_subplot_meta_with_new_split,
    measured_maps,
    normalize_plot_id,
    resolve_triplets,
    safe_name,
    seed_everything,
    torch_load,
)


def _device():
    if C.DEVICE.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(C.DEVICE)


def _recon_run_dir(method, triplet_name, recon_seed):
    return C.RECON_OUT / safe_name(method) / safe_name(triplet_name) / 'seed_{}'.format(recon_seed)


def _cube_map(root):
    files=sorted(Path(root).rglob('*.npy'));out={}
    for p in files:
        if p.stem in out: raise RuntimeError('Duplicate reconstructed cube stem {}'.format(p.stem))
        out[p.stem]=p
    return out


def _features_for_group(group,cube,sl_mask):
    out=np.zeros((len(group),81),np.float32)
    for j,(_,row) in enumerate(group.iterrows()):
        r0,r1,c0,c1=map(int,[row.row0,row.row1,row.col0,row.col1]);m=sl_mask[r0:r1,c0:c1];n=int(m.sum())
        if n!=int(row.n_SL_pixels):raise RuntimeError('SL count mismatch {}/{} subplot {}'.format(row.field,row.plot,row.subplot_index))
        px=cube[r0:r1,c0:c1,:][m]
        out[j]=np.concatenate([px.mean(0),px.std(0,ddof=0),np.array([float(n)],np.float32)]).astype(np.float32)
    return out


def build_measured_features(meta):
    C.CACHE.mkdir(parents=True,exist_ok=True);cache=C.CACHE/'measured40_subplot_features_newsplit.npy'
    if cache.exists():
        X=np.load(cache).astype(np.float32,copy=False)
        if X.shape==(len(meta),81):return X
    maps=measured_maps();X=np.zeros((len(meta),81),np.float32);groups=meta.groupby(['field','plot'],sort=False)
    for k,((field,plot),g) in enumerate(groups,start=1):
        field=str(field);plot=normalize_plot_id(plot);cube=np.load(maps[field][plot]).astype(np.float32,copy=False);mask=np.load(frozen_mask_path(field,plot)).astype(bool,copy=False);X[g.index.to_numpy()]=_features_for_group(g,cube,mask)
        if k==1 or k%100==0 or k==groups.ngroups:print('Measured yield features {}/{}'.format(k,groups.ngroups))
    np.save(cache,X);return X


def build_recon_test_features(meta,method,triplet,recon_seed,subdir='reconstructed40',cache_name='yield_test_features.npy'):
    rd=_recon_run_dir(method,triplet,recon_seed);cache=rd/cache_name
    if cache.exists():
        X=np.load(cache).astype(np.float32,copy=False)
        if X.shape==(int((meta.split=='test').sum()),81):return X
    root=rd/subdir
    if not root.exists():raise FileNotFoundError('Missing reconstructed test cubes: {}'.format(root))
    cmap=_cube_map(root);test=meta.loc[meta.split=='test'].copy().reset_index(drop=True);X=np.zeros((len(test),81),np.float32);groups=test.groupby(['field','plot'],sort=False)
    for k,((field,plot),g) in enumerate(groups,start=1):
        field=str(field);plot=normalize_plot_id(plot);p=cmap.get(plot)
        if p is None:raise FileNotFoundError('{} {} seed {} missing {}'.format(method,triplet,recon_seed,plot))
        cube=np.load(p).astype(np.float32,copy=False);mask=np.load(frozen_mask_path(field,plot)).astype(bool,copy=False);X[g.index.to_numpy()]=_features_for_group(g,cube,mask)
    np.save(cache,X);return X


class YieldMLP(nn.Module):
    def __init__(self,input_dim=81):
        super().__init__();layers=[];prev=input_dim
        for width in C.YIELD_HIDDEN:
            d=nn.Linear(prev,width);nn.init.xavier_uniform_(d.weight);nn.init.zeros_(d.bias);layers.extend([d,nn.ReLU()]);prev=width
        o=nn.Linear(prev,1);nn.init.xavier_uniform_(o.weight);nn.init.zeros_(o.bias);layers.append(o);self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x).reshape(-1)


def fit_scaler(X,idx):
    mean=X[idx].mean(0,dtype=np.float64);std=X[idx].std(0,dtype=np.float64,ddof=0);std[std<1e-12]=1;return mean.astype(np.float32),std.astype(np.float32)

def apply_scaler(X,mean,std):return ((X.astype(np.float64)-mean.astype(np.float64))/std.astype(np.float64)).astype(np.float32)

def loader(X,y,shuffle):return DataLoader(TensorDataset(torch.from_numpy(X).float(),torch.from_numpy(y).float()),batch_size=C.YIELD_BATCH_SIZE,shuffle=shuffle,drop_last=False)

def loader_mse(model,dl,device):
    model.eval();s=0.;n=0
    with torch.no_grad():
        for xb,yb in dl:
            xb=xb.to(device);yb=yb.to(device);p=model(xb);s+=float(torch.sum((p-yb)**2));n+=yb.numel()
    return s/max(n,1)

def predict(model,X,device):
    model.eval();out=[]
    with torch.no_grad():
        for st in range(0,len(X),4096):out.append(model(torch.from_numpy(X[st:st+4096]).float().to(device)).cpu().numpy())
    return np.concatenate(out)

def corr2(y,p):
    if len(y)<2 or np.std(y)==0 or np.std(p)==0:return np.nan
    r=np.corrcoef(y,p)[0,1];return float(r*r)

def metrics(y,p):
    rmse=float(math.sqrt(mean_squared_error(y,p)));return {'R2':float(r2_score(y,p)),'R2_corr2':corr2(y,p),'RMSE':rmse,'MAE':float(mean_absolute_error(y,p)),'NRMSE':float(rmse/np.mean(y))}

def plot_metrics(meta_test,y,p):
    d=meta_test.copy();d['true']=y;d['pred']=p;q=d.groupby(['field','plot','plot_yield'],as_index=False).agg(predicted=('pred','sum'));return metrics(q.plot_yield.to_numpy(float),q.predicted.to_numpy(float))


def train_or_load_yield_models(meta,X_measured,device):
    C.YIELD_OUT.mkdir(parents=True,exist_ok=True);ckdir=C.YIELD_OUT/'checkpoints_measured_hsi';ckdir.mkdir(parents=True,exist_ok=True)
    y=meta.subplot_yield.to_numpy(np.float32);train_idx=np.flatnonzero(meta.split.to_numpy()=='train');val_idx=np.flatnonzero(meta.split.to_numpy()=='val');test_idx=np.flatnonzero(meta.split.to_numpy()=='test')
    mean,std=fit_scaler(X_measured,train_idx);Z=apply_scaler(X_measured,mean,std);models={};baseline=[];preds={}
    for seed in C.YIELD_SEEDS:
        path=ckdir/'seed_{}_bestval.pth'.format(seed)
        if path.exists():
            ck=torch_load(path,map_location=device);model=YieldMLP(81).to(device);model.load_state_dict(ck['model_state_dict']);best_epoch=int(ck['best_epoch']);best_val=float(ck['best_validation_RMSE'])
        else:
            seed_everything(seed);model=YieldMLP(81).to(device);opt=torch.optim.Adam(model.parameters(),lr=C.YIELD_LR);crit=nn.MSELoss();tr=loader(Z[train_idx],y[train_idx],True);va=loader(Z[val_idx],y[val_idx],False);best=float('inf');best_epoch=-1;best_state=None;history=[]
            for ep in range(1,C.YIELD_EPOCHS+1):
                model.train();s=0.;n=0
                for xb,yb in tr:
                    xb=xb.to(device);yb=yb.to(device);opt.zero_grad(set_to_none=True);pr=model(xb);loss=crit(pr,yb);loss.backward();opt.step();s+=float(torch.sum((pr.detach()-yb)**2));n+=yb.numel()
                vm=loader_mse(model,va,device);tm=s/max(n,1)
                if vm<best:best=vm;best_epoch=ep;best_state=copy.deepcopy(model.state_dict())
                history.append({'epoch':ep,'train_RMSE':math.sqrt(tm),'validation_RMSE':math.sqrt(vm),'best_validation_RMSE':math.sqrt(best),'best_epoch':best_epoch})
                if ep==1 or ep%20==0 or ep==C.YIELD_EPOCHS:print('Yield seed {} epoch {:3d} train {:.4f} val {:.4f} best {:.4f}@{}'.format(seed,ep,math.sqrt(tm),math.sqrt(vm),math.sqrt(best),best_epoch))
            model.load_state_dict(best_state);best_val=math.sqrt(best);pd.DataFrame(history).to_csv(C.YIELD_OUT/'training_history_seed_{}.csv'.format(seed),index=False);torch.save({'seed':seed,'best_epoch':best_epoch,'best_validation_RMSE':best_val,'feature_mean':mean,'feature_std':std,'model_state_dict':best_state},path)
        model.eval();models[seed]=model;pr=predict(model,Z[test_idx],device);preds[seed]=pr;m=metrics(y[test_idx],pr);pm=plot_metrics(meta.iloc[test_idx].reset_index(drop=True),y[test_idx],pr);baseline.append({'yield_seed':seed,'best_epoch':best_epoch,'best_validation_RMSE':best_val,'subplot_R2':m['R2'],'subplot_RMSE':m['RMSE'],'subplot_MAE':m['MAE'],'plot_R2':pm['R2'],'plot_RMSE':pm['RMSE'],'plot_MAE':pm['MAE']});print('Measured-HSI yield seed {}: R2 {:.4f} RMSE {:.4f}'.format(seed,m['R2'],m['RMSE']))
    pd.DataFrame(baseline).to_csv(C.YIELD_OUT/'measured_hsi_baseline_5seeds.csv',index=False)
    np.save(C.YIELD_OUT/'measured_feature_mean.npy',mean);np.save(C.YIELD_OUT/'measured_feature_std.npy',std)
    return models,mean,std,preds,pd.DataFrame(baseline),train_idx,val_idx,test_idx,y,Z


def run_yield_transfer():
    device=_device();print('Yield-transfer device =',device);meta=load_subplot_meta_with_new_split();Xmeas=build_measured_features(meta);models,mean,std,meas_preds,base,train_idx,val_idx,test_idx,y,Zmeas=train_or_load_yield_models(meta,Xmeas,device);test_meta=meta.iloc[test_idx].reset_index(drop=True);ytest=y[test_idx]
    base_lookup=base.set_index('yield_seed').to_dict('index');rows=[]
    for triplet,_,actual in resolve_triplets():
        for method in C.ACTIVE_METHODS:
            for rseed in C.RECON_SEEDS:
                Xr=build_recon_test_features(meta,method,triplet,rseed);Zr=apply_scaler(Xr,mean,std)
                for yseed in C.YIELD_SEEDS:
                    pr=predict(models[yseed],Zr,device);m=metrics(ytest,pr);pm=plot_metrics(test_meta,ytest,pr);b=base_lookup[yseed];shift=float(math.sqrt(mean_squared_error(meas_preds[yseed],pr)));rows.append({'method':method,'triplet':triplet,'recon_seed':rseed,'yield_seed':yseed,'wl1_nm':actual[0],'wl2_nm':actual[1],'wl3_nm':actual[2],'measured_R2':b['subplot_R2'],'measured_RMSE':b['subplot_RMSE'],'recon_R2':m['R2'],'recon_RMSE':m['RMSE'],'recon_MAE':m['MAE'],'delta_R2':b['subplot_R2']-m['R2'],'delta_RMSE':m['RMSE']-b['subplot_RMSE'],'delta_MAE':m['MAE']-b['subplot_MAE'],'prediction_shift_RMSE':shift,'plot_R2':pm['R2'],'plot_RMSE':pm['RMSE']})
                print('Transfer done:',method,'|',triplet,'| recon seed',rseed)

    ref_rows=[]
    if 'Gram' in C.ACTIVE_METHODS and C.SAVE_GRAM_REFERENCE_MEAN_DIAGNOSTIC:
        for triplet,_,actual in resolve_triplets():
            for rseed in C.RECON_SEEDS:
                rd=_recon_run_dir('Gram',triplet,rseed)
                if not (rd/'reference_mean40').exists():
                    continue
                Xr=build_recon_test_features(meta,'Gram',triplet,rseed,subdir='reference_mean40',cache_name='yield_test_features_reference_mean.npy')
                Zr=apply_scaler(Xr,mean,std)
                for yseed in C.YIELD_SEEDS:
                    pr=predict(models[yseed],Zr,device);m=metrics(ytest,pr);b=base_lookup[yseed];shift=float(math.sqrt(mean_squared_error(meas_preds[yseed],pr)))
                    ref_rows.append({'method':'GramReferenceMean_DIAGNOSTIC','triplet':triplet,'recon_seed':rseed,'yield_seed':yseed,'wl1_nm':actual[0],'wl2_nm':actual[1],'wl3_nm':actual[2],'measured_R2':b['subplot_R2'],'measured_RMSE':b['subplot_RMSE'],'recon_R2':m['R2'],'recon_RMSE':m['RMSE'],'recon_MAE':m['MAE'],'delta_R2':b['subplot_R2']-m['R2'],'delta_RMSE':m['RMSE']-b['subplot_RMSE'],'delta_MAE':m['MAE']-b['subplot_MAE'],'prediction_shift_RMSE':shift,'uses_measured_test_hsi_mean':True})
        if ref_rows:
            pd.DataFrame(ref_rows).to_csv(C.YIELD_OUT/'gram_reference_mean_transfer_DIAGNOSTIC.csv',index=False)

    all_df=pd.DataFrame(rows);all_df.to_csv(C.YIELD_OUT/'frozen_transfer_all_25_combinations.csv',index=False)
    summary=all_df.groupby(['method','triplet'],as_index=False).agg(n=('delta_R2','size'),R2_mean=('recon_R2','mean'),R2_sd=('recon_R2','std'),RMSE_mean=('recon_RMSE','mean'),RMSE_sd=('recon_RMSE','std'),delta_R2_mean=('delta_R2','mean'),delta_R2_sd=('delta_R2','std'),delta_RMSE_mean=('delta_RMSE','mean'),delta_RMSE_sd=('delta_RMSE','std'),prediction_shift_RMSE_mean=('prediction_shift_RMSE','mean'))
    summary.to_csv(C.YIELD_OUT/'frozen_transfer_summary_25comb.csv',index=False)
    by_recon=all_df.groupby(['method','triplet','recon_seed'],as_index=False).agg(R2_mean5yield=('recon_R2','mean'),R2_sd5yield=('recon_R2','std'),delta_R2_mean5yield=('delta_R2','mean'),delta_RMSE_mean5yield=('delta_RMSE','mean'),prediction_shift_RMSE_mean5yield=('prediction_shift_RMSE','mean'))
    by_recon.to_csv(C.YIELD_OUT/'frozen_transfer_by_reconstruction_seed.csv',index=False)
    print('\nSaved transfer results under',C.YIELD_OUT);return all_df,summary
