from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from common import (
    abundance_from_centered,
    build_or_load_pixel_cache,
    key_of,
    load_frozen_decomposition,
    measured_maps,
    reconstruction_metrics,
    resolve_triplets,
    safe_name,
    save_json,
    seed_everything,
    split_rows,
    pixel_indices,
    streaming_train_mean_std,
    torch_load,
)


# =============================================================================
# Shared helpers
# =============================================================================

def _device():
    if C.DEVICE.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(C.DEVICE)


def _test_cube_rows(cube_table):
    return np.flatnonzero(cube_table['recon_split'].to_numpy() == 'test')


def _run_dir(method, triplet_name, seed):
    return C.RECON_OUT / safe_name(method) / safe_name(triplet_name) / 'seed_{}'.format(seed)


def _metrics_done(run_dir):
    return (run_dir / 'reconstruction_test_metrics.json').exists()


def _save_test_cube(output_root, field, plot, cube):
    d = Path(output_root) / str(field)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / '{}.npy'.format(plot), cube.astype(np.float32, copy=False))


def _global_test_metrics_from_saved(output_root, maps, cube_table):
    sq = abs_sum = sam_sum = 0.0
    nval = npix = 0
    for i in _test_cube_rows(cube_table):
        r = cube_table.iloc[int(i)]
        field, plot = str(r.field), str(r.plot)
        gt = np.load(maps[field][plot]).astype(np.float32, copy=False)
        pred = np.load(Path(output_root) / field / '{}.npy'.format(plot)).astype(np.float32, copy=False)
        valid = np.any(gt != 0, axis=2)
        g = gt[valid]; p = pred[valid]
        err = p.astype(np.float64) - g.astype(np.float64)
        sq += float(np.sum(err*err)); abs_sum += float(np.sum(np.abs(err))); nval += int(err.size)
        nom = np.sum(g.astype(np.float64)*p.astype(np.float64), axis=1)
        den = np.linalg.norm(g, axis=1) * np.linalg.norm(p, axis=1)
        ang = np.degrees(np.arccos(np.clip(nom/np.maximum(den,1e-12), -1, 1)))
        sam_sum += float(np.sum(ang)); npix += len(ang)
    return {
        'rmse': math.sqrt(sq/max(nval,1)),
        'mae': abs_sum/max(nval,1),
        'sam_deg': sam_sum/max(npix,1),
        'test_pixels': int(npix),
    }


# =============================================================================
# Simple MLP 3 -> 128 -> 128 -> 40
# =============================================================================
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,40),
        )
    def forward(self,x):
        return self.net(x)


def _simple_val_mse(model, pixel_matrix, val_idx, selected_idx, mean40, std40, device):
    model.eval(); sse=0.0; n=0
    im = mean40[list(selected_idx)]; istd = std40[list(selected_idx)]
    with torch.no_grad():
        for start in range(0,len(val_idx),C.INFERENCE_CHUNK):
            ii = val_idx[start:start+C.INFERENCE_CHUNK]
            y = np.asarray(pixel_matrix[ii], dtype=np.float32)
            x = (y[:,list(selected_idx)]-im.reshape(1,3))/istd.reshape(1,3)
            ys = (y-mean40.reshape(1,40))/std40.reshape(1,40)
            pr = model(torch.from_numpy(x).to(device))
            diff = pr - torch.from_numpy(ys).to(device)
            sse += float(torch.sum(diff*diff).item()); n += diff.numel()
    return sse/max(n,1)


def train_simple_mlp(pixel_matrix, train_idx, val_idx, selected_idx, mean40, std40, seed, run_dir, device):
    best_path = run_dir / 'best_model.pth'
    hist_path = run_dir / 'training_history.csv'
    if best_path.exists():
        ck = torch_load(best_path, map_location=device)
        m = SimpleMLP().to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
        return m, ck

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = SimpleMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.MLP_LR, weight_decay=C.MLP_WEIGHT_DECAY)
    crit = nn.MSELoss()
    im = mean40[list(selected_idx)]; istd = std40[list(selected_idx)]

    rng_val = np.random.RandomState(seed+999)
    if len(val_idx) > C.MLP_VAL_MAX_PIXELS:
        val_eval = val_idx[rng_val.choice(len(val_idx), C.MLP_VAL_MAX_PIXELS, replace=False)]
    else:
        val_eval = val_idx
    rng = np.random.RandomState(seed)
    best=float('inf'); best_epoch=-1; best_state=None; bad=0; history=[]

    for epoch in range(1,C.MLP_MAX_EPOCHS+1):
        model.train(); sse=0.0; n=0
        n_epoch = min(C.MLP_SAMPLES_PER_EPOCH, max(1,len(train_idx)))
        epoch_idx = train_idx[rng.randint(0,len(train_idx),size=n_epoch)]
        for start in range(0,n_epoch,C.MLP_BATCH_SIZE):
            ii=epoch_idx[start:start+C.MLP_BATCH_SIZE]
            y=np.asarray(pixel_matrix[ii],dtype=np.float32)
            x=(y[:,list(selected_idx)]-im.reshape(1,3))/istd.reshape(1,3)
            ys=(y-mean40.reshape(1,40))/std40.reshape(1,40)
            xb=torch.from_numpy(x).to(device); yb=torch.from_numpy(ys).to(device)
            opt.zero_grad(set_to_none=True); pr=model(xb); loss=crit(pr,yb); loss.backward(); opt.step()
            d=pr.detach()-yb; sse += float(torch.sum(d*d).item()); n += d.numel()
        train_mse=sse/max(n,1)
        val_mse=_simple_val_mse(model,pixel_matrix,val_eval,selected_idx,mean40,std40,device)
        improved=val_mse < best-C.MLP_MIN_DELTA
        if improved:
            best=float(val_mse); best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); bad=0
        else: bad += 1
        history.append({'epoch':epoch,'train_mse_scaled':train_mse,'validation_mse_scaled':val_mse,'best_validation_mse_scaled':best,'best_epoch':best_epoch,'bad_epochs':bad})
        if epoch==1 or epoch%10==0 or improved or bad>=C.MLP_PATIENCE:
            print('SimpleMLP seed {} epoch {:3d} train {:.8f} val {:.8f} best {:.8f}@{} patience {}/{}'.format(seed,epoch,train_mse,val_mse,best,best_epoch,bad,C.MLP_PATIENCE))
        if bad>=C.MLP_PATIENCE: break
    if best_state is None: raise RuntimeError('No SimpleMLP best state')
    model.load_state_dict(best_state); pd.DataFrame(history).to_csv(hist_path,index=False)
    payload={'method':'SimpleMLP','seed':seed,'selected_band_indices_40':list(map(int,selected_idx)),'input_mean':im.astype(np.float32),'input_std':istd.astype(np.float32),'output_mean40':mean40.astype(np.float32),'output_std40':std40.astype(np.float32),'best_epoch':best_epoch,'best_validation_mse_scaled':best,'model_state_dict':best_state}
    torch.save(payload,best_path)
    return model,payload


def simple_predict_pixels(model, measured_pixels, selected_idx, mean40, std40, device):
    im=mean40[list(selected_idx)]; istd=std40[list(selected_idx)]; out=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(measured_pixels),C.INFERENCE_CHUNK):
            y=np.asarray(measured_pixels[start:start+C.INFERENCE_CHUNK],dtype=np.float32)
            x=(y[:,list(selected_idx)]-im.reshape(1,3))/istd.reshape(1,3)
            ps=model(torch.from_numpy(x).to(device)).cpu().numpy()
            out.append((ps*std40.reshape(1,40)+mean40.reshape(1,40)).astype(np.float32))
    return np.concatenate(out) if out else np.empty((0,40),np.float32)


# =============================================================================
# Adjusted u2MDN: same-resolution 3-band -> 40-band Dirichlet reconstruction
# =============================================================================
class DenseDirichletEncoder(nn.Module):
    def __init__(self, input_dim=3, latent=C.U2_LATENT, base=C.U2_BASE_WIDTH):
        super().__init__()
        w=[base+i for i in range(5)]
        self.l1=nn.Linear(input_dim,w[0]); d1=input_dim+w[0]
        self.l2=nn.Linear(d1,w[1]); d2=d1+w[1]
        self.l3=nn.Linear(d2,w[2]); d3=d2+w[2]
        self.l4=nn.Linear(d3,w[3]); d4=d3+w[3]
        self.l5=nn.Linear(d4,w[4]); d5=d4+w[4]
        self.b1=nn.Linear(d2,w[2]); db1=d2+w[2]
        self.b2=nn.Linear(db1,w[3]); db2=db1+w[3]
        self.uniform=nn.Linear(d5,latent)
        self.beta=nn.Linear(db2,1)
        self.latent=latent
    def forward(self,x):
        a1=self.l1(x); s1=torch.cat([a1,x],1)
        a2=self.l2(s1); s2=torch.cat([a2,s1],1)
        a3=self.l3(s2); s3=torch.cat([a3,s2],1)
        a4=self.l4(s3); s4=torch.cat([a4,s3],1)
        a5=self.l5(s4); s5=torch.cat([a5,s4],1)
        b1=self.b1(s2); sb1=torch.cat([b1,s2],1)
        b2=self.b2(sb1); sb2=torch.cat([b2,sb1],1)
        uniform=torch.sigmoid(self.uniform(s5)).clamp(C.EPS,1-C.EPS)
        beta=F.softplus(self.beta(sb2))+C.EPS
        v=uniform.pow(1.0/beta)
        one_minus=1.0-v
        cumulative=torch.cumprod(one_minus,dim=1)
        previous=torch.cat([torch.ones_like(cumulative[:,:1]),cumulative[:,:-1]],dim=1)
        z=(v*previous).clamp(C.EPS,1-C.EPS)
        return z,uniform,beta


class U2Decoder(nn.Module):
    def __init__(self, latent=C.U2_LATENT):
        super().__init__(); self.d1=nn.Linear(latent,latent,bias=False); self.d2=nn.Linear(latent,40,bias=False)
    def forward(self,z): return self.d2(self.d1(z))
    def effective_matrix(self): return self.d1.weight.t().matmul(self.d2.weight.t())


class MICritic(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.f1=nn.Linear(dim,dim); self.f2=nn.Linear(dim,1,bias=False)
    def forward(self,x): return self.f2(self.f1(x)).reshape(-1)


class AdjustedU2MDN(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=DenseDirichletEncoder(); self.decoder=U2Decoder(); self.critic=MICritic(3+C.U2_LATENT)
    def forward(self,x):
        z,u,b=self.encoder(x); return self.decoder(z),z,u,b


def _u2_losses(model,x,y,selected_idx):
    pred,z,_,_=model(x)
    recon=F.mse_loss(pred,y)
    sparse_recon=F.mse_loss(pred[:,list(selected_idx)],x)
    eff=model.decoder.effective_matrix(); volume=torch.mean(torch.sum(torch.abs(eff),dim=1))
    p=z/(z.sum(1,keepdim=True)+C.EPS); entropy=-(p*torch.log(p+C.EPS)).sum(1).mean()
    perm=torch.randperm(len(x),device=x.device)
    pos=model.critic(torch.cat([x,z],1)); neg=model.critic(torch.cat([x[perm],z],1))
    mi=F.softplus(-pos).mean()+F.softplus(neg).mean()
    total=recon+C.U2_SPARSE_RECON_WEIGHT*sparse_recon+C.U2_VOLUME_WEIGHT*volume+C.U2_MI_WEIGHT*mi+C.U2_SPARSITY_WEIGHT*entropy
    return total,{'recon':recon,'sparse_recon':sparse_recon,'volume':volume,'mi':mi,'entropy':entropy}


def _u2_val_mse(model,pixel_matrix,val_idx,selected_idx,mean40,std40,device):
    model.eval(); sse=0.0;n=0; im=mean40[list(selected_idx)]; istd=std40[list(selected_idx)]
    with torch.no_grad():
        for st in range(0,len(val_idx),C.INFERENCE_CHUNK):
            ii=val_idx[st:st+C.INFERENCE_CHUNK]; y=np.asarray(pixel_matrix[ii],np.float32)
            x=(y[:,list(selected_idx)]-im.reshape(1,3))/istd.reshape(1,3); ys=(y-mean40.reshape(1,40))/std40.reshape(1,40)
            pr,_,_,_=model(torch.from_numpy(x).to(device)); d=pr-torch.from_numpy(ys).to(device); sse+=float(torch.sum(d*d).item());n+=d.numel()
    return sse/max(n,1)


def train_adjusted_u2mdn(pixel_matrix,train_idx,val_idx,selected_idx,mean40,std40,seed,run_dir,device):
    best_path=run_dir/'best_model.pth'; hist_path=run_dir/'training_history.csv'
    if best_path.exists():
        ck=torch_load(best_path,map_location=device); m=AdjustedU2MDN().to(device);m.load_state_dict(ck['model_state_dict']);m.eval();return m,ck
    run_dir.mkdir(parents=True,exist_ok=True);seed_everything(seed);model=AdjustedU2MDN().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=C.U2_LR,weight_decay=C.U2_WEIGHT_DECAY)
    im=mean40[list(selected_idx)];istd=std40[list(selected_idx)]
    rv=np.random.RandomState(seed+1999); val_eval=val_idx[rv.choice(len(val_idx),C.U2_VAL_MAX_PIXELS,replace=False)] if len(val_idx)>C.U2_VAL_MAX_PIXELS else val_idx
    rng=np.random.RandomState(seed);best=float('inf');best_epoch=-1;best_state=None;bad=0;history=[]
    for epoch in range(1,C.U2_MAX_EPOCHS+1):
        model.train(); totals={'loss':0.,'recon':0.,'mi':0.,'entropy':0.}; nb=0
        n_epoch=min(C.U2_SAMPLES_PER_EPOCH,max(1,len(train_idx))); eidx=train_idx[rng.randint(0,len(train_idx),size=n_epoch)]
        for st in range(0,n_epoch,C.U2_BATCH_SIZE):
            ii=eidx[st:st+C.U2_BATCH_SIZE]; y=np.asarray(pixel_matrix[ii],np.float32)
            x=(y[:,list(selected_idx)]-im.reshape(1,3))/istd.reshape(1,3); ys=(y-mean40.reshape(1,40))/std40.reshape(1,40)
            xb=torch.from_numpy(x).to(device); yb=torch.from_numpy(ys).to(device)
            opt.zero_grad(set_to_none=True);loss,parts=_u2_losses(model,xb,yb,selected_idx);loss.backward();opt.step();nb+=1
            totals['loss']+=float(loss.detach());totals['recon']+=float(parts['recon'].detach());totals['mi']+=float(parts['mi'].detach());totals['entropy']+=float(parts['entropy'].detach())
        val=_u2_val_mse(model,pixel_matrix,val_eval,selected_idx,mean40,std40,device); improved=val<best-C.U2_MIN_DELTA
        if improved: best=float(val);best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());bad=0
        else: bad+=1
        row={'epoch':epoch,'train_total_loss':totals['loss']/max(nb,1),'train_recon_mse':totals['recon']/max(nb,1),'train_mi_loss':totals['mi']/max(nb,1),'train_entropy':totals['entropy']/max(nb,1),'validation_mse_scaled':val,'best_validation_mse_scaled':best,'best_epoch':best_epoch,'bad_epochs':bad};history.append(row)
        if epoch==1 or epoch%10==0 or improved or bad>=C.U2_PATIENCE:
            print('AdjustedU2MDN seed {} epoch {:3d} total {:.6f} recon {:.6f} val {:.8f} best {:.8f}@{} patience {}/{}'.format(seed,epoch,row['train_total_loss'],row['train_recon_mse'],val,best,best_epoch,bad,C.U2_PATIENCE))
        if bad>=C.U2_PATIENCE:break
    if best_state is None:raise RuntimeError('No adjusted u2MDN best state')
    model.load_state_dict(best_state);pd.DataFrame(history).to_csv(hist_path,index=False)
    payload={'method':'AdjustedU2MDN','seed':seed,'selected_band_indices_40':list(map(int,selected_idx)),'input_mean':im.astype(np.float32),'input_std':istd.astype(np.float32),'output_mean40':mean40.astype(np.float32),'output_std40':std40.astype(np.float32),'best_epoch':best_epoch,'best_validation_mse_scaled':best,'model_state_dict':best_state,'adaptation':'same-resolution 3->40, dense encoder + stick-breaking Dirichlet latent + two-layer decoder','weights':{'volume':C.U2_VOLUME_WEIGHT,'mi':C.U2_MI_WEIGHT,'sparsity':C.U2_SPARSITY_WEIGHT,'sparse_reconstruction':C.U2_SPARSE_RECON_WEIGHT}}
    torch.save(payload,best_path);return model,payload


def u2_predict_pixels(model,measured_pixels,selected_idx,mean40,std40,device):
    im=mean40[list(selected_idx)];istd=std40[list(selected_idx)];out=[];model.eval()
    with torch.no_grad():
        for st in range(0,len(measured_pixels),C.INFERENCE_CHUNK):
            y=np.asarray(measured_pixels[st:st+C.INFERENCE_CHUNK],np.float32);x=(y[:,list(selected_idx)]-im.reshape(1,3))/istd.reshape(1,3)
            ps,_,_,_=model(torch.from_numpy(x).to(device)); pred=ps.cpu().numpy()*std40.reshape(1,40)+mean40.reshape(1,40);out.append(pred.astype(np.float32))
    return np.concatenate(out) if out else np.empty((0,40),np.float32)


# =============================================================================
# Gram mapper: exact original-style composite objective + frozen decomposition
# =============================================================================
class GramMapper(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(3,40,bias=False);self.register_buffer('repeat_counts',torch.tensor([13,13,14],dtype=torch.long))
    def forward(self,x): return self.linear(x)+torch.repeat_interleave(x,self.repeat_counts,dim=1)


def l21(a,b): return torch.sqrt(torch.sum((b-a)**2,dim=1)).sum()

def sam_loss(a,b):
    nom=torch.sum(a*b,dim=1);den=torch.sqrt(torch.sum(a*a,1))*torch.sqrt(torch.sum(b*b,1));cos=(nom/(den+C.EPS)).clamp(-1+C.EPS,1-C.EPS);return torch.rad2deg(torch.acos(cos)).mean()

def ergas(a,b):
    means=a.mean(0);mses=((a-b)**2).mean(0);return 100*torch.sqrt((mses/(means*means+C.EPS)).mean())

def gram_parts(ab):
    g=ab.t().mm(ab)/ab.shape[0];return g-g.mean(0,keepdim=True),torch.diag(g).unsqueeze(0)

def gram_forward(sparse_centered,mapper,enc,dec):
    mapped=mapper(sparse_centered).clamp(-1+C.EPS,1-C.EPS);ab=abundance_from_centered(mapped,enc);recon=dec(ab).clamp(-1+C.EPS,1-C.EPS);gc,gd=gram_parts(ab);return mapped,ab,recon,gc,gd

def gram_composite(mapped,recon,pgc,pgd,tgc,tgd):
    self_l=l21(mapped+C.EPS,recon);gl=l21(pgc,tgc);gs=sam_loss(pgc,tgc);gst=sam_loss(pgc.t(),tgc.t());ge=ergas(pgc,tgc);dl=l21(pgd,tgd);ds=sam_loss(pgd,tgd);multi=gl+ge+10*gs+gst;diag=dl+10*ds;total=.1*self_l+10*diag+100*multi;return total,{'gram_sam':gs}


def prepare_gram_targets(pixel_matrix,cube_table,enc,device):
    d=C.CACHE/'gram_targets_trainonly_decomposition';d.mkdir(parents=True,exist_ok=True); gc_path=d/'gram_c.npy';gd_path=d/'gram_d.npy';mean_path=d/'plot_mean40.npy'
    if gc_path.exists() and gd_path.exists() and mean_path.exists(): return np.load(mean_path,mmap_mode='r'),np.load(gc_path,mmap_mode='r'),np.load(gd_path,mmap_mode='r')
    n=len(cube_table);means=np.lib.format.open_memmap(mean_path,mode='w+',dtype=np.float32,shape=(n,40));gcs=np.lib.format.open_memmap(gc_path,mode='w+',dtype=np.float32,shape=(n,40,40));gds=np.lib.format.open_memmap(gd_path,mode='w+',dtype=np.float32,shape=(n,40))
    enc.eval()
    with torch.no_grad():
        for i,r in cube_table.iterrows():
            px=np.asarray(pixel_matrix[int(r.start):int(r.end)],np.float32);mu=px.mean(0,dtype=np.float64).astype(np.float32);means[i]=mu
            if str(r.recon_split)=='test':gcs[i]=0;gds[i]=0;continue
            x=torch.from_numpy(px-mu.reshape(1,40)).to(device);ab=abundance_from_centered(x,enc);gc,gd=gram_parts(ab);gcs[i]=gc.cpu().numpy();gds[i]=gd.squeeze(0).cpu().numpy()
            if i==0 or (i+1)%100==0 or i+1==n:print('Gram target {}/{}'.format(i+1,n))
    means.flush();gcs.flush();gds.flush();del means,gcs,gds
    return np.load(mean_path,mmap_mode='r'),np.load(gc_path,mmap_mode='r'),np.load(gd_path,mmap_mode='r')


def build_sparse_and_mean3(pixel_matrix,cube_table,selected_idx):
    sparse=np.asarray(pixel_matrix[:,list(selected_idx)],np.float32);m=np.zeros((len(cube_table),3),np.float32)
    for i,r in cube_table.iterrows():m[i]=sparse[int(r.start):int(r.end)].mean(0,dtype=np.float64)
    return sparse,m


def fit_plot_mean_ridge(plot_mean3,plot_mean40,cube_table,train_rows,val_rows):
    Xtr=plot_mean3[train_rows].astype(np.float64);Ytr=np.asarray(plot_mean40[train_rows],np.float64);Xv=plot_mean3[val_rows].astype(np.float64);Yv=np.asarray(plot_mean40[val_rows],np.float64)
    xm=Xtr.mean(0);ym=Ytr.mean(0);Xc=Xtr-xm;Yc=Ytr-ym;best=None
    for alpha in C.GRAM_MEAN_RIDGE_ALPHAS:
        A=Xc.T@Xc+float(alpha)*np.eye(3);W=np.linalg.pinv(A)@Xc.T@Yc;b=ym-xm@W;pred=Xv@W+b;rmse=float(np.sqrt(np.mean((pred-Yv)**2)))
        if best is None or rmse<best['val_rmse']:best={'alpha':float(alpha),'W':W,'b':b,'val_rmse':rmse}
    return best


def _eval_gram(mapper,enc,dec,sparse_gpu,mean3_gpu,gc_gpu,gd_gpu,cube_table,rows):
    mapper.eval();tot=gs=0.0
    with torch.no_grad():
        for ri in rows:
            i=int(ri);r=cube_table.iloc[i];x=sparse_gpu[int(r.start):int(r.end)]-mean3_gpu[i].reshape(1,3);mapped,ab,recon,pgc,pgd=gram_forward(x,mapper,enc,dec);loss,p=gram_composite(mapped,recon,pgc,pgd,gc_gpu[i],gd_gpu[i].unsqueeze(0));tot+=float(loss);gs+=float(p['gram_sam'])
    n=max(len(rows),1);return tot/n,gs/n


def train_gram(pixel_matrix,cube_table,train_rows,val_rows,selected_idx,seed,run_dir,enc,dec,target_gc,target_gd,device):
    best_path=run_dir/'best_model.pth';latest_path=run_dir/'latest_model.pth';hist_path=run_dir/'training_history.csv';run_dir.mkdir(parents=True,exist_ok=True)
    sparse_np,mean3_np=build_sparse_and_mean3(pixel_matrix,cube_table,selected_idx)
    sparse_gpu=torch.from_numpy(sparse_np).to(device);mean3_gpu=torch.from_numpy(mean3_np).to(device);gc_gpu=torch.from_numpy(np.asarray(target_gc,np.float32)).to(device);gd_gpu=torch.from_numpy(np.asarray(target_gd,np.float32)).to(device)
    seed_everything(seed);mapper=GramMapper().to(device);opt=torch.optim.Adam(mapper.parameters(),lr=C.GRAM_LR,weight_decay=C.GRAM_WEIGHT_DECAY)
    start=0;best=float('inf');best_epoch=-1;best_gs=float('inf');noimp=0
    if latest_path.exists():
        ck=torch_load(latest_path,map_location=device);mapper.load_state_dict(ck['model_state_dict']);opt.load_state_dict(ck['optimizer_state_dict']);start=int(ck['resume_epoch']);best=float(ck['best_validation_composite_loss']);best_epoch=int(ck['best_epoch']);best_gs=float(ck['best_validation_gram_sam']);noimp=int(ck.get('evals_without_improve',0));print('Resume Gram seed {} at {}'.format(seed,start))
    hist_exists=hist_path.exists() and start>0
    for epoch in range(start,C.GRAM_MAX_EPOCHS+1):
        t0=time.time();mapper.train();opt.zero_grad(set_to_none=True);tr=gs=0.;ntr=max(len(train_rows),1)
        for ri in train_rows:
            i=int(ri);r=cube_table.iloc[i];x=sparse_gpu[int(r.start):int(r.end)]-mean3_gpu[i].reshape(1,3);mapped,ab,recon,pgc,pgd=gram_forward(x,mapper,enc,dec);loss,p=gram_composite(mapped,recon,pgc,pgd,gc_gpu[i],gd_gpu[i].unsqueeze(0));(loss/ntr).backward();tr+=float(loss.detach());gs+=float(p['gram_sam'].detach())
        if C.GRAM_GRAD_CLIP>0:torch.nn.utils.clip_grad_norm_(mapper.parameters(),C.GRAM_GRAD_CLIP)
        do_val=epoch==start or epoch%C.GRAM_VAL_EVERY==0 or epoch==C.GRAM_MAX_EPOCHS
        vl=vgs=float('nan');improved=False
        if do_val:
            vl,vgs=_eval_gram(mapper,enc,dec,sparse_gpu,mean3_gpu,gc_gpu,gd_gpu,cube_table,val_rows)
            if vl<best:
                best=vl;best_epoch=epoch;best_gs=vgs;noimp=0;improved=True;torch.save({'method':'Gram','seed':seed,'epoch':epoch,'state_timing':'pre_step','selected_band_indices_40':list(map(int,selected_idx)),'best_validation_composite_loss':best,'best_validation_gram_sam':best_gs,'model_state_dict':copy.deepcopy(mapper.state_dict())},best_path)
            else:noimp+=1
        opt.step()
        latest={'resume_epoch':epoch+1,'best_epoch':best_epoch,'best_validation_composite_loss':best,'best_validation_gram_sam':best_gs,'evals_without_improve':noimp,'model_state_dict':mapper.state_dict(),'optimizer_state_dict':opt.state_dict()};torch.save(latest,latest_path)
        row={'epoch':epoch,'train_mean_composite_loss':tr/ntr,'train_mean_gram_sam':gs/ntr,'validation_mean_composite_loss':vl,'validation_mean_gram_sam':vgs,'best_validation_composite_loss':best,'best_epoch':best_epoch,'seconds':time.time()-t0}
        with open(hist_path,'a',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(row.keys()));
            if not hist_exists:w.writeheader();hist_exists=True
            w.writerow(row)
        if do_val and (epoch==start or epoch%10==0 or improved or epoch==C.GRAM_MAX_EPOCHS):print('Gram seed {} epoch {:5d} train {:.5f} val {:.5f} GramSAM {:.4f} best {:.5f}@{}{} {:.1f}s'.format(seed,epoch,tr/ntr,vl,vgs,best,best_epoch,' *' if improved else '',time.time()-t0))
        if C.GRAM_EARLY_STOP_PATIENCE>0 and noimp>=C.GRAM_EARLY_STOP_PATIENCE:break
    if not best_path.exists():raise RuntimeError('No Gram best checkpoint')
    ck=torch_load(best_path,map_location=device);mapper.load_state_dict(ck['model_state_dict']);mapper.eval();del sparse_gpu,mean3_gpu,gc_gpu,gd_gpu
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return mapper,mean3_np,ck


def gram_predict_centered(mapper,enc,dec,sparse_pixels,mean3,device):
    out=[];mapper.eval();enc.eval();dec.eval()
    with torch.no_grad():
        for st in range(0,len(sparse_pixels),C.INFERENCE_CHUNK):
            x=np.asarray(sparse_pixels[st:st+C.INFERENCE_CHUNK],np.float32)-mean3.reshape(1,3);_,_,rec,_,_=gram_forward(torch.from_numpy(x).to(device),mapper,enc,dec);out.append(rec.cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.empty((0,40),np.float32)


# =============================================================================
# Reconstruction output generation
# =============================================================================
def reconstruct_and_save_standard(method,model,selected_idx,mean40,std40,maps,cube_table,run_dir,device):
    root=run_dir/'reconstructed40';root.mkdir(parents=True,exist_ok=True)
    for ri in _test_cube_rows(cube_table):
        r=cube_table.iloc[int(ri)];field,plot=str(r.field),str(r.plot);cube=np.load(maps[field][plot]).astype(np.float32,copy=False);flat=cube.reshape(-1,40);valid=np.any(flat!=0,axis=1);px=flat[valid]
        pred=simple_predict_pixels(model,px,selected_idx,mean40,std40,device) if method=='SimpleMLP' else u2_predict_pixels(model,px,selected_idx,mean40,std40,device)
        out=np.zeros_like(flat,dtype=np.float32);out[valid]=pred;_save_test_cube(root,field,plot,out.reshape(cube.shape))
    metrics=_global_test_metrics_from_saved(root,maps,cube_table);save_json(metrics,run_dir/'reconstruction_test_metrics.json');return metrics


def reconstruct_and_save_gram(mapper,selected_idx,mean3_all,mean_model,maps,cube_table,run_dir,enc,dec,device):
    root=run_dir/'reconstructed40';refroot=run_dir/'reference_mean40';root.mkdir(parents=True,exist_ok=True)
    if C.SAVE_GRAM_REFERENCE_MEAN_DIAGNOSTIC:refroot.mkdir(parents=True,exist_ok=True)
    for ri in _test_cube_rows(cube_table):
        i=int(ri);r=cube_table.iloc[i];field,plot=str(r.field),str(r.plot);cube=np.load(maps[field][plot]).astype(np.float32,copy=False);flat=cube.reshape(-1,40);valid=np.any(flat!=0,axis=1);gt=flat[valid];sparse=gt[:,list(selected_idx)];m3=mean3_all[i]
        centered=gram_predict_centered(mapper,enc,dec,sparse,m3,device);pred_mean=(m3.astype(np.float64)@mean_model['W']+mean_model['b']).astype(np.float32);primary=centered+pred_mean.reshape(1,40)
        out=np.zeros_like(flat,dtype=np.float32);out[valid]=primary;_save_test_cube(root,field,plot,out.reshape(cube.shape))
        if C.SAVE_GRAM_REFERENCE_MEAN_DIAGNOSTIC:
            ref=centered+gt.mean(0,dtype=np.float64).astype(np.float32).reshape(1,40);rr=np.zeros_like(flat,dtype=np.float32);rr[valid]=ref;_save_test_cube(refroot,field,plot,rr.reshape(cube.shape))
    metrics=_global_test_metrics_from_saved(root,maps,cube_table);metrics['mean_restoration']='train/val-only ridge: observed 3-band plot mean -> 40-band plot mean';metrics['mean_ridge_alpha']=mean_model['alpha'];metrics['mean_ridge_val_rmse']=mean_model['val_rmse'];save_json(metrics,run_dir/'reconstruction_test_metrics.json')
    if C.SAVE_GRAM_REFERENCE_MEAN_DIAGNOSTIC:
        refm=_global_test_metrics_from_saved(refroot,maps,cube_table);refm['diagnostic_only']=True;refm['uses_measured_test_HSI_plot_mean']=True;save_json(refm,run_dir/'reference_mean_test_metrics.json')
    return metrics


# =============================================================================
# Public runner
# =============================================================================
def run_reconstruction():
    C.OUT.mkdir(parents=True,exist_ok=True);C.CACHE.mkdir(parents=True,exist_ok=True);C.RECON_OUT.mkdir(parents=True,exist_ok=True)
    device=_device();print('Reconstruction device =',device)
    pixel_matrix,cube_table,_,_=build_or_load_pixel_cache();train_rows=split_rows(cube_table,'train');val_rows=split_rows(cube_table,'val');train_idx=pixel_indices(cube_table,train_rows);val_idx=pixel_indices(cube_table,val_rows);mean40,std40=streaming_train_mean_std(pixel_matrix,cube_table,train_rows);maps=measured_maps();triplets=resolve_triplets()
    print('Frozen reconstruction split: train {} plots | val {} | test {}'.format(len(train_rows),len(val_rows),len(_test_cube_rows(cube_table))))
    print('Reconstruction seeds:',C.RECON_SEEDS)

    enc=dec=decomp_ck=None;plot_mean40=target_gc=target_gd=None
    if 'Gram' in C.ACTIVE_METHODS:
        enc,dec,decomp_ck,_=load_frozen_decomposition(device);plot_mean40,target_gc,target_gd=prepare_gram_targets(pixel_matrix,cube_table,enc,device)

    metric_rows=[]
    for triplet_name,selected_idx,actual in triplets:
        print('\n'+'#'*110);print('TRIPLET:',triplet_name,'indices',selected_idx,'wavelengths',actual);print('#'*110)
        sparse_for_mean=mean3_for_gram=None;mean_model=None
        if 'Gram' in C.ACTIVE_METHODS:
            sparse_for_mean,mean3_for_gram=build_sparse_and_mean3(pixel_matrix,cube_table,selected_idx);mean_model=fit_plot_mean_ridge(mean3_for_gram,plot_mean40,cube_table,train_rows,val_rows)
            print('Gram mean ridge alpha={} val RMSE={:.8f}'.format(mean_model['alpha'],mean_model['val_rmse']))

        for method in C.ACTIVE_METHODS:
            for seed in C.RECON_SEEDS:
                rd=_run_dir(method,triplet_name,seed);rd.mkdir(parents=True,exist_ok=True)
                if _metrics_done(rd):
                    print('SKIP completed:',method,triplet_name,'seed',seed);m=pd.read_json(rd/'reconstruction_test_metrics.json',typ='series').to_dict()
                else:
                    print('\nRUN',method,'|',triplet_name,'| seed',seed)
                    if method=='SimpleMLP':
                        model,_=train_simple_mlp(pixel_matrix,train_idx,val_idx,selected_idx,mean40,std40,seed,rd,device);m=reconstruct_and_save_standard(method,model,selected_idx,mean40,std40,maps,cube_table,rd,device)
                    elif method=='AdjustedU2MDN':
                        model,_=train_adjusted_u2mdn(pixel_matrix,train_idx,val_idx,selected_idx,mean40,std40,seed,rd,device);m=reconstruct_and_save_standard(method,model,selected_idx,mean40,std40,maps,cube_table,rd,device)
                    elif method=='Gram':
                        mapper,mean3_all,_=train_gram(pixel_matrix,cube_table,train_rows,val_rows,selected_idx,seed,rd,enc,dec,target_gc,target_gd,device);m=reconstruct_and_save_gram(mapper,selected_idx,mean3_all,mean_model,maps,cube_table,rd,enc,dec,device)
                    else: raise ValueError('Unknown method {}'.format(method))
                metric_rows.append({'method':method,'triplet':triplet_name,'recon_seed':seed,'wl1_nm':actual[0],'wl2_nm':actual[1],'wl3_nm':actual[2],**m})
                print('{} {} seed {} => SAM {:.4f} deg | RMSE {:.6f}'.format(method,triplet_name,seed,float(m['sam_deg']),float(m['rmse'])))

    out=pd.DataFrame(metric_rows);out.to_csv(C.OUT/'reconstruction_metrics.csv',index=False);print('\nSaved',C.OUT/'reconstruction_metrics.csv')
    return out
