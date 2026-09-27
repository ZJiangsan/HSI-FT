#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the leakage-controlled 40-band decomposition used by HSI-FT.

Compact reproduction of the study procedure:
  * deterministic plot split (seed 2026): 875 train / 96 val / 50 test;
  * decomposition fitted to TRAIN pixels only;
  * encoder 40->5, concat 45->5, concat 50->40, ReLU;
  * stick-breaking abundance representation;
  * bias-free Linear(40,40) decoder;
  * 10*L2,1 + 100*SID + 0.01*abundance entropy;
  * best checkpoint selected by centered training SAM before optimizer step.

The reported frozen checkpoint had its best state at epoch 12378, so this
script defaults to 12379 loop epochs (0..12378). Exact floating-point
reproducibility can depend on PyTorch/CUDA versions.
"""
import csv, json, math, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import config as C

EPS=1e-8
OUT=C.ROOT/"decomposition_train_only_seed2026"
SPLIT_SEED=2026
N_TRAIN,N_VAL,N_TEST=875,96,50
DECOMP_SEED=2026
EPOCHS=int(os.environ.get("HSIFT_DECOMP_EPOCHS","12379"))
LR0=float(os.environ.get("HSIFT_DECOMP_LR","1e-5"))
WEIGHT_DECAY=1e-6
LR_DECAY=0.98
LR_DECAY_EVERY=100
CHUNK=131072
RECON_COEF,SID_COEF,ENTROPY_COEF=10.0,100.0,0.01

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.conv11=nn.Linear(40,5,bias=False);self.conv12=nn.Linear(45,5,bias=False);self.conv13=nn.Linear(50,40,bias=False);self.relu=nn.ReLU()
    def forward(self,x):
        a=self.relu(self.conv11(x));s=torch.cat((x,a),1);b=self.relu(self.conv12(s));s=torch.cat((s,b),1);return self.relu(self.conv13(s))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.conv31=nn.Linear(40,40,bias=False)
    def forward(self,x):return self.conv31(x)

def stick(v):
    cp=torch.cumprod(1.0-v,dim=1);prev=torch.cat([torch.ones_like(cp[:,:1]),cp[:,:-1]],1);return v*prev

def entropy_loss(x):
    p=torch.abs(x);p=p/(p.sum(1,keepdim=True)+EPS);p=torch.clamp(p,min=EPS);return -(p*p.log()).sum(1).mean()

def l21(true,pred):return ((pred-true).pow(2).sum(1).sqrt()).sum()

def sid(true,pred):
    n=true.shape[0];a=true/(torch.linalg.norm(true,dim=1,keepdim=True)+EPS);b=pred/(torch.linalg.norm(pred,dim=1,keepdim=True)+EPS);a,b=a.clamp_min(EPS),b.clamp_min(EPS);return ((a-b)*(a.log()-b.log())).sum()/n

def sam_deg(true,pred):
    num=torch.sum(true*pred,dim=1);den=torch.linalg.norm(true,dim=1)*torch.linalg.norm(pred,dim=1)+EPS;return torch.rad2deg(torch.acos(torch.clamp(num/den,-1+EPS,1-EPS)))

def discover_cubes():
    records=[]
    for field in C.FIELDS:
        files=sorted(C.MEASURED40_ROOTS[field].rglob("*.npy"))
        if not files:raise RuntimeError(f"No HSI40 cubes under {C.MEASURED40_ROOTS[field]}")
        records.extend((field,p) for p in files)
    if len(records)!=1021:raise RuntimeError(f"Expected 1021 cubes, found {len(records)}")
    return sorted(records,key=lambda x:(str(x[0]),str(x[1])))

def make_or_load_split(records):
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/"reconstruction_plot_split.csv"
    if path.exists():
        rows=list(csv.DictReader(path.open("r",newline="",encoding="utf-8")));groups={"train":[],"val":[],"test":[]}
        if len(rows)!=1021:raise RuntimeError("Existing split does not contain 1021 plots")
        lookup={(str(f),str(Path(p).resolve())):(f,p) for f,p in records}
        for row in rows:
            key=(row["field"],str(Path(row["path"]).resolve()))
            if key in lookup:rec=lookup[key]
            else:
                matches=[(f,p) for f,p in records if str(f)==row["field"] and p.stem==row["plot"]]
                if len(matches)!=1:raise RuntimeError(f"Cannot resolve saved split row: {row}")
                rec=matches[0]
            groups[row["split"]].append(rec)
        return groups,path
    rng=np.random.default_rng(SPLIT_SEED);order=rng.permutation(len(records));shuffled=[records[int(i)] for i in order]
    groups={"train":shuffled[:N_TRAIN],"val":shuffled[N_TRAIN:N_TRAIN+N_VAL],"test":shuffled[N_TRAIN+N_VAL:]}
    label={(str(f),str(Path(p).resolve())):s for s,rs in groups.items() for f,p in rs}
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["split","split_seed","field","plot","path"]);w.writeheader()
        for field,p in records:w.writerow({"split":label[(str(field),str(Path(p).resolve()))],"split_seed":SPLIT_SEED,"field":field,"plot":p.stem,"path":str(Path(p).resolve())})
    return groups,path

def build_train_pixels(train_records):
    matrix_path=OUT/"train_valid_pixels_40band.npy";mean_path=OUT/"decomposition_train_global_mean.npy"
    if matrix_path.exists() and mean_path.exists():return np.load(matrix_path,mmap_mode="r"),np.load(mean_path).astype(np.float32)
    counts=[];total=0
    for field,p in train_records:
        cube=np.load(p,mmap_mode="r");flat=cube.reshape(-1,40);n=int(np.any(flat!=0,axis=1).sum());counts.append((field,p,n));total+=n
    arr=np.lib.format.open_memmap(matrix_path,mode="w+",dtype=np.float32,shape=(total,40));s=np.zeros(40,np.float64);pos=0
    for i,(field,p,n) in enumerate(counts,1):
        cube=np.load(p).astype(np.float32,copy=False);flat=cube.reshape(-1,40);px=flat[np.any(flat!=0,axis=1)];arr[pos:pos+len(px)]=px;s+=px.sum(0,dtype=np.float64);pos+=len(px)
        if i==1 or i%100==0 or i==len(counts):print(f"training pixel cache {i}/{len(counts)}")
    arr.flush();mean=(s/total).reshape(1,40).astype(np.float32);np.save(mean_path,mean);del arr
    return np.load(matrix_path,mmap_mode="r"),mean

def forward(enc,dec,x):
    v=enc(x).clamp(EPS,1e8);a=stick(v).clamp(EPS,1-EPS)
    with torch.no_grad():dec.conv31.weight.data.clamp_(-1+EPS,1-EPS)
    return a,dec(a).clamp(-1+EPS,1-EPS)

def one_epoch(enc,dec,opt_e,opt_d,pixels,mean_np,device):
    n_total=len(pixels);opt_e.zero_grad();opt_d.zero_grad();enc.train();dec.train();mean_t=torch.from_numpy(mean_np).to(device)
    sum_l21=sum_sid=sum_ent=0.0;sam_c_sum=sam_p_sum=sse=0.0;n_values=0
    for start in range(0,n_total,CHUNK):
        raw=np.asarray(pixels[start:start+CHUNK],np.float32);x=torch.from_numpy(raw-mean_np).to(device);abundance,rec=forward(enc,dec,x);ent=entropy_loss(abundance);ll=l21(x+EPS,rec);sl=sid(x+1.0+EPS,rec+1.0+EPS);weight=len(raw)/n_total
        loss=RECON_COEF*ll+SID_COEF*weight*sl+ENTROPY_COEF*weight*ent;loss.backward()
        with torch.no_grad():
            true_phys=x+mean_t;pred_phys=rec+mean_t;sum_l21+=float(ll);sum_sid+=float(sl)*weight;sum_ent+=float(ent)*weight;sam_c_sum+=float(sam_deg(x+EPS,rec).sum());sam_p_sum+=float(sam_deg(true_phys+EPS,pred_phys+EPS).sum());err=pred_phys-true_phys;sse+=float(torch.sum(err*err));n_values+=err.numel()
    return {"loss_recon_L21":sum_l21,"loss_sid":sum_sid,"entropy":sum_ent,"sam_centered_deg":sam_c_sum/n_total,"sam_physical_deg":sam_p_sum/n_total,"rmse_physical":math.sqrt(sse/n_values)}

def save_checkpoint(path,epoch,lr,enc,dec,best,mean,split_path):
    state={"epoch":int(epoch),"resume_epoch":int(epoch),"state_timing":"pre_step","lr":float(lr),"encoder_state_dict":enc.state_dict(),"decoder_state_dict":dec.state_dict(),"best_metrics":dict(best),"global_mean":mean,"global_mean40":mean,"settings":{"split_seed":SPLIT_SEED,"split_csv":str(split_path),"n_train":N_TRAIN,"n_val":N_VAL,"n_test":N_TEST,"decomposition_scope":"TRAIN cubes only","loss":"10*L21 + 100*SID + 0.01*entropy","seed":DECOMP_SEED,"epochs":EPOCHS}}
    tmp=Path(str(path)+".tmp");torch.save(state,tmp);os.replace(tmp,path)

OUT.mkdir(parents=True,exist_ok=True);random.seed(DECOMP_SEED);np.random.seed(DECOMP_SEED);torch.manual_seed(DECOMP_SEED)
if torch.cuda.is_available():torch.cuda.manual_seed_all(DECOMP_SEED)
device=torch.device(C.DEVICE if (not C.DEVICE.startswith("cuda") or torch.cuda.is_available()) else "cpu")
records=discover_cubes();groups,split_path=make_or_load_split(records);pixels,mean=build_train_pixels(groups["train"])
enc,dec=Encoder().to(device).float(),Decoder().to(device).float();lr=LR0;best={"sam_centered_deg":float("inf"),"epoch":-1};log_path=OUT/"training_log.csv"

for epoch in range(EPOCHS):
    opt_e=torch.optim.Adam(enc.parameters(),lr=lr,weight_decay=WEIGHT_DECAY);opt_d=torch.optim.Adam(dec.parameters(),lr=lr,weight_decay=WEIGHT_DECAY);m=one_epoch(enc,dec,opt_e,opt_d,pixels,mean,device)
    if m["sam_centered_deg"]<best["sam_centered_deg"]:
        best={**m,"epoch":epoch,"lr":lr};save_checkpoint(OUT/"best_decomposition_sam.pth",epoch,lr,enc,dec,best,mean,split_path);print("BEST",epoch,"SAMc",m["sam_centered_deg"],"SAMphys",m["sam_physical_deg"],"RMSE",m["rmse_physical"])
    exists=log_path.exists()
    with log_path.open("a",newline="",encoding="utf-8") as f:
        row={"epoch":epoch,"lr":lr,**m,"best_sam_centered_deg":best["sam_centered_deg"],"best_epoch":best["epoch"]};w=csv.DictWriter(f,fieldnames=list(row))
        if not exists:w.writeheader()
        w.writerow(row)
    opt_e.step();opt_d.step()
    if (epoch+1)%LR_DECAY_EVERY==0:lr*=LR_DECAY
    if epoch==0 or (epoch+1)%100==0 or epoch==EPOCHS-1:print(f"epoch {epoch:6d} SAMc {m['sam_centered_deg']:.6f} SAMphys {m['sam_physical_deg']:.6f} RMSE {m['rmse_physical']:.8f} best {best['sam_centered_deg']:.6f}@{best['epoch']}")

print("Finished. Best:",json.dumps(best,indent=2))
