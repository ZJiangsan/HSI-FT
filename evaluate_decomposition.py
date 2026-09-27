from pathlib import Path
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import config as C

EPS = 1e-8
BANDS = 40

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11=nn.Linear(40,5,bias=False)
        self.conv12=nn.Linear(45,5,bias=False)
        self.conv13=nn.Linear(50,40,bias=False)
        self.relu=nn.ReLU()
    def forward(self,x):
        a=self.relu(self.conv11(x))
        s=torch.cat((x,a),1)
        b=self.relu(self.conv12(s))
        s=torch.cat((s,b),1)
        return self.relu(self.conv13(s))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv31=nn.Linear(40,40,bias=False)
    def forward(self,x): return self.conv31(x)

def stick(v):
    c=torch.cumprod(1-v,dim=1)
    prev=torch.cat([torch.ones_like(c[:,:1]),c[:,:-1]],1)
    return v*prev

def forward_chunk(enc,dec,x):
    v=enc(x).clamp(EPS,1e8)
    s=stick(v).clamp(EPS,1-EPS)
    return dec(s).clamp(-1+EPS,1-EPS)

def load_split(path):
    groups={'train':[],'val':[],'test':[]}
    with open(path,newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            split=row['split'].strip().lower()
            p=Path(row['path'])
            if not p.exists():
                # Allow moving the prepared dataset root after the split was made.
                p=C.MEASURED40_ROOTS[row['field']] / (str(row['plot'])+'.npy')
            if not p.exists(): raise FileNotFoundError(p)
            groups[split].append({'field':row['field'],'plot':row['plot'],'path':p})
    for split,n in [('train',875),('val',96),('test',50)]:
        if len(groups[split])!=n: raise RuntimeError(f'Expected {n} {split} plots, found {len(groups[split])}')
    return groups

def load_checkpoint(path,device):
    ck=torch.load(path,map_location=device)
    enc,dec=Encoder().to(device).float(),Decoder().to(device).float()
    enc.load_state_dict(ck['encoder_state_dict']);dec.load_state_dict(ck['decoder_state_dict'])
    enc.eval();dec.eval()
    mean=ck.get('global_mean40',ck.get('global_mean',ck.get('mean40')))
    if mean is None: raise KeyError('checkpoint lacks global_mean40/global_mean/mean40')
    return enc,dec,np.asarray(mean,np.float32).reshape(1,40)

def evaluate_plot(path,enc,dec,mean,device,chunk=131072):
    cube=np.load(path,mmap_mode='r')
    flat=cube.reshape(-1,40)
    idx=np.flatnonzero(np.any(flat!=0,axis=1))
    if len(idx)==0: raise RuntimeError(f'No valid pixels: {path}')
    sam_sum=sse=0.;nval=0
    mt=torch.from_numpy(mean).to(device)
    with torch.no_grad():
        for st in range(0,len(idx),chunk):
            raw=np.asarray(flat[idx[st:st+chunk]],np.float32)
            x=torch.from_numpy(raw-mean).to(device)
            rec=forward_chunk(enc,dec,x)
            true=x+mt;pred=rec+mt
            num=torch.sum(true*pred,1)
            den=torch.linalg.norm(true,dim=1)*torch.linalg.norm(pred,dim=1)+EPS
            sam=torch.rad2deg(torch.acos(torch.clamp(num/den,-1+EPS,1-EPS)))
            sam_sum+=float(sam.sum())
            err=pred-true;sse+=float(torch.sum(err*err));nval+=err.numel()
    return len(idx),sam_sum, sse,nval

def evaluate_split(name,records,enc,dec,mean,device):
    pix=sam=sse=0.;nval=0;rows=[]
    for i,r in enumerate(records,1):
        p,ss,se,n=evaluate_plot(r['path'],enc,dec,mean,device)
        pix+=p;sam+=ss;sse+=se;nval+=n
        rows.append({'split':name,'field':r['field'],'plot':r['plot'],'n_valid_pixels':p,
                     'sam_physical_deg':ss/p,'rmse_physical':math.sqrt(se/n)})
        if i==1 or i%25==0 or i==len(records):
            print(name,i,'/',len(records),'SAM',sam/pix,'RMSE',math.sqrt(sse/nval))
    return {'split':name,'n_plots':len(records),'n_valid_pixels':pix,
            'sam_physical_deg':sam/pix,'rmse_physical':math.sqrt(sse/nval)},rows

def save_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

device=torch.device(C.DEVICE if (not C.DEVICE.startswith('cuda') or torch.cuda.is_available()) else 'cpu')
groups=load_split(C.PLOT_SPLIT_CSV)
enc,dec,mean=load_checkpoint(C.DECOMPOSITION_CKPT,device)
summaries=[];per_plot=[]
for split in ('train','val','test'):
    s,r=evaluate_split(split,groups[split],enc,dec,mean,device);summaries.append(s);per_plot.extend(r)
out=C.DECOMPOSITION_CKPT.parent
save_csv(out/'decomposition_frozen_split_metrics.csv',summaries)
save_csv(out/'decomposition_frozen_split_per_plot_metrics.csv',per_plot)
for r in summaries:
    print('{:<5s} plots {:4d} pixels {:10,d} SAM {:.6f} RMSE {:.8f}'.format(
        r['split'],r['n_plots'],r['n_valid_pixels'],r['sam_physical_deg'],r['rmse_physical']))
