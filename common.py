from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def safe_name(x):
    x = re.sub(r'[^A-Za-z0-9]+', '_', str(x)).strip('_')
    return x.lower()


def normalize_plot_id(x):
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def key_of(field, plot):
    return '{}::{}'.format(str(field), normalize_plot_id(plot))


def torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def map_npy_by_stem(root):
    root = Path(root)
    files = sorted(root.rglob('*.npy'))
    if not files:
        raise RuntimeError('No .npy files found under {}'.format(root))
    out = {}
    duplicates = []
    for p in files:
        if p.stem in out:
            duplicates.append(p.stem)
        else:
            out[p.stem] = p
    if duplicates:
        raise RuntimeError('Duplicate .npy stems under {}: {}'.format(root, duplicates[:20]))
    return out


def nearest_band_index(wavelengths, target_nm):
    return int(np.argmin(np.abs(np.asarray(wavelengths, dtype=np.float64) - float(target_nm))))


def resolve_triplets():
    wavelengths = np.load(C.WAVELENGTHS40_PATH).reshape(-1)
    if wavelengths.shape != (40,):
        raise ValueError('Expected wavelengths_40 shape (40,), got {}'.format(wavelengths.shape))
    rows = []
    wanted = None if C.ACTIVE_TRIPLETS == 'all' else set(C.ACTIVE_TRIPLETS)
    for name, nominal in C.TRIPLETS:
        if wanted is not None and name not in wanted:
            continue
        idx = tuple(nearest_band_index(wavelengths, x) for x in nominal)
        actual = tuple(float(wavelengths[i]) for i in idx)
        if len(set(idx)) != 3:
            raise RuntimeError('Triplet {} snapped to duplicate indices {}'.format(name, idx))
        rows.append((name, idx, actual))
    if not rows:
        raise RuntimeError('No active triplets.')
    return rows


def load_plot_split():
    df = pd.read_csv(C.PLOT_SPLIT_CSV)
    required = {'field', 'plot', 'split'}
    if not required.issubset(df.columns):
        raise ValueError('{} must contain {}'.format(C.PLOT_SPLIT_CSV, sorted(required)))
    df = df.copy()
    df['field'] = df['field'].astype(str)
    df['plot'] = df['plot'].map(normalize_plot_id)
    df['split'] = df['split'].astype(str).str.lower().str.strip().replace({
        'training': 'train', 'validation': 'val', 'valid': 'val', 'testing': 'test'
    })
    if df.duplicated(['field', 'plot']).any():
        raise RuntimeError('Duplicate field/plot entries in {}'.format(C.PLOT_SPLIT_CSV))
    counts = df['split'].value_counts().to_dict()
    expected = {'train': 875, 'val': 96, 'test': 50}
    for k, v in expected.items():
        if int(counts.get(k, 0)) != v:
            raise RuntimeError('Expected {} {} plots, found {}'.format(v, k, counts.get(k, 0)))
    return df


def load_subplot_meta_with_new_split():
    meta = pd.read_csv(C.SUBPLOT_META_CSV)
    required = {
        'field','plot','subplot_index','row0','row1','col0','col1',
        'n_SL_pixels','subplot_yield','plot_yield'
    }
    if not required.issubset(meta.columns):
        raise ValueError('subplot metadata missing {}'.format(sorted(required - set(meta.columns))))
    meta = meta.copy()
    meta['field'] = meta['field'].astype(str)
    meta['plot'] = meta['plot'].map(normalize_plot_id)
    if 'split' in meta.columns:
        meta = meta.drop(columns=['split'])
    split = load_plot_split()
    meta = meta.merge(split[['field','plot','split']], on=['field','plot'], how='left', validate='many_to_one')
    if meta['split'].isna().any():
        bad = meta.loc[meta['split'].isna(), ['field','plot']].drop_duplicates().head(20)
        raise RuntimeError('Some subplot plots missing from frozen plot split:\n{}'.format(bad))
    return meta.reset_index(drop=True)


def measured_maps():
    return {field: map_npy_by_stem(C.MEASURED40_ROOTS[field]) for field in C.FIELDS}


def _candidate_pair():
    for pix in C.PIXEL_CACHE_CANDIDATES:
        if not pix.exists():
            continue
        for tab in C.CUBE_TABLE_CANDIDATES:
            if tab.exists() and tab.parent == pix.parent:
                return pix, tab
    return None, None


def build_or_load_pixel_cache():
    C.CACHE.mkdir(parents=True, exist_ok=True)
    pix_path, table_path = _candidate_pair()
    if pix_path is None:
        pix_path = C.CACHE / 'all_valid_pixels_40.npy'
        table_path = C.CACHE / 'cube_ranges.csv'

    if pix_path.exists() and table_path.exists():
        pixel_matrix = np.load(pix_path, mmap_mode='r')
        table = pd.read_csv(table_path)
        print('Using pixel cache:', pix_path)
    else:
        maps = measured_maps()
        plot_split = load_plot_split()
        split_lookup = {key_of(r.field, r.plot): r.split for r in plot_split.itertuples(index=False)}
        records = []
        total = 0
        for field in C.FIELDS:
            for plot, p in sorted(maps[field].items()):
                cube = np.load(p, mmap_mode='r')
                if cube.ndim != 3 or cube.shape[-1] != 40:
                    raise ValueError('{} expected HxWx40, got {}'.format(p, cube.shape))
                flat = np.asarray(cube).reshape(-1, 40)
                n = int(np.any(flat != 0, axis=1).sum())
                split = split_lookup.get(key_of(field, plot))
                if split is None:
                    raise RuntimeError('No frozen split for {}/{}'.format(field, plot))
                records.append({'field': field, 'plot': plot, 'start': total, 'end': total+n, 'n_pixels': n, 'recon_split': split})
                total += n
        arr = np.lib.format.open_memmap(pix_path, mode='w+', dtype=np.float32, shape=(total, 40))
        cursor = 0
        for i, rec in enumerate(records):
            cube = np.load(maps[rec['field']][rec['plot']]).astype(np.float32, copy=False)
            flat = cube.reshape(-1, 40)
            valid = np.any(flat != 0, axis=1)
            px = flat[valid]
            arr[cursor:cursor+len(px)] = px
            cursor += len(px)
            if i == 0 or (i+1) % 100 == 0 or i+1 == len(records):
                print('cache {}/{} plots'.format(i+1, len(records)))
        arr.flush(); del arr
        pd.DataFrame(records).to_csv(table_path, index=False)
        pixel_matrix = np.load(pix_path, mmap_mode='r')
        table = pd.DataFrame(records)

    # Ignore any old split column and attach the exact current frozen plot split.
    table = table.copy()
    table['field'] = table['field'].astype(str)
    table['plot'] = table['plot'].map(normalize_plot_id)
    for c in ('recon_split','split'):
        if c in table.columns:
            table = table.drop(columns=[c])
    split = load_plot_split()
    table = table.merge(split[['field','plot','split']], on=['field','plot'], how='left', validate='one_to_one')
    table = table.rename(columns={'split':'recon_split'})
    if table['recon_split'].isna().any():
        raise RuntimeError('Pixel cube table does not match the frozen plot split.')
    if int(table.iloc[-1]['end']) != len(pixel_matrix):
        raise RuntimeError('Pixel cache/table length mismatch.')
    return pixel_matrix, table.reset_index(drop=True), pix_path, table_path


def split_rows(cube_table, split_name):
    return np.flatnonzero(cube_table['recon_split'].to_numpy() == split_name)


def pixel_indices(cube_table, row_indices):
    parts = [np.arange(int(cube_table.iloc[i]['start']), int(cube_table.iloc[i]['end']), dtype=np.int64) for i in row_indices]
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)


def streaming_train_mean_std(pixel_matrix, cube_table, train_rows):
    cache_mean = C.CACHE / 'train_mean40.npy'
    cache_std = C.CACHE / 'train_std40.npy'
    if cache_mean.exists() and cache_std.exists():
        mean = np.load(cache_mean).astype(np.float32)
        std = np.load(cache_std).astype(np.float32)
        if mean.shape == (40,) and std.shape == (40,):
            return mean, std
    s = np.zeros(40, dtype=np.float64)
    ss = np.zeros(40, dtype=np.float64)
    n = 0
    for i in train_rows:
        r = cube_table.iloc[int(i)]
        x = np.asarray(pixel_matrix[int(r.start):int(r.end)], dtype=np.float64)
        s += x.sum(0); ss += (x*x).sum(0); n += len(x)
    mean = s / max(n,1)
    var = ss / max(n,1) - mean*mean
    std = np.sqrt(np.maximum(var, 1e-12))
    np.save(cache_mean, mean.astype(np.float32)); np.save(cache_std, std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def reconstruction_metrics(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - gt
    rmse = math.sqrt(float(np.mean(err*err)))
    mae = float(np.mean(np.abs(err)))
    nom = np.sum(gt*pred, axis=1)
    den = np.linalg.norm(gt, axis=1) * np.linalg.norm(pred, axis=1)
    cos = np.clip(nom / np.maximum(den, 1e-12), -1.0, 1.0)
    sam = float(np.degrees(np.arccos(cos)).mean())
    return {'rmse': rmse, 'mae': mae, 'sam_deg': sam, 'pixels': int(len(gt))}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# Frozen decomposition architecture used by the Gram pipeline
# =============================================================================
class EncoderLRHSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Linear(40, 5, bias=False)
        self.conv12 = nn.Linear(45, 5, bias=False)
        self.conv13 = nn.Linear(50, 40, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        l11 = self.relu(self.conv11(x))
        s11 = torch.cat((x, l11), dim=1)
        l12 = self.relu(self.conv12(s11))
        s12 = torch.cat((s11, l12), dim=1)
        return self.relu(self.conv13(s12))


class DecoderHSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv31 = nn.Linear(40, 40, bias=False)
    def forward(self, x):
        return self.conv31(x)


def stick_segments(v):
    one_minus = 1.0 - v
    cumulative = torch.cumprod(one_minus, dim=1)
    previous = torch.cat([torch.ones_like(cumulative[:, :1]), cumulative[:, :-1]], dim=1)
    return v * previous


def load_frozen_decomposition(device):
    ckpt = torch_load(C.DECOMPOSITION_CKPT, map_location=device)
    enc = EncoderLRHSI().to(device).float()
    dec = DecoderHSI().to(device).float()
    enc.load_state_dict(ckpt['encoder_state_dict'])
    dec.load_state_dict(ckpt['decoder_state_dict'])
    for p in enc.parameters(): p.requires_grad_(False)
    for p in dec.parameters(): p.requires_grad_(False)
    enc.eval(); dec.eval()
    mean40 = ckpt.get('global_mean40', ckpt.get('global_mean', ckpt.get('mean40', None)))
    if mean40 is None:
        raise KeyError('Frozen decomposition checkpoint must contain global_mean40/global_mean/mean40.')
    mean40 = np.asarray(mean40, dtype=np.float32).reshape(-1)
    if mean40.shape != (40,):
        raise ValueError('Frozen decomposition mean shape is {}'.format(mean40.shape))
    return enc, dec, ckpt, mean40


def abundance_from_centered(x, encoder):
    v = encoder(x).clamp(C.EPS, 1.0-C.EPS)
    return stick_segments(v).clamp(C.EPS, 1.0-C.EPS)


def frozen_mask_path(field, plot):
    return C.SL_MASK_ROOT / str(field) / '{}_sl_mask.npy'.format(normalize_plot_id(plot))
