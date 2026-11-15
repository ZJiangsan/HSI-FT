#!/usr/bin/env python3
"""Create a reproducible original-paper-style, field/yield-stratified HSIwheat split.

Reads existing plot manifest and subplot metadata (no images, no training).
Outputs a new manifest and audit reports without altering existing experiments.
"""
from pathlib import Path
import os
import argparse
import json
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(os.environ.get('HSIWHEAT40_ROOT', '/home/nibio/HSIwheat_40'))
TEST_COUNTS = {'C3': 20, 'C4': 10, 'C9': 20}


def sample_by_yield_quantile(group, n, rng, bins=5):
    """Sample equal fractions from ranked within-field yield quintiles.

    Rank yields with plot-ID deterministic tie-breaking; split into exactly
    `bins` strata; proportional allocation by largest remainder.
    """
    group = group.sort_values(['plot_yield', 'plot'], kind='mergesort').copy()
    if n > len(group):
        raise ValueError(f'Requested {n} of {len(group)} plots')
    # Array splitting avoids qcut failures on repeated yield values.
    positions = np.array_split(np.arange(len(group)), bins)
    sizes = np.array([len(p) for p in positions], dtype=int)
    quotas = n * sizes / len(group)
    alloc = np.floor(quotas).astype(int)
    remaining = n - alloc.sum()
    remainder_order = np.argsort(-(quotas - alloc), kind='mergesort')
    for i in remainder_order[:remaining]:
        alloc[i] += 1
    selected = []
    for p, count in zip(positions, alloc):
        if count:
            selected.extend(rng.choice(p, size=int(count), replace=False).tolist())
    if len(selected) != n:
        raise AssertionError('Sampling size mismatch')
    return set(group.iloc[selected]['plotkey'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--validation-seed', type=int, default=2027)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()

    root = args.root
    old_csv = root/'decomposition_train_only_seed2026'/'reconstruction_plot_split.csv'
    meta_csv = root/'final_fourway_subplots_strict_sparsehsr'/'frozen_subplot_metadata_and_split.csv'
    out = args.output or root/f'original_style_yield_stratified_seed{args.seed}'
    if not old_csv.is_file() or not meta_csv.is_file():
        raise FileNotFoundError(f'Missing source file: {old_csv} or {meta_csv}')
    out.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(old_csv)
    meta = pd.read_csv(meta_csv)
    assert {'split','field','plot','path'}.issubset(old.columns)
    assert {'field','plot','plot_yield'}.issubset(meta.columns)
    if old.duplicated(['field','plot']).any():
        raise ValueError('Duplicate (field, plot) in original manifest')
    # Ensure every plot has a single truly measured whole-plot target.
    byplot = meta.groupby(['field','plot'], as_index=False).agg(
        plot_yield=('plot_yield','first'),
        n_unique_yields=('plot_yield','nunique'),
        n_subplots=('subplot_index','size'),
    )
    if (byplot.n_unique_yields != 1).any() or byplot.plot_yield.isna().any():
        raise ValueError('Inconsistent or missing measured plot yield')
    byplot = byplot.drop(columns='n_unique_yields')
    manifest = old.merge(byplot, on=['field','plot'], how='left', validate='one_to_one', indicator=True)
    if (manifest['_merge'] != 'both').any() or manifest.plot_yield.isna().any():
        bad = manifest.loc[(manifest['_merge'] != 'both')|manifest.plot_yield.isna(), ['field','plot']]
        raise ValueError(f'Missing plot metadata:\n{bad.head()}')
    manifest = manifest.drop(columns='_merge')
    manifest['plotkey'] = manifest['field'].astype(str)+'|'+manifest['plot'].astype(str)
    if len(manifest)!=1021:
        raise ValueError(f'Expected 1021 plots; found {len(manifest)}')
    if set(manifest.field) != set(TEST_COUNTS):
        raise ValueError(f'Unexpected fields: {sorted(manifest.field.unique())}')

    rng = np.random.default_rng(args.seed)
    tests = set()
    for field, count in TEST_COUNTS.items():
        tests |= sample_by_yield_quantile(manifest.loc[manifest.field == field], count, rng)
    manifest['new_split'] = 'train'
    manifest.loc[manifest.plotkey.isin(tests), 'new_split'] = 'test'
    if (manifest.new_split=='test').sum()!=50:
        raise AssertionError('Test count is not 50')

    # Keep 96 validation plots, sampled from the REMAINING plots only, and
    # stratified by field and yield so each field remains represented.
    remaining = manifest.loc[manifest.new_split=='train']
    quotas = 96 * remaining.field.value_counts().reindex(TEST_COUNTS.keys()).to_numpy() / len(remaining)
    alloc = np.floor(quotas).astype(int)
    for i in np.argsort(-(quotas-alloc), kind='mergesort')[:96-alloc.sum()]:
        alloc[i]+=1
    rng_val = np.random.default_rng(args.validation_seed)
    vals=set()
    for (field, _), count in zip(TEST_COUNTS.items(), alloc):
        vals |= sample_by_yield_quantile(remaining.loc[remaining.field==field], int(count), rng_val)
    manifest.loc[manifest.plotkey.isin(vals), 'new_split']='val'
    counts=manifest.new_split.value_counts()
    assert all(int(counts[s])==v for s,v in [('train',875),('val',96),('test',50)])
    assert manifest.plotkey.is_unique
    assert manifest.loc[manifest.new_split=='test','field'].value_counts().to_dict()==TEST_COUNTS

    # Full compatible manifest, same column names as original.
    new = manifest[['new_split','field','plot','path']].rename(columns={'new_split':'split'}).copy()
    new.insert(1,'split_seed',args.seed)
    new.to_csv(out/'reconstruction_plot_split.csv',index=False)
    audit=manifest[['field','plot','plotkey','plot_yield','n_subplots','split','new_split']].rename(columns={'split':'old_split'})
    audit.to_csv(out/'plot_split_with_yields.csv',index=False)
    summary=manifest.groupby(['new_split','field']).plot_yield.agg(['count','mean','std','min','max']).reset_index()
    summary.to_csv(out/'yield_distribution_by_split_field.csv',index=False)
    # Save measured plot yields for reproducibility, not only image paths.
    test=manifest.loc[manifest.new_split=='test', ['field','plot','plot_yield','n_subplots']].sort_values(['field','plot'])
    test.to_csv(out/'selected_test_plots.csv',index=False)
    old_test = set(manifest.loc[manifest.split=='test','plotkey'])
    overlap=len(old_test & tests)
    full_test=manifest.loc[manifest.new_split=='test','plot_yield']
    old_test_y=manifest.loc[manifest.split=='test','plot_yield']
    qa={'sampling':'rank-quintile stratified within each field; equal proportional allocation; RNG choice inside bins',
        'seed':args.seed,'validation_seed':args.validation_seed,
        'counts':{k:int(v) for k,v in counts.items()},
        'test_counts_by_field':TEST_COUNTS,
        'old_test_overlap_n':overlap,
        'old_test_mean_yield_g':float(old_test_y.mean()),
        'old_test_sd_population_g':float(old_test_y.std(ddof=0)),
        'new_test_mean_yield_g':float(full_test.mean()),
        'new_test_sd_population_g':float(full_test.std(ddof=0)),
        'new_test_range_g':[float(full_test.min()),float(full_test.max())],
        'important':'This is a documented-protocol approximation, not the original authors exact test IDs. This split evaluates downstream yield sensitivity; it does not refit the selected HSI reconstruction models.'}
    (out/'split_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print('OUTPUT:',out)
    print(json.dumps(qa,indent=2))
    print('\nYIELD DISTRIBUTION:\n',summary.to_string(index=False))
    print('\nRe-fit downstream yield models for the new partition; selected reconstruction weights remain fixed.')

if __name__ == '__main__':
    main()