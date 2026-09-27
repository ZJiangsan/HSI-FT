# Data layout

The HSI-FT scripts do not redistribute the HSIwheat imagery.

Download the public HSIwheat data from the University of Minnesota Data Repository:

**DOI: 10.13020/0ch0-vb18**

The final study code expects a prepared 40-band working root. By default:

```text
/home/nibio/HSIwheat_40/
```

or set:

```bash
export HSIWHEAT40_ROOT=/your/path/HSIwheat_40
```

Expected inputs:

```text
HSIwheat_40/
├── wavelengths_40.npy
├── C3/hsi40/*.npy
├── C4/hsi40/*.npy
├── C9/hsi40/*.npy
├── original190_paper_reproduction/
│   ├── subplot_metadata.csv
│   └── sl_masks/
└── decomposition_train_only_seed2026/
    ├── reconstruction_plot_split.csv
    └── best_decomposition_sam.pth
```

The 40-band cube and subplot-mask preparation follows the companion HSIwheat evaluation repository:

https://github.com/ZJiangsan/HSIRecon-Transfer

`train_decomposition_and_split.py` creates the new study's deterministic plot split (seed 2026; 875/96/50 plots) and train-only decomposition.
