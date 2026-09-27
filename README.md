# HSI-FT

Code for the study:

**From Spectral Recovery to Functional Compatibility: Why Hyperspectral Reconstructions Differ in Transfer to Genuine-HSI Models**

**HSI-FT** is a controlled comparison of three sparse-to-hyperspectral reconstruction strategies under one fixed three-band input, followed by an unchanged-model transfer test using a wheat-yield MLP trained only on genuine measured HSI.

## Central experimental question

When the sparse measurements, 40-band target, held-out plots, feature definition, scaler, and downstream predictor are held fixed, why can hyperspectral reconstructions with similar conventional spectral error differ substantially in compatibility with a model trained on genuine HSI?

The final paper compares:

- **Direct MLP** — pointwise 3-to-40 regression;
- **Adjusted u²-MDN** — same-resolution reconstruction with a stick-breaking / Dirichlet-style latent representation;
- **Gram-guided mapper** — sparse-to-latent mapping with a frozen HSI decomposition and abundance-Gram constraints.

## Final fixed protocol

### Sparse input

The same three measured bands are used by every reconstruction method:

- 460.669 nm
- 550.628 nm
- 801.674 nm

These correspond to 40-band indices `(3, 12, 36)` in the prepared HSIwheat representation.

### Plot split

A plot-disjoint split is created **before decomposition or reconstruction fitting**:

- 875 training plots
- 96 validation plots
- 50 held-out test plots

The split uses NumPy RNG seed `2026`. The exact frozen split used in the manuscript is committed in this repository as:

`splits/reconstruction_plot_split.csv`

For portability, the repository copy contains only `split`, `split_seed`, `field`, and `plot`; the original machine-specific absolute path column was removed.

### Reconstruction seeds

All methods use seeds `0,1,2,3,4`.

### Gram training budget

Every Gram mapper seed is trained for **5522 epochs**. The checkpoint with the minimum validation composite loss within that fixed budget is retained.

The Gram absolute spectral level is restored using a train/validation-only ridge mapping:

`3 observed-band plot means -> 40-band plot mean`

The ridge strength is selected only on reconstruction-validation plots. The held-out test plot's true 40-band mean is never used by the primary pipeline.

### Frozen downstream transfer

The yield predictor is trained **only on measured 40-band HSI**.

Each subplot is represented by 81 features:

- 40 band means;
- 40 population standard deviations (`ddof=0`);
- spike-and-leaf pixel count.

Under the new plot split this gives:

- 44,966 training subplots
- 4,973 validation subplots
- 2,565 held-out test subplots

Five yield-MLP initialization seeds (`0-4`) are trained on measured HSI. The measured-HSI feature scaler and model weights are then frozen and reused unchanged for every reconstructed representation.

No reconstruction-specific scaler fitting or downstream retraining is permitted in the primary transfer experiment.

## Repository layout

```text
HSI-FT/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── config.py
├── common.py
├── reconstruct.py
├── yield_transfer.py
├── train_decomposition_and_split.py
├── evaluate_decomposition.py
├── run_reconstruction.py
├── final_transfer_diagnostics.py
├── run_final_pipeline.py
├── splits/
│   └── reconstruction_plot_split.csv
├── preprocessing/
│   ├── prepare_40band_data.py
│   └── README.md
├── data/
│   └── README.md
├── docs/
│   └── REPRODUCIBILITY.md
└── results/
    ├── headline_summary.txt
    ├── primary_method_summary.csv
    ├── primary_seed_ranges.csv
    ├── primary_transfer_by_reconstruction_seed.csv
    ├── representation_diagnostics_by_recon_seed.csv
    └── subplot_split_counts.csv
```

## Data preparation

The public HSIwheat dataset is available from the University of Minnesota Data Repository:

**DOI: 10.13020/0ch0-vb18**

The new-study pipeline assumes the following prepared objects already exist under `HSIWHEAT40_ROOT`:

```text
wavelengths_40.npy
C3/hsi40/*.npy
C4/hsi40/*.npy
C9/hsi40/*.npy
original190_paper_reproduction/subplot_metadata.csv
original190_paper_reproduction/sl_masks/
```

`preprocessing/prepare_40band_data.py` creates the fixed 40-band cubes from the released 190-band cubes. The subplot-mask/metadata preprocessing is the same frozen upstream benchmark used in the companion repository:

https://github.com/ZJiangsan/HSIRecon-Transfer

See `data/README.md` for the expected local layout.

## Running the final study

Set the data root if it differs from the original experiment:

```bash
export HSIWHEAT40_ROOT=/path/to/HSIwheat_40
```

### 0. Prepare the 40-band cubes and frozen upstream subplot objects

Run `preprocessing/prepare_40band_data.py` for the 40-band cubes. The frozen subplot metadata and spike-and-leaf masks are shared with the companion HSIwheat benchmark; see `preprocessing/README.md`.

### 1. Create the frozen plot split and train the train-only decomposition

```bash
python train_decomposition_and_split.py
```

This deterministically creates the plot split (seed 2026) and trains the train-only decomposition. The reported run selected its best state at epoch 12378; the public script defaults to epochs 0–12378 and notes that exact floating-point identity can depend on the PyTorch/CUDA environment.

Outputs are written under:

```text
decomposition_train_only_seed2026/
```

The final reconstruction code expects:

```text
decomposition_train_only_seed2026/reconstruction_plot_split.csv
decomposition_train_only_seed2026/best_decomposition_sam.pth
```

### 2. Evaluate the frozen decomposition

```bash
python evaluate_decomposition.py
```

The paper reports approximately:

- train physical SAM 1.219°, RMSE 0.00397
- validation physical SAM 1.194°, RMSE 0.00389
- test physical SAM 1.242°, RMSE 0.00414

### 3. Run the three reconstruction methods

```bash
python run_reconstruction.py
```

This runs the fixed input condition for Direct MLP, adjusted u²-MDN and Gram with five reconstruction seeds each.

### 4. Run the genuine-HSI yield models, frozen transfer and mechanism diagnostics

```bash
python final_transfer_diagnostics.py
```

This stage:

1. builds the measured-HSI 81-feature representation;
2. trains or loads five genuine-HSI yield MLPs;
3. freezes their scaler and weights;
4. applies them unchanged to each reconstruction seed;
5. calculates `ΔR²`, `ΔRMSE`, `ΔMAE` and prediction-shift RMSE;
6. evaluates mean-feature and SD-feature distortion;
7. evaluates covariance/correlation preservation;
8. projects test features into a PCA basis fitted only on measured-HSI training features;
9. calculates model-sensitivity-weighted feature error;
10. saves manuscript-facing summaries and figures.

If the split and decomposition already exist, `run_final_pipeline.py` can run steps 3-4 consecutively.

## Expected headline results

The final five-seed analysis gives approximately:

| Input representation | Frozen R² | ΔR² | Spectral RMSE | SAM (°) |
|---|---:|---:|---:|---:|
| Measured 40-band HSI | 0.8469 | 0 | — | — |
| Direct MLP | 0.6562 | 0.1907 | 0.013910 | 3.7457 |
| Adjusted u²-MDN | 0.6712 | 0.1757 | 0.014264 | 3.8076 |
| Gram | 0.7360 | 0.1109 | 0.014956 | 4.0946 |

Gram therefore reduces mean `ΔR²` by approximately **41.8%** relative to Direct MLP and **36.9%** relative to adjusted u²-MDN, despite having higher conventional spectral error.

## Leakage controls

The held-out 50 plots are not used for:

- decomposition fitting;
- reconstruction fitting;
- reconstruction checkpoint selection;
- Gram ridge-mean fitting or alpha selection;
- yield-model training;
- yield-model checkpoint selection;
- feature-scaler estimation;
- hyperparameter selection.

Test labels are used only for final held-out performance calculation.

## Large generated files

Reconstructed test cubes and neural-network checkpoints are intentionally not committed. They can occupy many gigabytes. The final analysis caches `yield_test_features.npy` for each reconstruction run; after those caches are verified, full reconstructed cubes can be archived if disk space is needed.

## License

This repository is released under the **MIT License**. See [LICENSE](LICENSE).

## Citation

Please cite the associated manuscript when using this code. A final DOI/citation can be added after publication or preprint posting.
