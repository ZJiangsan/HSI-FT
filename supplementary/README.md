# Supplementary HSI-FT experiments

These scripts accompany the primary HSI-FT pipeline in the repository root. They were
recovered from the research project's saved analysis scripts and their machine-specific
absolute paths were replaced with environment variables. The primary reconstruction
method, frozen-transfer code, split, and published result summaries were not changed.

**Provenance and validation.** The original research scripts were used for the
supplementary experiments. These repository copies have been syntax-checked; they
have **not** been rerun end-to-end on the large original dataset following path
portability edits. The separate measured-HSI stratified prediction script below is
a reference implementation based on the public yield model and is *not* claimed
to be a byte-for-byte copy of the earlier Trinity `prediction_only.py`.

## Configuration

Run these scripts from a clone of this repository with its dependencies installed.
They import `config.py`, `common.py`, `reconstruct.py`, and `yield_transfer.py`
from the parent repository directory.

```bash
export HSIWHEAT40_ROOT=/path/to/HSIwheat_40
export HSIFT_SUPPLEMENTARY_BASE="$HSIWHEAT40_ROOT/new_manuscript_method_transfer_5seeds"
```

The default directory names are retained from the actual experiments. The
dataset's frozen subplot masks and metadata must already be prepared as documented
in the root `README.md` and `preprocessing/README.md`. Model checkpoints
and large reconstructed cubes are not distributed in GitHub.

**Never point `HSIFT_SUPPLEMENTARY_BASE` to an unrelated experiment.**
The first five scripts below refer to the **original** 875/96/50 plot split
and its already selected reconstruction checkpoints. The alternative yield
partition is a separate, downstream-only sensitivity analysis.

## Analyses and usage

| Script | Purpose | Example |
|---|---|---|
| `hsi_ft_reconstruction_specific_yield.py` | Independently fit 81-feature yield predictors to each fixed reconstruction seed | `python supplementary/hsi_ft_reconstruction_specific_yield.py --method gram --recon-seed 0` |
| `hsi_ft_yield_spectral80.py` | Repeat yield prediction after dropping the 81st (pixel-count) feature | `python supplementary/hsi_ft_yield_spectral80.py --method gram --recon-seed 0` |
| `hsi_ft_paired_bootstrap.py` | Paired plot-cluster bootstrap of held-out subplot predictions (default 2,000 samples) | `python supplementary/hsi_ft_paired_bootstrap.py --reps 2000 --seed 2026` |
| `hsi_ft_denoising_controls_fixed.py` | Raw HSI, Savitzky–Golay 7/2, or train-only PCA (10/20 components) | `python supplementary/hsi_ft_denoising_controls_fixed.py --denoiser sg7 --features 81` |
| `hsi_ft_plot_structure_controls.py` | Plot-centering/mean-restoration and within-plot-shrinkage diagnostics | `python supplementary/hsi_ft_plot_structure_controls.py --condition shrink05 --features 81` |
| `hsi_ft_original_style_split.py` | Generate the field/yield-stratified 20/10/20 held-out plot partition | `python supplementary/hsi_ft_original_style_split.py` |
| `hsi_ft_yield_stratified_measured.py` | Five-seed measured-HSI yield prediction for 80 and 81 features under the alternative partition | `python supplementary/hsi_ft_yield_stratified_measured.py` |

For the reconstruction-specific evaluation, run each of the three methods
(`simplemlp`, `adjustedu2mdn`, `gram`) with reconstruction seeds
0–4, then run the spectral-only evaluation for the same pairs. The bootstrap
expects the prediction CSV files produced by both stages. The full 81-feature
analysis *must* finish before the corresponding spectral80 script runs because
the latter reuses saved 81-feature matrices.

For denoising controls, `--denoiser` accepts `raw`, `sg7`, `pca10`,
and `pca20`. Both `--features 80` and `--features 81` are supported.

## Alternative yield-stratified sensitivity analysis

The original selected spectral reconstructions were **not retrained** for this
sensitivity analysis; the manuscript's additional plot-partition check applies
to **measured HSI downstream prediction**, not to reconstruction-method ranking.
The two steps are:

```bash
python supplementary/hsi_ft_original_style_split.py
python supplementary/hsi_ft_yield_stratified_measured.py
```

The stratified partition uses yield-ranked within-field quintiles, seed 2026,
and held-out counts C3=20, C4=10, C9=20; its 96 validation plots are selected
from the remainder with seed 2027. The output directory is
`original_style_yield_stratified_seed2026/`. The separate CSV
[`results/stratified_measured_hsi_5seeds.csv`](../results/stratified_measured_hsi_5seeds.csv)
contains the **actual ten-run results** uploaded from Trinity. The reference
script is supplied to make the method inspectable, but numerical identity with
those runs has not been reverified on Trinity.

## Interpretation cautions

- The 81st feature is **spike-and-leaf pixel count**, not a physical area measurement.
- Subplot yields were allocated from measured whole-plot yields in proportion
  to spike-and-leaf pixels. Subplot R² does not prove independent harvested
  subplot-yield accuracy.
- The bootstrap resamples **whole test plots** and uses matched examples; model
  seeds are not independent biological samples.
- The Gram `reference_mean40` branch uses measured test HSI means and is
  **diagnostic only**, not the primary Gram reconstruction.
- Some files and caches contain machine-dependent paths from the original
  experiment. Regenerate as needed from the published data and the documented
  preprocessing; do not treat absent large caches as missing source code.
