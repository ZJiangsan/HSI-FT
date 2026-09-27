from pathlib import Path
import os

# =============================================================================
# ROOTS
# =============================================================================
ROOT = Path(os.environ.get('HSIWHEAT40_ROOT', '/home/nibio/HSIwheat_40'))
FIELDS = ('C3', 'C4', 'C9')

# Exact frozen plot split created before the leakage-free decomposition.
PLOT_SPLIT_CSV = ROOT / 'decomposition_train_only_seed2026' / 'reconstruction_plot_split.csv'

# Frozen train-only decomposition. It is loaded and NEVER retrained here.
DECOMPOSITION_CKPT = ROOT / 'decomposition_train_only_seed2026' / 'best_decomposition_sam.pth'

# Frozen subplot geometry / yield targets / SL masks from the established HSIwheat pipeline.
STAGE1_ROOT = ROOT / 'original190_paper_reproduction'
SUBPLOT_META_CSV = STAGE1_ROOT / 'subplot_metadata.csv'
SL_MASK_ROOT = STAGE1_ROOT / 'sl_masks'

WAVELENGTHS40_PATH = ROOT / 'wavelengths_40.npy'
MEASURED40_ROOTS = {field: ROOT / field / 'hsi40' for field in FIELDS}

# Reuse an existing all-valid-pixel cache when available. If these do not exist,
# common.py builds a new cache under OUT/cache.
PIXEL_CACHE_CANDIDATES = [
    ROOT / 'experiment2_direct_mlp_3to40' / 'cache' / 'all_valid_pixels_40.npy',
    ROOT / 'decomposition_train_only_seed2026' / 'all_valid_pixels_40band.npy',
]
CUBE_TABLE_CANDIDATES = [
    ROOT / 'experiment2_direct_mlp_3to40' / 'cache' / 'cube_ranges.csv',
    ROOT / 'decomposition_train_only_seed2026' / 'cube_ranges.csv',
]

OUT = ROOT / 'hsi_ft_outputs'
CACHE = OUT / 'cache'
RECON_OUT = OUT / 'reconstruction'
YIELD_OUT = OUT / 'yield_transfer'
DIAG_OUT = OUT / 'diagnostics'

# =============================================================================
# FINAL STUDY SETTINGS
# =============================================================================
ACTIVE_METHODS = ('SimpleMLP', 'AdjustedU2MDN', 'Gram')
ACTIVE_TRIPLETS = ('Current',)

# Five reconstruction seeds and five downstream measured-HSI yield seeds.
RECON_SEEDS = (0, 1, 2, 3, 4)
YIELD_SEEDS = (0, 1, 2, 3, 4)

DEVICE = 'cuda:0'
EPS = 1e-8
INFERENCE_CHUNK = 131072

# =============================================================================
# FROZEN SPARSE-BAND DEFINITIONS
# Only 'Current' is active in this study; the remaining triplets are retained
# as provenance from the preceding HSIwheat evaluation.
# =============================================================================
TRIPLETS = [
    ('Current',                (460.7, 550.6, 801.7)),
    ('RGB-like',               (460.7, 550.6, 651.0)),
    ('Visible spread',         (431.4, 550.6, 682.4)),
    ('Visible cluster',        (460.7, 502.5, 550.6)),
    ('Green-red-edge-NIR',     (550.6, 711.7, 801.7)),
    ('Red-red-edge-NIR',       (651.0, 711.7, 801.7)),
    ('Blue-red-edge-NIR',      (460.7, 711.7, 854.0)),
    ('Red-edge cluster',       (682.4, 711.7, 741.0)),
    ('NIR cluster',            (801.7, 854.0, 868.6)),
    ('Wide spectral span',     (431.4, 711.7, 868.6)),
    ('T12 yield-optimized',    (441.8, 510.9, 611.3)),
]

# =============================================================================
# DIRECT MLP
# =============================================================================
MLP_MAX_EPOCHS = 500
MLP_PATIENCE = 40
MLP_MIN_DELTA = 1e-8
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-5
MLP_BATCH_SIZE = 16384
MLP_SAMPLES_PER_EPOCH = 1_000_000
MLP_VAL_MAX_PIXELS = 300_000

# =============================================================================
# ADJUSTED u2MDN
# Same-resolution 3->40 adaptation retaining the characteristic dense encoder,
# stick-breaking/Dirichlet latent representation and two-layer decoder.
# =============================================================================
U2_LATENT = 15
U2_BASE_WIDTH = 3
U2_MAX_EPOCHS = 500
U2_PATIENCE = 40
U2_MIN_DELTA = 1e-8
U2_LR = 1e-3
U2_WEIGHT_DECAY = 0.0
U2_BATCH_SIZE = 16384
U2_SAMPLES_PER_EPOCH = 1_000_000
U2_VAL_MAX_PIXELS = 300_000
U2_VOLUME_WEIGHT = 0.001
U2_MI_WEIGHT = 0.1
U2_SPARSITY_WEIGHT = 0.01
U2_SPARSE_RECON_WEIGHT = 1.0

# =============================================================================
# GRAM MAPPER
# Frozen train-only decomposition + Gram composite loss.
# =============================================================================
GRAM_MAX_EPOCHS = 5522
GRAM_LR = 1e-4
GRAM_WEIGHT_DECAY = 1e-4
GRAM_VAL_EVERY = 1
GRAM_CHECKPOINT_EVERY = 100
GRAM_EARLY_STOP_PATIENCE = 0  # fixed-budget training; validation-best checkpoint retained
GRAM_GRAD_CLIP = 0.0

# Primary Gram absolute output: centered Gram reconstruction + a 3->40 plot-mean
# ridge model fitted on TRAIN plots and alpha-selected on VAL plots only.
GRAM_MEAN_RIDGE_ALPHAS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)

# Save the true-measured-mean branch as a diagnostic only; it is never primary.
SAVE_GRAM_REFERENCE_MEAN_DIAGNOSTIC = True

# =============================================================================
# DOWNSTREAM YIELD MODEL
# =============================================================================
YIELD_HIDDEN = (10, 10, 10, 10)
YIELD_EPOCHS = 100
YIELD_BATCH_SIZE = 32
YIELD_LR = 1e-3

# =============================================================================
# DIAGNOSTICS
# =============================================================================
DIAG_GRADIENT_SUBPLOTS = 2500
DIAG_PIXEL_SAMPLE = 200000
