#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the final HSI-FT reconstruction experiment.

Final design:
    fixed triplet: Current (460.669 / 550.628 / 801.674 nm)
    methods: Direct MLP, adjusted u²-MDN, Gram
    reconstruction seeds: 0-4 for every method
    Gram budget: 5522 epochs per seed, validation-best checkpoint retained

After this finishes, run:
    python final_transfer_diagnostics.py
"""

import config as C

print("=" * 100)
print("HSI-FT: FINAL RECONSTRUCTION EXPERIMENT")
print("=" * 100)
print("Root             :", C.ROOT)
print("Frozen split     :", C.PLOT_SPLIT_CSV)
print("Decomposition    :", C.DECOMPOSITION_CKPT)
print("Triplets         :", C.ACTIVE_TRIPLETS)
print("Methods          :", C.ACTIVE_METHODS)
print("Recon seeds      :", C.RECON_SEEDS)
print("Gram max epochs  :", C.GRAM_MAX_EPOCHS)

from reconstruct import run_reconstruction

results = run_reconstruction()

print("\nFINAL RECONSTRUCTION COMPLETE")
print(results.to_string(index=False))
