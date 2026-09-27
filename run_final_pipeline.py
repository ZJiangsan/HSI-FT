#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the final reconstruction followed by the final frozen-transfer analysis.

Prerequisites:
  1) prepared 40-band HSIwheat cubes and subplot masks/metadata;
  2) frozen reconstruction_plot_split.csv;
  3) frozen best_decomposition_sam.pth.

See README.md.
"""

exec(open("run_reconstruction.py", "r", encoding="utf-8").read())
exec(open("final_transfer_diagnostics.py", "r", encoding="utf-8").read())
