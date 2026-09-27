# Upstream preprocessing

`prepare_40band_data.py` converts the released 190-band HSIwheat cubes into the fixed 40-band representation used by HSI-FT.

The subplot geometry, yield metadata and spike-and-leaf masks are shared with the preceding HSIwheat evaluation study and are documented/reproduced in the companion repository:

https://github.com/ZJiangsan/HSIRecon-Transfer

HSI-FT intentionally reuses those frozen upstream objects so that the new experiment changes the reconstruction strategy rather than redefining the downstream benchmark.
