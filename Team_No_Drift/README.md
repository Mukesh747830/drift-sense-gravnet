# KLA Problem Statement – AI-Based Restoration of Degraded Images

This repository contains the required submission structure for the KLA image restoration track.

## Execution

The script is strictly configured to execute without user interaction, API keys, or internet access.

```bash
python run.py <input-dir> <output-dir>
```

## Details
- **Inputs:** Reads all degraded `.npy` arrays from `<input-dir>`.
- **Processing:** Handles NaN/Inf corruption, normalizes to strictly `[0, 1]`, and applies non-linear spatial filtering to reconstruct missing data and remove physical noise artifacts.
- **Outputs:** Saves the restored `(H, W)` grayscale arrays to `<output-dir>` using identical filenames.
