# hnam simulation data on WCGPU1

All paths below are real directories under `/nfs/data/1/hnam`, also reachable from this repo via the `./hnam` symlink.

## PDHD — `train_data_PDHD_fixedbug_separateWC/`

- **Path:** `hnam/train_data_PDHD_fixedbug_separateWC/`
- **Size:** ~168 GB
- **Layout:** 60 paired `g4-rec-{i}.h5` / `g4-tru-{i}.h5` files, `i ∈ [0..59]`.
  - `g4-rec-{i}.h5` (~1.7–1.8 GB each) — reconstructed (signal-processed) frames
  - `g4-tru-{i}.h5` (~12–37 MB each) — truth charge frames
- **Processed copies:** `modified/` subdir holds `g4-rec-{i}_modified.h5` and `g4-tru-{i}_modified.h5` produced by the in-tree script `convert2.py` — a **channel-kill augmentation** step that injects 1–2 noisy channels per file (and zeros the matching truth). The output frames `frame_loose_lf0`, `frame_mp2_roi0`, `frame_mp3_roi0` are the three the model consumes. Details in [hnam_input_preprocessing.md](./hnam_input_preprocessing.md#stage-a--convert2py-channel-kill-augmentation-pdhd-only). The empty sibling dirs `modified_rec/`, `modified_tru/`, `noised/` look like staging scratch.
- **Train/val convention** used by `train4.py`: i ∈ [0..89] train, [90..99] val (with `i=2` skipped). Note that today there are only 60 raw files, so the upper indices are populated via `convert2.py` resampling. **Open question for hnam: confirm whether the [90..99] val files exist or if this is dead code.**
- **Top-level helper:** `hnam/move_g4_files.sh` — historical script partitioning into train (0–39) / val (40–59) subdirs (earlier convention).

## PDVD — two parallel sets

Both are organized by Charge-Readout-Plane (CRP), four subdirs each:

### 1. `train_data_PDVD_rate2M_window3p2/`  (default for training)
- **Path:** `hnam/train_data_PDVD_rate2M_window3p2/`
- **Size:** ~72 GB
- **Layout:** `crp1/`, `crp2/`, `crp3/`, `crp4/`. Each CRP holds 500 timestamped HDF5 files, e.g. `g4-rec-crp1-100-20260323T215241Z.h5`.
- **Used by:** `hnam/Pytorch-UNet/train4_pdvd.py` (default path).

### 2. `train_data_PDVD_rate2M_smallmpth_window3p2/`  (alternate)
- **Path:** `hnam/train_data_PDVD_rate2M_smallmpth_window3p2/`
- **Size:** ~72 GB
- **Layout:** same per-CRP structure.
- **Used by:** `train4_pdvd.py` (path is commented out by default; flip the path to use this set). "smallmpth" = smaller matched-filter threshold during signal processing.
- Also exists as a `.tar` archive sibling.

## Where do the raw files come from?

Not visible from the data directories themselves. The simulation pipeline (Geant4 + Wire-Cell sim → HDF5) is **not** captured in `/nfs/data/1/hnam` in an obvious form — `hnam/wct-dev-hnam` and `hnam/wire-cell-hnam` are the Wire-Cell forks but the actual jobsub configs that produced these files aren't in the data dirs. Flagged in [hnam_overview.md](./hnam_overview.md#simulation-generation).

## HDF5 dataset names you'll see inside

| Dataset (per event) | Notes |
| --- | --- |
| `frame_loose_lf0`, `frame_mp2_roi0`, `frame_mp3_roi0` | reconstructed signal-processed frames used as model **inputs** (PDHD) |
| `frame_deposplat0` | per-pixel truth charge — basis for the binary **mask** (PDHD) |
| `frame_ductor0` | truth charge for PDVD |
| `frame_*-crp{N}` variants | per-CRP versions used by PDVD loader |

Truth → mask conversion (thresholding + ROI padding) is documented in [hnam_truth_labeling.md](./hnam_truth_labeling.md).
