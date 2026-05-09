# Repository Overview: Pytorch-UNet for LArTPC Signal Processing

## Purpose

This repository implements deep learning-based pixel segmentation for **Liquid Argon Time Projection Chamber (LArTPC)** detectors, specifically targeting the **ProtoDUNE** and **DUNE** experiments at Fermilab. The goal is to distinguish true particle ionization signals from electronics noise in 2D detector readout frames.

The original Pytorch-UNet code (by milesial) has been adapted to:
- Accept multi-channel LArTPC detector frames as input
- Train on Monte Carlo simulation truth labels
- Export models for fast C++ inference in the reconstruction pipeline

Related talks:
- [ProtoDUNE DRA Meeting, Jan. 08 2020](https://indico.fnal.gov/event/22795/contribution/4)
- [DUNE Collaboration Meeting, Jan. 29 2020](https://indico.fnal.gov/event/20144/session/8/contribution/98)

---

## Directory Structure

```
Pytorch-UNet/
├── unet/                  # Classic U-Net architecture
│   ├── model.py           # UNet class
│   └── parts.py           # double_conv, down, up, outconv blocks
├── uresnet/               # U-Net with residual blocks (shallower)
│   ├── model.py           # UResNet class
│   └── parts.py           # residual_block, inconv, down, up blocks
├── nestedunet/            # UNet++ (nested dense skip connections)
│   └── model.py           # NestedUNet class with VGGBlock and PadCat
├── utils/                 # Data loading and preprocessing utilities
│   ├── load.py            # get_ids(), get_masks(), split_train_val()
│   ├── h5_utils.py        # HDF5 frame loading
│   ├── utils.py           # General helpers
│   ├── data_vis.py        # Visualization helpers
│   └── crf.py             # Conditional Random Field post-processing
├── scripts/               # Utility scripts
│   └── h5plot.py          # Visualize HDF5 frames as images
├── cpp/                   # C++ LibTorch inference
│   ├── test-ts.cpp        # C++ inference example using TorchScript
│   └── CMakeLists.txt     # Build config (LibTorch + Eigen)
├── pth-model/             # Saved PyTorch state dict models (.pth)
├── ts-model/              # TorchScript models (.ts) for deployment
├── ts-model-2.3/          # TorchScript models compatible with PyTorch 2.3
├── docs/                  # This documentation
│
├── train.py               # Training script v1 (percentage-based split)
├── train2.py              # Training script v2 (explicit index ranges)
├── train3.py              # Training script v3 (multi-file, PDVD detector)
├── train.sh / train2.sh / train3.sh   # Shell wrappers for training
│
├── eval.py                # Bulk evaluation across labeled datasets
├── eval-epoch.py          # Per-epoch loss/dice evaluation
├── eval_loss_sample.py    # Per-sample loss evaluation
├── eval_util.py           # Shared evaluation functions (dice, eff, purity)
├── eval.sh                # Shell wrapper for evaluation
│
├── predict.py             # Single-event inference, saves JPEG output
├── predict.sh             # Shell wrapper for prediction
│
├── dice_loss.py           # Custom Dice loss (autograd Function)
├── convert_loss.py        # Loss format conversion utilities
├── compare_ep.py          # Compare loss across training epochs
├── make_loss_dist.py      # Loss distribution analysis
├── plot_epoch.py          # Plot epoch-level metrics
├── tensor-board.py        # TensorBoard loss visualization
├── test-index.py          # Test HDF5 event indexing
├── roi-count.ipynb        # Jupyter notebook for ROI counting analysis
│
├── to-ts.py               # Convert .pth model → TorchScript .ts
├── to-pth.py              # Convert TorchScript .ts → .pth state dict
│
├── environment.yml        # Conda environment specification
└── requirements.txt       # pip requirements
```

---

## Model Naming Convention

Saved model files encode their configuration:

```
{architecture}-{input_channels}-{dataset}-{epochs}.{ext}
```

| Segment | Meaning | Examples |
|---|---|---|
| `unet` / `uresnet` / `nestedunet` | Architecture | `unet`, `nestedunet` |
| `l23` | Channels: loose + MP2 + MP3 (3-channel) | `l23` |
| `lt` | Channels: tight + loose (2-channel) | `lt` |
| `lt23` | Channels: tight + loose + MP2 + MP3 (4-channel) | `lt23` |
| `cosmic500` | Dataset: 500 cosmic-ray events | `cosmic500` |
| `e50` | Trained for 50 epochs | `e50` |

Example: `unet-l23-cosmic500-e50.pth` — U-Net, 3-channel input, 500 training events, 50 epochs.

---

## Quick Start

**Train:**
```bash
./train3.sh        # trains on multi-file g4 datasets, 50 epochs
./train2.sh        # trains on a single cosmic dataset, 50 epochs
```

**Predict (TorchScript model):**
```bash
./predict.sh       # runs inference on a single event, saves JPEG
```

**Convert model format:**
```bash
# PyTorch → TorchScript
./to-ts.py -m pth-model/unet-l23-cosmic500-e50.pth -o ts-model-2.3/unet-l23-cosmic500-e50.ts

# TorchScript → PyTorch
./to-pth.py -m ts-model/unet-l23-cosmic500-e50.ts -o pth-model/unet-l23-cosmic500-e50.pth
```

**Visualize HDF5 data:**
```bash
./scripts/h5plot.py data/g4-rec-r9.h5 /100/frame_loose_lf0
./scripts/h5plot.py data/g4-tru-r9.h5 /103/frame_ductor0
```

---

## Dependencies

- Python 3.9, PyTorch 1.10+ (CUDA 11.1), h5py, matplotlib, numpy
- For C++ inference: LibTorch, Eigen3
- See `environment.yml` (conda) or `requirements.txt` (pip) for full dependency lists
