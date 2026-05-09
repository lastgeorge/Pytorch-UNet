# Ground Truth Labeling

## Physics Background

In a **Liquid Argon Time Projection Chamber (LArTPC)**, charged particles traversing the detector ionize argon atoms along their trajectories. The freed electrons drift toward anode wire planes under an electric field, where they induce measurable signals. The readout is a 2D image: one axis represents wire channels (spatial position), the other represents time ticks (arrival time of the drifting electrons, proportional to depth).

The challenge is that the detector also records **electronics noise and induction artifacts** that are not related to true particle activity. The segmentation task is to identify which pixels in the readout correspond to genuine particle energy depositions versus noise.

The network produces a **per-pixel binary mask**: 1 = true particle signal, 0 = background/noise.

---

## HDF5 Data Format

All data is stored in HDF5 files organized as:

```
/{event_id}/{frame_tag}
```

- `event_id`: integer event index (e.g., `100`, `103`)
- `frame_tag`: string identifier for the readout type (see below)

Data comes in **paired file sets**:
- `*-rec-*.h5`: reconstructed detector readouts (network inputs)
- `*-tru-*.h5`: Monte Carlo truth information (network targets)

Example files:
```
data/cosmic-rec-0.h5   ←→   data/cosmic-tru-0.h5
data/g4-rec-r0.h5      ←→   data/g4-tru-r0.h5
...
data/g4-rec-r9.h5      ←→   data/g4-tru-r9.h5
```

To list the contents of an HDF5 file:
```bash
h5dump-shared -n data/g4-rec-r9.h5
```

To visualize a single frame:
```bash
./scripts/h5plot.py data/g4-rec-r9.h5 /100/frame_loose_lf0
./scripts/h5plot.py data/g4-tru-r9.h5 /103/frame_ductor0
```

---

## Input Channels (Reconstructed Frames)

Each input sample is a multi-channel 2D image formed by stacking several frame types. The channel configuration is encoded in the model name:

### `l23` configuration (3 channels) — most common

| Channel | Frame tag | Description |
|---|---|---|
| 0 | `frame_loose_lf0` | **Loose low-frequency ionization**: primary charge collection with minimal filtering. Broad signal, sensitive to particle tracks but also to noise and induction effects. |
| 1 | `frame_mp2_roi0` | **Micromegas plane 2 ROI**: secondary charge amplification readout from the MP2 anode, restricted to regions of interest (ROI) identified by the online filter. |
| 2 | `frame_mp3_roi0` | **Micromegas plane 3 ROI**: complementary amplification from the MP3 anode in its own ROI. |

### `lt` configuration (2 channels)

| Channel | Frame tag | Description |
|---|---|---|
| 0 | `frame_tight_lf0` | Tight low-frequency ionization — more aggressive filtering, lower noise but may miss some signal |
| 1 | `frame_loose_lf0` | Loose low-frequency ionization (as above) |

### `lt23` configuration (4 channels)

Combines all four: `frame_tight_lf0`, `frame_loose_lf0`, `frame_mp2_roi0`, `frame_mp3_roi0`.

The three frame types carry complementary information. The loose LF frame provides the raw signal; the MP2/MP3 ROI frames add localization from the secondary amplification stage, helping to separate real particle activity (which appears in multiple planes) from single-plane artifacts.

---

## Ground Truth: The Ductor Frame

Ground truth labels are derived exclusively from:

```
frame_ductor0   (from *-tru-*.h5)
```

The **Ductor** is the Monte Carlo simulation step in Wire-Cell Toolkit that models the actual ionization charge deposited by particles simulated by Geant4. It produces a 2D map of true energy depositions in the detector — the "ideal" signal that would be seen without any noise or detector effects.

This frame contains continuous ADC values representing the simulated deposited charge. Values are typically in the range 0–4000 ADC units, with zero indicating no true particle activity.

---

## Binary Mask Generation

The truth frame is converted to a binary segmentation mask by thresholding:

```python
mask = np.zeros_like(ductor_frame, dtype=np.float32)
mask[ductor_frame > threshold] = 1.0
```

The threshold separates true particle signal from residual simulation noise:

| Script | Threshold (ADC) | Notes |
|---|---|---|
| `train.py` | 100 | Original, conservative |
| `train2.py` | 100 | Same as train.py |
| `train3.py` | 10 | Lower threshold, captures fainter signals |

Pixels with deposited energy above the threshold are labeled **signal (1)**; all others are labeled **background (0)**.

This produces a sparse binary mask — signal pixels are typically a small fraction of the total image area, which motivates the use of Dice loss as a validation metric (Dice is robust to class imbalance; BCE can be dominated by the majority background class).

---

## Spatial Preprocessing

Each frame undergoes the following preprocessing before being fed to the network:

### 1. Cropping

Frames are cropped to a subregion of the detector:

| Detector | Wire range (`x_range`) | Tick range (`y_range`) |
|---|---|---|
| ProtoDUNE-SP (PDSP) | [800, 1600] | [0, 600] |
| ProtoDUNE-VD (PDVD) | [476, 952] | [0, 600] |

This focuses on an active detector region and produces a consistent input size.

### 2. Time-Axis Rebinning

The tick axis (time) is downsampled by a factor of 10 by default (`tick-rebin=10`), reducing the image height and computational cost. This can be disabled (`--tick-rebin 1`) for full-resolution inference.

### 3. Normalization

Input values are divided by `z_scale = 4000` (the expected maximum ADC value), mapping pixel values to approximately [0, 1].

### 4. Channel Stacking

Multiple frames are stacked into a `(C, H, W)` tensor where:
- C = number of input channels (2 or 3)
- H = cropped and rebinned tick range
- W = cropped wire range

---

## Data Utilities

| File | Key functions | Purpose |
|---|---|---|
| `utils/load.py` | `get_ids()`, `get_masks()`, `split_train_val()` | Event index enumeration, mask extraction, dataset splitting |
| `utils/h5_utils.py` | `load(file, event_id, tags)` | HDF5 frame retrieval by event ID and tag name |
| `utils/utils.py` | General array helpers | Misc. preprocessing |
| `utils/data_vis.py` | Visualization utilities | Frame display |
| `scripts/h5plot.py` | `main(file, path)` | Command-line HDF5 frame plotter |
| `test-index.py` | Index validation | Verify event index mapping across files |
| `roi-count.ipynb` | ROI counting analysis | Exploratory analysis of signal region counts |
