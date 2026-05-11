# Input data preprocessing in hnam's pipeline

There are **two stages** between a raw `g4-rec-*.h5` file and a model input tensor:

| Stage | Where | When | Effect |
| --- | --- | --- | --- |
| A. Channel-kill augmentation | `hnam/train_data_PDHD_fixedbug_separateWC/convert2.py` | **Offline**, run once to materialize `modified/` | Inject 1–2 "bad" channels per file as Gaussian noise; zero matching truth |
| B. Crop / rebin / normalize / stack | `hnam/Pytorch-UNet/utils/h5_utils.py` `get_chw_imgs()` (line 117) | **Online**, every batch | Build the (3, wire, tick) tensor the model sees |

Truth-mask construction (thresholding + ROI padding) is in [hnam_truth_labeling.md](./hnam_truth_labeling.md); this doc covers the image side.

---

## Stage A — `convert2.py` (channel-kill augmentation, PDHD only)

Each raw `g4-rec-{i}.h5` holds 10 events numbered `100..109`; the matching `g4-tru-{i}.h5` numbers them `0..9`. For every file `i ∈ [0..59]` (skipping `i=2`, which is missing), `convert2.py` does:

1. **Pick one event** (1 out of 10) at random.
2. **Pick 1 or 2 channels** out of 1600 at random.
3. In that event's `frame_loose_lf0`: replace those channels' waveforms with Gaussian noise of the **same pedestal** as the original waveform, but RMS scaled by a random factor in `[2, 15]`.
4. In the matching event's `frame_deposplat0` truth: set those same channels' values to 0 across all ticks.
5. Copy every other frame / event / channel through unchanged. In particular `frame_mp2_roi0` and `frame_mp3_roi0` are untouched even for the augmented event.
6. Output: `modified_rec/g4-rec-{i}_modified.h5` and `modified_tru/g4-tru-{i}_modified.h5` (gzip-compressed datasets).

RNG seed is fixed (`np.random.default_rng(seed=42)`), so the augmentation is reproducible. Comments in `convert2.py` are in Korean.

**Why this matters:** the network learns to ignore noisy / dead channels — the channel-kill case appears in the training set but with truth correctly saying "no signal here."

The earlier `convert.py` in the same dir is a precursor without the channel-kill logic; `convert2.py` is the version whose outputs are actually used by `train4.py`.

PDVD does **not** appear to have an equivalent offline augmentation step (no `convert*.py` script under either of the two `train_data_PDVD_*` dirs).

---

## Stage B — Online loader `get_chw_imgs()`

Per event, in this order:

### 1. Stack three input frames as channels

In `train4.py:75`:

```python
im_tags = ['frame_loose_lf0', 'frame_mp2_roi0', 'frame_mp3_roi0']
```

`load()` in `h5_utils.py:11` reads each tag, stacks them along a new last axis → shape `(wire, tick, 3)`.

The historical alternative `im_tags = ['frame_tight_lf0', 'frame_loose_lf0']` (a 2-channel input) is commented out in `train4.py:379`. The current default is the 3-frame "L23" set: **L**oose-LF, **MP2**-ROI, **MP3**-ROI.

### 2. Auto-transpose to `(wire, tick, ch)`

```python
if arr.shape[0] > arr.shape[1]:
    arr = arr.T
```

Handles the older WC convention of `(tick, wire)` storage.

### 3. Rebin in time

`rebin()` (h5_utils.py:66) is a mean-pool. With `rebin=[1, REBIN]` it pools `REBIN` consecutive ticks into 1.

`train4.py` sets the output tick count via `y_range_dict`:

| `REBIN` | output ticks (`y_range[1]`) |
| ---: | ---: |
| 1 | 6000 |
| 2 | 3000 |
| 3 | 2000 |
| 4 | **1500** (default) |
| 5 | 1200 |
| 6 | 1000 |
| 8 | 750 |
| 10 | 600 |

### 4. Normalize

```python
im = rebin(im, ...) / norm
```

where `norm = z_scale = 4000` (train4.py:218). All three input channels are divided by the same constant.

### 5. Crop in wire and tick

In `train4.py:213–217`:

```python
x_range = [0, 1600]    # full induction planes (U+V together) — current default
# x_range = [0, 800]   # U only
# x_range = [800, 1600] # V only
# x_range = [476, 952] # PDVD V (alt)

y_range = [0, y_range_dict[REBIN]]
```

### 6. HWC → CHW

`get_chw_imgs()` transposes the per-event image to `(ch, wire, tick)` to match PyTorch's `NCHW` convention.

### Resulting tensor

For the default PDHD config (`REBIN=4`, both induction planes):

```
shape (3, 1600, 1500)
ch 0: frame_loose_lf0 / 4000  (rebinned)
ch 1: frame_mp2_roi0  / 4000  (rebinned)
ch 2: frame_mp3_roi0  / 4000  (rebinned)
```

### PDVD differences

`get_chw_imgs_pdvd()` (h5_utils.py:128) does the same five steps but **per-file** detects the CRP suffix from the filename (e.g. `g4-rec-crp2-77-...h5` → suffix `1`) and uses `frame_loose_lf1`, `frame_mp2_roi1`, `frame_mp3_roi1` for that file. Output shape is the same `(3, wire, tick)`.

---

## File layout you'll see inside HDF5

```
g4-rec-{i}.h5
└── /<event_id>/
      ├── frame_loose_lf0     # (wire, tick) — wire-cell SP "loose-lf" output
      ├── frame_mp2_roi0      # (wire, tick) — multi-plane #2 ROI
      ├── frame_mp3_roi0      # (wire, tick) — multi-plane #3 ROI
      └── (other diagnostic frames)

g4-tru-{i}.h5
└── /<event_id - 100>/
      └── frame_deposplat0    # (wire, tick) — Geant4 truth deposited charge
```

For PDVD the suffix `0` is replaced by the CRP index (`0..3`) and the truth dataset is `frame_ductor*`.
