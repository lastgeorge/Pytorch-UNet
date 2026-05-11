# hnam model architecture + reproduce PDHD training

## Final architecture: MobileNetV3-large U-Net

- **File:** `hnam/Pytorch-UNet/mobilenetv3/model.py`, class `MobileNetV3_UNet`.
- **Encoder:** torchvision MobileNetV3-large (pretrained ImageNet weights by default).
  - Input goes through a 1×1 conv that projects from `in_channels` to 3 channels before entering the MobileNet stem.
  - 5 downsample stages (strides 1/2, 1/4, 1/8, 1/16, 1/32) — feature maps from each are kept for the skip connections.
- **Decoder:** 4 `UpBlock` stages (`ConvTranspose2d` → concat with the matching encoder skip → 2× `ConvBNAct`).
- **Head:** 1×1 conv to `out_channels=1`, then sigmoid.
- **Checkpoint size:** ~21 MB per `CP*.pth` (vs ~52 MB for the vanilla U-Net it replaced).
- **Selected in launcher:** `hnam/Pytorch-UNet/train4.sh` line 48 — `MODELS="mobilenetv3"`.

Other architectures present in the repo (`unet/`, `nestedunet/`, `mobilenetv2/`, `transformer/`) were experimented with but are not the current choice.

## Design discussion: isotropic 2D convolution on (wire, time)

The model uses standard 2D convolutions that treat the wire and time axes identically — same kernel sizes (3 or 5 in MobileNetV3-large), same strides, same padding. This is worth examining because the **output** is functionally a per-wire ROI mask in time: the downstream Wire-Cell pipeline consumes the prediction as binary signal/no-signal intervals along the tick axis on each channel, and `get_masks()` in `utils/h5_utils.py:141` literally documents itself as "padding on time axis" (per-wire `for h in range(H)` loops over the time axis only).

So the **task is 1D-along-time per channel**, while the **representation** is 2D. Is the isotropic 2D kernel choice optimal?

### Physical scales after rebin=4

| Axis | Pitch | Comment |
| --- | --- | --- |
| Wire (induction U / V) | ≈ 4.79 mm (PDHD) | Fixed by detector geometry. |
| Tick (raw, @ 500 ns) | ≈ 0.8 mm of drift (1.6 mm/μs at 500 V/cm) | Continuous in principle. |
| Tick (effective, `rebin=4`) | ≈ 3.2 mm | This is what the network sees. |

With `rebin=4` the two axes have **comparable physical units** (3.2 mm/tick vs 4.79 mm/wire, ratio ≈ 0.67). At smaller rebin the asymmetry grows: `rebin=1` gives a ratio of 0.17 — the time axis becomes ~6× finer than the wire axis in physical units, and isotropic 3×3 / 5×5 kernels would over-smooth in time.

### Arguments for the current isotropic 2D design

1. **Tracks are genuinely 2D objects.** A diagonal track traces an inclined line in (wire, time) space; the *evidence* for an ROI at one position comes from neighbouring (wire, tick) pixels. Even though the *label* is 1D per channel, joint 2D context helps localise it.
2. **Roughly isotropic physical units after rebin=4** (table above) — the strongest physics-side argument. Pulses are ~10–15 raw ticks (~3 effective ticks) wide in time and light up ~3–5 wires at a given instant; the two natural scales are within a factor of two.
3. **Pretrained transfer.** MobileNetV3-large's ImageNet weights are useful for low-level edge / texture features; reusing them requires the 2D structure the backbone was designed around. A from-scratch anisotropic net loses this benefit.
4. **Literature precedent.** Standard DNN-ROI and DNN-SP variants (Yu et al. 2021, `docs/2007.12743v3.pdf`) all use 2D convs on (channel, tick).

### Arguments against

1. **The labeling itself is 1D along time.** `get_masks()` pads, gap-merges and run-length-filters strictly along the tick axis (`min_run`, `padding`, `padding_side`, `min_gap`). There is no wire-axis padding. The *task prior* — "per-channel temporal segmentation, with wire context as a hint" — is not symmetric.
2. **The wire-plane prior is already supplied by the inputs.** Two of the three input channels are `frame_mp2_roi0` and `frame_mp3_roi0` — Wire-Cell's 2-plane and 3-plane coincidence ROI candidates, which already encode cross-plane wire-axis correlations. The network does not need a large wire-RF to *rediscover* this prior; it only needs to combine the three input channels appropriately at each spatial location.
3. **U/V plane boundary at wire 800.** Training crops `[0, 1600]` = U(800) + V(800) concatenated. Any wire-axis kernel that spans wire 800 mixes signals from two physically uncorrelated induction planes. Isotropic 5×5 kernels at deep stages with jumps of 16–32 actively bleed information across this discontinuity; smaller wire kernels avoid it for free.
4. **Wasted capacity on wire-axis smoothness.** Every isotropic 3×3 / 5×5 conv encodes a smoothness prior on the wire axis. The network has to "spend capacity" to *un*-smooth at sharp wire boundaries of an ROI; anisotropic kernels wouldn't.
5. **Bottleneck RF is square — and far too wide.** From [hnam_inference_input_shape.md](./hnam_inference_input_shape.md#3-minimum-input-size-and-bottleneck-receptive-field) the theoretical RF at the bottleneck is ≈ 595×595 input pixels. A pulse covers ~3–5 wires; a track is contiguous over tens of wires. Reaching 595 wires of context to decide one ROI's edges is grossly more than the task needs.
6. **Smaller rebin breaks the isotropy assumption.** If you ever want finer temporal resolution (rebin=1 or 2), isotropic kernels become a poor fit (see table above).

### Direction of improvement: **less** wire-axis convolution, **same or more** time-axis

The reasoning above (especially points 2, 3, 5 of "Arguments against") points one way: shrink the wire-axis kernel extent and/or wire-axis receptive field, while preserving (or growing) the time-axis reach. Aim for a wire-RF of ~10–20 wires (covering a pulse's instantaneous wire-spread plus a track's local contiguity) rather than the current ~595.

Don't go all the way to *zero* wire context — cross-wire noise rejection (correlated EMI across neighbouring channels) and the "real-track-is-contiguous" prior still want a few wires of receptive field. Less, not none.

### Alternatives worth A/B testing

Easiest to boldest. All are simplest inside the from-scratch `unet/` architecture; the MobileNetV3 backbone is harder to modify without sacrificing ImageNet pretraining.

| Variant | What changes | Effect |
| --- | --- | --- |
| **Anisotropic kernels (mild)** | Deeper-stage depthwise k×k → **3×k** or **1×k** (wire × time) | Cuts wire-RF growth at deep stages without touching strides. Pulse-aspect-ratio aware. |
| **Asymmetric strides** | One of the wire-axis stride-2 stages becomes stride-1 (so wire effective downsample = /16, time stays /32) | Halves wire-jump → halves wire-RF; per-wire resolution stays sharper. Costs more compute since wide feature maps survive deeper. |
| **Factorized 2D conv** | Replace k×k with k×1 followed by 1×k | Lets `k_wire` and `k_time` be set independently. Fewer params than k×k. |
| **1D temporal bottleneck** | At the bottleneck two stages, use 1×k (pure time) depthwise convs | Drops wire context entirely past mid-encoder; early stages still do channel mixing. Most aligned with the per-wire-temporal-mask task; riskiest in case the wire prior in MP2/MP3 inputs is insufficient. |

The repo already contains alternative encoders (`unet/`, `nestedunet/`, `mobilenetv2/`, `transformer/`). A transformer-style attention model with **separate wire- and time-axis attention** is the natural place to test the "decouple the axes" hypothesis.

### Bottom line

For the current `rebin=4` PDHD configuration, the isotropic 2D design is **defensible but not provably optimal**: physical scales roughly align, pretrained transfer is a real benefit, and the network is well-tested. If the current eval metrics ([hnam_evaluation.md](./hnam_evaluation.md)) are good enough for the downstream physics, don't fix what isn't broken.

If you want to squeeze more performance out — especially at smaller rebin where the wire/time aspect ratio diverges, or to mitigate the U/V boundary bleed at wire 800 — the single highest-leverage change to try is replacing the **deepest two stages' 5×5 depthwise convs with 1×5 or 3×5 (time-dominant)** kernels. That concentrates capacity on the axis where the output actually lives, naturally shrinks cross-plane bleed at wire 800, and minimally disturbs the rest of the architecture.

## Reproduce hnam's PDHD training

### 0. Environment

hnam's conda env is named **`dnnroi`** (see `hnam/Pytorch-UNet/environment.yml`):
- Python 3.9
- PyTorch + torchvision built against CUDA 11.1
- h5py 3.10, numpy 1.26, matplotlib 3.8

If you don't already have it on WCGPU1, recreate from the file:

```bash
conda env create -f hnam/Pytorch-UNet/environment.yml
conda activate dnnroi
```

(Confirm with hnam whether there's a shared env already activated for him — the `python-packages/` directory at `/nfs/data/1/hnam/python-packages` suggests he may pip-install on top.)

### 1. Launch PDHD training

```bash
cd hnam/Pytorch-UNet     # use his copy so the hard-coded data paths resolve
./train4.sh              # answer "1" at the PDHD/PDVD prompt
```

The launcher echoes the active parameters and then runs (default values shown):

```bash
python train4.py \
  --gpu \
  --model mobilenetv3 \
  --start-epoch 0 --nepoch 50 \
  --start-train 0 --ntrain 90 \
  --start-val 90  --nval 10 \
  --learning-rate 0.1 \
  --truth-th 100 \
  --padding 1 --min-run 2 --padding-side both \
  --avoid-merge --min-gap 1 \
  --rebin 4 \
  --mobilenetv3-variant large
```

Parameter knobs (defined at the top of `train4.sh`):

| Block | Knob | Default |
| --- | --- | --- |
| Model | `MODELS` | `mobilenetv3` (or `unet`, `nestedunet`, `mobilenetv2`, `transformer`) |
|  | `MOBILENETV3_VARIANT` | `large` (or `small`) |
|  | `MOBILENETV3_PRETRAINED` | `1` (set to `0` to add `--no-pretrained`) |
| Training | `SEPOCH` / `NEPOCH` | `0` / `50` |
|  | `LR` | `0.1` |
|  | `STRAIN` / `NTRAIN` | `0` / `90` |
|  | `SVAL` / `NVAL` | `90` / `10` |
| Data / mask | `TRUTH_TH` | `100` |
|  | `REBINS` | `4` (space-separated list sweeps multiple values) |
| Padding | `PADDING`, `MIN_RUN`, `PADDING_SIDE`, `AVOID_MERGE`, `MIN_GAP` | `1`, `2`, `both`, `1`, `1` |

If you want to run with your own data path/output dir, copy `train4.sh` into your own workdir, edit the parameter block, and update the hard-coded paths inside `train4.py` (`/nfs/data/1/hnam/train_data_PDHD_fixedbug_separateWC/modified/g4-rec-{i}_modified.h5` and the truth counterpart).

### 2. Outputs

Each run creates a fresh checkpoint directory of the form:

```
chk_mobilenetv3_<YYYYMMDD>_<HHMMSS>/
├── config.json        # mirrors the param block above
├── CP0.pth, CP1.pth, …
├── loss.csv           # per-epoch train+val loss
├── loss-batch.csv     # per-batch loss
├── eval-loss.csv      # per-epoch eval-set loss
├── eval-dice.csv      # per-epoch Dice
├── ep-75-75.csv, …    # per-epoch eff/pur on each eval dataset
└── log
```

The most recent PDHD checkpoint dir is `hnam/Pytorch-UNet/chk_mobilenetv3_20260511_061959/` (May 11, 2026). Many earlier ones (`chk_mobunetv3_*`, `checkpoints_*`) exist under `hnam/Pytorch-UNet/`.

### 3. Switching to PDVD later

The same launcher does it. Pick `2` at the prompt; it runs `train4_pdvd.py`, which reads from `train_data_PDVD_rate2M_window3p2/` by default (with a commented alt line for the small-mpth variant). All other knobs in `train4.sh` apply.

PDVD checkpoints follow the naming `chk_mobunetv3_pdvd_rebin<N>_thr<T>_padding<P>/`.

## Looking at logs

Top-level training logs:
- `hnam/log.txt` — recent training session output (e.g. "PDHD: 531 train batches, 59 val batches").
- `hnam/log_wcpy_02142025.txt` — older session (PDVD variant: 472 train, 118 val batches).

These are stdout dumps; useful for sanity-checking what data shape and batch count to expect.
