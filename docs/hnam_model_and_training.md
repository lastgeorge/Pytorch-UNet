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
