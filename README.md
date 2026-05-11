# Pytorch-UNet For LArTPC Signal Processing

Original Repository: [here](https://github.com/milesial/Pytorch-UNet)

Example usage in `train.sh` and `predict.sh`

## install

### use `conda`
prerequisite: conda
https://docs.anaconda.com/free/anaconda/install/linux/

use environment.yml
```
conda env create -f environment.yml
```

manually
```
conda create --name pt110 python=3.9 numpy
conda activate pt110
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install matplotlib
pip install h5py
```

### use `pip`

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## talks:
 - [ProtoDUNE DRA Meeting, Jan. 08 2020](https://indico.fnal.gov/event/22795/contribution/4)
 - [DUNE Collaboration Meeting, Jan. 29 2020](https://indico.fnal.gov/event/20144/session/8/contribution/98)

## notes
```bash
h5dump-shared -n data/g4-rec-r9.h5
./scripts/h5plot.py data/g4-rec-r9.h5 /100/frame_loose_lf0
./scripts/h5plot.py data/g4-rec-r9.h5 /100/frame_mp3_roi0
./scripts/h5plot.py data/g4-rec-r9.h5 /100/frame_mp2_roi0
./scripts/h5plot.py data/g4-tru-r9.h5 /103/frame_ductor0
./scripts/h5plot.py data/g4-rec-r9.h5 /103/frame_gauss0
./train3.sh
python plot_epoch.py 1
./to-ts.py -m test0/CP49.pth
```

ts to pth
```bash
./to-pth.py -m ts-model/unet-l23-cosmic500-e50.ts -o pth-model/unet-l23-cosmic500-e50.pth
./to-pth.py -m ts-model/nestedunet-l23-cosmic500-e50.ts -o pth-model/nestedunet-l23-cosmic500-e50.pth -t nestedunet
./to-pth.py -m ts-model/unet-lt-cosmic500-e50.ts -o pth-model/unet-lt-cosmic500-e50.pth -i 2
```
pth to ts
```bash
./to-ts.py -m pth-model/unet-l23-cosmic500-e50.pth -o ts-model-2.3/unet-l23-cosmic500-e50.ts
./to-ts.py -m pth-model/nestedunet-l23-cosmic500-e50.pth -o ts-model-2.3/nestedunet-l23-cosmic500-e50.ts -t nestedunet
./to-ts.py -m pth-model/unet-lt-cosmic500-e50.pth -o ts-model-2.3/unet-lt-cosmic500-e50.ts -i 2
```

## Updated training and TorchScript conversion scripts

This branch adds updated scripts for model selection, PDHD/PDVD training, and TorchScript conversion.

- `train4.py`: PDHD training script with command-line DNN model selection
- `train4_pdvd.py`: PDVD training script with command-line DNN model selection
- `train4.sh`: interactive launcher for selecting PDHD or PDVD training
- `to-ts4.py`: updated PyTorch checkpoint to TorchScript conversion script with MobileNetV3 and Transformer support

## Training with `train4.sh`

Run:

```bash
./train4.sh
```

The script asks which detector configuration to use:

```bash
1) PDHD  -> train4.py
2) PDVD  -> train4_pdvd.py
```

Selecting `PDHD` runs `train4.py`, while selecting `PDVD` runs `train4_pdvd.py`.

Training parameters can be edited directly in `train4.sh`, including:

```bash
MODELS="mobilenetv3"
REBINS="4"
NEPOCH=50
TRUTH_TH=100
PADDING=1
MIN_RUN=2
PADDING_SIDE="both"
AVOID_MERGE=1
MIN_GAP=1
```

Available DNN models are:

```bash
unet
uresnet
nestedunet
mobilenetv2
mobilenetv3
transformer
```

For MobileNetV3, the following options are separated in `train4.sh`:

```bash
MOBILENETV3_VARIANT="large"
MOBILENETV3_PRETRAINED=1
```

## Manual training examples

PDHD example:

```bash
python train4.py --gpu \
  --model mobilenetv3 \
  --mobilenetv3-variant large \
  --start-epoch 0 \
  --nepoch 50 \
  --start-train 0 \
  --ntrain 90 \
  --start-val 90 \
  --nval 10 \
  --learning-rate 0.1 \
  --truth-th 100 \
  --padding 1 \
  --min-run 2 \
  --padding-side both \
  --avoid-merge \
  --min-gap 1 \
  --rebin 4
```

PDVD example:

```bash
python train4_pdvd.py --gpu \
  --model mobilenetv3 \
  --mobilenetv3-variant large \
  --start-epoch 0 \
  --nepoch 50 \
  --start-train 0 \
  --ntrain 90 \
  --start-val 90 \
  --nval 10 \
  --learning-rate 0.1 \
  --truth-th 100 \
  --padding 1 \
  --min-run 2 \
  --padding-side both \
  --avoid-merge \
  --min-gap 1 \
  --rebin 4
```

## Convert PyTorch checkpoint to TorchScript with `to-ts4.py`

`to-ts4.py` converts a trained PyTorch checkpoint into a TorchScript model.

Example for MobileNetV3:

```bash
python to-ts4.py \
  --model chk_mobunetv3_pdvd_20260421_234631/CP37.pth \
  --arch mobilenetv3 \
  --mv3-variant large \
  --mv3-pretrained \
  --output pdvd-test-CP37_rebin5_thr100_padding1_mpth.ts
```

In this script:

```bash
--model          path to the trained checkpoint file
--arch           model architecture, such as unet, uresnet, nestedunet, mobilenetv2, mobilenetv3, or transformer
--mv3-variant    MobileNetV3 variant, either large or small
--mv3-pretrained use pretrained MobileNetV3 backbone
--output         output TorchScript file path
```

Example for Transformer U-Net:

```bash
python to-ts4.py \
  --model chk_transformer_YYYYMMDD_HHMMSS/CP37.pth \
  --arch transformer \
  --output transformer-CP37.ts
```
