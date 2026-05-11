#!/usr/bin/env bash

set -e

# ============================================================
# GPU setting
# ============================================================
# Uncomment this line if you want to force a specific GPU.
# export CUDA_VISIBLE_DEVICES="0"


# ============================================================
# Detector selection
# ============================================================
echo "============================================================"
echo "Select detector type"
echo "  1) PDHD  -> train4.py"
echo "  2) PDVD  -> train4_pdvd.py"
echo "============================================================"

while true
do
  read -rp "Enter detector type [1/2, default: 1]: " DETECTOR_CHOICE

  if [ -z "${DETECTOR_CHOICE}" ]; then
    DETECTOR_CHOICE="1"
  fi

  if [ "${DETECTOR_CHOICE}" = "1" ]; then
    DETECTOR="PDHD"
    TRAIN_SCRIPT="train4.py"
    break
  elif [ "${DETECTOR_CHOICE}" = "2" ]; then
    DETECTOR="PDVD"
    TRAIN_SCRIPT="train4_pdvd.py"
    break
  else
    echo "Invalid input. Please enter 1 for PDHD or 2 for PDVD."
  fi
done


# ============================================================
# Model selection
# ============================================================
# Available models:
#   unet | uresnet | nestedunet | mobilenetv2 | mobilenetv3 | transformer
MODELS="mobilenetv3"


# ============================================================
# MobileNetV3-specific options
# ============================================================
# These options are used only when MODEL is mobilenetv3.
#
# MobileNetV3 variant:
#   large | small
MOBILENETV3_VARIANT="large"

# Use pretrained weights for MobileNetV3:
#   1 = use pretrained weights
#   0 = add --no-pretrained
MOBILENETV3_PRETRAINED=1


# ============================================================
# Training parameters
# ============================================================
SEPOCH=0
NEPOCH=50
STRAIN=0
NTRAIN=90
SVAL=90
NVAL=10
LR=0.1


# ============================================================
# Data and mask parameters
# ============================================================
TRUTH_TH=100
REBINS="4"          # Example: "4" or "4 6 10"


# ============================================================
# Mask padding parameters
# ============================================================
PADDING=1
MIN_RUN=2
PADDING_SIDE="both"     # both | left | right
AVOID_MERGE=1           # 1 = enable --avoid-merge, 0 = disable
MIN_GAP=1


# ============================================================
# Check training script
# ============================================================
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "Error: ${TRAIN_SCRIPT} was not found."
  exit 1
fi


# ============================================================
# Run training
# ============================================================
for MODEL in ${MODELS}
do
  for REBIN in ${REBINS}
  do
    # ------------------------------------------------------------
    # Build optional flags
    # ------------------------------------------------------------
    AM_FLAG=""
    if [ "${AVOID_MERGE}" -eq 1 ]; then
      AM_FLAG="--avoid-merge"
    fi

    MOBILENETV3_FLAGS=""
    if [ "${MODEL}" = "mobilenetv3" ]; then
      MOBILENETV3_FLAGS="--mobilenetv3-variant ${MOBILENETV3_VARIANT}"

      if [ "${MOBILENETV3_PRETRAINED}" -eq 0 ]; then
        MOBILENETV3_FLAGS="${MOBILENETV3_FLAGS} --no-pretrained"
      fi
    fi

    # ------------------------------------------------------------
    # Print run information
    # ------------------------------------------------------------
    echo "============================================================"
    echo "Detector                : ${DETECTOR}"
    echo "Training script          : ${TRAIN_SCRIPT}"
    echo "Model                   : ${MODEL}"
    echo "Rebin                   : ${REBIN}"
    echo "Truth threshold          : ${TRUTH_TH}"
    echo "Learning rate            : ${LR}"
    echo "Start epoch              : ${SEPOCH}"
    echo "Number of epochs         : ${NEPOCH}"
    echo "Training start index     : ${STRAIN}"
    echo "Training samples         : ${NTRAIN}"
    echo "Validation start index   : ${SVAL}"
    echo "Validation samples       : ${NVAL}"
    echo "Padding                  : ${PADDING}"
    echo "Minimum run length       : ${MIN_RUN}"
    echo "Padding side             : ${PADDING_SIDE}"
    echo "Avoid merge              : ${AVOID_MERGE}"
    echo "Minimum gap              : ${MIN_GAP}"

    if [ "${MODEL}" = "mobilenetv3" ]; then
      echo "MobileNetV3 variant      : ${MOBILENETV3_VARIANT}"
      echo "MobileNetV3 pretrained   : ${MOBILENETV3_PRETRAINED}"
    fi

    echo "============================================================"

    # ------------------------------------------------------------
    # Run command
    # ------------------------------------------------------------
    echo "[CMD] python ${TRAIN_SCRIPT} --gpu --model ${MODEL} --rebin ${REBIN}"

    time python "${TRAIN_SCRIPT}" \
      --gpu \
      --model "${MODEL}" \
      --start-epoch "${SEPOCH}" \
      --nepoch "${NEPOCH}" \
      --start-train "${STRAIN}" \
      --ntrain "${NTRAIN}" \
      --start-val "${SVAL}" \
      --nval "${NVAL}" \
      --learning-rate "${LR}" \
      --truth-th "${TRUTH_TH}" \
      --padding "${PADDING}" \
      --min-run "${MIN_RUN}" \
      --padding-side "${PADDING_SIDE}" \
      ${AM_FLAG} \
      --min-gap "${MIN_GAP}" \
      --rebin "${REBIN}" \
      ${MOBILENETV3_FLAGS}

  done
done
