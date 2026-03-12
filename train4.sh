#!/bin/bash

#export CUDA_VISIBLE_DEVICES="0"

# ====== Params ======
SEPOCH=0
NEPOCH=50
STRAIN=0
NTRAIN=90
SVAL=90
NVAL=10
LR=0.1

TRUTH_TH=100
PADDING=1
MIN_RUN=2
PADDING_SIDE="both"     # both | left | right
AVOID_MERGE=1           # 1(True) / 0(False)
MIN_GAP=1
REBINS="4"        # 4 | 6 | 10


# ====== Run ======
AM_FLAG=""
if [[ "${AVOID_MERGE}" -eq 1 ]]; then
  AM_FLAG="--avoid-merge"
fi

#for TRUTH_TH in $TRUTH_THS
for REBIN in $REBINS
do
  echo "Run w/ truth_th: ${TRUTH_TH}"
  echo "Run w/ rebin: ${REBIN}"
  time python train4.py --gpu \
    --start-epoch "${SEPOCH}" --nepoch "${NEPOCH}" \
    --start-train "${STRAIN}" --ntrain "${NTRAIN}" \
    --start-val "${SVAL}" --nval "${NVAL}" \
    --learning-rate "${LR}" --truth-th "${TRUTH_TH}" \
    --padding "${PADDING}" --min-run "${MIN_RUN}" \
    --padding-side "${PADDING_SIDE}" ${AM_FLAG} --min-gap "${MIN_GAP}" \
    --rebin "${REBIN}"
done
