# hnam deployment: TorchScript export + Wire-Cell integration (PDHD)

How a trained `.pth` checkpoint becomes a model loaded by the official Wire-Cell processing chain on PDHD.

## Step 1 — Convert `.pth` → `.ts` (TorchScript)

Conversion scripts (newest first):

| Script | Notes |
| --- | --- |
| `hnam/Pytorch-UNet/to-ts_my4.py` | Current; supports all architectures (`unet`, `nestedunet`, `mobilenetv2`, `mobilenetv3`, `transformer`); falls back from `torch.jit.script` to `torch.jit.trace` if scripting fails. |
| `hnam/Pytorch-UNet/to-ts_my3.py`, `to-ts_my2.py`, `to-ts_my.py` | Earlier versions, kept for reference. |
| `hnam/Pytorch-UNet/to-ts4.py`, `to-ts.py` | Simpler variants. |

`to-ts_my4.py` flags:

```
--model/-m   <path.pth>                (required)
--arch/-a    {unet|nestedunet|mobilenetv2|mobilenetv3|transformer}
--gpu/-g
--output/-o  <path.ts>                 (default: derived from model name)
--input-ch   <int>                     (default 3)
--output-ch  <int>                     (default 1)
--height     <int>                     (default 800)
--width      <int>                     (default 600)
--mv3-variant {large|small}            (default large)
--mv3-pretrained                       (only matters for instantiating the wrapper)
--tr-embed-dim / --tr-heads / …        (transformer-specific)
```

Example command (PDHD MobileNetV3):

```bash
cd hnam/Pytorch-UNet
conda activate dnnroi
python to-ts_my4.py \
  --model chk_mobilenetv3_20260511_061959/CP0.pth \
  --arch  mobilenetv3 \
  --mv3-variant large \
  --input-ch 3 --output-ch 1 \
  --height 800 --width 600 \
  --output mobilenetv3_pdhd_rebin4_thr100_pad1.ts
```

### Existing exports

`hnam/Pytorch-UNet/torchscript3/` already contains ~21 MB `.ts` files matching the various rebin/threshold/padding sweeps, e.g.:

- `CP49_mobileunetv3_rebin10_lr0p1_thr100_padding0_UV.ts`
- `CP49_mobileunetv3_rebin10_lr0p1_thr10_padding0_UV.ts`
- `CP49_mobileunetv3_rebin4_lr0p1_thr100_padding0_UV.ts`
- `CP38_mobileunetv3_rebin6_lr0p1_thr100_padding1_UV.ts`
- … (rebin 1/4/6/10 × thr 10/50/100 × padding 0/1)

Plus a few one-off models in `hnam/Pytorch-UNet/` itself (e.g. `CP47_mobileunetv3_rebin10_lr0p1_thr100_padding0_NF_UV.ts`, `pdvd-test-CP37_rebin5_thr100_padding1_mpth.ts`).

**Open question for hnam:** which of these is the "blessed" production model for PDHD? (Captured in [hnam_overview.md](./hnam_overview.md#models).)

## Step 2 — Wire-Cell C++ loader

The C++ components live in hnam's Wire-Cell Toolkit fork:

| Role | Path |
| --- | --- |
| TorchScript wrapper (`ITensorSetFilter`) — header | `hnam/wct-dev-hnam/toolkit/pytorch/inc/WireCellPytorch/TorchScript.h` |
| TorchScript wrapper — impl | `hnam/wct-dev-hnam/toolkit/pytorch/src/TorchScript.cxx` (calls `torch::jit::load(model_path, at::kCUDA)`) |
| DNN-ROI node — header | `hnam/wct-dev-hnam/toolkit/pytorch/inc/WireCellPytorch/DNNROIFinding.h` |
| DNN-ROI node — impl | `hnam/wct-dev-hnam/toolkit/pytorch/src/DNNROIFinding.cxx` |

`DNNROIFinding` is the per-plane (U/V/W) consumer of the model: it takes signal-processed traces, builds the input tensor, runs through the `TorchScript` forward, and writes the result back as new traces tagged `dnnspN`.

## Step 3 — Jsonnet wiring (PDHD)

Under `hnam/wire-cell-hnam/cfg/wire-cell-cfg/pgrapher/experiment/pdhd/`:

| File | Role |
| --- | --- |
| `dnnroi.jsonnet` | Builder function for one DNN-ROI node per anode. Signature: `function (anode, ts, prefix="dnnroi", output_scale=1.0, nticks=6000, tick_per_slice=10, nchunks=1)`. The `ts` argument is the `TorchScript` component to share across anodes. |
| `wcls-rawdigit-dnnsp.jsonnet` | Full PDHD processing flow: noise filter → SP (Gauss/Wiener) → DNN-ROI; emits frame tag `dnnspN`. |
| `wcls-rawdigit-sp.jsonnet` | Equivalent flow without DNN — useful as a baseline diff. |
| `sp.jsonnet`, `nf.jsonnet`, `chndb-resp.jsonnet`, `sp-filters.jsonnet`, `params.jsonnet` | Building blocks called by the above. |

The `output_scale`, `nticks`, `tick_per_slice`, `nchunks` knobs in `dnnroi.jsonnet` are the main run-time levers.

## Step 4 — LArSoft FCL drop-in

The production handoff to PDHD reconstruction is an FCL file used by LArSoft jobs. Examples under `hnam/wire-cell-hnam/pdhd-data/`:

```
util/standard_reco_stage2_calibration_protodunehd_keepup_dnnroi.fcl
util/standard_reco_stage2_calibration_protodunehd_keepup_dnnroi_ori.fcl
util/standard_reco_stage2_calibration_protodunehd_keepup_dnnroi_pandorareco.fcl
util/my_standard_reco_stage2_calibration_protodunehd_keepup_dnnroi.fcl
util/my_standard_reco_stage2_calibration_protodunehd_keepup_dnnroi_pandorareco.fcl
```

Per-test drops live in `pdhd-data/test0519/`, `test0523/`, `test0924/`, each under a `dnnroi_sp/` (and sometimes `dnnroi_sp_pandorareco/`) subdir. These are the most concrete examples of "model checked into the official chain" — they bundle the FCL config and presumably reference the `.ts` model and the Wire-Cell job config.

**Open question for hnam:** confirm which test directory (and which `.ts` file inside) is the recipe to follow for a current PDHD deployment. ([hnam_overview.md](./hnam_overview.md#deployment-recipe).)

## End-to-end skeleton

```
.pth checkpoint  ──to-ts_my4.py──>  .ts file
                                       │
                                       v
                         Wire-Cell job (Jsonnet)
                         dnnroi.jsonnet wires a TorchScript node + DNNROIFinding
                         wcls-rawdigit-dnnsp.jsonnet stitches the full PDHD flow
                                       │
                                       v
                         LArSoft job (FCL)
                         standard_reco_stage2_calibration_protodunehd_keepup_dnnroi.fcl
                                       │
                                       v
                         Reco output with `recob::Wire` "dnnsp" stream
```
