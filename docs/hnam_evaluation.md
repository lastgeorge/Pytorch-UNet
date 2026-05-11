# hnam evaluation code and metrics

## Evaluation entry points

All under `hnam/Pytorch-UNet/`:

| Script | Purpose |
| --- | --- |
| `eval.py` | Main offline evaluation. Loads a checkpoint (`.pth` or `.ts`), runs over hard-coded eval datasets, writes per-event CSV to `out-eval/`. |
| `eval-epoch.py` | Sweeps across all `CP*.pth` in a checkpoint dir; useful for "best epoch" curves. |
| `eval_loss_sample.py` + `eval_loss_sample.sh` | Variant that also reports the loss alongside metrics. |
| `eval_util.py` | Holds the actual metric implementations (called by all the above). |
| `predict.py` | Inference only — produces output masks/plots, no metrics. |
| `eval.sh` | Example launcher (historical, references older `unet-l23-cosmic500-e50` etc. checkpoints under `model/<name>/CP50-450.pth`). |

`eval.py` CLI (most-used flags):

```
--model/-m   <path.pth or path.ts>
--input/-i   <h5 file(s)>          # optional, otherwise uses hard-coded list (see below)
--output/-o  <name>                # used to name the CSV in out-eval/
--gpu/-g
--mask-threshold/-t <float>        # default 0.5 binarization threshold on model output
--scale/-s   <float>               # input scaling
```

## Metrics implemented (`eval_util.py`)

`eval_eff_pur(net, dataset, th=0.5, gpu)` returns four numbers, averaged over the events in `dataset`:

| Symbol | Meaning |
| --- | --- |
| `eff_pix` | **Pixel-level efficiency** — fraction of truth-signal pixels that the model also flags. |
| `pur_pix` | **Pixel-level purity** — fraction of predicted-signal pixels that are actually truth-signal. |
| `eff_roi` | **ROI-level efficiency** — fraction of truth ROIs that contain at least one predicted-signal pixel. |
| `pur_roi` | **ROI-level purity** — fraction of predicted ROIs that contain at least one truth-signal pixel. |

Other helpers in the same file:

- `eval_dice` — Dice coefficient on the binary masks.
- `eval_loss` — re-runs the loss function for sanity check.
- `eval_roi`, `eval_pixel` — building blocks the higher-level metrics call.

The four eff/pur numbers are appended per epoch into `ep-<label>.csv` files inside each checkpoint dir (e.g. `chk_mobilenetv3_20260511_061959/ep-75-75.csv`).

## NOT implemented locally: charge bias, completeness

A grep for `charge.bias`, `q_bias`, `charge_ratio`, `completeness` across `hnam/Pytorch-UNet/` returns **no matches**. These metrics — which the user's question list specifically asks about — are presumably computed downstream once the DNN-ROI output has been processed back into a `frame_dnnspN` tag inside Wire-Cell. **Open question for hnam, captured in [hnam_overview.md](./hnam_overview.md#evaluation).**

## Eval datasets

`eval.py` hard-codes paths of the form:

```python
eval_imgs.append('eval-jinst-init-sub/eval-' + label + '/g4-rec-0.h5')
eval_masks.append('eval-jinst-init-sub/eval-' + label + '/g4-tru-0.h5')
```

with `label ∈ {75-75, 87-85, 80-80, 82-82, 87-87}`. These directories **do not exist** under `hnam/Pytorch-UNet/` today (the older `eval.sh` references `model/<name>/CP50-450.pth` checkpoints that also aren't here). Treat these as test sets we still need from hnam — flagged in the overview.

## Outputs

Per-event CSVs land in:

- `out-eval/<output-name>.csv` (when invoking `eval.py` directly), or
- `<checkpoint-dir>/ep-<label>.csv` (when invoked as part of training, e.g. via `eval-epoch.py`).

Both follow the pattern `event_id, eff_pix, pur_pix, eff_roi, pur_roi`.

## Quick-start (when datasets are in place)

```bash
cd hnam/Pytorch-UNet
conda activate dnnroi
python eval.py -g \
  --model chk_mobilenetv3_20260511_061959/CP0.pth \
  --output mobilenetv3-CP0
# Results land in out-eval/mobilenetv3-CP0.csv
```

For TorchScript instead of `.pth`, just point `--model` at the `.ts` file — `eval.py` detects the suffix and uses `torch.jit.load`.
