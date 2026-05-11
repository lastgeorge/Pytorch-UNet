# Truth labeling in hnam's pipeline

How the per-pixel binary masks fed to the U-Net are built from Geant4 truth.

## Where in the pipeline does labeling actually happen?

**Not in `convert2.py`** — a common point of confusion. The mask construction is **online**, inside the training data loader. The full chain is four layers:

| Step | Where | When | Role for the truth |
| --- | --- | --- | --- |
| 1. Geant4 + Wire-Cell sim | `hnam/wct-dev-hnam` (not captured locally) | one-off | Writes per-pixel deposited-charge frames `frame_deposplat0` (PDHD) / `frame_ductor0` (PDVD) into `g4-tru-*.h5`. |
| 2. **Channel-kill augmentation (offline)** | `hnam/train_data_PDHD_fixedbug_separateWC/convert2.py` | run once to materialize `modified/` | Zeroes `frame_deposplat0` at the augmented channels for the one augmented event per file (so the truth says "no signal" exactly where the input says "noisy channel"). Everything else is copied through unchanged. **No thresholding, no padding here** — the output is still continuous charge. |
| 3. **Threshold + run-length pad (online)** | `hnam/Pytorch-UNet/utils/h5_utils.py` `get_masks()` (line 141, PDHD); `get_masks_pdvd()` (line 249, PDVD) | per batch during training | Reads the (possibly-augmented) charge truth, crops + rebins to match the image tensor, thresholds at `truth_th`, finds contiguous per-wire tick-runs, pads each run, optionally enforces a gap between neighbours. **This is the labeling.** |
| 4. Launcher | `hnam/Pytorch-UNet/train4.sh` | shell-level | Just *exports the labeling parameters* (`TRUTH_TH`, `PADDING`, `MIN_RUN`, `PADDING_SIDE`, `AVOID_MERGE`, `MIN_GAP`) and forwards them to `train4.py`, which threads them into the `get_masks()` calls at `train4.py:230, 234, 242`. |

**Practical consequence:** a single set of `g4-tru-*_modified.h5` files trains many models with different threshold / padding choices — the offline step does not have to be re-run when the labeling parameters change. Just edit `train4.sh` and relaunch.

The image side has the same offline-vs-online split: `convert2.py` injects noise into `frame_loose_lf0` offline; `get_chw_imgs()` builds the input tensor online. See [hnam_input_preprocessing.md](./hnam_input_preprocessing.md) for the image side.

## Code path (online labeling)

- **PDHD:** `hnam/Pytorch-UNet/utils/h5_utils.py` `get_masks()` (line 141)
- **PDVD:** same file, `get_masks_pdvd()` (line 249) — the two functions share their algorithm; only the HDF5 truth dataset names differ.

The truth HDF5 dataset is `frame_deposplat0` for PDHD and `frame_ductor0` for PDVD. Both store per-pixel deposited charge.

## Pipeline summary (what `get_masks()` does per event)

Executed inside the training loop, on each batch:

1. **Read** the truth charge array `Q[wire, tick]` from the `modified/` HDF5 file (so the channel-kill zeroing from step 2 of the chain above is already baked in).
2. **Rebin** in the time direction by `REBIN` (mean-pool, same factor as the image side) and **crop** to the training ROI (`crop0` = wire range, `crop1` = tick range).
3. **Threshold:** `mask = Q > truth_th`. With `truth_th=100` (default), every pixel below 100 ADC-equivalent units becomes background.
4. **Find contiguous signal runs** per wire row (a "run" is a stretch of consecutive `True` ticks). Implemented by the `_find_runs()` helper using `np.diff` on the row.
5. **Pad each run** that is at least `min_run` ticks long by `padding` ticks on the configured `padding_side` (`both` / `left` / `right`). If `avoid_merge` is set, shrink the padding so neighbouring runs stay separated by at least `min_gap` ticks.
6. Return the binary mask, shape `(wire_crop, tick_crop / REBIN)`, dtype `float32`.

## Parameters (as wired in `train4.sh`)

These are the defaults you'll inherit if you run `./train4.sh` unmodified:

| Param | Default | Meaning |
| --- | ---: | --- |
| `TRUTH_TH` | `100` | charge threshold for `Q > th → signal` |
| `REBINS` | `4` | time-direction rebin factor (one value or a space-separated list to sweep) |
| `PADDING` | `1` | ticks to extend each run on the chosen side(s) |
| `MIN_RUN` | `2` | minimum run length (in rebinned ticks) eligible for padding |
| `PADDING_SIDE` | `both` | `both` / `left` / `right` |
| `AVOID_MERGE` | `1` | if 1, padding stops short to keep `min_gap` between runs |
| `MIN_GAP` | `1` | minimum gap (in ticks) between two separate signal runs after padding |

The same parameters are recorded in each run's `config.json`, e.g. `hnam/Pytorch-UNet/chk_mobilenetv3_20260511_061959/config.json`:

```json
{
  "model": "mobilenetv3",
  "rebin": 4,
  "truth_th": 100,
  "padding": 1,
  "min_run": 2,
  "padding_side": "both",
  "avoid_merge": true,
  "min_gap": 1
}
```

## Algorithm sketch

```python
# pseudo-code distilled from get_masks() / get_masks_pdvd()
for row in mask:
    intervals = find_runs(row)              # list of (lo, hi)
    for j, (lo, hi) in enumerate(intervals):
        if hi - lo + 1 < min_run:
            continue
        pad_lo = lo - padding if padding_side in ('left',  'both') else lo
        pad_hi = hi + padding if padding_side in ('right', 'both') else hi
        if avoid_merge:
            # clamp pad_lo so prev_hi + min_gap <= pad_lo
            # clamp pad_hi so pad_hi + min_gap <= next_lo
        row[pad_lo:pad_hi+1] = True
```

`_find_runs` is a small helper inside both functions.

## Upstream truth (Geant4 → HDF5)

The "truth tagging" step that turns Geant4 energy deposits into the `frame_deposplat0` / `frame_ductor0` HDF5 datasets happens **before** this file — inside the Wire-Cell simulation chain (`hnam/wct-dev-hnam`, `hnam/wire-cell-hnam`). The exact jobsub / configuration that produced today's training files is **not** captured locally — see [hnam_overview.md](./hnam_overview.md#simulation-generation).

After Geant4/WC writes the raw `g4-tru-*.h5` files, `convert2.py` rewrites them into `modified/g4-tru-*_modified.h5` with the channel-kill zeros applied (step 2 of the chain at the top of this doc). Those zeros propagate naturally through the threshold + run-length pass — a zeroed channel has no `Q > truth_th` pixels, so it generates no positive labels and the network learns to expect no signal there. See [hnam_input_preprocessing.md](./hnam_input_preprocessing.md#stage-a--convert2py-channel-kill-augmentation-pdhd-only) for the augmentation details.
