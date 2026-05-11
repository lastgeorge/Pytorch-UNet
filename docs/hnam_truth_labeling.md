# Truth labeling in hnam's pipeline

How the per-pixel binary masks fed to the U-Net are built from Geant4 truth.

## Code path

- **PDHD:** `hnam/Pytorch-UNet/utils/h5_utils.py` `get_masks()` (line 141)
- **PDVD:** same file, `get_masks_pdvd()` (line 249) — the two functions share their algorithm; only the HDF5 truth dataset names differ.

The truth HDF5 dataset is `frame_deposplat0` for PDHD and `frame_ductor0` for PDVD. Both store per-pixel deposited charge.

## Pipeline summary

For each event:

1. **Read** the truth charge array `Q[wire, tick]` from the HDF5 file.
2. **Crop** to the training ROI (`crop0`, `crop1`) and **rebin** in the time direction by `REBIN`.
3. **Threshold:** `mask = Q > truth_th`. With `truth_th=100` (default), every pixel below 100 ADC-equivalent units becomes background.
4. **Find contiguous signal runs** per wire/channel row (a "run" is a stretch of consecutive `True` ticks).
5. **Pad each run** that is at least `min_run` ticks long by `padding` ticks on the configured `padding_side`. Optionally enforce `min_gap` between adjacent runs to avoid merging them (`avoid_merge`).
6. Return the binary mask.

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
