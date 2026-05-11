# Train ↔ deploy tick-count compatibility

**Question.** Training fixes the input at 6000 raw ticks (1500 after `rebin=4`). At deployment the readout window may differ — e.g. 8000 ticks (a clean multiple of `rebin`), or something slightly off like 5999. How does this work?

**Short answer.** The MobileNetV3-UNet is fully convolutional, so the *model itself* is shape-generic. Whether the deployed `.ts` file inherits that genericity depends on whether it was exported with `torch.jit.script` (yes) or fell through to `torch.jit.trace` (maybe — verify). On the Wire-Cell side, `nticks` and `tick_per_slice` in the Jsonnet config decide what shape reaches the model; the C++ component does **not** auto-pad to a multiple of `tick_per_slice`, so a non-divisible tick count needs an explicit crop or pad in the WC config.

---

## 1. Training-time shape is fixed

The online loader produces tensors of shape `(3, 1600, 1500)`:

- `train4.py:216` — `x_range = [0, 1600]` crops to U+V planes.
- `train4.py:212` — `y_range = [0, y_range_dict.get(rebin_factor, 600)]`; for the current PDHD config `rebin_factor=4` → `y_range=[0, 1500]`.
- `utils/h5_utils.py:66` — `rebin()` mean-pools every `REBIN` ticks. With raw 6000 / 4 = 1500.

Full pipeline in [hnam_input_preprocessing.md](./hnam_input_preprocessing.md). What follows assumes that doc as context.

---

## 2. The model is mathematically shape-generic

`hnam/Pytorch-UNet/mobilenetv3/model.py` defines the MobileNetV3-large + U-Net used by the current PDHD checkpoint:

- **No `Linear` / `Flatten` / `AdaptiveAvgPool`** in the forward path — pure conv / batchnorm / activation.
- **Skip alignment is interpolation-based**, not crop-based:

  ```python
  # mobilenetv3/model.py:47  (UpBlock)
  if x.shape[-2:] != skip.shape[-2:]:
      x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
  ```

  So odd downsamples don't break skip concatenation.

- **Output is restored to input size** by interpolation:

  ```python
  # mobilenetv3/model.py:163
  x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
  ```

- **Soft constraint**: MobileNetV3 has 5 stride-2 stages, so each spatial dim must survive 5 halvings (≥ 32). 1600 wires and any tick count ≥ ~50 satisfy this comfortably.

Conclusion: the network in PyTorch can take any `(1, 3, H, T)` with `H, T ≥ 32` and produce `(1, 1, H, T)` back. No hidden fixed-size assumption.

---

## 3. Does the exported `.ts` keep that genericity?

Depends on how it was exported. `hnam/Pytorch-UNet/to-ts_my4.py` uses:

```python
# to-ts_my4.py:113
def try_script_then_trace(net, example, use_cuda):
    try:
        sm = torch.jit.script(net)        # preferred — shape-generic graph
    except Exception:
        sm = torch.jit.trace(net, ...)    # fallback — records example shape
```

- **If scripting succeeded** → the `.ts` carries the Python control flow and is genuinely shape-generic.
- **If it fell through to tracing** → the trace records the example shape (`--height`/`--width`, default `800 × 600`). PyTorch will warn ("Tensor-shape-dependent control flow") when you feed a different shape; the call **often** still runs for fully-conv graphs but this is *not guaranteed*.

You can tell after the fact:

```python
import torch
m = torch.jit.load("path/to/model.ts", map_location="cpu")
print(type(m).__name__)            # ScriptModule or RecursiveScriptModule
print(m.original_name)             # 'MobileNetV3_UNet' if scripted; '<traced>' if traced
print(m.code[:500])                # scripted modules carry Python source; traced ones show a flat graph
```

If you find the deployed `.ts` was traced and you need a different shape, the cleanest fix is to re-export with the deployment shape:

```bash
python to-ts_my4.py \
  --model chk_mobilenetv3_<…>/CP<…>.pth \
  --arch  mobilenetv3 --mv3-variant large \
  --input-ch 3 --output-ch 1 \
  --height 1600 --width 2000 \
  --output mobilenetv3_pdhd_8000ticks.ts
```

A 10-line smoke test you can run on any candidate `.ts`:

```python
import torch
m = torch.jit.load("path/to/model.ts", map_location="cpu").eval()
for H, T in [(1600, 1500), (1600, 2000), (1600, 1499)]:
    try:
        y = m(torch.zeros(1, 3, H, T))
        print(H, T, "->", tuple(y.shape))
    except Exception as e:
        print(H, T, "FAIL:", type(e).__name__, str(e)[:120])
```

---

## 4. How deployment picks the tick count

In Wire-Cell, the DNN-ROI component (`hnam/wct-dev-hnam/toolkit/pytorch/src/DNNROIFinding.cxx`) reads three knobs from config:

| Knob | Where | Default | Meaning |
| --- | --- | ---: | --- |
| `tick0` | `DNNROIFinding.h:54` | 0 | start of the raw-tick window taken from the frame |
| `nticks` | `DNNROIFinding.h:55` | 6000 | length of the raw-tick window |
| `tick_per_slice` | `DNNROIFinding.h:79` | 10 | downsample factor before the model (≡ training `rebin`) |

Pipeline per call:

```
frame[tick0 : tick0+nticks]                       # raw window, shape (C_apa, nticks)
  → Array::downsample(arr, tick_per_slice, 1)     # DNNROIFinding.cxx:250
  → tensor shape (1, ntags, nchannels, nticks/tick_per_slice)
  → m_forward->forward(...)                       # DNNROIFinding.cxx:271 — calls the .ts model
  → Array::upsample(out, tick_per_slice, 0)       # DNNROIFinding.cxx:282
  → ROI mask back at raw tick resolution
```

Two things to note:

- **No sliding window in time.** The full `nticks` window goes through in one (or `nchunks`-many) forward pass(es), where `nchunks` only chunks the **channel** axis (`DNNROIFinding.cxx:263`). If `nticks=8000, tick_per_slice=4`, the model receives `(1, 3, 1600, 2000)` in one shot.
- **No auto-pad to a multiple of `tick_per_slice`.** `Array::downsample` produces `floor(nticks / tick_per_slice)` rows. Anything not divisible is silently truncated.

The Jsonnet entry point that exposes these is `wire-cell-hnam/cfg/wire-cell-cfg/pgrapher/experiment/pdhd/dnnroi.jsonnet:14`:

```jsonnet
function (anode, ts, prefix="dnnroi", output_scale=1.0,
          nticks=6000, tick_per_slice=10, nchunks=1)
```

---

## 5. The three scenarios concretely

| Detector tick count | Recommended WC config | Model sees | Verdict |
| --- | --- | --- | --- |
| **6000** (training match) | `nticks=6000, tick_per_slice=4` | `(1, 3, 1600, 1500)` | Just works. |
| **8000** (rebin-aligned) | `nticks=8000, tick_per_slice=4` | `(1, 3, 1600, 2000)` | Works **iff** the `.ts` is shape-generic. Run the smoke test in §3 once; if scripted, you're done. If traced and it fails, re-export with `--height 1600 --width 2000`. |
| **5999** (not a multiple of 4) | `nticks=5996, tick_per_slice=4` (truncate to 1499) **or** pad the raw frame to 6000 upstream and use `nticks=6000` | `(1, 3, 1600, 1499)` or `(1, 3, 1600, 1500)` | Works after explicit crop/pad. The C++ does NOT auto-pad — you'll lose the last 3 ticks silently if you set `nticks=5999, tick_per_slice=4`. |

Generalising: pick `nticks = floor(raw_ticks / rebin) * rebin` (largest multiple of `tick_per_slice` ≤ what the frame holds), and the `.ts` will run as long as it's shape-generic.

---

## 6. Critical config alignment gotcha

The Jsonnet default `tick_per_slice = 10` is the legacy DNN-SP value. **The current PDHD MobileNetV3 model was trained with `rebin = 4`** (per `chk_mobilenetv3_20260511_061959/config.json`). You must override it:

```jsonnet
dnnroi(tools.anodes[n], ts,
       nticks = params.daq.nticks,
       tick_per_slice = 4,                   // <-- not 10
       nchunks = 1)
```

If you forget, the model still runs (the shapes work out) but every input is downsampled 10× instead of 4× — the result is silently wrong. Cross-check: the old rebin=10 checkpoints (`torchscript3/CP49_mobileunetv3_rebin10_*.ts`) coexist with rebin=4 exports; pair each `.ts` with the matching `tick_per_slice`.

---

## 7. Channel dimension — no user action needed

The wire-axis count is computed deployment-side from the anode geometry: `m_chlist = Aux::plane_channels(anode, plane)`. For PDHD that yields 800 U + 800 V (= 1600) per APA, identical to the training crop. No knob to set, no risk of mismatch.

---

## 8. Open questions for hnam

- Was the deployed PDHD `.ts` exported via `torch.jit.script` (preferred) or did `try_script_then_trace` fall through to `torch.jit.trace`?
- In the production `wcls-rawdigit-dnnsp.jsonnet`, what `nticks` and `tick_per_slice` are actually set? (Defaults `6000` / `10` do **not** match training `rebin=4`.)
- Is the ProtoDUNE-HD readout window guaranteed to be 6000 ticks, or does it vary with data-taking mode? If it varies, is the variation always rebin-aligned?
- Same questions for PDVD (out of scope here, same machinery applies).
