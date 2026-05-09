# Model Architectures

This repository implements three encoder-decoder segmentation architectures for LArTPC signal processing. All three share the same interface: they accept multi-channel 2D detector frames and output a per-pixel probability map (sigmoid-activated), where values > 0.5 indicate true particle signal.

---

## UNet (`unet/`)

The classic U-Net from Ronneberger et al. (2015), adapted for LArTPC frames.

### Architecture

5-level encoder-decoder with 4 skip connections.

```
Input (C_in, H, W)
   │
   └─ inconv ─────────────────────────────────────┐
       64 ch                                       │  skip x1
       └─ down1 ────────────────────────────────┐  │
           128 ch                               │  │  skip x2
           └─ down2 ─────────────────────────┐  │  │
               256 ch                        │  │  │  skip x3
               └─ down3 ──────────────────┐  │  │  │
                   512 ch                 │  │  │  │  skip x4
                   └─ down4 ─────────┐   │  │  │  │
                       512 ch        │   │  │  │  │
                       └─ up1 ───────┘   │  │  │  │  (concat x4: 512+512→256)
                           256 ch        │  │  │  │
                           └─ up2 ───────┘  │  │  │  (concat x3: 256+256→128)
                               128 ch       │  │  │
                               └─ up3 ──────┘  │  │  (concat x2: 128+128→64)
                                   64 ch       │  │
                                   └─ up4 ──────┘  │  (concat x1: 64+64→64)
                                       64 ch       │
                                       └─ outconv ──┘
                                           C_out ch
                                           └─ sigmoid
```

### Building Blocks (`unet/parts.py`)

- **`double_conv`**: `Conv2d(3×3, padding=1) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU`
- **`inconv`**: wraps `double_conv` for the input stem
- **`down`**: `MaxPool2d(2×2) → double_conv` — halves spatial resolution
- **`up`**: `Upsample(bilinear, ×2) → pad to skip shape → concat skip → double_conv`; alternatively `ConvTranspose2d` if `bilinear=False`
- **`outconv`**: `Conv2d(1×1)` — maps to output channels

### Key Properties

| Property | Value |
|---|---|
| Depth | 5 levels (4 downsamples) |
| Base filters | 64 |
| Max filters | 512 |
| Downsampling | MaxPool2d |
| Upsampling | Bilinear (default) or ConvTranspose2d |
| Skip connections | 4 (concat) |
| Output activation | `torch.sigmoid` |

---

## UResNet (`uresnet/`)

A shallower variant that replaces double-conv blocks with **pre-activation residual blocks** and uses strided convolutions for downsampling instead of pooling.

### Architecture

4-level encoder-decoder with 3 skip connections.

```
Input (C_in, H, W)
   │
   └─ inconv ──────────────────────────────────┐
       64 ch (residual stem)                    │  skip x1
       └─ down1 (stride-2 residual) ─────────┐  │
           128 ch                            │  │  skip x2
           └─ down2 (stride-2 residual) ──┐  │  │
               256 ch                     │  │  │  skip x3
               └─ down3 (stride-2) ──┐   │  │  │
                   256 ch            │   │  │  │
                   └─ up2 ───────────┘   │  │  │  (concat x3: 256+256→128)
                       128 ch            │  │  │
                       └─ up3 ───────────┘  │  │  (concat x2: 128+128→64)
                           64 ch            │  │
                           └─ up4 ───────────┘  │  (concat x1: 64+64→64)
                               64 ch            │
                               └─ outconv ───────┘
                                   C_out ch
                                   └─ sigmoid
```

### Building Blocks (`uresnet/parts.py`)

- **`residual_block`**: pre-activation style — `BN → ReLU → Conv3×3(stride) → BN → ReLU → Conv3×3` with a `1×1 Conv + BN` projection shortcut. Output = shortcut + main path (no final activation)
- **`inconv`**: two-conv residual stem with 1×1 shortcut (no final ReLU)
- **`down`**: a `residual_block` with `stride=2` — downsampling built into the convolution
- **`up`**: `Upsample(bilinear) → pad → concat skip → residual_block`
- **`outconv`**: `Conv2d(1×1)`

### Key Properties

| Property | Value |
|---|---|
| Depth | 4 levels (3 downsamples) |
| Base filters | 64 |
| Max filters | 256 |
| Downsampling | Strided Conv (in residual block) |
| Upsampling | Bilinear (default) |
| Skip connections | 3 (concat) |
| Output activation | `torch.sigmoid` |

---

## NestedUNet / UNet++ (`nestedunet/`)

An implementation of **UNet++** (Zhou et al., 2018), which replaces single skip connections with dense nested sub-networks connecting every encoder node to every decoder node at the same resolution.

### Architecture

A 5×5 grid of nodes `conv{i}_{j}` where `i` is the depth (0 = shallowest) and `j` is the "column" index. The standard U-Net backbone occupies column 0 (encoder) and the path `i=0, j=1..4` (final decoder). Intermediate nodes form dense shortcuts.

```
Depth 0:  x0_0 ─── x0_1 ─── x0_2 ─── x0_3 ─── x0_4 → output
            │   ↗     │   ↗     │   ↗     │   ↗
Depth 1:  x1_0 ─── x1_1 ─── x1_2 ─── x1_3
            │   ↗     │   ↗     │   ↗
Depth 2:  x2_0 ─── x2_1 ─── x2_2
            │   ↗     │   ↗
Depth 3:  x3_0 ─── x3_1
            │   ↗
Depth 4:  x4_0
```

Each node `conv{i}_{j}` receives:
- All previous nodes on the same row: `x{i}_0, x{i}_1, ..., x{i}_{j-1}` (concatenated via `PadCat`)
- The upsampled output of the node below: `upsample(x{i+1}_{j-1})`

Filter widths: `nb_filter = [32, 64, 128, 256, 512]` — note the base starts at 32 (half of classic UNet).

### Building Blocks (`nestedunet/model.py`)

- **`VGGBlock`**: `Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → ReLU` (no residual connection). Input channels configurable.
- **`PadCat`**: pads all input tensors to match the height and width of the largest tensor, then concatenates along the channel dimension — handles small spatial size mismatches from repeated upsampling.

### Output Modes

| Mode | Output | Sigmoid? |
|---|---|---|
| `deepsupervision=False` (default) | Single output from `x0_4` | Yes |
| `deepsupervision=True` | List of 4 outputs: `x0_1, x0_2, x0_3, x0_4` | No (apply in loss) |

### Key Properties

| Property | Value |
|---|---|
| Depth | 5 levels (4 downsamples) |
| Base filters | 32 |
| Max filters | 512 |
| Downsampling | MaxPool2d (shared) |
| Upsampling | Bilinear (shared) |
| Skip connections | Dense nested (up to 4 per node) |
| Output activation | `torch.sigmoid` (single output mode) |

---

## Architecture Comparison

| Aspect | UNet | UResNet | NestedUNet |
|---|---|---|---|
| Depth (levels) | 5 | 4 | 5 |
| Conv block | double_conv | pre-act residual | VGGBlock |
| Downsampling | MaxPool2d | Strided Conv | MaxPool2d |
| Base filters | 64 | 64 | 32 |
| Max filters | 512 | 256 | 512 |
| Skip connections | 4, single | 3, single | Dense nested |
| Deep supervision | No | No | Optional |
| Output activation | sigmoid | sigmoid | sigmoid / none |
| Parameters (approx.) | ~31M | ~12M | ~9M |

All three accept arbitrary `input_channels` (e.g., 2 or 3) and `output_channels` (number of segmentation classes). The sigmoid output means each output channel is treated independently as a binary signal, which is appropriate for the LArTPC case where a single binary mask is the target.
