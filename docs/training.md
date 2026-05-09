# Training

## Overview

Three training scripts are provided, each building on the previous with additional features for controlling data splits, resuming from checkpoints, and handling multiple input files.

| Script | Dataset | Split method | Checkpoint naming | Use case |
|---|---|---|---|---|
| `train.py` | Single file | Percentage-based random | `CP{epoch}-{batch}.pth` | Quick experiments |
| `train2.py` | Single file | Explicit index ranges | `CP{epoch}.pth` | Reproducible runs |
| `train3.py` | Multiple files (10) | Explicit ranges + multi-file mapping | `CP{epoch}.pth` | Full dataset training |

---

## Command-Line Arguments

All scripts share a common set of arguments:

| Argument | Flag | Default | Description |
|---|---|---|---|
| Epochs | `-e / --epochs` | 5 | Number of training epochs |
| Batch size | `-b / --batch-size` | 1 | Samples per gradient update |
| Learning rate | `-l / --learning-rate` | 0.1 | Initial SGD learning rate |
| GPU | `-g / --gpu` | False | Enable CUDA training |
| Load checkpoint | `-c / --load` | None | Path to `.pth` file to resume from |
| Scale | `-s / --scale` | 0.5 | Input downscaling factor |

`train2.py` and `train3.py` additionally support:

| Argument | Description |
|---|---|
| `--start-epoch / --nepoch` | Epoch range (start and count) for checkpoint resumption |
| `--start-train / --ntrain` | Training sample start index and count |
| `--start-val / --nval` | Validation sample start index and count |

Example from `train3.sh`:
```bash
time python train3.py -g -e 50 --ntrain 90 --nval 10
```

---

## Loss Function

Training uses **Binary Cross-Entropy (BCE) loss** applied to flattened predictions and targets:

```python
criterion = nn.BCELoss()
masks_pred_flat = masks_pred.view(-1)
true_masks_flat = true_masks.view(-1)
loss = criterion(masks_pred_flat, true_masks_flat)
```

This treats each pixel independently as a binary classification (signal vs. noise), which is appropriate for LArTPC segmentation where signal pixels are sparse.

**Note**: The custom `dice_loss.py` implements a differentiable Dice coefficient via `torch.autograd.Function`, but it is used only as a **validation metric**, not in the backpropagation loss.

---

## Optimizer

SGD with momentum and weight decay:

```python
optimizer = optim.SGD(net.parameters(),
                      lr=lr,
                      momentum=0.9,
                      weight_decay=0.0005)
```

The learning rate is fixed throughout training (no scheduler). Optimizer state is **not saved** in checkpoints — resuming training restarts with the original learning rate.

---

## Data Loading

Training data is stored in paired HDF5 files:
- `*-rec-*.h5`: reconstructed detector frames (inputs)
- `*-tru-*.h5`: Monte Carlo truth frames (labels)

`train.py` / `train2.py` use a single file pair:
```python
"data/cosmic-rec-0.h5", "data/cosmic-tru-0.h5"
```

`train3.py` loads multiple files for a larger dataset:
```python
rec_files = [f"data/g4-rec-r{i}.h5" for i in range(10)]
tru_files = [f"data/g4-tru-r{i}.h5" for i in range(10)]
```

Event indices are mapped across files using `id_gen()`, which converts a flat sample index into a `(file_id, event_id)` tuple.

Within each batch, frames are:
1. Cropped to a spatial subregion (e.g., wires 800–1600, ticks 0–600)
2. Rebinned along the time axis (factor of 10×)
3. Stacked into a CHW tensor

---

## Training Loop

```
for epoch in range(epochs):
    for batch in DataLoader(train_set):
        images, masks = batch
        if gpu: images, masks = images.cuda(), masks.cuda()

        masks_pred = net(images)
        loss = criterion(masks_pred.view(-1), masks.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint
    torch.save(net.state_dict(), f"checkpoints/CP{epoch+1}.pth")

    # Validation
    eval_loss = eval_util.eval_loss(net, val_set, gpu)
    eval_dice = eval_util.eval_dice(net, val_set, gpu)
```

---

## Validation Metrics

Computed at the end of each epoch using `eval_util.py`:

### Dice Coefficient
Measures overlap between predicted and ground-truth masks. Robust to class imbalance (important since signal pixels are sparse).
```
Dice = 2 × |pred ∩ truth| / (|pred| + |truth|)
```
Predictions are thresholded at 0.5 before computing Dice.

### Efficiency (Recall)
Fraction of true signal pixels that are correctly predicted as signal.
```
Efficiency = TP / (TP + FN)
```

### Purity (Precision)
Fraction of predicted signal pixels that are truly signal.
```
Purity = TP / (TP + FP)
```

Efficiency and purity are computed both at the **pixel level** and at the **ROI level** (contiguous signal regions detected as connected components).

---

## Checkpointing

- Saved after each epoch: `checkpoints/CP{epoch}.pth` (or `CP{epoch}-{batch}.pth` in train.py)
- Contains only model weights (`net.state_dict()`), not optimizer state
- On `KeyboardInterrupt`: saves to `INTERRUPTED.pth`
- To resume training, pass the checkpoint path with `-c`:
  ```bash
  python train2.py -g -c checkpoints/CP10.pth --start-epoch 10 -e 40
  ```

---

## Differences Between Training Scripts

### train.py
- Splits data randomly by percentage via `split_train_val()`
- Single HDF5 file pair
- Checkpoint names include batch count: `CP{epoch}-{batchcount}.pth`
- Dice evaluation active during training

### train2.py
- User specifies exact sample index ranges (`--start-train`, `--ntrain`, `--start-val`, `--nval`)
- Supports loading a checkpoint before resuming (if `--start-epoch > 0`)
- Simpler checkpoint naming: `CP{epoch}.pth`

### train3.py
- Adds multi-file support using `id_gen()` to map flat indices across 10 HDF5 files
- Supports PDVD detector spatial layout: `x_range=[476, 952]` (vs. PDSP `[800, 1600]`)
- Lower truth threshold: 10 ADC (vs. 100 ADC in earlier scripts)
- Dice evaluation commented out; only BCE loss used for validation

---

## Visualization

**TensorBoard** (`tensor-board.py`): reads epoch-level loss logs and plots training curves.

**plot_epoch.py**: generates per-epoch loss/dice plots directly from checkpoint directories.

**compare_ep.py** / **make_loss_dist.py**: tools for comparing loss distributions across runs or epochs.
