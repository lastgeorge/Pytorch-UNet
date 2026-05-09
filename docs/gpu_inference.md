# GPU Training and Inference

## GPU Training

Training is single-GPU. There is no multi-GPU support (no `DataParallel` or `DistributedDataParallel`).

### Enabling GPU

Pass the `-g` flag when calling any training script:
```bash
python train3.py -g -e 50 --ntrain 90 --nval 10
```

Internally, this moves the model and data to the default CUDA device:
```python
net = net.cuda()
images = images.cuda()
masks  = masks.cuda()
```

### Resource Management on Shared Nodes

Two settings are used to control CPU/GPU resource consumption:
```python
torch.set_num_threads(1)   # limit CPU thread contention on HPC shared nodes
# torch.backends.cudnn.benchmark = True  # disabled — avoids extra GPU memory allocation
```

These are appropriate for training on shared cluster nodes where memory is limited.

---

## Model Formats

Models are stored and used in two formats:

| Format | Extension | Use case |
|---|---|---|
| PyTorch state dict | `.pth` | Training, checkpoint resumption, Python inference |
| TorchScript | `.ts` | C++ deployment, framework-independent inference |

Saved model directories:
- `pth-model/` — PyTorch weights
- `ts-model/` — TorchScript (original, PyTorch 1.x)
- `ts-model-2.3/` — TorchScript recompiled for PyTorch 2.3 compatibility

---

## PyTorch → TorchScript Conversion (`to-ts.py`)

Converts a `.pth` state dict to a traced TorchScript model using `torch.jit.trace()`:

```bash
./to-ts.py -m pth-model/unet-l23-cosmic500-e50.pth -o ts-model-2.3/unet-l23-cosmic500-e50.ts
./to-ts.py -m pth-model/nestedunet-l23-cosmic500-e50.pth -o ts-model-2.3/nestedunet-l23-cosmic500-e50.ts -t nestedunet
./to-ts.py -m pth-model/unet-lt-cosmic500-e50.pth -o ts-model-2.3/unet-lt-cosmic500-e50.ts -i 2
```

Key flags:
| Flag | Description |
|---|---|
| `-m` | Input `.pth` file |
| `-o` | Output `.ts` file |
| `-t` | Model type: `unet` (default), `uresnet`, or `nestedunet` |
| `-i` | Number of input channels (default 3) |

The trace uses a random example input of shape `(1, 3, 800, 600)` (or `(1, 2, ...)` for 2-channel models). `torch.jit.trace` records all operations on this example input to produce a static computation graph.

**Warning**: `torch.jit.trace` does not capture conditional branches. The traced graph is specific to the input shape used during tracing — different spatial sizes may require re-tracing.

---

## TorchScript → PyTorch Conversion (`to-pth.py`)

Extracts weights from a TorchScript model and saves them as a standard state dict:

```bash
./to-pth.py -m ts-model/unet-l23-cosmic500-e50.ts -o pth-model/unet-l23-cosmic500-e50.pth
./to-pth.py -m ts-model/nestedunet-l23-cosmic500-e50.ts -o pth-model/nestedunet-l23-cosmic500-e50.pth -t nestedunet
./to-pth.py -m ts-model/unet-lt-cosmic500-e50.ts -o pth-model/unet-lt-cosmic500-e50.pth -i 2
```

This uses `named_parameters()` and `named_buffers()` to reconstruct a compatible state dict.

---

## Python Inference (`predict.py`)

Single-event inference from the command line. Supports both `.pth` and `.ts` models.

```bash
./predict.sh
# Internally calls something like:
python predict.py \
  --model ts-model/unet-l23-cosmic500-e50.ts \
  --input data/g4-rec-r9.h5 \
  --range 100 101 \
  --tick-rebin 1 \
  --threshold 0.5
```

Key arguments:

| Argument | Description |
|---|---|
| `--model` | Path to `.pth` or `.ts` model file |
| `--input` | Input HDF5 file |
| `--range START END` | Event index range to process |
| `--tick-rebin` | Time axis rebinning factor (default 10) |
| `--threshold` | Binary output threshold (default 0.5) |
| `--gpu` / no flag | CPU by default (use `map_location='cpu'` for CPU-mode load) |

Output is saved as a JPEG image of the binary segmentation mask.

**Model loading logic:**
```python
if model_path.endswith('.ts'):
    net = torch.jit.load(model_path)
else:
    net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
net.eval()
```

---

## C++ Inference (`cpp/`)

For production use in the reconstruction pipeline, inference runs via LibTorch C++ API using TorchScript models.

### Build

Dependencies: **LibTorch** and **Eigen3**.

```bash
cd cpp
source cuda-env.sh      # set up CUDA environment variables
bash cmake.sh           # configure with CMake
make -j4
```

`CMakeLists.txt` links against:
- `Torch` (LibTorch)
- `Eigen3::Eigen`
- `pthread`

### Usage (`test-ts.cpp`)

The C++ inference example:
1. Loads a TorchScript model: `torch::jit::load(model_path)`
2. Moves to CUDA if available: `model.to(torch::kCUDA, 0)`
3. Constructs an input tensor from Eigen arrays representing wire data:
   ```cpp
   auto input = torch::stack({ch0, ch1, ch2}).unsqueeze(0);
   // Shape: {1, 3, H, W}
   ```
4. Runs inference:
   ```cpp
   auto output = model.forward({input}).toTensor();
   ```
5. Maps the output back to Eigen arrays for downstream processing

### Device Selection

```cpp
if (torch::cuda::is_available()) {
    model.to(torch::kCUDA, 0);
    input = input.to(torch::kCUDA, 0);
} else {
    // CPU fallback
}
```

### Performance Notes

- TorchScript models are ~54 MB per model (UNet l23)
- C++ inference avoids Python overhead for real-time reconstruction
- For batch throughput, inference can loop over events without re-loading the model
