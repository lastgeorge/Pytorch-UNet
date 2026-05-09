# Algorithm: DNN-Augmented LArTPC Signal Processing

> Based on: "Augmented Signal Processing in Liquid Argon Time Projection Chambers with a Deep Neural Network"
> H. W. Yu et al. (Brookhaven National Laboratory), arXiv:2007.12743, submitted to JINST.

---

## 1. The Signal Processing Problem

A LArTPC detector records the detector readout as 2D images: one axis is the wire channel index (spatial position), the other is the time tick (proportional to drift depth). The raw measurement on each wire is a convolution of the true ionization charge distribution S with the known detector response R, plus electronics noise N:

```
M(t, x) = ∫∫ R(t - t', x - x') · S(t', x') dt'dx'  +  N(t, x)
```

The goal of signal processing is to **recover S** from M given R, in the presence of N. This is done by 2D deconvolution in the frequency domain, followed by a **Region of Interest (ROI)** detection step.

### Why ROI Detection is Critical

An ROI is a window in the time domain bounding a region of genuine ionization activity. Defining tight ROIs before the final deconvolution suppresses low-frequency noise — the dominant noise source for induction-plane wires — and dramatically improves the signal-to-noise ratio.

LArTPCs typically have three wire planes:
- **U, V** (induction planes): wires at oblique angles; field response is **bipolar** (induced current flows both toward and away from the wire as the electron cloud passes). This makes deconvolution and ROI detection more challenging.
- **W** (collection plane): wires collect the ionization; field response is **unipolar**, so simple thresholding is sufficient for ROI finding.

### The Hard Case: Prolonged Tracks

Tracks traveling nearly parallel to wire planes (track angle θ_xz close to 90°) present the greatest challenge. For such tracks:
- The ionization cloud sweeps past many wires over an extended time window
- The bipolar induction signal on each wire partially cancels → very low amplitude
- The dominant signal energy shifts to low frequencies, where electronics noise is highest
- SNR drops severely; the traditional ROI algorithm misses most of the signal

The figure below illustrates the effect: a 45° track produces a clear waveform spike, while an 80° track produces a nearly invisible signal buried in noise.

---

## 2. Traditional Algorithm (Baseline)

The state-of-the-art baseline (implemented in the Wire-Cell Toolkit) follows this pipeline:

```
Raw waveform → Noise filtering → 2D Deconvolution (tight LF)
                                                      ↓
                              2D Deconvolution (loose LF)  →  ROI Detection  →  ROI mask
                                                                     ↑
                              2D Deconvolution (for charge) ──────────────────────────────→ Charge
```

- **Tight LF filter**: suppresses low-frequency content → high purity but low efficiency (misses prolonged tracks)
- **Loose LF filter**: includes more low-frequency content → higher efficiency but lower purity (more noise)
- **ROI detection**: heuristic connectivity-based logic combining tight and loose deconvolution results

The heuristic logic cannot be straightforwardly extended to handle the prolonged-track case and fails in busy environments (many overlapping tracks, e.g., near neutrino interaction vertices).

---

## 3. The DNN Approach: Two Key Innovations

This repository replaces the heuristic ROI detection step with a deep neural network that combines two complementary information sources:

### Innovation 1: U-Net Semantic Segmentation

ROI detection is recast as **per-pixel binary classification**: label each pixel of the 2D detector image as "signal" (inside ROI) or "noise" (outside ROI). This is a standard semantic segmentation task. The U-Net architecture (Ronneberger et al., 2015) is adopted — an encoder-decoder CNN with skip connections that propagates both coarse, high-level patterns and fine spatial detail to the output.

### Innovation 2: Multi-Plane Geometry Constraint

The key domain-knowledge contribution. In a real LArTPC, every ionization electron is independently sensed by **all wire planes simultaneously**. A genuine particle signal must therefore produce consistent activity across all three planes at the same drift time — a strong geometric constraint that noise cannot satisfy.

Instead of feeding only the deconvolved waveform to the network, two additional **coincidence images** (MP2 and MP3) are constructed from cross-plane wire matching and stacked as extra input channels. This encodes physics knowledge directly into the input representation.

---

## 4. Multi-Plane Coincidence Images (MP2 and MP3)

For each induction-plane target (e.g., V plane), the coincidence images are built as follows, processing one time slice at a time:

```
For each time slice (4 ticks ≈ 2 μs):
  1. Identify channels in V, U, W planes that fall inside the initial ROIs
     (from the traditional tight+loose LF deconvolution step)

  2. MP3 (three-plane coincidence):
     For each V-plane channel that IS inside the initial ROI,
     check if its wire geometrically overlaps wires from both U and W subsets.
     If yes → MP3 pixel = 1 (three-plane coincident signal)

  3. MP2 (two-plane coincidence):
     For each V-plane channel that is NOT inside the initial ROI,
     check if its wire geometrically overlaps wires from both U and W subsets.
     If yes → MP2 pixel = 1 (two-plane coincident signal, missed by initial ROI)

  4. Repeat for all time slices.
```

The MP2 image is the critical recovery channel: it identifies genuine signal pixels that the initial heuristic ROI missed, guided purely by cross-plane geometric consistency. For prolonged tracks — where the initial ROI algorithm fails but the track still appears in the other planes — MP2 restores the missed signal regions.

**Computational efficiency**: Wire overlap determination (finding whether two angled wires intersect) naively requires trigonometric operations per wire pair. The **Ray Grid technique** (Appendix A of the paper) exploits the fixed pitch and direction of an idealized wire plane to replace all sin/cos/sqrt calls with pre-computed tensor lookups and integer index arithmetic, making the computation scale with O(N_layers²) rather than O(N_wires²).

After construction, MP2 and MP3 are rebinned from the 4-tick slice resolution to the 10-tick bin resolution of the deconvolved image by averaging.

---

## 5. Network Input: Three-Channel Image

The three input channels fed to the U-Net (the `l23` configuration):

| Channel | Source | Content | dtype |
|---|---|---|---|
| 0 | `frame_loose_lf` | 2D deconvolved ionization charge (loose LF filter) | float, normalized by 4000 |
| 1 | `frame_mp2_roi` | Two-plane wire coincidence mask | Boolean (0 or 1) |
| 2 | `frame_mp3_roi` | Three-plane wire coincidence mask | Boolean (0 or 1) |

- Channel 0 provides the raw signal content — high efficiency but also high noise
- Channels 1–2 provide the geometric cross-plane veto — separating real tracks from noise artifacts

The network learns to combine these: where channel 0 shows activity AND channels 1/2 confirm cross-plane consistency → signal; where channel 0 shows activity but channels 1/2 are empty → likely noise.

---

## 6. U-Net Architecture

A five-level encoder-decoder network (see `unet/model.py`, `unet/parts.py`):

```
Input: (3, 800, 600)  [channels, wires, time bins]
  │
  ├─ Level 1 encoder: 64 ch   ──────────────────────────── Level 1 decoder: 64 ch ─→ 1×1 conv → sigmoid
  │     (MaxPool↓)                                               (↑bilinear, concat)
  ├─ Level 2 encoder: 128 ch  ──────────────────────── Level 2 decoder: 128 ch
  │     (MaxPool↓)                                        (↑bilinear, concat)
  ├─ Level 3 encoder: 256 ch  ─────────────────── Level 3 decoder: 256 ch
  │     (MaxPool↓)                                  (↑bilinear, concat)
  ├─ Level 4 encoder: 512 ch  ──────────── Level 4 decoder: 512 ch
  │     (MaxPool↓)                           (↑bilinear, concat)
  └─ Bottleneck:     512 ch ──────────────┘
```

- Each encoder block: `Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU`
- Skip connections: output of each encoder level concatenated with the corresponding decoder level
- Effective receptive field at bottleneck: 48 × 48 pixels (= 3 × 2⁴)
- Output: single-channel sigmoid probability map; threshold at 0.5 → binary ROI mask

Two alternative architectures are also implemented (`uresnet/`, `nestedunet/`) and perform similarly, but UNet is preferred for its ~10% lower GPU memory footprint.

---

## 7. Training

| Parameter | Value |
|---|---|
| Dataset | Cosmic ray simulation: CORSIKA → Geant4 → Wire-Cell |
| Events per epoch | 500 (450 train / 50 validation) |
| Cosmic tracks per event | 5–20 (avg. ~3500 ROIs per event) |
| Truth label source | Geant4 ionization electron distribution (`frame_ductor`) |
| Truth threshold | 100 electrons per 10-tick rebinned pixel |
| Loss function | Binary Cross-Entropy (`nn.BCELoss`) |
| Optimizer | SGD, momentum 0.9, learning rate 0.1 |
| Epochs | 50 (loss converges by ~epoch 30) |
| Hardware | Intel i9-9900K + NVIDIA RTX 2080 Ti (11 GB VRAM) |
| Time per epoch | ~6 minutes |

The truth label is constructed by smearing the ideal Geant4 ionization distribution with a Gaussian (simulating diffusion), then applying the 100-electron threshold. This threshold is well below the electronics noise (~400 electrons per tick), so the label captures genuine signal while remaining stable.

---

## 8. Performance

Evaluated on simulated prolonged tracks — the most challenging case. Two metrics:

- **Pixel Efficiency** = correctly labeled signal pixels / total signal pixels from truth
- **Pixel Purity** = correctly labeled signal pixels / total predicted signal pixels

| Algorithm | Efficiency @ 87° | Purity @ 87° |
|---|---|---|
| Traditional heuristic (Ref.) | ~5% | ~40% |
| DNN without multi-plane info | ~25% | ~65% |
| **DNN with multi-plane info** | **~35%** | **~80%** |

The multi-plane constraint provides the largest gain at very large angles where the traditional algorithm collapses entirely.

**Inference speed** (per wire plane):

| Method | Time | Memory | VRAM |
|---|---|---|---|
| Traditional (CPU) | 0.40 s | 1.3 GB | — |
| DNN CPU | 16.7 s | 4.8 GB | — |
| **DNN GPU** | **0.14 s** | 3.7 GB | 3.7 GB |

GPU inference is ~3× faster than the traditional algorithm while achieving superior physics performance.

---

## 9. Deployment

- Trained PyTorch models are converted to **TorchScript** (`.ts`) format using `torch.jit.trace()`
- Loaded in the **Wire-Cell Toolkit** C++ reconstruction pipeline via LibTorch (`torch::jit::load()`)
- The DNN ROI detection is applied only to induction planes (U and V); the collection plane (W) uses simple thresholding due to its unipolar signal response
- See `to-ts.py`, `cpp/test-ts.cpp`, and `docs/gpu_inference.md` for implementation details

---

## Summary

The algorithm achieves its improvement by treating ROI detection as a supervised learning problem and injecting detector-specific domain knowledge (multi-plane wire geometry) directly into the input representation. The U-Net is not just replacing a heuristic with a black box — it is given richer, physics-informed inputs that make the task tractable for the hardest track topologies.
