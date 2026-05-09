# Improvement Ideas for the LArTPC DNN ROI Detection

This document brainstorms concrete ideas to improve the system in two complementary directions:

1. **Memory / compute efficiency** — make the model lighter so it fits on smaller GPUs and runs faster in the production C++ pipeline.
2. **Physics performance** — push pixel efficiency and purity higher, especially in the hardest regimes (prolonged tracks on induction planes, busy interaction vertices, electromagnetic showers).

The ideas are grouped by lever: architecture, loss, input features, and training data. A short prioritized "if I could only do 3 things" list is at the end.

---

## 1. Memory & Compute Efficiency

### 1.1 Lightweight convolutions
- **Depthwise separable convolutions (MobileNet-style)** — replace each `Conv3×3` in `unet/parts.py::double_conv` with a `Conv3×3 (depthwise) + Conv1×1 (pointwise)` pair. Roughly **9× fewer FLOPs and parameters** at each block. Easiest single change for memory.
- **MobileNetV2 inverted residuals** — `Conv1×1 (expand) → Conv3×3 (depthwise) → Conv1×1 (project)`, with linear bottleneck and residual skip. More expressive than plain depthwise separable.
- **MobileNetV3 / EfficientNet blocks** — add Squeeze-Excitation and h-swish; usually 1–2% more efficiency at the same FLOPs.
- **GhostNet / ShuffleNetV2 blocks** — generate redundant feature maps from cheap operations; very effective when channels are large.

### 1.2 Quantization
- **Post-training INT8 quantization** of the TorchScript model — typical ~4× memory reduction and ~2–3× CPU speedup, often <1% accuracy drop on segmentation.
- **Quantization-aware training (QAT)** — recovers any drop. Fits naturally with the existing TorchScript export path used by Wire-Cell Toolkit.
- **Mixed precision training** (`torch.cuda.amp`) — FP16 weights and activations during training, halving memory and giving ~1.5× speedup on Tensor Cores.

### 1.3 Pruning & distillation
- **Channel pruning** based on BatchNorm γ-magnitudes followed by a short fine-tune. The current 64–512-channel U-Net is likely overparameterized for binary segmentation with sparse positive class.
- **Knowledge distillation** — train a heavy U-Net (teacher) with rich inputs (Section 4), distill its outputs into a small MobileNet-U-Net (student) for production. Especially attractive because MP2/MP3 features are computed offline anyway.

### 1.4 Sparse representations
- **Submanifold sparse convolutions** (MinkowskiEngine, spconv) — LArTPC images are >95% empty. Switching to a sparse CNN can cut compute and memory by an order of magnitude. Already standard in the LArTPC physics community (e.g., MicroBooNE 3D networks).
- **Trade-off**: only worth it if the dense input is not also given to the network (sparse-dense interfaces are awkward).

### 1.5 Other memory tricks
- **Gradient checkpointing** (`torch.utils.checkpoint`) — recompute activations during the backward pass. Trades compute for memory; useful when training with larger crops or richer inputs (Section 4).
- **Tiled inference** — process overlapping patches at full resolution rather than the whole 800×600 image, then stitch. Lets a constrained-VRAM card handle full DUNE FD frames.

### 1.6 Expected wins (rough order)

| Change | Param ↓ | VRAM ↓ | Latency ↓ | Accuracy hit |
|---|---|---|---|---|
| Depthwise separable | ~9× | ~4× | ~3× | small |
| Channel pruning (50%) | ~4× | ~2× | ~2× | small if fine-tuned |
| INT8 quantization | ~4× | ~4× | ~2–3× CPU | <1% |
| Sparse conv | depends | ~10× | ~5× | minimal if done right |
| Mixed precision | 1× | 2× | 1.5× | none |

---

## 2. Network Architecture for Physics Performance

The dominant failure mode of the current network is **prolonged tracks** (track angle θ_xz close to 90°): they are very long in time, very faint in amplitude, and the U-Net's bottleneck only "sees" a 48×48 receptive field — not enough context to confirm a long, faint structure is genuine.

### 2.1 Larger global context
- **Extra downsampling levels** (5 → 6 or 7) — doubles or quadruples the receptive field at the cost of more parameters and lower bottleneck spatial resolution.
- **Dilated / atrous convolutions** in the bottleneck (DeepLab v3+ style ASPP) — large effective receptive field without losing resolution. Direct fix for the prolonged-track receptive-field problem.
- **Self-attention / transformer bottleneck** (e.g., 2 transformer blocks at the lowest level, or a Swin Transformer encoder) — captures arbitrary long-range correlations along a track.
- **Non-local blocks** at one or two levels — cheaper than full self-attention.

### 2.2 Better skip pathways
- **Attention U-Net** — gated attention on the skip connections; suppresses background features in skips so the decoder focuses on track-relevant content. Drop-in upgrade with a small parameter cost.
- **UNet 3+ full-scale skips** — connect every encoder level to every decoder level. Already partially explored via `nestedunet/`.
- **HRNet-style parallel branches** — keep a high-resolution branch alive throughout the network. Best for fine localization at the boundary of a faint signal.

### 2.3 Track-aware architectural priors
- **Anisotropic / strip kernels** — replace some 3×3 kernels with 1×N or N×1 kernels along the dominant track direction. Tracks are inherently 1D structures; standard 3×3 kernels waste capacity on rotational symmetry the data does not have.
- **Strip pooling** — pooling along single rows or columns; natural fit for line-like signals.
- **Multi-orientation conv banks** — apply convs rotated for U/V/W wire orientations and combine.

### 2.4 Channel attention
- **Squeeze-and-Excitation / CBAM blocks** — when input channel count grows (Section 4), the network needs to learn how much to trust each filter view; SE blocks make this explicit.

### 2.5 Two-stage / iterative architectures
- **Cascade**: a fast lightweight Stage-1 produces a coarse ROI; Stage-2 refines pixel labels conditioned on Stage-1 (Mask R-CNN style). Saves compute since Stage-2 only runs in proposed regions.
- **Iterative refinement** — feed the DNN's first-pass output back into the deconvolution chain (use it to define a new ROI → cleaner LF deconvolution → re-run DNN). 2–3 iterations typically saturate. This is especially natural here because the DNN's output is exactly what the traditional pipeline needs as input to its next step.

### 2.6 Cross-plane joint network
The current design treats each induction plane as an independent inference, with cross-plane info injected only via MP2/MP3. A truly joint architecture:
- Shared encoder consuming all three planes (with MP2/MP3 still as side info)
- Three decoder heads, one per plane
- Allows the encoder to learn cross-plane consistency end-to-end rather than only through the precomputed binary masks.

---

## 3. Loss Function Improvements

The signal pixel fraction is small (~few % of pixels), so plain BCE is suboptimal — it lets the easy background dominate the gradient.

### 3.1 Class-imbalance-aware losses
- **Dice / Generalized Dice loss** — directly optimizes overlap; robust to class imbalance. `dice_loss.py` already exists for evaluation; just wire it into the training loop.
- **Focal loss** — `(1−p_t)^γ · BCE` down-weights easy pixels and concentrates learning on hard examples (faint prolonged tracks).
- **Tversky loss** — generalizes Dice with separate FN/FP weights (α, β); choose α > β to prefer high efficiency over high purity (or vice versa) per analysis goal.
- **Focal Tversky** — combines focal weighting with the Tversky knobs.

### 3.2 Topology-preserving losses
Tracks are tubular structures; the most damaging errors are not isolated pixel flips but **broken** tracks.
- **clDice** (centerline Dice) — combines Dice with a soft-skeleton overlap term; explicitly preserves connectivity. Drop-in addition; ~30 lines of code.
- **Persistent-homology-based topology loss** — penalizes spurious holes and disconnected components. Heavier, but theoretically the right thing.
- **Hausdorff distance loss** — punishes boundary errors at the worst pixel; useful for tight ROIs.

### 3.3 Weighting tricks
- **Distance-weighted BCE** — Ronneberger's original U-Net paper used per-pixel weights based on distance to the boundary; particularly effective when tracks are close together.
- **Per-angle stratified weighting** — over-weight examples with θ_xz > 80°. Combats the natural angular distribution that under-represents the hard regime.
- **Combo loss** — `BCE + λ_dice · Dice + λ_cl · clDice`; empirically robust and simple to tune.

### 3.4 Physics-informed losses
- **Reconstruction consistency**: given the known response R and the predicted mask m, the implied charge `m · S_decon` must satisfy `R ⊗ (m · S_decon) ≈ M_raw`. Penalize the residual. This adds a physical prior that the mask is consistent with the observed waveform without needing more labels.
- **Cross-plane geometric consistency** — penalize predictions on plane V that have no geometrically corresponding active region on planes U and W. Soft version of the MP2/MP3 constraint, applied at the loss level.

---

## 4. Input Channel Choices (the highest-leverage area)

The current network input has **3 channels**: `frame_loose_lf` + `MP2` + `MP3`. The space of useful inputs is much larger and largely unexplored. The hypothesis: just **stacking richer inputs** likely gives the largest single gain on prolonged-track performance, with no architectural change.

### 4.1 Multiple software filters (efficiency–purity trade-off)

Different software filters expose different aspects of the same waveform:

| Filter | Efficiency | Purity | Best at |
|---|---|---|---|
| Tight LF | low | high | clean signal regions |
| Loose LF (current) | high | low | prolonged tracks |
| Extra-loose LF | very high | very low | extreme angles |
| Various HF cutoffs | varies | varies | trading off electronics noise |
| Wiener filter | optimal SNR | adaptive | data-dependent |

**Concrete proposal**: stack 3–5 deconvolution variants as separate channels — e.g., {tight LF, medium LF, loose LF, extra-loose LF, no-LF charge}. Each is one HDF5 frame; cost is only memory. The network learns which filter to trust where.

This directly addresses the user's observation that low- and high-frequency filter choices change SNR and signal preservation in different ways.

### 4.2 Charge image after ROI applied (currently unused)

The traditional Wire-Cell pipeline computes a `Decon. for Charge` image — deconvolution **without** any LF filter, but with the ROI mask applied to suppress everything outside the regions of interest. This image carries the **most physically faithful charge values** (no LF filter distortion) within the ROI.

- **Static use**: stack the post-ROI charge image (from the heuristic ROI) as an extra channel — gives the network access to undistorted charge magnitude.
- **Iterative use**: run the DNN once, apply its predicted ROI to the no-LF deconvolution, recompute the charge image, feed it back as a channel for a second forward pass. The network uses its own first-pass output to "clean" its second-pass input.

This is one of the highest-impact ideas because the charge information is currently fed to the network only in distorted, LF-filtered form.

### 4.3 Connectivity & topology features

A real track is a **connected, elongated** structure. Almost all noise pixels are isolated or form small blobs. Pre-computing connectivity features and stacking them as channels gives the network this prior for free:

- **Connected-component map** — each pixel labeled with its component size after a simple threshold. Small components = noise; large components = candidate track.
- **Component aspect ratio** — track-like (elongated) vs. blob-like (circular/EM-shower-like).
- **Skeleton / medial axis** of the thresholded loose-LF image — emphasizes track centerlines, very informative for tubular structures.
- **Top-hat / morphological features** — directional opening/closing along candidate track angles.
- **Edge maps** (Sobel / Canny) — captures track edges, which are sharper than noise gradients.
- **Hough transform output** — explicit line/segment detector; gives the network a strong line prior.
- **Distance transform** — distance from each pixel to nearest active region; smooths sparse activity into a denser feature.

These features are cheap to compute (vectorized OpenCV/scikit-image) and provide complementary information that no amount of network depth would extract from raw waveforms quickly.

### 4.4 Extended multi-plane information

The current MP2/MP3 use a single initial-ROI definition (tight + loose connectivity heuristic). Extensions:

- **MP2/MP3 from multiple LF filter outputs** — recompute geometric coincidence using initial ROIs from tight, loose, and extra-loose deconvolutions. Stack as extra channels: MP2_tight, MP2_loose, MP3_tight, MP3_loose. Each variant catches a different efficiency/purity regime.
- **Soft coincidence** — replace the binary 0/1 MP2/MP3 with a continuous "coincidence strength" (e.g., sum of charges from matched wires on other planes). Carries gradient information.
- **Direct cross-plane projections** — resample U/W signals into V's coordinate system and stack as channels, not just the binary coincidence mask. Lets the network correlate continuous waveforms across planes rather than just binary overlap.

### 4.5 Time-frequency representations

- **Wavelet decomposition** of each waveform — each scale becomes a channel; captures time-localized frequency structure better than fixed-bandwidth filtering.
- **Short-time Fourier transform per wire** — multiple time-frequency channels.

This is more speculative; the gain will depend on whether prolonged-track signals have characteristic time-frequency signatures that simple LF filtering misses.

### 4.6 Summary input pack (concrete recommendation)

A concrete 8–10-channel stack worth trying:

1. Deconvolved with tight LF
2. Deconvolved with loose LF (current)
3. Deconvolved with extra-loose LF
4. No-LF charge image after heuristic ROI applied
5. MP2 (current)
6. MP3 (current)
7. Connected-component-size map (computed from channel 2 thresholded)
8. Skeleton (medial axis) of channel 2 thresholded
9. (Optional) MP2 from extra-loose initial ROI
10. (Optional) Soft coincidence strength

Memory cost: ~3× current. Likely the largest single physics gain in this document.

---

## 5. Training Data Improvements

### 5.1 Data augmentation appropriate to LArTPC
- **Time-axis shift** — translate the entire image along the drift-time axis. Free invariance.
- **Wire-axis shift** — translate along the wire-channel axis.
- **Realistic noise injection** — add measured noise profiles from real ProtoDUNE data; helps generalize from simulation to data.
- **Mixup / CutMix between events** — interpolate between two event images and their masks. Cheap regularization.
- **No rotation** — wire orientations are fixed by detector geometry, so rotation is unphysical and would hurt.

### 5.2 Hard-example focus
- **Oversample large-angle tracks** (θ_xz > 80°) — they are the failure mode. Build a separate "hard" sub-dataset and mix it in 50/50 with normal events.
- **Synthetic prolonged tracks** — generate Geant4 muons deliberately at extreme angles to populate the regime that natural cosmic-ray distributions under-sample.
- **Curriculum learning** — start training on small-angle (easy) tracks, gradually shift the angular distribution toward the hard regime. Often more stable than focal loss.
- **Online hard example mining** — at each epoch, identify the events with worst loss and over-sample them in the next epoch.

### 5.3 Multi-task and self-supervised auxiliaries
- **Charge regression head** — alongside the binary mask, predict the charge value at each pixel. Forces the encoder to retain quantitative information.
- **Track angle estimation head** — predict per-pixel θ_xz. Correlates with the MP2/MP3 input and gives the network an explicit notion of "this region is a prolonged track".
- **Particle ID head** (muon vs. electron vs. shower) — coarse semantic labels regularize the representation.
- **Self-supervised pretraining**:
  - Masked reconstruction (mask out random patches of the input, reconstruct them) — MAE-style.
  - Cross-plane contrastive — two planes of the same event are positive pairs, planes from different events are negatives.

### 5.4 Multi-detector training
- Combine PDSP, PDVD, MicroBooNE, DUNE-FD simulations in one training pool.
- Improves generalization and exposes the network to a wider range of noise spectra and wire geometries.
- Modest extra effort; significant generalization payoff.

### 5.5 Domain adaptation to real data
- The simulation-to-data gap is the dominant remaining systematic.
- **Adversarial domain confusion** — a small discriminator tries to distinguish simulation-feature from real-feature representations; the U-Net is trained to fool it.
- **Self-training** — run the model on real data, use confident predictions as pseudo-labels, retrain.

---

## 6. Top Recommendations (if you can only do three things)

These are ranked by **expected physics gain per unit engineering effort**.

### A. Stack richer inputs (Section 4.6)
**Engineering**: ~1 week. Modify `utils/load.py` and `utils/h5_utils.py` to read and stack 8–10 channels. Re-train.
**Expected gain**: large on prolonged tracks (>5–10% absolute efficiency at θ_xz > 85°). The current network is starved of information.

### B. Combo loss BCE + Dice + clDice (Section 3.1, 3.2)
**Engineering**: ~1 day. The Dice loss is already implemented in `dice_loss.py`; clDice is ~30 lines.
**Expected gain**: better connectivity preservation (fewer broken tracks) and modest efficiency/purity gains across the board.

### C. Attention U-Net + dilated bottleneck (Section 2.1, 2.2)
**Engineering**: ~3 days. New blocks in `unet/parts.py` and `unet/model.py`.
**Expected gain**: directly addresses the small-receptive-field weakness; expected solid improvement on prolonged tracks. If memory is a concern, also swap in depthwise separable convs (Section 1.1) — this gives a smaller and stronger network.

### Stretch: iterative refinement (Section 2.5)
**Engineering**: requires modifying the Wire-Cell C++ pipeline to feed DNN output back into deconvolution. Larger systems-engineering cost.
**Expected gain**: potentially the largest physics gain because it closes the feedback loop that the current pipeline cannot.

---

## 7. Evaluation Suggestions

When evaluating any of the above, beyond the existing pixel efficiency / pixel purity metrics:

- **Per-angle binned metrics** — efficiency as a function of θ_xz. The interesting regime is θ_xz > 75°.
- **ROI-level metrics** (already in `eval_util.eval_eff_pur`) — measure recall and precision of contiguous regions, not just pixels. Catches the "broken track" failure mode that pixel metrics hide.
- **Connectivity metric** — number of connected components in predicted vs. truth masks. A perfect pixel-wise score with double the components count means broken tracks.
- **Downstream physics impact** — track reconstruction efficiency, calorimetric resolution after the full Wire-Cell pipeline, neutrino interaction vertex resolution. The DNN is one step in a long chain; ultimate value is measured at the top of the chain.

The most informative single plot for this work remains Figure 8 from the paper (efficiency/purity vs. θ_xz). Any new idea should be evaluated against that baseline.
