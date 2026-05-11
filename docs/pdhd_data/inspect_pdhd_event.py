#!/usr/bin/env python3
"""Inspect one PDHD training event and produce PNG quick-look figures.

Generates per-frame heatmaps, a truth binary mask, value histograms,
and a channel-kill comparison vs the raw (pre-augmentation) file.
Writes a small text summary alongside the figures.

Defaults to the augmented event 100 in
train_data_PDHD_fixedbug_separateWC/modified/g4-rec-0_modified.h5,
because that is the event where convert2.py injects noise into
channels 702 and 1046 -- making the comparison panel meaningful.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "train_data_PDHD_fixedbug_separateWC")

INPUT_FRAMES = ["frame_loose_lf0", "frame_mp2_roi0", "frame_mp3_roi0"]
# Intermediate Wire-Cell SP stages present only in the *raw* rec file.
# Order roughly matches the SP pipeline: low-frequency filters, decon,
# ROI builders, gauss, then the dnnsp slot.
INTERMEDIATE_FRAMES = [
    "frame_tight_lf0",
    "frame_decon_charge0",
    "frame_cleanup_roi0",
    "frame_break_roi_1st0",
    "frame_break_roi_2nd0",
    "frame_extend_roi0",
    "frame_shrink_roi0",
    "frame_gauss0",
    "frame_dnnsp0",
]
TRUTH_FRAME = "frame_deposplat0"
TRUTH_TH = 100  # matches hnam's default in train4.sh

# 2560 PDHD APA channels: U[0..800), V[800..1600), W[1600..2560)
PLANE_BOUNDARIES = (800, 1600)


def load_frame(path, event_id, dataset):
    """Read one (tick, ch) frame, auto-transpose to (ch, tick) like h5_utils.load."""
    with h5py.File(path, "r") as f:
        arr = np.array(f[str(event_id)][dataset])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D dataset, got shape {arr.shape} for {dataset}")
    # Native storage in hnam's files is (tick, channel) with tick > channel.
    # h5_utils.load() flips to (channel, tick) when shape[0] > shape[1].
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr  # (channel, tick)


def heatmap(arr, title, outpath, vmin=None, vmax=None, cmap="bwr", symmetric=False):
    """Plot a (channel, tick) array as an image with plane separators."""
    n_ch, n_tick = arr.shape
    if symmetric and vmin is None and vmax is None:
        v = np.nanpercentile(np.abs(arr), 99) or 1.0
        vmin, vmax = -v, v
    elif vmin is None and vmax is None:
        vmin, vmax = np.nanpercentile(arr, [1, 99])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    for bound in PLANE_BOUNDARIES:
        ax.axhline(bound, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("tick (0..6000)")
    ax.set_ylabel("channel (0..2560)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    # Plane labels
    labels = [("U", 400), ("V", 1200), ("W", 2080)]
    for name, ypos in labels:
        ax.text(
            -0.02,
            ypos / n_ch,
            name,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def histograms_panel(frames, outpath):
    """2x2 grid of log-y histograms for the four (name, array) pairs."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, arr) in zip(axes.flat, frames):
        flat = arr.ravel()
        finite = flat[np.isfinite(flat)]
        ax.hist(finite, bins=200, log=True)
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("count (log)")
        # Annotate sparsity
        zero_frac = float(np.mean(np.abs(finite) < 1e-6))
        ax.text(
            0.97,
            0.95,
            f"exact-zero: {100 * zero_frac:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
    fig.suptitle("Pixel-value distributions (log y)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def intermediate_stages_panel(raw_rec, event_id, outpath):
    """3x3 grid of heatmaps for the intermediate Wire-Cell SP stages.

    Read from the raw rec file (these datasets are stripped from the
    modified rec). Per-panel symmetric colormap for signed quantities,
    sequential for clearly-nonnegative ones.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    summary_rows = []
    with h5py.File(raw_rec, "r") as f:
        g = f[str(event_id)]
        for ax, name in zip(axes.flat, INTERMEDIATE_FRAMES):
            if name not in g:
                ax.set_visible(False)
                continue
            arr = np.array(g[name])
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            finite = arr[np.isfinite(arr)]
            nz = finite[np.abs(finite) > 1e-6]
            # For dense panels (<10% zeros), use 1/99 percentile of full data.
            # For sparse ones, use 1/99 percentile of NON-zero values so the
            # colorbar isn't pinned to 0 by the mostly-zero background.
            if nz.size == 0:
                lo, hi = -1.0, 1.0
            elif (finite.size - nz.size) / finite.size < 0.1:
                lo, hi = np.nanpercentile(finite, [1, 99])
            else:
                lo, hi = np.nanpercentile(nz, [1, 99])
            if lo < 0 and hi > 0:
                v = max(abs(lo), abs(hi))
                vmin, vmax, cmap = -v, v, "bwr"
            else:
                vmin, vmax, cmap = max(0, lo), hi, "viridis"
            im = ax.imshow(
                arr,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            for bound in PLANE_BOUNDARIES:
                ax.axhline(bound, color="k", lw=0.4, alpha=0.4)
            zero_pct = 100 * float(np.mean(np.abs(finite) < 1e-6))
            ax.set_title(f"{name}\nzero={zero_pct:.1f}%  range=[{finite.min():.0f},{finite.max():.0f}]", fontsize=9)
            ax.set_xlabel("tick", fontsize=8)
            ax.set_ylabel("channel", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.04)
            summary_rows.append(summarize(name, arr, TRUTH_TH))
    fig.suptitle(
        f"Intermediate Wire-Cell SP stages -- raw rec event {event_id}\n"
        "(present in g4-rec-*.h5, stripped from modified/g4-rec-*_modified.h5)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    return summary_rows


def noise_injection_panel(raw_rec, mod_rec, raw_tru, mod_tru, event_id, outpath):
    """Side-by-side raw-vs-modified for the two known noise-injected channels.

    convert2.py replaces channels 702 and 1046 of frame_loose_lf0 with
    Gaussian noise for event 100; truth is zeroed for those channels.
    Find the channels empirically (any |diff| > 0) so the panel still
    works if hnam re-runs the augmentation with a different seed.
    """
    raw = load_frame(raw_rec, event_id, "frame_loose_lf0")
    mod = load_frame(mod_rec, event_id, "frame_loose_lf0")
    diff = np.abs(raw - mod).max(axis=1)
    affected = np.flatnonzero(diff > 1e-3)
    if affected.size == 0:
        print(f"  noise_injection: no augmented channels in event {event_id}, skipping panel")
        return False

    truth_id = event_id - 100
    raw_t = load_frame(raw_tru, truth_id, TRUTH_FRAME)
    mod_t = load_frame(mod_tru, truth_id, TRUTH_FRAME)

    n = len(affected)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)
    for row, ch in enumerate(affected):
        axes[row, 0].plot(raw[ch], lw=0.5, label="raw", color="tab:blue")
        axes[row, 0].plot(mod[ch], lw=0.5, label="modified", color="tab:orange", alpha=0.7)
        axes[row, 0].set_title(f"frame_loose_lf0 -- channel {ch} (waveform)")
        axes[row, 0].set_xlabel("tick")
        axes[row, 0].set_ylabel("value")
        axes[row, 0].legend(loc="upper right", fontsize=8)

        axes[row, 1].plot(raw_t[ch], lw=0.6, label="raw truth", color="tab:blue")
        axes[row, 1].plot(mod_t[ch], lw=0.6, label="modified truth", color="tab:orange", alpha=0.7)
        axes[row, 1].set_title(f"frame_deposplat0 -- channel {ch} (truth)")
        axes[row, 1].set_xlabel("tick")
        axes[row, 1].set_ylabel("charge")
        axes[row, 1].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"convert2.py channel-kill: event {event_id} -- {n} channel(s) replaced"
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  noise_injection: augmented channels = {list(affected)}")
    return True


def summarize(name, arr, threshold):
    flat = arr.ravel()
    finite = flat[np.isfinite(flat)]
    zero_frac = float(np.mean(np.abs(finite) < 1e-6))
    above_frac = float(np.mean(np.abs(finite) > threshold))
    return {
        "name": name,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "zero_pct": 100 * zero_frac,
        "above_th_pct": 100 * above_frac,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def write_summary(rows, threshold, outpath):
    lines = [
        f"# PDHD event inspection -- truth threshold = {threshold}",
        "",
        f"{'dataset':28s} {'shape':>14s} {'dtype':>8s} {'zero%':>8s} {'>th%':>8s}"
        f" {'min':>10s} {'max':>12s} {'mean':>10s}",
    ]
    for r in rows:
        lines.append(
            f"{r['name']:28s} {str(r['shape']):>14s} {r['dtype']:>8s}"
            f" {r['zero_pct']:8.2f} {r['above_th_pct']:8.2f}"
            f" {r['min']:10.1f} {r['max']:12.1f} {r['mean']:10.1f}"
        )
    txt = "\n".join(lines) + "\n"
    with open(outpath, "w") as fh:
        fh.write(txt)
    print(txt)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--file",
        default=os.path.join(DEFAULT_DATA_DIR, "modified", "g4-rec-0_modified.h5"),
        help="Rec HDF5 file (what the network sees).",
    )
    p.add_argument(
        "--truth",
        default=os.path.join(DEFAULT_DATA_DIR, "modified", "g4-tru-0_modified.h5"),
        help="Truth HDF5 file (matching rec).",
    )
    p.add_argument(
        "--raw-file",
        default=os.path.join(DEFAULT_DATA_DIR, "g4-rec-0.h5"),
        help="Raw rec HDF5 file (pre-convert2.py), for the noise-injection panel.",
    )
    p.add_argument(
        "--raw-truth",
        default=os.path.join(DEFAULT_DATA_DIR, "g4-tru-0.h5"),
        help="Raw truth HDF5 file.",
    )
    p.add_argument("--event", type=int, default=100, help="Rec event id (rec=event, tru=event-100).")
    p.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "png"),
        help="Where to write PNGs and summary.txt.",
    )
    p.add_argument("--truth-th", type=float, default=TRUTH_TH, help="Truth charge threshold for the binary mask.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"rec   : {args.file}")
    print(f"tru   : {args.truth}")
    print(f"event : {args.event} (truth event {args.event - 100})")
    print(f"outdir: {args.outdir}")
    print()

    # ---- load the four frames we want to plot ----
    rec_frames = {name: load_frame(args.file, args.event, name) for name in INPUT_FRAMES}
    truth = load_frame(args.truth, args.event - 100, TRUTH_FRAME)

    # ---- heatmaps ----
    heatmap(
        rec_frames["frame_loose_lf0"],
        f"frame_loose_lf0 (event {args.event}) -- dense signal-processed input",
        os.path.join(args.outdir, "frame_loose_lf0.png"),
        symmetric=True,
        cmap="bwr",
    )
    heatmap(
        rec_frames["frame_mp2_roi0"],
        f"frame_mp2_roi0 (event {args.event}) -- sparse ROI mask",
        os.path.join(args.outdir, "frame_mp2_roi0.png"),
        vmin=0,
        vmax=4000,
        cmap="hot",
    )
    heatmap(
        rec_frames["frame_mp3_roi0"],
        f"frame_mp3_roi0 (event {args.event}) -- sparse ROI mask",
        os.path.join(args.outdir, "frame_mp3_roi0.png"),
        vmin=0,
        vmax=4000,
        cmap="hot",
    )
    heatmap(
        truth,
        f"frame_deposplat0 (truth event {args.event - 100}) -- sparse charge",
        os.path.join(args.outdir, "truth_deposplat0.png"),
        vmin=0,
        vmax=np.nanpercentile(truth, 99.5),
        cmap="viridis",
    )
    heatmap(
        (truth > args.truth_th).astype(np.float32),
        f"truth binary mask (truth > {args.truth_th:g})",
        os.path.join(args.outdir, "truth_mask.png"),
        vmin=0,
        vmax=1,
        cmap="Greys",
    )

    # ---- histograms ----
    histograms_panel(
        [
            ("frame_loose_lf0", rec_frames["frame_loose_lf0"]),
            ("frame_mp2_roi0", rec_frames["frame_mp2_roi0"]),
            ("frame_mp3_roi0", rec_frames["frame_mp3_roi0"]),
            (f"frame_deposplat0 (truth)", truth),
        ],
        os.path.join(args.outdir, "value_histograms.png"),
    )

    # ---- noise-injection comparison ----
    if os.path.exists(args.raw_file) and os.path.exists(args.raw_truth):
        noise_injection_panel(
            args.raw_file,
            args.file,
            args.raw_truth,
            args.truth,
            args.event,
            os.path.join(args.outdir, "noise_injection.png"),
        )
    else:
        print("  noise_injection: raw rec/tru not found, skipping panel")

    # ---- intermediate Wire-Cell SP stages (raw file only) ----
    intermediate_rows = []
    if os.path.exists(args.raw_file):
        intermediate_rows = intermediate_stages_panel(
            args.raw_file,
            args.event,
            os.path.join(args.outdir, "intermediate_stages.png"),
        )
    else:
        print("  intermediate_stages: raw rec not found, skipping panel")

    # ---- summary table ----
    rows = []
    for name, arr in rec_frames.items():
        rows.append(summarize(name, arr, args.truth_th))
    rows.append(summarize(TRUTH_FRAME, truth, args.truth_th))
    rows.extend(intermediate_rows)
    write_summary(rows, args.truth_th, os.path.join(args.outdir, "summary.txt"))


if __name__ == "__main__":
    sys.exit(main())
