# hnam DNN-ROI chain — overview

Reference notes for Hokyeong Nam's ("hnam") DNN-ROI training and deployment work for DUNE on WCGPU1. hnam's home directory is symlinked into this repo as `./hnam` (target: `/nfs/data/1/hnam`).

The five topic docs each answer one of the questions sent to hnam:

| Topic | Doc | Question answered |
| --- | --- | --- |
| Simulation data on WCGPU1 | [hnam_data.md](./hnam_data.md) | "Where is the simulation data?" |
| Input preprocessing (offline + online) | [hnam_input_preprocessing.md](./hnam_input_preprocessing.md) | "How are the inputs processed?" |
| Truth labeling + ROI extension | [hnam_truth_labeling.md](./hnam_truth_labeling.md) | "How was truth labeled?" |
| Model architecture + how to train | [hnam_model_and_training.md](./hnam_model_and_training.md) | "What is the final model architecture?" |
| Evaluation scripts & metrics | [hnam_evaluation.md](./hnam_evaluation.md) | "What is the evaluation code and what metrics?" |
| TorchScript export + Wire-Cell integration | [hnam_deployment.md](./hnam_deployment.md) | "How is the model integrated into the official chain?" |

## Top-level map of `/nfs/data/1/hnam`

```
hnam/
├── Pytorch-UNet/                                # training, eval, TS export (mirror of HokyeongNam:feature/mobileunet)
├── wct-dev-hnam/                                # Wire-Cell Toolkit fork — C++ DNN-ROI components live here
├── wire-cell-hnam/                              # WC cfg fork — Jsonnet + LArSoft FCL deployment files
├── wire-cell-python/                            # WC Python fork
├── wcpy-hnam/                                   # additional Python-side glue
├── python-packages/                             # local Python installs
├── train_data_PDHD_fixedbug_separateWC/         # PDHD training data (~168 GB)
├── train_data_PDVD_rate2M_window3p2/            # PDVD training data (default, ~72 GB)
├── train_data_PDVD_rate2M_smallmpth_window3p2/  # PDVD alt-SP variant (~72 GB)
├── demo/  myScript/  test/  tmp/                # scratch
├── log.txt, log_wcpy_02142025.txt               # training/eval session logs
├── move_g4_files.sh                             # partitions PDHD g4 files train/val
└── epoch_time*.csv, epoch_time_extraction.py    # per-epoch timing analysis
```

The Conda env hnam uses is named **`dnnroi`** (Python 3.9, PyTorch on CUDA 11.1) — see `hnam/Pytorch-UNet/environment.yml`.

## Open questions for hnam

Items that the local files don't fully answer; ready to forward.

### Simulation generation
- **Where do the raw `g4-rec-*.h5` / `g4-tru-*.h5` HDF5 files come from?** No Geant4/Wire-Cell job submission configs were found alongside the data dirs (only post-processing scripts `convert.py` / `convert2.py`).
- **Which Wire-Cell version produced them?** `wct-dev-hnam` and `wire-cell-hnam` are present locally; please point us at the specific commit/tag and jobsub recipe.

### Models
- Of the many `chk_mobilenetv3_*` / `chk_mobunetv3_pdvd_*` checkpoints under `hnam/Pytorch-UNet/`, **which one is the "blessed" model** to use for PDHD vs PDVD?
- For PDHD, **which `.ts` file in `hnam/Pytorch-UNet/torchscript3/` (or elsewhere) is currently deployed in the official chain?** What conversion command produced it (rebin, threshold, input shape)?

### Data variants
- Does `train_data_PDVD_rate2M_smallmpth_window3p2` supersede `train_data_PDVD_rate2M_window3p2`, or are they used in parallel? What does "small mpth" buy us?

### Evaluation
- **Charge bias** and **completeness** were not found in `hnam/Pytorch-UNet/` code (grep for `charge.bias`, `q_bias`, `charge_ratio`, `completeness` returned nothing in `*.py`). Are these computed downstream in Wire-Cell (e.g., on the dnnsp frame output), or in a notebook/script that lives elsewhere?
- The test datasets referenced in `eval.py` (`eval-jinst-init-sub/eval-<label>/g4-rec-0.h5` etc.) do **not** exist under `hnam/Pytorch-UNet/`. Where do those evaluation files live, and how should we obtain them?

### Deployment recipe
- The LArSoft FCL fragments under `hnam/wire-cell-hnam/pdhd-data/*/dnnroi_sp/` look like the production drop-in for PDHD. Could you confirm the exact recipe (FCL name, WC version, `.ts` model expected) that's been handed to Xuyang / production processing?

## Verification commands used to build these docs

```bash
ls /nfs/data/1/hnam
du -sh /nfs/data/1/hnam/train_data_PDHD_fixedbug_separateWC /nfs/data/1/hnam/train_data_PDVD_rate2M_window3p2
cat hnam/Pytorch-UNet/train4.sh
cat hnam/Pytorch-UNet/chk_mobilenetv3_20260511_061959/config.json
grep -n "def \|truth_th\|padding" hnam/Pytorch-UNet/utils/h5_utils.py
grep -n "eff_\|pur_" hnam/Pytorch-UNet/eval_util.py
ls hnam/Pytorch-UNet/torchscript3/
find hnam/wct-dev-hnam -name "TorchScript.*" -o -name "DNNROI*"
find hnam/wire-cell-hnam -name "*dnnroi*" -o -name "*dnnsp*"
```
