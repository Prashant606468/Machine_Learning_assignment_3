# ML Assignment 3 — MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR10

Prashant · M24MA2009 · IIT Jodhpur

## Reproduce everything

```bash
conda env create -f environment.yml && conda activate ml-a3   # or: pip install -r requirements.txt
bash run_all.sh            # uses python3; override with PY=python bash run_all.sh. Trains, evaluates, writes tables/figs, builds report/main.pdf
```

CIFAR10 is downloaded by torchvision into `data/` on first run. Expected time on an RTX 3050 (6 GB):
about 2 hours. Q1/Q2 MLPs take seconds per epoch, CNNs and AEs a few seconds to ~15 s per epoch,
Q4 SVMs run on the CPU in parallel.

Tested on Python 3.9 (torch 2.6) and 3.13 (torch 2.15).

Individual questions (each one reuses cached results of the earlier ones, so order matters):

```bash
python src/q1.py   # MLP-base / MLP-wide / LocalMLP, permutation control, memorisation
python src/q2.py   # CNN vs MLP: clean, block_shuffle, pixel_perm, lowpass, highpass
python src/q3.py   # AEs (d = 32, 256, 1024), DAE, linear probe + kNN
python src/q4.py   # SVM C x kernel sweep, difficult pairs, clustering
python src/make_numbers.py
```

Sanity check without CIFAR10 or a GPU (synthetic data, 2 epochs, writes to `results_smoke/`):
`SMOKE=1 python src/q1.py` (and so on).

## Split

`splits/split_seed242009.npz` — stratified 45,000 / 5,000 train/val split of the official training set,
`random_state = 242009` (digits of roll number M24MA2009, in order). Created on the first run and
verified on every later run. `splits/q4_subset_10000_seed242009.npz` — fixed stratified subset used for
every SVM in Q4.

## Layout

| Path | Contents |
|---|---|
| `configs/config.yaml` | every hyperparameter |
| `src/common.py` | data, split, transforms, trainer (lr sweep + val selection + one test eval), calibration |
| `src/models.py` | MLP, LocalMLP, CNN, ConvAE (`python src/models.py` prints parameter counts) |
| `src/q1.py` … `src/q4.py` | one script per question; each writes logs, LaTeX tables and figures |
| `src/make_numbers.py` | collects key numbers as LaTeX macros for the report |
| `results/logs/` | per-epoch histories, sweeps, test logits, summaries, stdout logs, pip freeze |
| `results/tables/`, `results/figs/` | generated, included by `report/main.tex` |
| `report/` | CVPR-format report |

## Reuse across questions

| Model / result | Introduced | Reused in |
|---|---|---|
| `mlp_base` | Q1(a) | Q2(a) MLP |
| `mlp_base__pixel_perm` | Q1(a) | Q2(b) |
| `cnn` (clean) | Q2(a) | Q3(b) fixed classifier |
| `ae_{d*}` selected AE | Q3(a) | Q3(b) baseline, Q3(c), Q4 representation |
| `dae_{d*}` | Q3(b) | Q3(c) |

## Rules followed

- Test set never used for any choice; in every lr sweep only the selected run is evaluated on test.
- No pretrained networks, no external data, no augmentation, no architecture search.
- GPU runs use cuDNN autotuning, so retraining may move accuracies by a few tenths of a point.
