#!/usr/bin/env bash
# Exact reproduction of every number in the report. Run from the repo root.
# Total on an RTX 3050 6GB laptop: roughly 2 h (Q4 SVMs run on CPU).
set -euo pipefail
PY=${PY:-python3}   # override: PY=python bash run_all.sh
cd "$(dirname "$0")"
mkdir -p results/logs
$PY -m pip freeze > results/logs/pip_freeze.txt
$PY -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\")" > results/logs/hardware.txt
$PY src/models.py                 | tee results/logs/param_counts.txt
$PY src/q1.py                     2>&1 | tee results/logs/q1_stdout.log
$PY src/q2.py                     2>&1 | tee results/logs/q2_stdout.log   # reuses mlp_base from Q1
$PY src/q3.py                     2>&1 | tee results/logs/q3_stdout.log   # reuses clean CNN from Q2
$PY src/q4.py                     2>&1 | tee results/logs/q4_stdout.log   # reuses selected AE from Q3
$PY src/make_numbers.py
echo "\\newcommand{\\CommitHash}{$(git rev-parse --short HEAD 2>/dev/null || echo UNCOMMITTED)}" > results/tables/commit.tex
cd report && latexmk -pdf -interaction=nonstopmode main.tex
