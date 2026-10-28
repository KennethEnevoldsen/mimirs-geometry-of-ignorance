#!/usr/bin/env bash
# Reproduce everything: unigram counts -> prior fit -> GSM runs -> figures.
# ~1.5 h on 4x Quadro RTX 8000 (48 GB). Drop languages / lower --n to go faster.
set -euo pipefail
mkdir -p results logs

uv run src/count_unigrams.py pile 1e8 &
uv run src/count_unigrams.py dynaword 1e8 &
wait
uv run src/fit_prior.py

uv run src/run_gsm.py --langs eng dan --device cuda:0 --n 2 --batch-size 12 &
uv run src/run_gsm.py --langs nob swe --device cuda:1 --n 2 --batch-size 12 &
uv run src/run_gsm.py --langs deu isl --device cuda:2 --n 2 --batch-size 12 &
uv run src/run_gsm.py --langs fra zho --device cuda:3 --n 2 --batch-size 12 &
wait
uv run src/analyze.py
