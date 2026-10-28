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

# §6: does steering along d_prior change accuracy?
C="final:0 final:-0.5 final:-0.25 final:-1 final:0.5 random:-0.5 h1:-0.5 h1:-0.25"
uv run src/steer_gsm.py --lang eng --device cuda:0 --configs $C &
uv run src/steer_gsm.py --lang dan --device cuda:1 --configs $C &
uv run src/steer_gsm.py --lang nob --device cuda:2 --configs $C &
uv run src/steer_gsm.py --lang deu --device cuda:3 --configs $C &
wait
uv run src/analyze_steer.py

# §7: EuroEval, baseline and two doses of the intervention (≈80 min each on one GPU)
./src/run_euroeval.sh 0     results/euroeval/base
./src/run_euroeval.sh -0.25 results/euroeval/m025
./src/run_euroeval.sh -0.5  results/euroeval/m05
uv run src/analyze_euroeval.py results/euroeval/base results/euroeval/m025 results/euroeval/m05
