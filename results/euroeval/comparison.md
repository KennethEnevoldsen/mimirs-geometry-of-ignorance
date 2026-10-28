| dataset | task | base | m025 | m05 | Δ m025 | Δ m05 |
|---|---|---|---|---|---|---|
| hellaswag | common-sense-reasoning | 53.62 | 53.73 | 53.52 | +0.11 | -0.10 |
| winogrande-da | common-sense-reasoning | 25.50 | 25.50 | 27.91 | +0.00 | +2.41 |
| contamination-canary | contamination-detection | 100.00 | 100.00 | 100.00 | +0.00 | +0.00 |
| valeu-da | european-values | 87.39 | 91.23 | 89.45 | +3.84 | +2.06 |
| valeu-en | european-values | 0.51 | 0.42 | 0.12 | -0.09 | -0.39 |
| ragtruth-da | hallucination | 1.46 | 1.37 | 1.57 | -0.09 | +0.11 |
| ragtruth-en | hallucination | 5.85 | 5.54 | 5.05 | -0.31 | -0.81 |
| danish-citizen-tests | knowledge | 53.51 | 54.30 | 54.31 | +0.79 | +0.80 |
| danske-talemaader | knowledge | 40.41 | 40.41 | 35.73 | +0.00 | -4.68 |
| life-in-the-uk | knowledge | 35.33 | 35.27 | 36.34 | -0.05 | +1.01 |
| dala | linguistic-acceptability | 98.43 | 98.43 | 97.65 | +0.00 | -0.78 |
| scala-en | linguistic-acceptability | 61.87 | 62.26 | 61.68 | +0.39 | -0.19 |
| zebra-puzzles-easy-da | logical-reasoning | 50.78 | 58.20 | 60.42 | +7.42 | +9.64 |
| zebra-puzzles-easy-en | logical-reasoning | 61.59 | 66.80 | 66.67 | +5.21 | +5.08 |
| conll-en | named-entity-recognition | 63.83 | 64.93 | 65.08 | +1.10 | +1.25 |
| dansk | named-entity-recognition | 43.99 | 45.99 | 47.40 | +2.00 | +3.41 |
| multi-wiki-qa-da | reading-comprehension | 80.87 | 79.22 | 72.97 | -1.65 | -7.89 |
| squad | reading-comprehension | 84.10 | 83.56 | 82.34 | -0.54 | -1.76 |
| angry-tweets | sentiment-classification | 58.21 | 59.62 | 59.71 | +1.41 | +1.50 |
| sst5 | sentiment-classification | 58.08 | 57.44 | 56.19 | -0.64 | -1.89 |
| cnn-dailymail | summarization | 36.54 | 39.61 | 38.13 | +3.07 | +1.59 |
| nordjylland-news | summarization | 37.23 | 36.70 | 35.85 | -0.53 | -1.38 |
| bfcl-v2 | tool-calling | 4.80 | 4.40 | 3.60 | -0.40 | -1.20 |
| multi-bfcl-da | tool-calling | 7.81 | 7.42 | 6.25 | -0.39 | -1.56 |
| flores-en-da | translation | 64.46 | 63.50 | 60.34 | -0.96 | -4.12 |
| wmt24pp-en-da | translation | 53.41 | 55.31 | 53.33 | +1.90 | -0.08 |
| danwic | word-in-context | 26.26 | 22.91 | 23.74 | -3.35 | -2.51 |

m025: n=27 datasets, mean Δ=+0.68, wins/ties/losses=11/4/12, mean base=47.99, mean steered=48.67
   by task: common-sense-reasoning +0.05 (n=2), contamination-detection +0.00 (n=1), european-values +1.88 (n=2), hallucination -0.20 (n=2), knowledge +0.25 (n=3), linguistic-acceptability +0.19 (n=2), logical-reasoning +6.32 (n=2), named-entity-recognition +1.55 (n=2), reading-comprehension -1.09 (n=2), sentiment-classification +0.39 (n=2), summarization +1.27 (n=2), tool-calling -0.40 (n=2), translation +0.47 (n=2), word-in-context -3.35 (n=1)
m05: n=27 datasets, mean Δ=-0.02, wins/ties/losses=11/1/15, mean base=47.99, mean steered=47.98
   by task: common-sense-reasoning +1.15 (n=2), contamination-detection +0.00 (n=1), european-values +0.84 (n=2), hallucination -0.35 (n=2), knowledge -0.96 (n=3), linguistic-acceptability -0.48 (n=2), logical-reasoning +7.36 (n=2), named-entity-recognition +2.33 (n=2), reading-comprehension -4.83 (n=2), sentiment-classification -0.19 (n=2), summarization +0.10 (n=2), tool-calling -1.38 (n=2), translation -2.10 (n=2), word-in-context -2.51 (n=1)
