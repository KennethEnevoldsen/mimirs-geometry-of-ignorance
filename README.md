# Mimir's geometry of ignorance

Does [DFM Mimir](https://huggingface.co/danish-foundation-models/DFM-Mimir) (a 1B Danish/English
Hierarchical Reasoning Model, [Schneider-Kamp et al. 2026](https://arxiv.org/abs/2608.13517)) have a
*direction of ignorance*: one direction of the unembedding that encodes the unigram prior, which the model
leans on when it is unsure ([Liu et al. 2026](https://arxiv.org/abs/2609.02959))? We test it while Mimir solves
math word problems from [multilingual GSM-symbolic](https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic)
in 8 languages.

**TL;DR**

1. **The direction exists, and there is one per language.** A single direction of Mimir's unembedding reconstructs the
   English unigram prior (R² = 0.73, nKL = 0.11) and another the Danish one (R² = 0.81, nKL = 0.14).
   An emulated random initialisation gets R² = 0.05, nKL = 0.95. The two directions are orthogonal
   (cos = −0.02). The paper studied English-dominant models only, where one direction was enough.
2. **The prior is causally active.** Patching λ along `d_prior` at the most uncertain response token moves the
   prediction toward the unigram prior: KL drops from 5.6 to 3.2 nats as λ\* goes from −2 to 2. A random
   direction with the same norm does not do this.
3. **Mimir barely uses the prior when it does math** (|λ| ≤ 0.12, against ≈ 0.4 for Llama‑3.2‑3B on WikiText).
   What little prior it uses matches the answer's language: Nordic answers load on the Danish prior and
   English answers on the English one. That language signature only shows up in the last H‑module
   application. Earlier applications in the recurrence all sit at λ < 0.
4. **λ does not predict errors** (pooled AUROC 0.52; mean token entropy gets 0.73). Replacing the question
   with a different one raises λ in all 8 languages, but only by +0.006 to +0.025.
5. **Side finding: in Danish, Norwegian and Icelandic, Mimir often skips the reasoning.** It answers
   straight away with `\boxed{…}` in 32% / 52% / 35% of cases, and those answers are mostly wrong (17–22%).
   When it does reason, it gets 89% / 93% / 40% right. It never does this in English.

## Setup

* **Model**: `danish-foundation-models/DFM-Mimir` (v1) in fp32 with the Gemma‑4 chat template and a PrefixLM mask
  (the prompt is one bidirectional block), as the model card requires. Decoding is greedy, with at most 768 new
  tokens.
* **Data**: [`danish-foundation-models/multilingual-gsm-symbolic`](https://huggingface.co/datasets/danish-foundation-models/multilingual-gsm-symbolic):
  the first 2 synthetic variants of each of the 100 templates, giving 200 problems per language for
  eng, dan, nob, swe, isl, deu, fra and zho (1,600 in total). Each prompt uses the instruction for its language
  from the `multilingual-gsm-symbolic` package. We score the last `\boxed{}`, or the last number if there is none
  (Mimir answers English with "The final answer is: $X$").
* **Prior fit** (paper §3.3 / App. B): we count unigrams over 100M tokens of `monology/pile-uncopyrighted` (English) and
  100M tokens of `danish-foundation-models/danish-dynaword` (Danish), plus an equal-weight mix. We then fit
  `W d ≈ log p_unigram` by OLS on mean-centred rows and set `λ = ⟨d̂, y⟩ / β`.
  **Deviation from the paper:** the support is the 32k most frequent tokens instead of 70% of the vocabulary. Gemma‑4's
  262k vocabulary is massively multilingual, and most of it never occurs in English or Danish text: only 169k / 101k
  distinct tokens appeared in 100M tokens. With 70% coverage the fit degrades (nKL 0.60 for English). The sweep
  is in `results/prior_fit.json`.
* **Reading λ through the recurrence**: Mimir runs `L L L H L L L H`. Each stack ends in a weightless RMSNorm, so
  the output of every module application lives in the same space as the final prediction state. We read λ from
  all 8 applications, a logit lens that comes for free.
* **Controls / interventions**: (a) teacher-force the same response after a *different* question; (b) patch
  `y ← y + (λ* − λ0) d_prior` (paper eq. 8) against a random unit direction with the same norm. We patch at two
  places: the most uncertain response token and the first token of the final answer.

## Results

### 1. Mimir encodes an English and a Danish unigram prior on (nearly) orthogonal directions

![prior fit](figures/prior_fit.png)

| prior | R² | nKL | β |
|---|---|---|---|
| English (Pile) | 0.73 | 0.110 | 19.9 |
| Danish (Dynaword) | 0.81 | 0.144 | 35.8 |
| equal mix | 0.77 | 0.165 | 20.4 |
| mix, emulated random init | 0.05 | 0.948 | 11.2 |

For comparison, the paper reports R² 0.77–0.83 and nKL 0.06–0.08 for Gemma‑3 and Qwen‑2.5. cos(d_en, d_da) = −0.02.

### 2. Accuracy and the "no reasoning" failure mode

| lang | acc | direct-answer rate | acc (direct) | acc (with reasoning) |
|---|---|---|---|---|
| eng | 0.92 | 0.00 | – | 0.92 |
| dan | 0.68 | 0.32 | 0.22 | 0.89 |
| nob | 0.55 | 0.52 | 0.18 | 0.93 |
| swe | 0.69 | 0.09 | 0.06 | 0.76 |
| isl | 0.32 | 0.35 | 0.17 | 0.40 |
| deu | 0.69 | 0.01 | 0.00 | 0.69 |
| fra | 0.73 | 0.00 | – | 0.73 |
| zho | 0.63 | 0.00 | – | 0.63 |

### 3. Prior loading λ while answering

| lang | λ (mix) | λ (English prior) | λ (Danish prior) | λ after a different question | AUROC error (mean λ) | AUROC error (mean entropy) |
|---|---|---|---|---|---|---|
| eng | −0.113 | 0.079 | −0.095 | −0.097 | 0.60 | 0.65 |
| dan | 0.070 | 0.005 | 0.090 | 0.095 | 0.14 | 0.92 |
| nob | 0.047 | 0.040 | 0.060 | 0.066 | 0.09 | 0.94 |
| swe | 0.056 | 0.015 | 0.068 | 0.082 | 0.51 | 0.74 |
| isl | 0.051 | 0.044 | 0.052 | 0.075 | 0.41 | 0.60 |
| deu | 0.030 | 0.009 | 0.046 | 0.045 | 0.60 | 0.66 |
| fra | −0.027 | 0.011 | 0.003 | −0.017 | 0.52 | 0.57 |
| zho | −0.058 | 0.027 | −0.024 | −0.052 | 0.46 | 0.49 |

Pooled AUROC: mean λ 0.52, mean entropy 0.73. On reasoning responses only: 0.54 and 0.69. The Danish and
Norwegian λ AUROCs are far *below* 0.5 because direct answers are short, carry almost no prior (λ ≈ 0.01) and are
mostly wrong. In other words, λ there picks up "did not reason", not "is uncertain".

![lambda vs position](figures/lambda_vs_position.png)

![language priors](figures/language_priors.png)

### 4. Prior loading through the HRM recurrence

![lambda per module](figures/lambda_per_module.png)

Every intermediate module application has λ < 0 in every language. The final H‑module application is where λ
splits by language: English goes down to −0.11 on the mix prior, while the Nordic languages go up to +0.05–0.07.

### 5. Steering along `d_prior`

![steering](figures/steering.png)

At the most uncertain token, raising λ\* pulls the prediction toward the unigram prior, and a random direction with
the same norm does not. This matches the paper's Fig. 4. At the first token of the answer, the prediction is very
confident: it only changes at extreme values. At λ\* = −2, the probability of the original answer token drops from
0.90 to 0.09.

## Reproduce

```bash
uv sync
bash run.sh          # ≈1.5 h on 4× Quadro RTX 8000; writes results/ and figures/
uv run src/analyze.py  # only re-aggregates (no GPU): tables + figures from results/
```

| file | what |
|---|---|
| `src/count_unigrams.py` | stream a corpus and count Gemma‑4 tokens |
| `src/fit_prior.py` | OLS fit of `d_prior`, R², nKL, random-init control, support sweep |
| `src/run_gsm.py` | generate, score, teacher-force with hooks on every L/H application, control and steering |
| `src/analyze.py` | tables (`results/summary.json`) and `figures/` |

All raw generations are in `results/gsm/<lang>.jsonl`, and per-token λ / entropy / KL are in `results/gsm/<lang>.npz`.

## Caveats

* We used one checkpoint (Mimir v1, not v1.5), 200 problems per language and greedy decoding.
* The unigram priors come from reference corpora (Pile and Dynaword), not from Mimir's training mixture.
* The support is 32k tokens, not 70% of the vocabulary (see above).
* We only measure λ on response tokens. Under the PrefixLM mask the prompt is bidirectional, so prompt positions do
  not do next-token prediction and the paper's analysis of how λ falls as context builds up does not carry over
  directly.
* Math responses are low-entropy text, so the small λ we see is not directly comparable to the paper's WikiText
  numbers.

## References

* T. J. B. Liu et al. *The Geometry of Ignorance: LLMs Know When to Temper Bayesian Priors.* arXiv:2609.02959 (2026).
* P. Schneider-Kamp et al. *DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data.* arXiv:2608.13517 (2026).
* [multilingual-gsm-symbolic](https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic).
