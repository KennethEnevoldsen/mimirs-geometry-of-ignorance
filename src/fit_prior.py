"""Fit the direction of ignorance d_prior on Mimir's unembedding (Liu et al. 2026, Sec. 3.3 / App. B).

OLS on mean-centred unembedding rows -> mean-centred log unigram probabilities, restricted to the
support of reasonably frequent tokens. Writes results/prior_<corpus>.pt and results/prior_fit.json.
"""

import json

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download

MODEL = "danish-foundation-models/DFM-Mimir"
# Support = the K most frequent tokens. The paper uses 70% of the vocab, but Gemma 4's 262k vocab is
# massively multilingual: most of it never occurs in English/Danish text (see README for a K sweep).
SUPPORT_K = 32_000


def load_unembedding() -> torch.Tensor:
    path = hf_hub_download(MODEL, "model.safetensors")
    with safe_open(path, "pt") as f:
        return f.get_tensor("lm_head.weight").double()


def fit(W: torch.Tensor, counts: np.ndarray, support_k: int = SUPPORT_K) -> dict:
    support = np.sort(np.argsort(-counts, kind="stable")[:support_k])
    support = support[counts[support] > 0]
    p = counts[support] / counts[support].sum()
    ell = torch.from_numpy(np.log(p))
    Wf = W[support]
    Wc, ec = Wf - Wf.mean(0), ell - ell.mean()
    d = torch.linalg.lstsq(Wc, ec.unsqueeze(1), driver="gelsd").solution.squeeze(1)
    pred = Wc @ d
    r2 = 1 - ((ec - pred) ** 2).sum() / (ec**2).sum()
    p_t = torch.from_numpy(p)
    log_pprior = torch.log_softmax(Wf @ d, 0)  # restricted + renormalised on the support
    kl = (p_t * (p_t.log() - log_pprior)).sum()
    kl_unif = (p_t * (p_t.log() + np.log(len(support)))).sum()
    beta = d.norm()
    return {
        "d_prior": d,
        "beta": beta.item(),
        "support": torch.from_numpy(support),
        "log_punigram": ell,
        "r2": r2.item(),
        "nkl": (kl / kl_unif).item(),
        "n_support": len(support),
        "n_tokens": int(counts.sum()),
    }


def main() -> None:
    W = load_unembedding()
    pile = np.load("results/unigram_pile.npy")
    dyna = np.load("results/unigram_dynaword.npy")
    # bilingual reference prior: equal weight English (Pile) and Danish (Dynaword)
    mix = pile / pile.sum() + dyna / dyna.sum()
    mix = mix * (pile.sum() + dyna.sum()) / 2
    corpora = {"pile": pile, "dynaword": dyna, "mix": mix}

    stats, dirs = {}, {}
    for name, counts in corpora.items():
        res = fit(W, counts)
        dirs[name] = res["d_prior"] / res["beta"]
        torch.save({k: v for k, v in res.items()}, f"results/prior_{name}.pt")
        stats[name] = {k: v for k, v in res.items() if isinstance(v, (int, float))}
        print(name, stats[name])

    # control: emulated initialisation (rows i.i.d. N(0, init_range^2)), as in the paper's Fig. 2
    g = torch.Generator().manual_seed(0)
    W_rand = torch.randn(W.shape, generator=g, dtype=torch.float64) * 0.025515518153991442
    res = fit(W_rand, mix)
    stats["mix_random_init"] = {k: v for k, v in res.items() if isinstance(v, (int, float))}
    print("random init", stats["mix_random_init"])

    stats["support_sweep"] = {
        f"{name}_K{k}": {kk: v for kk, v in fit(W, corpora[name], k).items() if kk in ("r2", "nkl")}
        for name in ("pile", "dynaword")
        for k in (8_000, 16_000, 32_000, 64_000, 128_000)
    }
    stats["cosine"] = {
        "pile_vs_dynaword": float(dirs["pile"] @ dirs["dynaword"]),
        "pile_vs_mix": float(dirs["pile"] @ dirs["mix"]),
        "dynaword_vs_mix": float(dirs["dynaword"] @ dirs["mix"]),
    }
    print(stats["cosine"])
    json.dump(stats, open("results/prior_fit.json", "w"), indent=2)


if __name__ == "__main__":
    main()
