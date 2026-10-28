"""Aggregate results/gsm/*.{jsonl,npz} into figures/ and results/summary.json."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_gsm import final_answer  # noqa: E402

RES, FIG = Path("results"), Path("figures")
FIG.mkdir(exist_ok=True)
META = json.load(open(RES / "meta.json"))
MODULES, PRIORS, GRID = META["modules"], META["priors"], np.array(META["steer_grid"])
LANG_ORDER = ["eng", "dan", "nob", "swe", "isl", "deu", "fra", "zho"]
P_MIX = PRIORS.index("mix")


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) (Mann-Whitney)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_reasoning(recs) -> dict:
    rr = [r for r in recs if not r["direct"]]
    c = np.array([r["correct"] for r in rr], dtype=bool)
    lam = np.array([r["lam"][P_MIX, -1].mean() for r in rr])
    ent = np.array([r["ent"].mean() for r in rr])
    return {"auroc_error_mean_lambda_reasoning_only": auroc(lam[~c], lam[c]),
            "auroc_error_mean_entropy_reasoning_only": auroc(ent[~c], ent[c]),
            "lambda_direct": float(np.mean([r["lam"][P_MIX, -1].mean() for r in recs if r["direct"]])) if any(r["direct"] for r in recs) else None,
            "lambda_reasoning": float(lam.mean()) if len(lam) else None}


def load_lang(lang: str):
    recs = [json.loads(line) for line in open(RES / "gsm" / f"{lang}.jsonl")]
    for r in recs:  # re-score with the current parser so scoring fixes don't need a GPU rerun
        pred, _ = final_answer(r["response"])
        r["pred"] = pred
        r["correct"] = pred is not None and abs(pred - r["gold"]) < 1e-4 * max(1, abs(r["gold"]))
        r["direct"] = r["response"].strip().startswith("\\boxed")  # answer given without any reasoning
    arr = np.load(RES / "gsm" / f"{lang}.npz")
    for r in recs:
        i = r["idx"]
        r["lam"], r["ent"], r["kl"] = arr[f"lam_{i}"], arr[f"ent_{i}"], arr[f"kl_{i}"]
        r["lammis"] = arr[f"lammis_{i}"]
    return recs


def fig_prior_fit(summary):
    path = hf_hub_download("danish-foundation-models/DFM-Mimir", "model.safetensors")
    with safe_open(path, "pt") as f:
        W = f.get_tensor("lm_head.weight").double()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, name, title in zip(axes, ["pile", "dynaword"], ["English (Pile)", "Danish (Dynaword)"]):
        fit = torch.load(RES / f"prior_{name}.pt", weights_only=False)
        Wf = W[fit["support"]]
        x = (Wf - Wf.mean(0)) @ fit["d_prior"]
        y = fit["log_punigram"] - fit["log_punigram"].mean()
        ax.hexbin(x.numpy(), y.numpy(), gridsize=70, bins="log", cmap="viridis", mincnt=1)
        lim = [min(x.min(), y.min()).item(), max(x.max(), y.max()).item()]
        ax.plot(lim, lim, "r--", lw=1)
        s = summary["prior_fit"][name]
        ax.set_title(f"{title}: $R^2$={s['r2']:.2f}, nKL={s['nkl']:.3f}", fontsize=10)
        ax.set_xlabel(r"$W\,d_{prior}$ (centred)")
        ax.set_ylabel(r"$\log p_{unigram}$ (centred)")
    fig.suptitle(
        f"Mimir's unembedding encodes each language's unigram prior along one direction "
        f"(cos(d_en, d_da) = {summary['prior_fit']['cosine']['pile_vs_dynaword']:.2f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / "prior_fit.png", dpi=150)
    plt.close(fig)


def main():
    langs = [lang for lang in LANG_ORDER if (RES / "gsm" / f"{lang}.jsonl").exists()]
    data = {lang: load_lang(lang) for lang in langs}
    summary = {"prior_fit": json.load(open(RES / "prior_fit.json")), "languages": {}}

    # ---- per language table -------------------------------------------------------------------
    for lang, recs in data.items():
        c = np.array([r["correct"] for r in recs])
        mean_lam = np.array([r["lam"][P_MIX, -1].mean() for r in recs])
        mean_ent = np.array([r["ent"].mean() for r in recs])
        onset = [(r["correct"], r["lam"][P_MIX, -1, r["onset"]]) for r in recs if r["onset"] is not None]
        oc = np.array([o[0] for o in onset], dtype=bool)
        ol = np.array([o[1] for o in onset])
        summary["languages"][lang] = {
            "n": len(recs),
            "accuracy": float(c.mean()),
            "finished": float(np.mean([r["finished"] for r in recs])),
            "direct_answer_rate": float(np.mean([r["direct"] for r in recs])),
            "accuracy_direct": float(np.mean([r["correct"] for r in recs if r["direct"]])) if any(r["direct"] for r in recs) else None,
            "accuracy_reasoning": float(np.mean([r["correct"] for r in recs if not r["direct"]])) if not all(r["direct"] for r in recs) else None,
            "mean_lambda": {p: float(np.mean([r["lam"][k, -1].mean() for r in recs])) for k, p in enumerate(PRIORS)},
            "lambda_per_module": [float(np.mean([r["lam"][P_MIX, m].mean() for r in recs])) for m in range(len(MODULES))],
            "mean_lambda_mismatch": float(np.mean([r["lammis"].mean() for r in recs])),
            "lambda_correct": float(mean_lam[c].mean()) if c.any() else None,
            "lambda_incorrect": float(mean_lam[~c].mean()) if (~c).any() else None,
            "auroc_error_mean_lambda": auroc(mean_lam[~c], mean_lam[c]),
            "auroc_error_mean_entropy": auroc(mean_ent[~c], mean_ent[c]),
            "auroc_error_onset_lambda": auroc(ol[~oc], ol[oc]),
            # direct answers are short and mostly wrong, so also check within reasoning responses only
            **auroc_reasoning(recs),
        }
    # pooled error detection
    allr = [r for recs in data.values() for r in recs]
    c = np.array([r["correct"] for r in allr])
    ml = np.array([r["lam"][P_MIX, -1].mean() for r in allr])
    me = np.array([r["ent"].mean() for r in allr])
    rr = [r for r in allr if not r["direct"]]
    cr = np.array([r["correct"] for r in rr], dtype=bool)
    mlr = np.array([r["lam"][P_MIX, -1].mean() for r in rr])
    mer = np.array([r["ent"].mean() for r in rr])
    summary["pooled_reasoning_only"] = {
        "n": len(rr),
        "accuracy": float(cr.mean()),
        "auroc_error_mean_lambda": auroc(mlr[~cr], mlr[cr]),
        "auroc_error_mean_entropy": auroc(mer[~cr], mer[cr]),
    }
    summary["pooled"] = {
        "n": len(allr),
        "accuracy": float(c.mean()),
        "auroc_error_mean_lambda": auroc(ml[~c], ml[c]),
        "auroc_error_mean_entropy": auroc(me[~c], me[c]),
    }

    # ---- steering -----------------------------------------------------------------------------
    for key in ("steering", "steering_maxent"):
        S = [r[key] for r in allr if r.get(key)]
        summary[key] = {
            "n": len(S),
            "lam0_mean": float(np.mean([s["lam0"] for s in S])),
            **{f"{m}_{d}": np.mean([s[f"{m}_{d}"] for s in S], 0).tolist()
               for m in ("kl", "p_orig") for d in ("prior", "random")},
        }
    json.dump(summary, open(RES / "summary.json", "w"), indent=2)
    fig_prior_fit(summary)

    # ---- figure: lambda per module application ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for lang in langs:
        ax.plot(MODULES, summary["languages"][lang]["lambda_per_module"], marker="o", label=lang)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"mean prior loading $\lambda$ (mix prior)")
    ax.set_xlabel("module application (execution order)")
    ax.set_title("Prior loading across Mimir's recurrence (response tokens)", fontsize=10)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "lambda_per_module.png", dpi=150)
    plt.close(fig)

    # ---- figure: lambda vs response position, and the mismatched-question control --------------
    T, k = 300, 9  # positions shown, smoothing window

    def curve(lang, fn):
        M = np.full((len(data[lang]), T), np.nan)
        for i, r in enumerate(data[lang]):
            v = fn(r)[:T]
            M[i, : len(v)] = v
        mu = np.nanmean(M, 0)
        mu[(~np.isnan(M)).sum(0) < 20] = np.nan
        return np.convolve(mu, np.ones(k) / k, mode="same")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for lang in langs:
        axes[0].plot(curve(lang, lambda r: r["lam"][P_MIX, -1]), label=lang)
        axes[1].plot(curve(lang, lambda r: r["lammis"] - r["lam"][P_MIX, -1]), label=lang)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("response token position")
    axes[0].set_ylabel(r"$\lambda$ (final state, mix prior)")
    axes[0].set_title("prior loading while answering", fontsize=10)
    axes[1].set_ylabel(r"$\lambda_{other\ question} - \lambda_{own\ question}$")
    axes[1].set_title("control: same response after a different question", fontsize=10)
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "lambda_vs_position.png", dpi=150)
    plt.close(fig)

    # ---- figure: language-specific priors -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(langs))
    w = 0.38
    ax.bar(x - w / 2, [summary["languages"][lang]["mean_lambda"]["pile"] for lang in langs], w, label="English prior (Pile)")
    ax.bar(x + w / 2, [summary["languages"][lang]["mean_lambda"]["dynaword"] for lang in langs], w, label="Danish prior (Dynaword)")
    ax.set_xticks(x, langs)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"mean $\lambda$ over response tokens")
    ax.set_title("Which language's prior does Mimir load on?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "language_priors.png", dpi=150)
    plt.close(fig)

    # ---- figure: steering ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in zip(axes, ["steering_maxent", "steering"],
                              ["most uncertain response token", "answer onset (first token of the final answer)"]):
        s = summary[key]
        ax.plot(GRID, s["kl_prior"], "o-", label=r"along $d_{prior}$")
        ax.plot(GRID, s["kl_random"], "s--", label="random direction (matched norm)")
        ax.axvline(s["lam0_mean"], color="gray", ls=":", label=fr"natural $\lambda_0$={s['lam0_mean']:.2f}")
        ax.set_xlabel(r"patched prior loading $\lambda^*$")
        ax.set_title(f"{title} (n={s['n']})", fontsize=10)
    axes[0].set_ylabel(r"$D_{KL}(p_{\lambda^*} \| p_{unigram})$ (nats)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "steering.png", dpi=150)
    plt.close(fig)

    # ---- print a markdown table for the README -------------------------------------------------
    print("| lang | acc | direct-answer rate | acc direct | acc w/ reasoning |")
    print("|---|---|---|---|---|")
    for lang in langs:
        s = summary["languages"][lang]
        f = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"| {lang} | {s['accuracy']:.2f} | {s['direct_answer_rate']:.2f} | {f(s['accuracy_direct'])} | {f(s['accuracy_reasoning'])} |")
    print()
    print("| lang | n | acc | λ mix | λ en-prior | λ da-prior | λ mismatch | AUROC err (λ) | AUROC err (entropy) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lang in langs:
        s = summary["languages"][lang]
        print(f"| {lang} | {s['n']} | {s['accuracy']:.2f} | {s['mean_lambda']['mix']:.3f} | {s['mean_lambda']['pile']:.3f} | "
              f"{s['mean_lambda']['dynaword']:.3f} | {s['mean_lambda_mismatch']:.3f} | "
              f"{s['auroc_error_mean_lambda']:.2f} | {s['auroc_error_mean_entropy']:.2f} |")
    print(json.dumps(summary["pooled"]))
    print(json.dumps(summary["pooled_reasoning_only"]))
    for lang in langs:
        s = summary["languages"][lang]
        print(lang, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if "reasoning" in k or "direct" in k or "onset" in k})
    for key in ("steering", "steering_maxent"):
        s = summary[key]
        print(key, "lam0", round(s["lam0_mean"], 3), "kl_prior", np.round(s["kl_prior"], 2).tolist(),
              "kl_random", np.round(s["kl_random"], 2).tolist(), "p_orig_prior", np.round(s["p_orig_prior"], 2).tolist())


if __name__ == "__main__":
    main()
