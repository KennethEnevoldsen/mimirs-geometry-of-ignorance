"""Summarise results/steer/*.jsonl: accuracy per (config, language), paired exact McNemar vs. baseline."""

import json
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RES = Path("results/steer")
LANGS = ["eng", "dan", "nob", "deu"]
BASE = "final:0"


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    rows = {}
    for f in RES.glob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            rows.setdefault((r["config"], r["lang"]), []).append(r)
    configs = sorted({c for c, _ in rows}, key=lambda c: (c.split(":")[0] != "final", c.split(":")[0], float(c.split(":")[1])))
    summary = {}
    print("| config | " + " | ".join(LANGS) + " | all | Δ vs baseline | McNemar p | direct-answer rate | mean tokens |")
    print("|---" * (len(LANGS) + 6) + "|")
    for c in configs:
        accs, b_all, c_all, corr, base_corr, direct, ntok = [], 0, 0, [], [], [], []
        for lang in LANGS:
            if (c, lang) not in rows or (BASE, lang) not in rows:
                accs.append(None)
                continue
            x = np.array([r["correct"] for r in rows[(c, lang)]])
            y = np.array([r["correct"] for r in rows[(BASE, lang)]])
            accs.append(x.mean())
            corr += list(x)
            base_corr += list(y)
            b_all += int((x & ~y).sum())
            c_all += int((~x & y).sum())
            direct += [r["direct"] for r in rows[(c, lang)]]
            ntok += [r["n_resp_tokens"] for r in rows[(c, lang)]]
        if not corr:
            continue
        delta = np.mean(corr) - np.mean(base_corr)
        p = mcnemar_p(b_all, c_all)
        summary[c] = {"acc": dict(zip(LANGS, accs)), "acc_all": float(np.mean(corr)), "delta": float(delta),
                      "wins": b_all, "losses": c_all, "mcnemar_p": p, "direct_rate": float(np.mean(direct)),
                      "mean_tokens": float(np.mean(ntok)), "n": len(corr)}
        cells = " | ".join("–" if a is None else f"{a:.3f}" for a in accs)
        print(f"| {c} | {cells} | {np.mean(corr):.3f} | {delta:+.3f} ({b_all}↑/{c_all}↓) | {p:.3g} | {np.mean(direct):.2f} | {np.mean(ntok):.0f} |")
    # where does the change come from? split by whether the *baseline* answered without reasoning
    print()
    print("| config | lang | direct rate | acc (baseline reasoned) | acc (baseline answered directly) |")
    print("|---|---|---|---|---|")
    for c in configs:
        for lang in LANGS:
            if (c, lang) not in rows or (BASE, lang) not in rows:
                continue
            x, base = rows[(c, lang)], rows[(BASE, lang)]
            reasoned = [a["correct"] for a, b in zip(x, base) if not b["direct"]]
            direct = [a["correct"] for a, b in zip(x, base) if b["direct"]]
            f = lambda v: f"{np.mean(v):.3f} (n={len(v)})" if v else "–"  # noqa: E731
            print(f"| {c} | {lang} | {np.mean([a['direct'] for a in x]):.2f} | {f(reasoned)} | {f(direct)} |")
            summary[c].setdefault("by_lang", {})[lang] = {
                "direct_rate": float(np.mean([a["direct"] for a in x])),
                "acc_where_baseline_reasoned": float(np.mean(reasoned)) if reasoned else None,
                "acc_where_baseline_direct": float(np.mean(direct)) if direct else None,
            }
    json.dump(summary, open("results/steer_summary.json", "w"), indent=2)

    # figure: accuracy vs delta for the final-state intervention, per language
    finals = sorted((float(c.split(":")[1]), c) for c in summary if c.startswith("final:"))
    if finals:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for lang in LANGS:
            xs = [d for d, c in finals if summary[c]["acc"][lang] is not None]
            ys = [summary[c]["acc"][lang] for d, c in finals if summary[c]["acc"][lang] is not None]
            ax.plot(xs, ys, "o-", label=lang)
        ax.plot([d for d, _ in finals], [summary[c]["acc_all"] for _, c in finals], "k--", lw=2, label="all (mean)")
        for c, m in (("random:-0.5", "x"), ("h1:-0.5", "^"), ("h1:-0.25", "v")):
            if c in summary:
                ax.scatter([float(c.split(":")[1])], [summary[c]["acc_all"]], marker=m, color="k", zorder=5,
                           s=60, label=f"{c} (all)")
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"shift along $d_{prior}$, $\Delta\lambda$  (<0: rely less on the prior)")
        ax.set_ylabel("GSM-symbolic accuracy")
        ax.set_title("Does relying less on the unigram prior help?", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig("figures/steer_accuracy.png", dpi=150)


if __name__ == "__main__":
    main()
