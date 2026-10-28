"""Mimir on multilingual GSM-symbolic: generate answers, then read the prior loading factor lambda.

For each language: instantiate every template N times, generate greedily with the chat template and a
PrefixLM mask (as Mimir requires), score the \\boxed{} answer, and teacher-force prompt+response once
more with hooks on every L/H module application to compute, for each response token:

  lam[prior, module, t] = <d_hat, y_module(t)> / beta     (Liu et al. 2026, eq. 5)

for the three fitted priors (pile=English, dynaword=Danish, mix), plus the entropy and the KL of the
final next-token distribution to the unigram prior. Controls / interventions:
  * mismatch: the same response teacher-forced after a *different* question (context made useless)
  * steering at the answer onset (first token inside \\boxed{) and at the highest-entropy response
    token: y + (lam* - lam0) d_prior vs. a magnitude-matched random direction (paper eq. 8 / Fig. 4)

Usage: python src/run_gsm.py --langs eng dan --device cuda:0 --n 2
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import get_dataset_split_names, load_dataset
from multilingual_gsm_symbolic import load_instruction

sys.path.insert(0, str(Path(__file__).parent))
from common import StateRecorder, load, module_labels  # noqa: E402

PRIORS = ("pile", "dynaword", "mix")
STEER_GRID = np.round(np.arange(-2.0, 4.01, 0.5), 2)
OUT = Path("results/gsm")
DATASET = "danish-foundation-models/multilingual-gsm-symbolic"


def load_questions(lang: str, n: int, max_templates: int | None) -> list[dict]:
    """The first n pre-generated synthetic variants of every template (seed 42 in the dataset)."""
    split = next(s for s in get_dataset_split_names(DATASET, lang) if s.startswith("test_synthetic"))
    ds = load_dataset(DATASET, lang, split=split)
    per_template: dict[str, list[dict]] = {}
    for row in ds:
        per_template.setdefault(row["source_id"], []).append(row)
    out = []
    for sid in sorted(per_template, key=int)[:max_templates]:
        for row in per_template[sid][:n]:
            out.append({"template": sid, "question": row["question"], "gold": float(row["target"])})
    return out


def parse_number(s: str) -> float | None:
    while re.search(r"\d,\d{3}(?!\d)", s):  # thousands separators: 1,112 / 12,345,678
        s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)  # decimal comma (da/de/fr/...): 2,5
    s = s.replace("\\%", "").replace("%", "").replace("$", "").replace("\\$", "")
    s = re.sub(r"\\(text|mathrm)\{[^}]*\}", "", s)
    frac = re.fullmatch(r"\s*-?\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}\s*", s)
    if frac:
        return float(frac.group(1)) / float(frac.group(2))
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None


def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 1, i + len("\\boxed{")
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + len("\\boxed{") : j - 1]


def build_prompt(tok, instruction: str, question: str) -> list[int]:
    msgs = [{"role": "user", "content": f"{instruction}\n\n{question}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def generate(tok, model, prompts: list[list[int]], device, batch_size=32, max_new_tokens=768):
    outs = []
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b : b + batch_size]
        L = max(map(len, chunk))
        ids = torch.full((len(chunk), L), tok.pad_token_id)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, p in enumerate(chunk):  # left padding
            ids[i, L - len(p) :] = torch.tensor(p)
            att[i, L - len(p) :] = 1
        tt = att.clone()  # PrefixLM: the whole (non-pad) prompt is one bidirectional block
        gen = model.generate(
            input_ids=ids.to(device),
            attention_mask=att.to(device),
            token_type_ids=tt.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        for row in gen[:, L:].tolist():
            if tok.eos_token_id in row:
                row = row[: row.index(tok.eos_token_id) + 1]
            outs.append([t for t in row if t != tok.pad_token_id])
        print(f"  generated {len(outs)}/{len(prompts)}", flush=True)
    return outs


class Readout:
    def __init__(self, model, device):
        self.W = model.lm_head.weight.detach().float()
        self.p = {}
        for name in PRIORS:
            fit = torch.load(f"results/prior_{name}.pt", weights_only=False)
            d = fit["d_prior"].float().to(device)
            self.p[name] = {
                "d": d,
                "beta": d.norm(),
                "dhat": d / d.norm(),
                "support": fit["support"].to(device),
                "log_uni": fit["log_punigram"].float().to(device),
            }

    def lam(self, Y: torch.Tensor, name: str) -> torch.Tensor:  # Y [..., d]
        pr = self.p[name]
        return Y @ pr["dhat"] / pr["beta"]

    def kl_to_unigram(self, Y: torch.Tensor, name="mix") -> torch.Tensor:
        pr = self.p[name]
        logp = torch.log_softmax(Y @ self.W[pr["support"]].T, -1)  # restricted & renormalised
        return (logp.exp() * (logp - pr["log_uni"])).sum(-1)

    def entropy(self, Y: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(Y @ self.W.T, -1)
        return -(logp.exp() * logp).sum(-1)


@torch.no_grad()
def teacher_force(model, rec: StateRecorder, prompt: list[int], resp: list[int], device):
    ids = torch.tensor([prompt + resp], device=device)
    tt = torch.zeros_like(ids)
    tt[0, : len(prompt)] = 1
    rec.clear()
    model(input_ids=ids, token_type_ids=tt, use_cache=False)
    # state at position k predicts token k+1 -> the response tokens are predicted from P-1 .. P+R-2
    sl = slice(len(prompt) - 1, len(prompt) + len(resp) - 1)
    return torch.stack([s[0, sl] for s in rec.states])  # [modules, R, d]


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_answer(text: str) -> tuple[float | None, int | None]:
    """The model's answer and its char offset: the last \\boxed{} if present (as instructed), else the
    last number in the response (Mimir often ends English answers with 'The final answer is: $X$')."""
    i = text.rfind("\\boxed{")
    if i >= 0:
        boxed = extract_boxed(text)
        return (parse_number(boxed) if boxed is not None else None), i + len("\\boxed{")
    nums = list(NUMBER.finditer(text))
    if not nums:
        return None, None
    return parse_number(nums[-1].group()), nums[-1].start()


def answer_onset(tok, resp: list[int], char_pos: int | None) -> int | None:
    """Index t of the response token in which the final answer starts (Y[:, t] predicts resp[t])."""
    if char_pos is None:
        return None
    for j in range(len(resp)):
        if len(tok.decode(resp[: j + 1], skip_special_tokens=True)) > char_pos:
            return j
    return None


@torch.no_grad()
def steer(ro: Readout, y: torch.Tensor, orig_tok: int, gen: torch.Generator, name="mix"):
    pr = ro.p[name]
    lam0 = (y @ pr["dhat"] / pr["beta"]).item()
    u = torch.randn(y.shape, generator=gen).to(y.device)
    u = u / u.norm()
    res = {"lam0": lam0, "kl_prior": [], "kl_random": [], "p_orig_prior": [], "p_orig_random": []}
    for target in STEER_GRID:
        delta = (target - lam0) * pr["beta"]
        for key, direction in (("prior", pr["dhat"]), ("random", u)):
            ys = y + delta * direction
            res[f"kl_{key}"].append(ro.kl_to_unigram(ys[None], name).item())
            res[f"p_orig_{key}"].append(torch.softmax(ys @ ro.W.T, -1)[orig_tok].item())
    return res


def run_language(lang: str, tok, model, ro: Readout, n: int, device: str, max_templates=None, batch_size=32) -> None:
    questions = load_questions(lang, n, max_templates)
    instruction = load_instruction(lang)
    prompts = [build_prompt(tok, instruction, q["question"]) for q in questions]
    print(f"[{lang}] {len(prompts)} questions", flush=True)
    responses = generate(tok, model, prompts, device, batch_size=batch_size)

    rec = StateRecorder(model)
    rng = random.Random(0)
    gen = torch.Generator().manual_seed(0)
    records, arrays = [], {}
    for i, (q, prompt, resp) in enumerate(zip(questions, prompts, responses)):
        text = tok.decode(resp, skip_special_tokens=True)
        pred, char_pos = final_answer(text)
        correct = pred is not None and abs(pred - q["gold"]) < 1e-4 * max(1, abs(q["gold"]))
        Y = teacher_force(model, rec, prompt, resp, device)  # [8, R, d]
        lam = np.stack([ro.lam(Y, p).cpu().numpy() for p in PRIORS])  # [3, 8, R]
        yf = Y[-1]
        ent = ro.entropy(yf).cpu().numpy()
        kl = ro.kl_to_unigram(yf).cpu().numpy()

        # control: same response, but conditioned on another (random) question of the same language
        j = rng.choice([k for k in range(len(questions)) if questions[k]["template"] != q["template"]])
        Ym = teacher_force(model, rec, prompts[j], resp, device)
        lam_mismatch = ro.lam(Ym[-1], "mix").cpu().numpy()

        onset = answer_onset(tok, resp, char_pos)
        steering = steer(ro, yf[onset], resp[onset], gen) if onset is not None else None
        hi = int(ent.argmax())  # the most uncertain response position
        steering_maxent = steer(ro, yf[hi], resp[hi], gen)

        arrays[f"lam_{i}"] = lam.astype(np.float32)
        arrays[f"ent_{i}"] = ent.astype(np.float32)
        arrays[f"kl_{i}"] = kl.astype(np.float32)
        arrays[f"lammis_{i}"] = lam_mismatch.astype(np.float32)
        arrays[f"tok_{i}"] = np.array(resp, dtype=np.int32)
        records.append(
            {
                "lang": lang,
                "idx": i,
                **q,
                "response": text,
                "pred": pred,
                "correct": bool(correct),
                "boxed": "\\boxed{" in text,
                "n_resp_tokens": len(resp),
                "finished": resp[-1] == tok.eos_token_id if resp else False,
                "onset": onset,
                "steering": steering,
                "maxent_pos": hi,
                "steering_maxent": steering_maxent,
            }
        )
    rec.remove()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{lang}.npz", **arrays)
    with open(OUT / f"{lang}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = np.mean([r["correct"] for r in records])
    print(f"[{lang}] accuracy {acc:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=2, help="instances per template")
    ap.add_argument("--batch-size", type=int, default=32, help="lower it if the KV cache OOMs")
    ap.add_argument("--max-templates", type=int, default=None, help="for smoke tests")
    args = ap.parse_args()
    torch.manual_seed(0)
    tok, model = load(args.device)
    ro = Readout(model, args.device)
    json.dump({"modules": module_labels(model), "priors": PRIORS, "steer_grid": STEER_GRID.tolist()},
              open(Path("results") / "meta.json", "w"))
    for lang in args.langs:
        run_language(lang, tok, model, ro, args.n, args.device, args.max_templates, args.batch_size)


if __name__ == "__main__":
    main()
