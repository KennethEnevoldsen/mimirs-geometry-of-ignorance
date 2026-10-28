"""Does relying less on the prior make Mimir better at math? Steer d_prior during generation.

At every forward pass we shift a module output along the (mix) direction of ignorance,
    y <- y + delta * d_prior          (delta in lambda units; d_prior has norm beta)
and re-run the GSM-symbolic evaluation. Configs:
  final:<delta>   shift the final prediction state (H2 output). Because y feeds the unembedding
                  directly, this is exactly logits + delta * W d_prior ~= logits + delta * log p_unigram
                  (+ const), i.e. a tempered prior subtraction (delta < 0) / addition (delta > 0).
  h1:<delta>      shift the output of the first H-module application, which then flows through the
                  second cycle (an internal intervention, not a logit transform).
  random:<delta>  like final, but along a fixed random unit direction with the same norm (control).

Usage: python src/steer_gsm.py --lang dan --device cuda:0 --configs final:0 final:-0.5 random:-0.5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from multilingual_gsm_symbolic import load_instruction

sys.path.insert(0, str(Path(__file__).parent))
from common import load  # noqa: E402
from run_gsm import build_prompt, final_answer, generate, load_questions  # noqa: E402

OUT = Path("results/steer")


class Steerer:
    def __init__(self, model, device, prior="mix"):
        fit = torch.load(f"results/prior_{prior}.pt", weights_only=False)
        self.d = fit["d_prior"].float().to(device)
        g = torch.Generator().manual_seed(0)
        u = torch.randn(self.d.shape, generator=g).to(device)
        self.u = u / u.norm() * self.d.norm()  # random direction, same norm as d_prior
        self.n_h = model.config.H_cycles
        self.mode, self.delta, self.calls = "final", 0.0, 0
        model.model.register_forward_pre_hook(self._reset)
        model.model.H_module.register_forward_hook(self._hook)

    def set(self, config: str):
        mode, delta = config.split(":")
        self.mode, self.delta = mode, float(delta)

    def _reset(self, *_):
        self.calls = 0

    def _hook(self, _mod, _inp, out):
        idx, self.calls = self.calls, self.calls + 1
        if self.delta == 0:
            return out
        target = 0 if self.mode == "h1" else self.n_h - 1
        if idx != target:
            return out
        vec = self.u if self.mode == "random" else self.d
        return out + self.delta * vec.to(out.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()
    tok, model = load(args.device)
    st = Steerer(model, args.device)
    questions = load_questions(args.lang, args.n, None)
    instruction = load_instruction(args.lang)
    prompts = [build_prompt(tok, instruction, q["question"]) for q in questions]
    OUT.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        path = OUT / f"{config.replace(':', '_')}__{args.lang}.jsonl"
        if path.exists():
            continue
        st.set(config)
        print(f"[{args.lang}] {config}", flush=True)
        responses = generate(tok, model, prompts, args.device, batch_size=args.batch_size)
        rows = []
        for q, resp in zip(questions, responses):
            text = tok.decode(resp, skip_special_tokens=True)
            pred, _ = final_answer(text)
            correct = pred is not None and abs(pred - q["gold"]) < 1e-4 * max(1, abs(q["gold"]))
            rows.append({**q, "config": config, "lang": args.lang, "response": text, "pred": pred,
                         "correct": bool(correct), "direct": text.strip().startswith("\\boxed"),
                         "n_resp_tokens": len(resp), "finished": bool(resp) and resp[-1] == tok.eos_token_id})
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{args.lang}] {config} acc={sum(r['correct'] for r in rows) / len(rows):.3f}", flush=True)


if __name__ == "__main__":
    main()
