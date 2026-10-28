"""Patch EuroEval's model loading so every forward pass is steered along Mimir's direction of ignorance.

Imported automatically by Python because this directory is put on PYTHONPATH (see `src/run_euroeval.sh`).

Env vars:
  MIMIR_STEER_DELTA  shift in lambda units (default 0 = no intervention)
  MIMIR_STEER_PRIOR  which fitted prior to use: pile | dynaword | mix (default mix)

The intervention is applied to the output of the *final* H-module application, i.e. to the state that
the unembedding turns into logits. Because the unembedding is linear this is exactly
`logits + delta * W d_prior`, a tempered subtraction of the unigram prior (see README, section 6).
"""

import os

import torch

DELTA = float(os.environ.get("MIMIR_STEER_DELTA", "0"))
PRIOR = os.environ.get("MIMIR_STEER_PRIOR", "mix")
PRIOR_FILE = os.environ.get("MIMIR_STEER_PRIOR_FILE", "results/prior_mix.pt")
SHIFT_FILE = os.environ.get("MIMIR_SHIFT_FILE", f"results/Wd_{PRIOR}.npy")
BACKEND = os.environ.get("MIMIR_STEER_BACKEND", "vllm")
DEBUG = os.environ.get("MIMIR_STEER_DEBUG", "") not in ("", "0")

_patched = False
_installed: set[int] = set()


def _attach(model: torch.nn.Module) -> torch.nn.Module:
    if DELTA == 0 or id(model) in _installed:
        return model
    if not hasattr(model, "model") or not hasattr(model.model, "H_module"):
        return model  # not an HRM model
    fit = torch.load(PRIOR_FILE, weights_only=False)
    direction = fit["d_prior"].float()
    n_h = model.config.H_cycles
    counter = {"n": 0}

    def _reset(*_args, **_kwargs):
        counter["n"] = 0

    def _hook(_module, _inputs, output):
        idx, counter["n"] = counter["n"], counter["n"] + 1
        if idx != n_h - 1:
            return output
        return output + DELTA * direction.to(device=output.device, dtype=output.dtype)

    model.model.register_forward_pre_hook(_reset)
    model.model.H_module.register_forward_hook(_hook)
    _installed.add(id(model))
    print(f"[mimir-steer] attached delta={DELTA} prior={PRIOR} ({PRIOR_FILE})", flush=True)
    return model


def _patch() -> None:
    global _patched
    if _patched:
        return
    from transformers import AutoModel, AutoModelForCausalLM

    for cls in (AutoModelForCausalLM, AutoModel):
        original = cls.from_pretrained.__func__  # classmethod's underlying function

        def wrapper(cls_, *args, __original=original, **kwargs):
            return _attach(__original(cls_, *args, **kwargs))

        cls.from_pretrained = classmethod(wrapper)
    _patched = True
    if DEBUG:
        print("[mimir-steer] from_pretrained patched", flush=True)


def _force_hf_backend() -> None:
    """Stop EuroEval from picking vLLM: our intervention needs the HF module's forward passes."""
    from euroeval.benchmark_modules.vllm import VLLMModel

    VLLMModel.model_exists = classmethod(lambda cls, model_id, benchmark_config: False)
    print("[mimir-steer] forced the Hugging Face backend (vLLM disabled)", flush=True)


def _patch_vllm() -> None:
    """Steer inside EuroEval's vLLM backend.

    vLLM computes logits with `HrmTextForCausalLM.compute_logits`, which feeds both sampling and the
    prompt/token log-probabilities used to score multiple-choice answers. Adding `delta * W d_prior`
    there is exactly the same intervention as the HF hook below (the unembedding is linear).
    """
    import numpy as np
    from vllm.model_executor.models.hrm_text import HrmTextForCausalLM

    shift = torch.from_numpy(np.load(SHIFT_FILE)).float()
    original = HrmTextForCausalLM.compute_logits

    def compute_logits(self, hidden_states):
        logits = original(self, hidden_states)
        if logits is None:
            return logits
        return logits + DELTA * shift.to(device=logits.device, dtype=logits.dtype)

    HrmTextForCausalLM.compute_logits = compute_logits
    print(f"[mimir-steer] vLLM compute_logits patched: delta={DELTA} prior={PRIOR} ({SHIFT_FILE})", flush=True)


if DELTA != 0:
    if BACKEND == "vllm":
        _patch_vllm()
    else:
        _patch()
if os.environ.get("MIMIR_FORCE_HF", "0") not in ("", "0"):
    _force_hf_backend()


