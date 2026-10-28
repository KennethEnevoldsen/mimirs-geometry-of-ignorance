"""Precompute the logit shift induced by the direction of ignorance: W @ d_prior (shape: vocab)."""

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL = "danish-foundation-models/DFM-Mimir"


def main() -> None:
    path = hf_hub_download(MODEL, "model.safetensors")
    with safe_open(path, "pt") as f:
        W = f.get_tensor("lm_head.weight").double()
    for prior in ("pile", "dynaword", "mix"):
        fit = torch.load(f"results/prior_{prior}.pt", weights_only=False)
        shift = (W @ fit["d_prior"]).numpy().astype(np.float32)
        shift -= shift.mean()  # a constant added to all logits is irrelevant; centring keeps it small
        np.save(f"results/Wd_{prior}.npy", shift)
        print(prior, "shift std", shift.std().round(3), "range", shift.min().round(2), shift.max().round(2))


if __name__ == "__main__":
    main()
