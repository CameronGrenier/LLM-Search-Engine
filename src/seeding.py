"""Seed every random number generator the pipeline can reach.

No stage is stochastic today: generation decodes greedily with do_sample=False,
the embedding model runs in eval mode so dropout is off, and IndexFlatIP is an
exact index with no trained quantizer. Seeding anyway is cheap insurance, so
that the reproducibility claim keeps holding if any of those change -- sampled
decoding, a dropout-enabled model, or an approximate FAISS index.

PYTHONHASHSEED is deliberately not set here. CPython reads it once at
interpreter startup, so assigning it from inside a running process has no
effect on that process. reproduce.py sets it in the environment it hands to
each stage subprocess, which is the only point where it can take effect.

Call seed_everything() at import time in any module that loads a model.
"""

from __future__ import annotations

from config import RANDOM_SEED


def seed_everything(seed: int = RANDOM_SEED) -> int:
    """Seed the stdlib, numpy and torch generators.

    numpy and torch are imported lazily so that stages which need neither
    (preprocessing, chunking, the BM25 index) can still call this without
    paying for the import.

    Args:
      seed: The seed to apply. Defaults to config.RANDOM_SEED.

    Returns:
      The seed that was applied, so callers can log it.
    """
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        # Seeds the CPU generator and every initialized accelerator generator.
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # torch.mps has no manual_seed on every build, so probe before calling.
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "manual_seed"):
            if torch.backends.mps.is_available():
                mps.manual_seed(seed)
    except ImportError:
        pass

    return seed
