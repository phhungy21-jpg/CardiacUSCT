"""Fixed-seed helper for fourier_inverse/ (numpy-only; no torch here)."""

import random

import numpy as np

SEED = 42


def set_all_seeds(seed: int = SEED) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
