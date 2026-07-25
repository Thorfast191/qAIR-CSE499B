import os
import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=False):
    """
    Seeds python, numpy and torch.

    This function existed for the project's whole history and was never
    called from any entry point, so no run was reproducible: collate_fn
    draws a fresh torch.randperm per batch to shuffle options, dropout is
    active throughout, and hypothesis generation samples. main.py,
    runner.py and ablations.py all call it now.

    deterministic=True additionally forces deterministic cuDNN kernels.
    That costs throughput and is not needed for run-to-run comparability
    (seeding alone gives that on a fixed device), so it is off by default
    -- turn it on for the final numbers that go in a paper.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[SEED] Set to {seed} (deterministic={deterministic})")


def seeded_generator(seed=42):
    """
    Generator for DataLoader shuffling, so batch ORDER is reproducible
    too and not just weight init.
    """

    g = torch.Generator()
    g.manual_seed(seed)
    return g
