"""Reproducible random seeding across ``random``, NumPy and (optionally) PyTorch."""

from __future__ import annotations

import os
import random

from dork.utils.logging import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int = 1337, deterministic: bool = False) -> int:
    """Seed all RNGs we rely on for reproducibility.

    Args:
        seed: The integer seed.
        deterministic: If True, request deterministic CuDNN/cuBLAS kernels.
            This can slow training but makes runs bit-for-bit reproducible.

    Returns:
        The seed that was set (for logging/round-tripping).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is a core dep
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch is an optional extra
        pass

    logger.debug("Seeded all RNGs with seed=%d (deterministic=%s)", seed, deterministic)
    return seed
