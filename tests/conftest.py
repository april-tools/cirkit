import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _setup_global_state() -> None:
    # Seed all RNGs.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set deterministic algorithms.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # Use high precision computation.
    torch.set_default_dtype(torch.float64)  # type: ignore[no-untyped-call]
    # Disable autograd because we do not need it in most cases.
    torch.set_grad_enabled(False)
