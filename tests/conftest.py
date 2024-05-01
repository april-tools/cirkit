import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _setup_reproducible_global_state() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float64)  # type: ignore[no-untyped-call]
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
