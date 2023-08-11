import pytest
import torch

from cirkit.utils import set_determinism


@pytest.fixture(autouse=True)
def _setup_global_state() -> None:
    torch.set_grad_enabled(False)
    set_determinism()
