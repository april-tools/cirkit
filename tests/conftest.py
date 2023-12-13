import pytest
import torch

from cirkit.utils import set_determinism


@pytest.fixture(autouse=True)
def _setup_global_state() -> None:
    torch.set_default_dtype(torch.float64)  # type: ignore[no-untyped-call]
    set_determinism()
