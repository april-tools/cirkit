import random
from typing import Tuple

import numpy as np
import torch

from cirkit.utils import RandomCtx


def rand() -> Tuple[float, float, float]:
    return (random.random(), np.random.rand(), torch.rand(()).item())


def test_random_ctx() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)  # type: ignore[no-untyped-call]

    assert rand() == (0.8444218515250481, 0.5488135039273248, 0.49625658988952637)

    with RandomCtx(42):
        assert rand() == (0.6394267984578837, 0.3745401188473625, 0.8822692632675171)

    assert rand() == (0.7579544029403025, 0.7151893663724195, 0.7682217955589294)


def test_random_ctx_decorator() -> None:
    torch.set_default_dtype(torch.float32)  # type: ignore[no-untyped-call]
    assert RandomCtx(42)(rand)() == (0.6394267984578837, 0.3745401188473625, 0.8822692632675171)
