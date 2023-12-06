# pylint: disable=missing-function-docstring,missing-return-doc
# TODO: disable checking for docstrings for every test file in tests/

from typing import Union

import numpy as np
import torch
from numpy.typing import NDArray


def isclose(
    a: Union[float, NDArray[Union[np.float16, np.float32, np.float64]], torch.Tensor],
    b: Union[float, NDArray[Union[np.float16, np.float32, np.float64]], torch.Tensor],
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> NDArray[np.bool_]:
    if isinstance(a, float):
        a = np.array(a)
    elif isinstance(a, torch.Tensor):  # type: ignore[misc]
        a = a.detach().cpu().numpy()
    if isinstance(b, float):
        b = np.array(b)
    elif isinstance(b, torch.Tensor):  # type: ignore[misc]
        b = b.detach().cpu().numpy()
    return np.isclose(a, b, rtol=rtol, atol=atol)


def allclose(
    a: Union[float, NDArray[Union[np.float16, np.float32, np.float64]], torch.Tensor],
    b: Union[float, NDArray[Union[np.float16, np.float32, np.float64]], torch.Tensor],
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> bool:
    return np.all(isclose(a, b, rtol=rtol, atol=atol)).item()
