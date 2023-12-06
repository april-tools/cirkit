from typing import Union

import numpy as np
import torch


def isclose(
        a: Union[float, np.ndarray, torch.Tensor],
        b: Union[float, np.ndarray, torch.Tensor],
        rtol: float = 1e-10,
        atol: float = 1e-12,
) -> Union[bool, np.ndarray, torch.Tensor]:
    if isinstance(a, float):
        a = np.array(a)
    if isinstance(b, float):
        b = np.array(b)
    if isinstance(a, np.ndarray) and isinstance(b, torch.Tensor):
        a = torch.tensor(a, device=b.device)
    elif isinstance(a, torch.Tensor) and isinstance(b, np.ndarray):
        b = torch.tensor(b, device=a.device)

    if isinstance(a, np.ndarray):  # iff isinstance(b, np.ndarray)
        res = np.isclose(a, b, rtol=rtol, atol=atol)
    else: # isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        res = torch.isclose(a, b, rtol=rtol, atol=atol)
    if res.numel() == 1:
        return res.squeeze().item()
    return res


def allclose(
    a: Union[float, np.ndarray, torch.Tensor],
    b: Union[float, np.ndarray, torch.Tensor],
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> bool:
    res = isclose(a, b, rtol=rtol, atol=atol)
    if isinstance(res, np.ndarray):
        return np.all(res).item()
    elif isinstance(res, torch.Tensor):
        return torch.all(res).item()
    return res
