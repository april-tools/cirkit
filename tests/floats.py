import numpy as np
import torch
from numpy.typing import NDArray

DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-12


def isclose(
    a: float | NDArray[np.float16 | np.float32 | np.float64] | torch.Tensor,
    b: float | NDArray[np.float16 | np.float32 | np.float64] | torch.Tensor,
    /,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> NDArray[np.bool_]:
    """Proxy torch.isclose/np.isclose with a different global default rtol and atol.

    Args:
        a (Union[float, NDArray[Union[np.float16, np.float32, np.float64]], torch.Tensor]): \
            The first to compare.
        b (Union[float, NDArray[Union[np.float16, np.float32, np.float64]], torch.Tensor]): \
            The second to compare.
        rtol (float, optional): The relative tolerance. Defaults to 1e-8.
        atol (float, optional): The absolute tolerance. Defaults to 1e-12.

    Returns:
        NDArray[np.bool_]: The result of isclose.
    """
    if isinstance(a, float):
        a = np.array(a)
    elif isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, float):
        b = np.array(b)
    elif isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    return np.isclose(a, b, rtol=rtol, atol=atol)


def allclose(
    a: float | NDArray[np.float16 | np.float32 | np.float64] | torch.Tensor,
    b: float | NDArray[np.float16 | np.float32 | np.float64] | torch.Tensor,
    /,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> bool:
    """Proxy torch.allclose/np.allclose with a different global default rtol and atol.

    Args:
        a (Union[float, NDArray[Union[np.float16, np.float32, np.float64]], torch.Tensor]): \
            The first to compare.
        b (Union[float, NDArray[Union[np.float16, np.float32, np.float64]], torch.Tensor]): \
            The second to compare.
        rtol (float, optional): The relative tolerance. Defaults to 1e-8.
        atol (float, optional): The absolute tolerance. Defaults to 1e-12.

    Returns:
        NDArray[np.bool_]: The result of allclose.
    """
    return np.all(isclose(a, b, rtol=rtol, atol=atol)).item()
