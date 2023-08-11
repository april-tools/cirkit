from typing import Callable, Optional

import torch

#: A re-parametrization function takes as input a parameters tensor,
#: an optional folding mask (see folding), and returns the re-parametrized tensor.
ReparamFunction = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


def reparam_id(p: torch.Tensor, fold_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """No reparametrization.

    This is an identity function on tensors.

    Args:
        p: The parameters tensor.
        fold_mask: The folding mask.

    Returns:
        torch.Tensor: No-op, p itself.
    """
    if fold_mask is not None:
        p = p * fold_mask  # pylint: disable=consider-using-augmented-assign
    return p


def reparam_exp(p: torch.Tensor, fold_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reparametrize parameters via exponentiation.

    Args:
        p: The parameters tensor.
        fold_mask: The folding mask.

    Returns:
        torch.Tensor: The element-wise exponentiation of p.
    """
    p = torch.exp(p)
    if fold_mask is not None:
        p = p * fold_mask  # pylint: disable=consider-using-augmented-assign
    return p


def reparam_square(p: torch.Tensor, fold_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reparametrize parameters via squaring.

    Args:
        p: The parameters tensor.
        fold_mask: The folding mask.

    Returns:
        torch.Tensor: The element-wise squaring of p.
    """
    p = torch.square(p)
    if fold_mask is not None:
        p = p * fold_mask  # pylint: disable=consider-using-augmented-assign
    return p


def reparam_softmax(
    p: torch.Tensor, fold_mask: Optional[torch.Tensor] = None, *, dim: int = 0
) -> torch.Tensor:
    """Reparametrize parameters via softmax.

    Args:
        p: The parameters tensor.
        fold_mask: The folding mask.
        dim: The dimension along which apply the softmax.

    Returns:
        torch.Tensor: The softmax of p along the given dimension.
    """
    if fold_mask is not None:
        p = p + torch.log(fold_mask)
    return torch.softmax(p, dim=dim)


def reparam_log_softmax(
    p: torch.Tensor, fold_mask: Optional[torch.Tensor] = None, *, dim: int = 0
) -> torch.Tensor:
    """Reparametrize parameters via log-softmax.

    Args:
        p: The parameters tensor.
        fold_mask: The folding mask.
        dim: The dimension along which apply the log-softmax.

    Returns:
        torch.Tensor: The log-softmax of p along the given dimension.
    """
    if fold_mask is not None:
        p = p + torch.log(fold_mask)
    return torch.log_softmax(p, dim=dim)


def reparam_positive(
    p: torch.Tensor, fold_mask: Optional[torch.Tensor] = None, *, eps: float = 1e-15
) -> torch.Tensor:
    """Reparameterize parameters to be positive with a given threshold.

    Args:
        p: The parameters tensor.
        fold_mask: The folding mask.
        eps: The minimum positive value.

    Returns:
        torch.Tensor: The element-wise clamping of p with eps as minimum value.
    """
    assert eps > 0.0, "The epsilon value should be positive"
    p = torch.clamp(p, min=eps)
    if fold_mask is not None:
        p = p * fold_mask  # pylint: disable=consider-using-augmented-assign
    return p
