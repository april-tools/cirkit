import torch


def reparam_id(p: torch.Tensor) -> torch.Tensor:
    """No reparametrization.

    This is an identity function on tensors.

    Args:
        p: The parameters tensor.

    Returns:
        torch.Tensor: No-op, p itself.
    """
    return p


def reparam_exp(p: torch.Tensor) -> torch.Tensor:
    """Reparametrize parameters via exponentiation.

    Args:
        p: The parameters tensor.

    Returns:
        torch.Tensor: The element-wise exponentiation of p.
    """
    return torch.exp(p)


def reparam_square(p: torch.Tensor) -> torch.Tensor:
    """Reparametrize parameters via squaring.

    Args:
        p: The parameters tensor.

    Returns:
        torch.Tensor: The element-wise squaring of p.
    """
    return torch.square(p)


def reparam_softmax(p: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Reparametrize parameters via softmax.

    Args:
        p: The parameters tensor.
        dim: The dimension along which apply the softmax.

    Returns:
        torch.Tensor: The softmax of p along the given dimension.
    """
    return torch.softmax(p, dim=dim)


def reparam_log_softmax(p: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Reparametrize parameters via log-softmax.

    Args:
        p: The parameters tensor.
        dim: The dimension along which apply the log-softmax.

    Returns:
        torch.Tensor: The log-softmax of p along the given dimension.
    """
    return torch.log_softmax(p, dim=dim)


def reparam_positive(p: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Reparameterize parameters to be positive with a given threshold.

    Args:
        p: The parameters tensor.
        eps: The minimum positive value.

    Returns:
        torch.Tensor: The element-wise clamping of p with eps as minimum value.
    """
    assert eps > 0.0, "The epsilon value should be positive"
    return torch.clamp(p, min=eps)
