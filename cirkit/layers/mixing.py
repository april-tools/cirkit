from typing import Optional

import torch
from torch import Tensor, nn

from cirkit.layers.layer import Layer
from cirkit.utils.reparams import ReparamFunction, reparam_id

# TODO: rework docstrings


class MixingLayer(Layer):
    # TODO: how we fold line here?
    r"""Implement the Mixing Layer, in order to handle sum nodes with multiple children.

    Recall Figure II from above:

           S          S
        /  |  \      / \ 
       P   P  P     P  P
      /\   /\  /\  /\  /\ 
     N  N N N N N N N N N

    Figure II


    We implement such excerpt as in Figure III, splitting sum nodes with multiple \
        children in a chain of two sum nodes:

            S          S
        /   |  \      / \ 
       S    S   S    S  S
       |    |   |    |  |
       P    P   P    P  P
      /\   /\  /\   /\  /\ 
     N  N N N N N N N N N

    Figure III


    The input nodes N have already been computed. The product nodes P and the \
        first sum layer are computed using an
    SumProductLayer, yielding a log-density tensor of shape
        (batch_size, vector_length, num_nodes).
    In this example num_nodes is 5, since the are 5 product nodes (or 5 singleton \
        sum nodes). The MixingLayer
    then simply mixes sums from the first layer, to yield 2 sums. This is just an \
        over-parametrization of the original
    excerpt.
    """

    # TODO: num_output_units is num_input_units
    def __init__(
        self,
        num_input_components: int,
        num_output_units: int,
        num_folds: int = 1,
        fold_mask: Optional[torch.Tensor] = None,
        *,
        reparam: ReparamFunction = reparam_id,
    ) -> None:
        """Init class.

        Args:
            num_input_components (int): The number of mixing components.
            num_output_units (int): The number of output units.
            num_folds (int): The number of folds.
            fold_mask (Optional[torch.Tensor]): The mask to apply to the folded parameter tensors.
            reparam: The reparameterization function.
        """
        super().__init__(num_folds=num_folds, fold_mask=fold_mask)
        self.reparam = reparam
        self.num_input_components = num_input_components
        self.num_output_units = num_output_units

        self.params = nn.Parameter(
            torch.empty(self.num_folds, num_input_components, num_output_units)
        )
        self.param_clamp_value["min"] = torch.finfo(self.params.dtype).smallest_normal
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization: U(0.01, 0.99) with normalization."""
        with torch.no_grad():
            nn.init.uniform_(self.params, 0.01, 0.99)
            self.params /= self.params.sum(dim=1, keepdim=True)  # type: ignore[misc]

    def _forward(self, x: Tensor) -> Tensor:
        fold_mask = self.fold_mask.unsqueeze(dim=-1) if self.fold_mask is not None else None
        weight = self.reparam(self.params, fold_mask)
        return torch.einsum("fck,fckb->fkb", weight, x)

    # TODO: make forward return something
    # pylint: disable-next=arguments-differ
    def forward(self, log_input: Tensor) -> Tensor:  # type: ignore[override]
        """Do the forward.

        Args:
            log_input (Tensor): The input.

        Returns:
            Tensor: the output.
        """
        m: Tensor = torch.max(log_input, dim=1, keepdim=True)[0]  # (F, 1, K, B)
        x = torch.exp(log_input - m)  # (F, C, K, B)
        x = self._forward(x)  # (F, K, B)
        x = torch.log(x)
        if self.fold_mask is not None:
            x = torch.nan_to_num(x, nan=0)
            m = torch.nan_to_num(m, neginf=0)
        return x + m.squeeze(dim=1)  # (F, K, B)

    # TODO: see commit 084a3685c6c39519e42c24a65d7eb0c1b0a1cab1 for backtrack
