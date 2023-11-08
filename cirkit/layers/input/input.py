from abc import abstractmethod
from typing import Literal

from torch import Tensor

from cirkit.layers import Layer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.type_aliases import ReparamFactory


class InputLayer(Layer):
    """The abstract base for all input layers."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_channels: int = 1,
        num_replicas: int = 1,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        num_folds: Literal[-1] = -1,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Init class.

        Args:
            num_vars (int): The number of variables of the circuit.
            num_channels (int, optional): The number of channels of each variable. Defaults to 1.
            num_replicas (int, optional): The number of replicas for each variable. Defaults to 1.
            num_input_units (Literal[1], optional): The number of input units, must be 1. \
                Defaults to 1.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            num_folds (Literal[-1], optional): The number of folds, unused. The number of folds \
                should be num_vars*num_replicas. Defaults to -1.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        # The input to input layer is the variable itself
        assert num_input_units == 1, "We define num_input_units is 1 for InputLayer."
        assert arity == 1, "We define arity is 1 for InputLayer."
        assert fold_mask is None, "InputLayer should not mask folds."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_vars * num_replicas,
            fold_mask=fold_mask,
            reparam=reparam,
        )
        self.num_vars = num_vars
        self.num_channels = num_channels
        self.num_replicas = num_replicas

    # TODO: integrate() interface and docstring is not checked.
    # TODO: this should be in Layer? for some layers it's no-op but interface should exist
    @abstractmethod
    def integrate(self) -> Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            Tensor: The integration of the layer over all variables.
        """
