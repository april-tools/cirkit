from abc import abstractmethod
from typing import Any, Literal

from torch import Tensor

from cirkit.layers import Layer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.type_aliases import ReparamFactory


class InputLayer(Layer):
    """The abstract base for all input layers."""

    def __init__(  # type: ignore[misc]  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_channels: int = 1,
        num_replicas: int = 1,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        num_folds: Literal[0] = 0,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamIdentity,
        **_: Any,
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
            num_folds (Literal[0], optional): The number of folds. Should not be provided and will \
                be calculated as num_vars*num_replicas. Defaults to 0.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        # The input to input layer is the variable itself
        assert num_vars > 0, "The number of variables must be positive."
        assert num_channels > 0, "The number of channels must be positive."
        assert num_replicas > 0, "The number of replicas must be positive."
        assert num_input_units == 1, "We define num_input_units is 1 for InputLayer."
        assert arity == 1, "We define arity is 1 for InputLayer."
        assert not num_folds, "We calculate num_folds instead of pass-in for InputLayer."
        assert fold_mask is None, "InputLayer should not mask folds."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_vars * num_replicas,
            fold_mask=None,
            reparam=reparam,
        )
        self.num_vars = num_vars
        self.num_channels = num_channels
        self.num_replicas = num_replicas

    # TODO: any good way to sync the docstring?
    # TODO: temp solution to accomodate IntegralInputLayer
    # TODO: should InputLayer be a subclass of Layer? We can remove Layer and use two top-level
    #       layer class InputLayer and InnerLayer, because input is different in many ways.
    #       OR, we can make Layer more abstract and introduce InnerLayer to be as concrete as the
    #           current Layer.
    #       This affects the pylint error of __call__ and how we design integrate and other ops.
    # The shapes for InputLayer is different, so we use an override to change the docstring, which
    # triggers pylint error.
    # pylint: disable-next=useless-parent-delegation
    def __call__(self, x: Tensor, *_: Any) -> Tensor:  # type: ignore[misc]
        """Invoke the forward function.

        Args:
            x (Tensor): The input to this layer, shape (*B, D, C).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """
        return super().__call__(x, *_)  # type: ignore[misc]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (*B, D, C).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """

    # TODO: integrate() interface and docstring is not checked. (shapes?)
    # TODO: this should be in Layer? for some layers it's no-op but interface should exist
    @abstractmethod
    def integrate(self) -> Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            Tensor: The integration of the layer over all variables.
        """
