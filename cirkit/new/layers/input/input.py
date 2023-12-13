from abc import abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Type

from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import Reparameterization


class InputLayer(Layer):
    """The abstract base class for input layers."""

    # NOTE: We use exactly the same interface (H, *B, K)->(*B, K) for __call__ of input layers:
    #           1. Define arity(H)=1, which is simply an unsqueeze of input.
    #           2. Define num_input_units(K)=num_channels(C), which reuses the K dimension.
    #       For dimension D (variables), we should parse the input in circuit according to the
    #       scope of the corresponding region node/symbolic input layer.
    # TODO: currently we only support fully factorized input, so the input layer only works as
    #       univariate functions. If we want to extend to more complicated cases, we can allow H>1,
    #       and reuse the H dim as num_vars.

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[1] = 1,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if the layer has no params. Defaults to None.
        """
        assert arity == 1, "We define arity=1 for InputLayer."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

    @classmethod
    @abstractmethod
    def get_integral(  # type: ignore[misc]  # Ignore: Unavoidable for kwargs.
        cls, **kwargs: Any
    ) -> Tuple[Type["InputLayer"], Dict[str, Any]]:
        """Get the config to construct the integral of the input layer.

        Args:
            **kwargs (Any): The additional kwargs for this layer,

        Returns:
            Tuple[Type[InputLayer], Dict[str, Any]]: The class of the integral layer and its \
                additional kwargs.
        """
