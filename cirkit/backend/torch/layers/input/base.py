from abc import ABC

from cirkit.backend.torch.layers import TorchLayer


class TorchInputLayer(ABC, TorchLayer):
    """The abstract base class for input layers."""

    # NOTE: We use exactly the sae interface (H, *B, K) -> (*B, K) for __call__ of input layers:
    #           1. Define arity(H)=num_channels(C), reusing the H dimension.
    #           2. Define num_input_units(K)=num_vars(D), which reuses the K dimension.
    #       For dimension D (variables), we should parse the input in circuit according to the
    #       scope of the corresponding region node/symbolic input layer.

    def __init__(self, *, num_variables: int, num_output_units: int, num_channels: int) -> None:
        """Init class.

        Args:
            num_variables (int): The number of variables.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels.
        """
        super().__init__(
            num_input_units=num_variables, num_output_units=num_output_units, arity=num_channels
        )

    @property
    def num_variables(self) -> int:
        return self.num_input_units

    @property
    def num_channels(self) -> int:
        return self.num_channels
