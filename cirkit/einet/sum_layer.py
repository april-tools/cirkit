from abc import abstractmethod

from .layer import Layer

# TODO: rework docstrings


class SumLayer(Layer):
    """Implements an abstract SumLayer class. Takes care of parameters and EM.

    EinsumLayer and MixingLayer are derived from SumLayer.
    """

    @abstractmethod
    def num_of_param(self) -> int:
        """Get the number of params.

        Returns:
            int: the number of params
        """
        # TODO: use a property instead for this kind of thing?
