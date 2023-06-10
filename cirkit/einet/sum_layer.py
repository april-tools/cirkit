from abc import abstractmethod
from typing import Any, List, Literal, Optional

from torch import Tensor

from .layer import Layer

# TODO: rework docstrings


class SumLayer(Layer):
    """Implements an abstract SumLayer class. Takes care of parameters and EM.

    EinsumLayer and MixingLayer are derived from SumLayer.
    """

    # TODO: this __init__ is useless
    # # TODO: the best way to allow generic signature?
    # def __init__(self) -> None:  # TODO: restore params mask ", params_mask=None):"
    #     """Init class."""
    #     # """
    #     # :param params_shape: shape of tensor containing all sum weights (tuple of ints).
    #     # :param normalization_dims: the dimensions (axes) of the sum-weights
    #     #                   which shall be normalized
    #     #                            (int of tuple of ints)
    #     # :param params_mask: binary mask for masking out certain parameters
    #     #                   (tensor of shape params_shape).
    #     # """
    #     super().__init__()

    @abstractmethod
    def num_of_param(self) -> int:
        """Get the number of params.

        Returns:
            int: the number of params
        """
        # TODO: use a property instead for this kind of thing?

    @abstractmethod
    def _forward(self, params: Optional[Tensor] = None) -> Tensor:
        """Implement the actual sum operation.

        :param params: sum-weights to use.
        :return: result of the sum layer. Must yield a \
            (batch_size, num_dist, num_nodes) tensor of log-densities.
                 Here, num_dist is the vector length of vectorized sums \
                    (K in the paper), and num_nodes is the number
                 of sum nodes in this layer.
        """

    @abstractmethod
    # pylint: disable=too-many-arguments
    def _backtrack(  # type: ignore[misc]
        self,
        dist_idx: List[int],
        node_idx: List[int],
        sample_idx: List[int],
        params: Tensor,
        use_evidence: bool = False,
        mode: Literal["sample", "argmax"] = "sample",
        **kwargs: Any,
    ) -> Any:
        """Is a helper routine to implement EiNet backtracking, for sampling or MPE approximation.

        dist_idx, node_idx, sample_idx are lists of indices, all of the same length.

        :param dist_idx: list of indices, indexing into vectorized sums.
        :param node_idx: list of indices, indexing into node list of this layer.
        :param sample_idx: list of sample indices; representing the identity of \
                            the samples the EiNet is about to
                           generate. We need this, since not every SumLayer necessarily \
                            gets selected in the top-down
                           sampling process.
        :param params: sum-weights to use (Tensor).
        :param use_evidence: incorporate the bottom-up evidence (Bool)? For conditional sampling.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: Additional keyword arguments.
        :return: depends on particular implementation.
        """

    # TODO: the forward() return type is a mess
    def forward(self, x: Optional[Tensor] = None) -> Tensor:  # type: ignore[override]
        """Evaluate this SumLayer.

        :param x: unused
        :return: tensor of log-densities. Must be of shape (batch_size, num_dist, num_nodes).
                 Here, num_dist is the vector length of vectorized sum nodes \
                    (K in the paper), and num_nodes is the
                 number of sum nodes in this layer.
        """
        return self._forward()

    @abstractmethod
    # pylint: disable-next=arguments-differ
    def backtrack(  # type: ignore[misc]
        self,
        dist_idx: List[int],
        node_idx: List[int],
        sample_idx: List[int],
        *args: Any,
        use_evidence: bool = False,
        mode: Literal["sample", "argmax"] = "sample",
        **kwargs: Any,
    ) -> Tensor:
        """Is helper routine for backtracking in EiNets, see _sample(...) for details.

        Args:
            dist_idx (_type_): Some indices.
            node_idx (_type_): Some indices.
            sample_idx (_type_): Some indices.
            use_evidence (bool, optional): Whether to use evidence. Defaults to False.
            mode (Literal["sample", "argmax"], optional): The mode.. Defaults to "sample".
            args: Any.
            kwargs: Any.

        Returns:
            Tensor: The return.
        """
