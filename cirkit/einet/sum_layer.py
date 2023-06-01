import functools
from abc import abstractmethod
from typing import Any, Callable, Literal, Optional

from torch import Tensor
from torch.nn import functional as F

from .layer import Layer


class SumLayer(Layer):
    """Implements an abstract SumLayer class. Takes care of parameters and EM.

    EinsumLayer and MixingLayer are derived from SumLayer.
    """

    # TODO: the best way to allow generic signature?
    def __init__(self) -> None:  # TODO: restore params mask ", params_mask=None):"
        """Init class."""
        # """
        # :param params_shape: shape of tensor containing all sum weights (tuple of ints).
        # :param normalization_dims: the dimensions (axes) of the sum-weights
        #                   which shall be normalized
        #                            (int of tuple of ints)
        # :param use_em: use the on-board EM algorithm?
        # :param params_mask: binary mask for masking out certain parameters
        #                   (tensor of shape params_shape).
        # """
        super().__init__()
        self.frozen = False

    def freeze(self, freeze: bool = True) -> None:
        """Freeze all params from bw.

        Args:
            freeze (bool, optional): Whether to freeze or unfreeze. Defaults to True.
        """
        for param in self.parameters():
            param.requires_grad = not freeze
        self.frozen = freeze

    # TODO: do we need this? or use a property?
    def is_frozen(self) -> bool:
        """Test if is frozen.

        Returns:
            bool: Whether is frozen.
        """
        return self.frozen

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
    def _backtrack(
        self,
        dist_idx,  # TODO: types
        node_idx,
        sample_idx,
        params: Tensor,
        use_evidence: bool = False,
        mode: Literal["sample", "argmax"] = "sample",
        **kwargs: Any,
    ) -> Tensor:
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

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """Evaluate this SumLayer.

        :param x: unused
        :return: tensor of log-densities. Must be of shape (batch_size, num_dist, num_nodes).
                 Here, num_dist is the vector length of vectorized sum nodes \
                    (K in the paper), and num_nodes is the
                 number of sum nodes in this layer.
        """
        # TODO the distinction em or not has not to be done here
        # if self._use_em:
        #     params = self.params
        # else:
        #     reparam = self.reparam(self.params)
        #     params = reparam
        # params = self.params
        self._forward()

    @abstractmethod
    def backtrack(
        self,
        dist_idx,
        node_idx,
        sample_idx,
        *_: Any,
        use_evidence: bool = False,
        mode: Literal["sample", "argmax"] = "sample",
        **kwargs,
    ) -> Tensor:
        """Is helper routine for backtracking in EiNets, see _sample(...) for details.

        Args:
            dist_idx (_type_): Some indices.
            node_idx (_type_): Some indices.
            sample_idx (_type_): Some indices.
            use_evidence (bool, optional): Whether to use evidence. Defaults to False.
            mode (Literal["sample", "argmax"], optional): The mode.. Defaults to "sample".

        Returns:
            Tensor: The return.
        """

    # TODO: deprecated
    def em_purge(self) -> None:
        """Discard em statistics."""
        raise NotImplementedError

    # TODO: deprecated
    def em_process_batch(self) -> None:
        """Accumulate EM statistics of current batch. This should be called after \
            call to backwards() on the output of the EiNet."""
        raise NotImplementedError

    # TODO: deprecated
    def em_update(self, _triggered: bool = False) -> None:
        """Do an EM update.
        
        If the setting is online EM (online_em_stepsize is not None), \
            then this function does nothing, \
            since updates are triggered automatically. Thus, leave the private \
            parameter _triggered alone.

        pylint not happy param _triggered for internal use, don't set.
        """
        raise NotImplementedError

    def reparam_function(self) -> Callable:
        """Reparametrize function, transforming unconstrained parameters \
            into valid sum-weight (non-negative, normalized).

        Returns:
            Callable: The function.
        """

        def reparam(params_in: Tensor) -> Tensor:
            other_dims = tuple(
                i for i in range(len(params_in.shape)) if i not in self.normalization_dims
            )

            permutation = other_dims + self.normalization_dims
            unpermutation = tuple(
                c for i in range(len(permutation)) for c, j in enumerate(permutation) if j == i
            )

            numel = functools.reduce(
                lambda x, y: x * y, [params_in.shape[i] for i in self.normalization_dims]
            )

            other_shape = tuple(params_in.shape[i] for i in other_dims)
            params_in = params_in.permute(permutation)
            orig_shape = params_in.shape
            params_in = params_in.reshape(other_shape + (numel,))
            out = F.softmax(params_in, -1)
            out = out.reshape(orig_shape).permute(unpermutation)
            return out

        return reparam

    # TODO: deprecated
    def project_params(self, params: Tensor) -> None:
        """Is currently not required.

        Args:
            params (Tensor): Not used.
        """
        raise NotImplementedError
