import functools
from abc import abstractmethod
from typing import Tuple
from typing_extensions import Self  # FUTURE: in typing from 3.11

import torch
from torch import Tensor, nn

from cirkit.new.layers.input.constant import ConstantLayer
from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


class ExpFamilyLayer(InputLayer):
    """The abstract base class for Exponential Family distribution layers.

    Exponential Family dist:
        f(x|θ) = exp(η(θ) · T(x) + log_h(x) - A(η)).
    Ref: https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions.

    However here we directly parameterize η instead of θ:
        f(x|η) = exp(η · T(x) + log_h(x) - A(η)).
    Implemtations provide inverse mapping from η to θ.

    This implementation is fully factorized over the variables if used as multivariate, i.e., \
    equivalent to num_vars (arity) univariate EF distributions followed by a Hadamard product of \
    the same arity.
    """

    # NOTE: suff_stats_shape should not be part of the interface for users, but should be set by
    #       subclasses based on the implementation. We assume it is already set before entering
    #       ExpFamilyLayer.__init__.
    suff_stats_shape: Tuple[int, ...]
    """The shape for sufficient statistics, as dim S (or *S). The last dim is for normalization, \
    if relevant."""

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        assert all(
            s for s in self.suff_stats_shape
        ), "The number of sufficient statistics must be positive."

        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.params = reparam
        if self.params.materialize((arity, num_output_units, *self.suff_stats_shape), dim=-1):
            self.reset_parameters()  # Only reset if newly materialized.

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default: N(0, 1)."""
        for child in self.children():
            if isinstance(child, Reparameterization):
                child.initialize(functools.partial(nn.init.normal_, mean=0, std=1))

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        return self.comp_space.from_log(self.log_prob(x))

    def log_prob(self, x: Tensor) -> Tensor:
        """Calculate log-probability log(f(x)) from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The log-prob log(f), shape (*B, K).
        """
        # TODO: if we just propagate unnormalized values, we can remove log_part here and move it to
        #       integration -- by definition integration is partition.
        eta = self.params()  # shape (H, K, *S).
        suff_stats = self.sufficient_stats(x)  # shape (*B, H, S).
        log_h = self.log_base_measure(x)  # shape (*B, H).
        log_part = self.log_partition(eta)  # shape (H, K).
        # We need to flatten because we cannot have two ... in einsum for suff_stats as (*B, H, *S).
        eta = eta.flatten(start_dim=-len(self.suff_stats_shape))  # shape (H, K, S).
        return torch.sum(
            torch.einsum("hks,...hs->...hk", eta, suff_stats)  # shape (*B, H, K).
            + log_h.unsqueeze(dim=-1)  # shape (*B, H, 1).
            - log_part,  # shape (*1, H, K), 1s automatically prepended.
            dim=-2,
        )  # shape (*B, H, K) -> (*B, K).

    @abstractmethod
    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The sufficient statistics T, shape (*B, H, S).
        """

    @abstractmethod
    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The natural parameters eta, shape (*B, H).
        """

    @abstractmethod
    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (H, K, *S).

        Returns:
            Tensor: The log partition function A, shape (H, K).
        """

    @classmethod
    def get_integral(cls, symb_cfg: SymbLayerCfg[Self]) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the definite integral of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer. Unused here.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the integral.
        """
        # TODO: for unnormalized EF, should be ParameterizedConstantLayer
        # We have already normalized with log_partition in forward(), so integral is always 1.
        # IGNORE: Unavoidable for kwargs.
        return SymbLayerCfg(
            layer_cls=ConstantLayer, layer_kwargs={"const_value": 1.0}  # type: ignore[misc]
        )

    # NOTE: Here we define the differential of all EF layers, but some subclasses, e.g. Categorical,
    #       should not have differentials due to discrete variable. Those subclasses should override
    #       this method and raise an error.
    @classmethod
    def get_partial(
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the partial differential w.r.t. the given channel \
        of the given variable in the scope of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            order (int, optional): The order of differentiation. Defaults to 1.
            var_idx (int, optional): The variable to diffrentiate. The idx is counted within this \
                layer's scope but not global variable id. Defaults to 0.
            ch_idx (int, optional): The channel of variable to diffrentiate. Defaults to 0.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the partial differential w.r.t. the \
                given channel of the given variable.
        """
        assert order >= 0, "The order of differential must be non-negative."
        if not order:
            return symb_cfg

        # TODO: pylint bug? should not raise cyclic-import?
        # DISABLE: We must import here to avoid cyclic import.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from cirkit.new.layers.input.exp_family.diff_ef import DiffEFLayer

        # IGNORE: DiffEFLayer contains Any.
        # IGNORE: Unavoidable for kwargs.
        return SymbLayerCfg(
            layer_cls=DiffEFLayer,  # type: ignore[misc]
            layer_kwargs={
                "ef_cls": symb_cfg.layer_cls,
                "ef_kwargs": symb_cfg.layer_kwargs,  # type: ignore[misc]
                "order": order,
                "var_idx": var_idx,
                "ch_idx": ch_idx,
            },
            reparam=symb_cfg.reparam,  # Reuse the same reparam to share params.
        )
