from typing import NoReturn, Optional, cast
from typing_extensions import Self  # FUTURE: in typing from 3.11

import torch
from torch import Tensor

from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import BinaryReparam, Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


class CategoricalProductPartitionLayer(InputLayer):
    """Layer calculates the partition function for product input layer of categorical distribution."""

    # DISABLE: It's designed to have these arguments.
    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,  # TODO: change to tuple?
        arity: int = 1,
        reparam: BinaryReparam,
        num_categories: int,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Optional[Reparameterization], optional): The reparameterization for this layer parameters. \
                Guaranteed to be BinaryReparam, for product layer.
            num_categories (int): The number of categories for Categorical distribution.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )
        assert (
            num_categories > 0
        ), "The number of categories for Categorical distribution must be positive."
        self.num_categories = num_categories
        self.suff_stats_shape = (num_input_units, num_categories)

        assert (
            reparam.is_materialized
        ), "parameters for partition function must be materialized beforehand"
        self.params = reparam

    def reset_parameters(self) -> None:
        """Do nothing, as constant layers do not have parameters."""

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass, calculate the partition function of the. \
        layer product, given the parameters.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        assert isinstance(self.params(), list), "parameter must be the product parameters"
        assert all(
            param.ndim == 4
            and param.shape[0] == self.arity
            and param.shape[2] == self.num_input_units
            and param.shape[3] == self.num_categories
            for param in self.params()
        ), "All categorical parameters must have shape (arity, num_units, num_input_units, categories)"

        # shape (H, K_out_n, K_in, cat) -> (H, K_out_n, *S=(K_in, cat))
        flatten_params = [
            param.flatten(start_dim=-len(self.suff_stats_shape)) for param in self.params()
        ]
        num_params = len(self.params())

        # Generate the einsum notation: product of 2 is acb,adb->acd
        operand_parts = ["a" + chr(ord("c") + i) + "b" for i in range(num_params)]
        output_parts = ["a"] + [chr(ord("c") + i) for i in range(num_params)]
        einsum_notation = f"{','.join(operand_parts)}->{''.join(output_parts)}"

        # [*(H, K_out_n, S)] -> (H, K_out_1, K_out_2, ...)
        part_func = self.comp_space.sum(
            lambda *ps: torch.einsum(einsum_notation, *ps), *flatten_params, dim=-1, keepdim=True
        )
        # (H, K_out_1, K_out_2, ...) -> (K_out_1, K_out_2, ...)
        part_func = self.comp_space.sum(
            lambda p: torch.sum(p, dim=0), part_func, dim=0, keepdim=True
        )
        return self.comp_space.from_log(part_func.to(x))

    # IGNORE: SymbLayerCfg contains Any.
    @classmethod
    def get_integral(cls, symb_cfg: SymbLayerCfg[Self]) -> NoReturn:  # type: ignore[misc]
        """Do nothing, as constant layers do not have parameters."""
        raise TypeError("Cannot differentiate over discrete variables.")

    # IGNORE: SymbLayerCfg contains Any.
    @classmethod
    def get_partial(  # type: ignore[misc]
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> NoReturn:
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
        raise TypeError("Cannot differentiate over discrete variables.")


class NormalProductPartitionLayer(InputLayer):
    """Layer calculates the partition function for product input layer of normal distribution."""

    # DISABLE: It's designed to have these arguments.
    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,  # TODO: change to tuple?
        arity: int = 1,
        reparam: BinaryReparam,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Optional[Reparameterization], optional): The reparameterization for this layer parameters. \
                Guaranteed to be BinaryReparam, for product layer.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )
        self.suff_stats_shape = (2, num_input_units)  # 2 for eta_1 and eta_2 in exponential family
        assert (
            reparam.is_materialized
        ), "parameters for partition function must be materialized beforehand"
        self.params = reparam

    def reset_parameters(self) -> None:
        """Do nothing, as constant layers do not have parameters."""

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        assert isinstance(self.params(), list), "parameter must be the product parameters"
        assert (
            len(self.params()) == 2
        ), "normal partition function with number of product > 2 is not supported yet"
        assert all(
            param.ndim == 4
            and param.shape[0] == self.arity
            and param.shape[2] == 2
            and param.shape[3] == self.num_input_units
            for param in self.params()
        ), "All categorical parameters must have shape (arity, num_units, 2, num_input_units)"

        assert self.num_input_units == 1, "num_input_units > 1 is not supported yet"

        log1 = torch.log(torch.tensor(1.0))
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi))

        # (H,K,2,1) -> (H,K), (H,K)
        # TODO: support num_input_units > 1, number of product > 2
        log_eta_11 = torch.log(self.params()[0][:, :, 0, :].squeeze(dim=-1))
        eta_12 = self.params()[0][:, :, 1, :].squeeze(dim=-1)
        log_eta_21 = torch.log(self.params()[1][:, :, 0, :].squeeze(dim=-1))
        eta_22 = self.params()[1][:, :, 1, :].squeeze(dim=-1)

        # mu = -eta_1/(2*eta_2)
        # var = -1/(2*eta_2)
        mu_1 = torch.exp(log_eta_11 - torch.log(-2.0 * eta_12))
        mu_2 = torch.exp(log_eta_21 - torch.log(-2.0 * eta_22))
        var_1 = torch.exp(log1 - torch.log(-2.0 * eta_12))
        var_2 = torch.exp(log1 - torch.log(-2.0 * eta_22))

        # log partition function for product of Gaussians
        # \int_{-inf}^{inf} N(mu_1,var_1) * N(mu_2,var_2) dx
        # = (1/sqrt(2*pi*(var_1+var_2))) * exp(-(mu_1-mu_2)^2/2*(var_1+var_2))
        log_var_1_sum_var_2 = torch.log(
            var_1.unsqueeze(dim=-2) + var_2.unsqueeze(dim=-1)
        )  # (H,K,K)
        log_mu_1_sum_mu_2_sq = 2.0 * torch.log(
            torch.abs(mu_1.unsqueeze(dim=-2) - mu_2.unsqueeze(dim=-1))
        )  # (H,K,K)
        exp_term = torch.exp(log_mu_1_sum_mu_2_sq - log_var_1_sum_var_2)
        log_part_func = -0.5 * (log_two_pi + log_var_1_sum_var_2 + exp_term)

        # (H,K,K) -> (1,K,K)
        log_part_func = self.comp_space.sum(
            lambda p: torch.sum(p, dim=0), log_part_func, dim=0, keepdim=True
        )
        return self.comp_space.from_log(log_part_func.to(x))

    # IGNORE: SymbLayerCfg contains Any.
    @classmethod
    def get_integral(cls, symb_cfg: SymbLayerCfg[Self]) -> NoReturn:  # type: ignore[misc]
        """Do nothing, as constant layers do not have parameters."""
        raise TypeError("Cannot differentiate over discrete variables.")

    # IGNORE: SymbLayerCfg contains Any.
    @classmethod
    def get_partial(  # type: ignore[misc]
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> NoReturn:
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
        raise TypeError("Cannot differentiate over discrete variables.")
