import math
from collections import defaultdict
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Type, Union, cast

import torch
from torch import Tensor, nn

from cirkit.layers.einsum.mixing import EinsumMixingLayer
from cirkit.region_graph import RegionGraph, RegionNode
from cirkit.region_graph.rg_node import _EiNetAddress

from ..layers.einsum import EinsumLayer
from ..layers.exp_family import ExpFamilyLayer
from ..layers.layer import Layer

# TODO: check all type casts. There should not be any without a good reason
# TODO: rework docstrings


class _TwoInputs(NamedTuple):
    """Provide names for left and right inputs."""

    left: RegionNode
    right: RegionNode


# TODO: might be a good idea. but how to design better
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class _Args:
    """Arguments for EinsumNetwork class.

    num_var: number of random variables (RVs). An RV might be multidimensional \
        though -- see num_dims.
    num_dims: number of dimensions per RV. E.g. you can model an 32x32 RGB \
        image as an 32x32 array of three dimensional RVs.
    num_input_distributions: number of distributions per input region (K in the paper).
    num_sums: number of sum nodes per internal region (K in the paper).
    num_classes: number of outputs of the PC.
    exponential_family: which exponential family to use; (sub-class ExponentialFamilyTensor).
    exponential_family_args: arguments for the exponential family, e.g. trial-number N for Binomial.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]
        self,
        layer_type: Type[EinsumLayer],
        num_var: int,
        num_sums: int,
        num_input: int,
        exponential_family: Type[ExpFamilyLayer],
        exponential_family_args: Dict[str, Any],
        r: int = 1,
        prod_exp: bool = False,
        pd_num_pieces: int = 0,  # TODO: this is not used?
        shrink: bool = False,
    ):
        """Init class.

        Args:
            layer_type (Type[EinsumLayer]): I don't know.
            num_var (int): I don't know.
            num_sums (int): I don't know.
            num_input (int): I don't know.
            exponential_family (Type[ExponentialFamilyArray]): I don't know.
            exponential_family_args (Dict[str, Any]): I don't know.
            r (int, optional): I don't know. Defaults to 1.
            prod_exp (bool, optional): I don't know. Defaults to False.
            pd_num_pieces (int, optional): I don't know. Defaults to 0.
            shrink (bool, optional): I don't know. Defaults to False.
        """
        self.r = r
        self.num_var = num_var
        self.num_dims = 1
        self.num_input_distributions = num_input
        self.num_sums = num_sums
        self.num_classes = 1
        self.exponential_family = exponential_family
        self.exponential_family_args = exponential_family_args  # type: ignore[misc]
        self.prod_exp = prod_exp
        self.pd_num_pieces = pd_num_pieces
        self.shrink = shrink
        self.layer_type = layer_type


class LowRankEiNet(nn.Module):
    """EiNet with low rank impl."""

    # TODO: why graph is not in args?
    # pylint: disable=too-complex,too-many-locals,too-many-statements
    def __init__(self, graph: RegionGraph, args: _Args) -> None:
        """Make an EinsumNetwork.

        Args:
            graph (RegionGraph): The region graph.
            args (Args): The args.
        """
        super().__init__()

        # TODO: check graph. but do we need it?

        self.graph = graph
        self.args = args

        assert (
            len(list(graph.output_nodes)) == 1
        ), "Currently only EinNets with single root node supported."

        # TODO: return a list instead?
        output_node = list(graph.output_nodes)[0]
        assert output_node.scope == set(
            range(args.num_var)
        ), "The graph should be over range(num_var)."

        # TODO: don't bind it to RG
        for node in graph.input_nodes:
            node.k = args.num_input_distributions

        for node in graph.inner_region_nodes:
            node.k = args.num_sums if node is not output_node else args.num_classes

        # Algorithm 1 in the paper -- organize the PC in layers  NOT BOTTOM UP !!!
        self.graph_layers = graph.topological_layers(bottom_up=False)

        # input layer
        einet_layers: List[Layer] = [
            args.exponential_family(
                self.graph_layers[0][1],
                args.num_var,
                args.num_dims,
                **args.exponential_family_args,  # type: ignore[misc]
            )
        ]  # note: enforcing this todo: restore  # TODO: <-- what does this mean
        self.bookkeeping: List[
            Union[
                Tuple[List[_EiNetAddress], List[_EiNetAddress]],
                Tuple[Tensor, Tensor],
                Tuple[None, None],
            ]
        ] = [(None, None)]

        def _k_gen() -> Generator[int, None, None]:
            k_list = (
                # TODO: another mypy ** bug
                [cast(int, 2**i) for i in range(5, int(math.log2(args.num_sums)))]
                if args.shrink
                else [args.num_sums]
            )
            while True:  # pylint: disable=while-used
                if len(k_list) > 1:
                    yield k_list.pop(-1)
                else:
                    yield k_list[0]

        k = _k_gen()

        # internal layers
        for partition_layer, region_layer in self.graph_layers[1:]:
            # TODO: duplicate check with einet layer, but also useful here?
            # out_k = set(
            #     out_region.k for partition in partition_layer for out_region in partition.outputs
            # )
            # assert len(out_k) == 1, f"For internal {c} there are {len(out_k)} nums sums"
            # out_k = out_k.pop()

            # TODO: this can be a wrong layer, refer to back up code
            # assert out_k > 1
            einsum_layer = args.layer_type(
                partition_layer, k=next(k), prod_exp=args.prod_exp, r=args.r
            )
            einet_layers.append(einsum_layer)

            # get pairs of nodes which are input to the products (list of lists)
            # length of the outer list is same as self.products, length of inner lists is 2
            # "left child" has index 0, "right child" has index 1
            two_inputs = [_TwoInputs(*sorted(partition.inputs)) for partition in partition_layer]
            # TODO: again, why do we need sorting
            # collect all layers which contain left/right children
            left_addr = [inputs.left.einet_address for inputs in two_inputs]
            right_addr = [inputs.right.einet_address for inputs in two_inputs]
            self.bookkeeping.append((left_addr, right_addr))

            # when the EinsumLayer is followed by a EinsumMixingLayer, we produce a
            # dummy "node" which outputs 0 (-inf in log-domain) for zero-padding.
            dummy_idx: Optional[int] = None

            # the dictionary mixing_component_idx stores which nodes (axis 2 of the
            # log-density tensor) need to get mixed
            # in the following EinsumMixingLayer
            mixing_component_idx: Dict[RegionNode, List[int]] = defaultdict(list)

            for part_idx, partition in enumerate(partition_layer):
                # each product must have exactly 1 parent (sum node)
                assert len(partition.outputs) == 1
                out_region = partition.outputs[0]

                if len(out_region.inputs) == 1:
                    out_region.einet_address.layer = einsum_layer
                    out_region.einet_address.idx = part_idx
                else:  # case followed by EinsumMixingLayer
                    mixing_component_idx[out_region].append(part_idx)
                    dummy_idx = len(partition_layer)

            # the Mixing layer is only for regions which have multiple partitions as children.
            if multi_sums := [region for region in region_layer if len(region.inputs) > 1]:
                assert dummy_idx is not None
                mixing_layer = EinsumMixingLayer(multi_sums, einsum_layer)
                einet_layers.append(mixing_layer)

                # The following code does some bookkeeping.
                # padded_idx indexes into the log-density tensor of the previous
                # EinsumLayer, padded with a dummy input which
                # outputs constantly 0 (-inf in the log-domain), see class EinsumLayer.
                padded_idx: List[List[int]] = []
                max_components = mixing_layer.param.shape[-1]  # TODO: duplicated
                for reg_idx, region in enumerate(region_layer):
                    num_components = len(mixing_component_idx[region])
                    this_idx = mixing_component_idx[region] + [dummy_idx] * (
                        max_components - num_components
                    )
                    padded_idx.append(this_idx)
                    if max_components > num_components:
                        mixing_layer.params_mask[:, reg_idx, num_components:] = 0.0
                    region.einet_address.layer = mixing_layer
                    region.einet_address.idx = reg_idx
                mixing_layer.reset_parameters()
                self.bookkeeping.append((mixing_layer.params_mask, torch.tensor(padded_idx)))

        # TODO: can we annotate a list here?
        # TODO: actually we should not mix all the input/mix/ein different types in one list
        self.einet_layers = nn.ModuleList(einet_layers)
        self.exp_reparam: bool = False
        self.mixing_softmax: bool = False

    # TODO: find a better way to do this. should be in Module? (what about multi device?)
    # TODO: maybe we should stick to some device agnostic impl rules
    def get_device(self) -> torch.device:
        """Get the device that params is on.

        Returns:
            torch.device: the device.
        """
        # TODO: ModuleList is not generic type
        return cast(ExpFamilyLayer, self.einet_layers[0]).params.device

    def initialize(
        self,
        exp_reparam: bool = False,
        mixing_softmax: bool = False,
    ) -> None:
        """Initialize layers.

        :param exp_reparam: I don't know
        :param mixing_softmax: I don't know
        """
        # TODO: do we need this function? because each layer inits itself in __init__
        self.exp_reparam = exp_reparam
        assert not mixing_softmax  # TODO: then why have this?
        assert not exp_reparam  # TODO: then why have this?

        # TODO: I really don't know how to avoid force cast
        for layer in cast(List[Layer], self.einet_layers):
            layer.reset_parameters()

    # TODO: this get/set is not good
    def set_marginalization_idx(self, idx: Tensor) -> None:
        """Set indicices of marginalized variables.

        Args:
            idx (Tensor): The indices.
        """
        cast(ExpFamilyLayer, self.einet_layers[0]).set_marginalization_idx(idx)

    def get_marginalization_idx(self) -> Optional[Tensor]:
        """Get indicices of marginalized variables.

        Returns:
            Tensor: The indices.
        """
        return cast(ExpFamilyLayer, self.einet_layers[0]).get_marginalization_idx()

    def partition_function(self, x: Optional[Tensor] = None) -> Tensor:
        """Do something that I don't know.

        Args:
            x (Optional[Tensor], optional): The input. Defaults to None.

        Returns:
            Tensor: The output.
        """
        old_marg_idx = self.get_marginalization_idx()
        # assert old_marg_idx is not None  # TODO: then why return None?
        self.set_marginalization_idx(torch.arange(self.args.num_var))

        if x is not None:
            z = self.forward(x)
        else:
            # TODO: check this, size=(1, self.args.num_var, self.args.num_dims) is appropriate
            # TODO: above is original, but size is not stated?
            # TODO: use tuple as shape because of line folding? or everywhere?
            fake_data = torch.ones((1, self.args.num_var), device=self.get_device())
            z = self.forward(fake_data)  # TODO: why call forward but not __call__

        # TODO: can indeed be None
        self.set_marginalization_idx(old_marg_idx)  # type: ignore[arg-type]
        return z

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke the forward.

        Args:
            x (Tensor): The input.

        Returns:
            Tensor: The output.
        """
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    # TODO: originally there's a plot_dict. REMOVED assuming not ploting.
    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the EinsumNetwork feed forward.

        Args:
            x (Tensor): the shape is (batch_size, self.num_var, self.num_dims)

        Returns:
            Tensor: Return value.
        """
        input_layer = self.einet_layers[0]  # type: ignore[misc]
        input_layer(x)

        # TODO: use zip instead
        for i, einsum_layer in enumerate(self.einet_layers[1:], 1):  # type: ignore[misc]
            if isinstance(einsum_layer, EinsumLayer):  # type: ignore[misc]
                left_addr, right_addr = self.bookkeeping[i]
                assert isinstance(left_addr, list) and isinstance(right_addr, list)
                # TODO: we should use dim=2, check all code
                log_left_prob = torch.stack(
                    [addr.layer.prob[:, :, addr.idx] for addr in left_addr], dim=2
                )
                log_right_prob = torch.stack(
                    [addr.layer.prob[:, :, addr.idx] for addr in right_addr], dim=2
                )
                einsum_layer(log_left_prob, log_right_prob)
            elif isinstance(einsum_layer, EinsumMixingLayer):  # type: ignore[misc]
                _, padded_idx = self.bookkeeping[i]
                assert isinstance(padded_idx, Tensor)  # type: ignore[misc]
                log_input_prob = einsum_layer.input_layer_as_list[0].prob[:, :, padded_idx]
                einsum_layer(log_input_prob)

        # TODO: why use prob but not directly return?
        return cast(EinsumMixingLayer, self.einet_layers[-1]).prob[:, :, 0]

    # TODO: and what's the meaning of this?
    # def backtrack(self, num_samples=1, class_idx=0, x=None, mode='sampling', **kwargs):
    # TODO: there's actually nothing to doc
    # pylint: disable=missing-param-doc
    def backtrack(self, *_: Any, **__: Any) -> None:  # type: ignore[misc]
        """Raise an error.

        Raises:
            NotImplementedError: Not implemented.
        """
        raise NotImplementedError

    def sample(  # type: ignore[misc]
        self, num_samples: int = 1, class_idx: int = 0, x: Optional[Tensor] = None, **_: Any
    ) -> None:
        """Cause an error anyway.

        Args:
            num_samples (int, optional): I don't know/care now. Defaults to 1.
            class_idx (int, optional): I don't know/care now. Defaults to 0.
            x (Optional[Tensor], optional): I don't know/care now. Defaults to None.
        """
        self.backtrack(num_samples=num_samples, class_idx=class_idx, x=x, mode="sample")

    def mpe(  # type: ignore[misc]
        self, num_samples: int = 1, class_idx: int = 0, x: Optional[Tensor] = None, **_: Any
    ) -> None:
        """Cause an error anyway.

        Args:
            num_samples (int, optional):  I don't know/care now. Defaults to 1.
            class_idx (int, optional):  I don't know/care now. Defaults to 0.
            x (Optional[Tensor], optional):  I don't know/care now. Defaults to None.
        """
        self.backtrack(num_samples=num_samples, class_idx=class_idx, x=x, mode="argmax")
