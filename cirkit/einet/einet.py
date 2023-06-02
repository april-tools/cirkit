import math
from typing import Any, Dict, Generator, List, Optional, Type, cast

import torch
from torch import Tensor, nn

from cirkit.region_graph import PartitionNode, RegionGraph, RegionNode

from .einsum_layer import GenericEinsumLayer
from .exp_family import ExponentialFamilyArray
from .input_layer import FactorizedInputLayer
from .layer import Layer
from .mixing_layer import EinsumMixingLayer

# TODO: should be split this file from the "layer" folder?
# TODO: check all type casts. There should not be any without a good reason


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
    use_em: determines if the internal em algorithm shall be used; otherwise you might use e.g. SGD.
    online_em_frequency: how often shall online be triggered in terms, of \
        batches? 1 means after each batch, None means batch EM. In the latter \
            case, EM updates must be triggered manually after each epoch.
    online_em_stepsize: stepsize for inline EM. Only relevant if online_em_frequency not is None.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(  # type: ignore[misc]
        self,
        rg_structure: str,
        layer_type: Type[GenericEinsumLayer],
        num_var: int,
        num_sums: int,
        num_input: int,
        exponential_family: Type[ExponentialFamilyArray],
        exponential_family_args: Dict[str, Any],
        r: int = 1,
        prod_exp: bool = False,
        pd_num_pieces: int = 0,  # TODO: this is not used?
        shrink: bool = False,
    ):
        """Init class.

        Args:
            rg_structure (str): I don't know.
            layer_type (Type[GenericEinsumLayer]): I don't know.
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
        self.use_em = False
        self.online_em_frequency = 1  # TODO: unused ?
        self.online_em_stepsize = 0.05  # TODO: unused ?
        self.prod_exp = prod_exp
        self.rg_structure = rg_structure
        self.pd_num_pieces = pd_num_pieces
        self.shrink = shrink
        self.layer_type = layer_type


class LowRankEiNet(nn.Module):
    """EiNet with low rank impl."""

    # TODO: why graph is not in args?
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
            len(graph.output_nodes) == 1
        ), "Currently only EinNets with single root node supported."

        # TODO: return a list instead?
        output_node = list(graph.output_nodes)[0]
        assert output_node.scope == set(
            range(args.num_var)
        ), "The graph should be over range(num_var)."

        # TODO: don't bind it to RG
        for node in graph.input_nodes:
            node.num_dist = args.num_input_distributions

        for node in graph.inner_region_nodes:
            node.num_dist = args.num_sums if node is not output_node else args.num_classes

        # Algorithm 1 in the paper -- organize the PC in layers  NOT BOTTOM UP !!!
        self.graph_layers = graph.topological_layers(bottom_up=False)

        # input layer
        # TODO: a better way to represent the interleaved layers
        einet_layers: List[Layer] = [
            FactorizedInputLayer(
                cast(List[RegionNode], self.graph_layers[0]),
                args.num_var,
                args.num_dims,
                args.exponential_family,
                args.exponential_family_args,  # type: ignore[misc]
                use_em=False,
            )
        ]  # note: enforcing this todo: restore  # TODO: <-- what does this mean

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
        for c, layer in enumerate(self.graph_layers[1:]):
            # TODO: is it real that graph_layers[1] is sum?
            if c % 2:
                # sum layer, that's a region layer in the graph
                # the Mixing layer is only for regions which have multiple partitions as children.
                # TODO: find a better way to interleave
                layer = cast(List[RegionNode], layer)  # pylint: disable=redefined-loop-name
                # TODO: return list?
                if multi_sums := [n for n in layer if len(list(graph.get_node_input(n))) > 1]:
                    einet_layers.append(
                        # TODO: abstract method "backtrack", "get_shape_dict"
                        # pylint: disable-next=abstract-class-instantiated
                        EinsumMixingLayer(  # type: ignore[abstract]
                            graph, multi_sums, cast(GenericEinsumLayer, einet_layers[-1])
                        )  # TODO: good type?
                    )
            else:
                # product layer, that's a partition layer in the graph
                # TODO: find a better way to interleave
                layer = cast(List[PartitionNode], layer)  # pylint: disable=redefined-loop-name
                num_sums = set(n.num_dist for p in layer for n in graph.get_node_output(p))
                assert len(num_sums) == 1, f"For internal {c} there are {len(num_sums)} nums sums"
                num_sum = num_sums.pop()

                # TODO: this can be a wrong layer, refer to back up code
                assert num_sum > 1
                einet_layers.append(
                    args.layer_type(
                        self.graph,
                        layer,
                        einet_layers,
                        # r=args.r,  # TODO: how to put this in generic __init__
                        prod_exp=args.prod_exp,
                        k=next(k),
                    )
                )

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
        # TODO: but -1 is not necessary a mixing layer
        return cast(EinsumMixingLayer, self.einet_layers[-1]).params.device

    def initialize(
        self,
        init_dict: Optional[Dict[object, Tensor]] = None,  # TODO: definitely should not be object
        exp_reparam: bool = False,
        mixing_softmax: bool = False,
    ) -> None:
        """Initialize layers.

        :param init_dict: None; or
                          dictionary int->initializer; mapping layer index to initializers; or
                          dictionary layer->initializer;
                          the init_dict does not need to have an initializer for all layers
        :param exp_reparam: I don't know
        :param mixing_softmax: I don't know
        """
        self.exp_reparam = exp_reparam
        assert not mixing_softmax  # TODO: then why have this?
        assert not exp_reparam  # TODO: then why have this?
        if init_dict is None:
            init_dict = {}
        # TODO: I don't know what's the best to do here.
        # but WHY so many possibilities for the dict in the first palce???
        if all(isinstance(k, int) for k in init_dict.keys()):
            init_dict = {
                self.einet_layers[cast(int, k)]: v  # type: ignore[misc]
                for k, v in init_dict.items()
            }
        # TODO: I really don't know how to avoid force cast
        for layer in cast(List[Layer], self.einet_layers):
            layer.initialize(init_dict.get(layer))

    # TODO: this get/set is not good
    def set_marginalization_idx(self, idx: Tensor) -> None:
        """Set indicices of marginalized variables.

        Args:
            idx (Tensor): The indices.
        """
        cast(FactorizedInputLayer, self.einet_layers[0]).set_marginalization_idx(idx)

    def get_marginalization_idx(self) -> Optional[Tensor]:
        """Get indicices of marginalized variables.

        Returns:
            Tensor: The indices.
        """
        return cast(FactorizedInputLayer, self.einet_layers[0]).get_marginalization_idx()

    def partition_function(self, x: Optional[Tensor] = None) -> Tensor:
        """Do something that I don't know.

        Args:
            x (Optional[Tensor], optional): The input. Defaults to None.

        Returns:
            Tensor: The output.
        """
        old_marg_idx = self.get_marginalization_idx()
        assert old_marg_idx is not None  # TODO: then why return None?
        self.set_marginalization_idx(torch.arange(self.args.num_var))

        if x is not None:
            z = self.forward(x)
        else:
            # TODO: check this, size=(1, self.args.num_var, self.args.num_dims) is appropriate
            # TODO: above is original, but size is not stated?
            fake_data = torch.ones(
                (1, self.args.num_var),  # TODO: use tuple here? or everywhere?
                device=cast(EinsumMixingLayer, self.einet_layers[-1]).params.device,
            )  # TODO: cast might not be safe
            z = self.forward(fake_data)  # TODO: why call forward but not __call__

        self.set_marginalization_idx(old_marg_idx)
        return z

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

        for einsum_layer in self.einet_layers[1:]:  # type: ignore[misc]
            einsum_layer()

        # TODO: why use prob but not directly return?
        # TODO: why prob can be None? but not matter, should use return
        return cast(Tensor, cast(EinsumMixingLayer, self.einet_layers[-1]).prob)[:, :, 0]

    # TODO: why not directly access?
    def get_layers(self) -> nn.ModuleList:
        """Get the layers.

        Returns:
            nn.ModuleList: The layers.
        """
        return self.einet_layers

    # TODO: what's the meaning of this?
    def forward_layer(self, layer: Layer, x: Optional[Tensor] = None) -> None:
        """Do something that I don't know.

        Args:
            layer (Layer): The Layer.
            x (Optional[Tensor], optional): I don't know. Defaults to None.
        """
        # TODO: why something is any here
        if layer is self.einet_layers[0]:  # type: ignore[misc]
            assert x is not None
            layer(x)
        else:
            layer()

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
