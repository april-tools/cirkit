from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import torch
from torch import Tensor, nn

from cirkit.layers.exp_family import ExpFamilyLayer
from cirkit.layers.layer import Layer
from cirkit.layers.mixing import MixingLayer
from cirkit.layers.sum_product import SumProductLayer
from cirkit.region_graph import RegionGraph, RegionNode

# TODO: check all type casts. There should not be any without a good reason
# TODO: rework docstrings


class _TwoInputs(NamedTuple):
    """Provide names for left and right inputs."""

    left: RegionNode
    right: RegionNode


class TensorizedPC(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Tensorized and folded PC implementation."""

    # pylint: disable-next=too-many-locals,too-many-statements,too-many-arguments
    def __init__(  # type: ignore[misc]
        self,
        graph: RegionGraph,
        num_vars: int,
        layer_cls: Type[SumProductLayer],
        efamily_cls: Type[ExpFamilyLayer],
        layer_kwargs: Dict[str, Any],
        efamily_kwargs: Dict[str, Any],
        num_inner_units: int,
        num_input_units: int,
        num_channels: int = 1,
        num_classes: int = 1,
    ) -> None:
        """Make an TensorizedPC.

        Args:
            graph (RegionGraph): The region graph.
            num_vars (int): The number of variables.
            layer_cls (Type[SumProductLayer]): The inner layer class.
            efamily_cls (Type[ExpFamilyLayer]): The exponential family class.
            layer_kwargs (Dict[str, Any]): The parameters for the inner layer class.
            efamily_kwargs (Dict[str, Any]): The parameters for the exponential family class.
            num_inner_units (int): The number of units for each layer.
            num_input_units (int): The number of input units in the input layer.
            num_channels (int): The number of channels (e.g., 3 for RGB pixel). Defaults to 1.
            num_classes (int): The number of classes of the PC. Defaults to 1.
        """
        assert num_inner_units > 0, "The number of output units per layer should be positive"
        assert (
            num_input_units > 0
        ), "The number of input untis in the input layer should be positive"
        assert num_classes > 0, "The number of classes should be positive"
        super().__init__()

        # TODO: check graph. but do we need it?
        self.graph = graph
        self.num_vars = num_vars
        num_output_units = num_inner_units  # TODO: clean up relationship among all num_*_units

        assert (
            len(list(graph.output_nodes)) == 1
        ), "Currently only circuits with single root region node are supported."

        # TODO: return a list instead?
        output_node = list(graph.output_nodes)[0]
        assert output_node.scope == set(
            range(num_vars)
        ), "The region graph should have been defined on the same number of variables"

        # Algorithm 1 in the paper -- organize the PC in layers  NOT BOTTOM UP !!!
        self.graph_layers = graph.topological_layers(bottom_up=False)

        # Initialize input layer
        self.input_layer = efamily_cls(
            self.graph_layers[0][1],
            num_vars,
            num_channels,
            num_input_units,
            **efamily_kwargs,  # type: ignore[misc]
        )

        # A dictionary mapping each region node ID to
        #   (i) its index in the corresponding fold, and
        #   (ii) the layer that computes such fold.
        region_id_fold: Dict[int, Tuple[int, Layer]] = {}
        for i, region in enumerate(self.graph_layers[0][1]):
            region_id_fold[region.get_id()] = (i, self.input_layer)

        # Book-keeping: None for input, Tensor for mixing, Tuple for einsum
        self.bookkeeping: List[
            Union[
                Tuple[Tuple[List[Layer], Tensor], Tuple[List[Layer], Tensor]], Tuple[Layer, Tensor]
            ]
        ] = []

        # Build inner layers
        inner_layers: List[Layer] = []
        # TODO: use start as kwarg?
        for idx, (partition_layer, region_layer) in enumerate(self.graph_layers[1:], start=1):
            # TODO: duplicate check with einet layer, but also useful here?
            # out_k = set(
            #     out_region.k for partition in partition_layer for out_region in partition.outputs
            # )
            # assert len(out_k) == 1, f"For internal {c} there are {len(out_k)} nums sums"
            # out_k = out_k.pop()

            # TODO: this can be a wrong layer, refer to back up code
            # assert out_k > 1
            num_outputs = num_output_units if idx < len(self.graph_layers) - 1 else num_classes
            num_inputs = num_input_units if idx == 1 else num_output_units
            inner_layer = layer_cls(
                partition_layer, num_inputs, num_outputs, **layer_kwargs  # type: ignore[misc]
            )
            inner_layers.append(inner_layer)

            # get pairs of nodes which are input to the products (list of lists)
            # length of the outer list is same as self.products, length of inner lists is 2
            # "left child" has index 0, "right child" has index 1
            two_inputs = [_TwoInputs(*sorted(partition.inputs)) for partition in partition_layer]
            # TODO: again, why do we need sorting
            # collect all layers which contain left/right children
            # TODO: duplicate code
            left_region_ids = list(r.left.get_id() for r in two_inputs)
            right_region_ids = list(r.right.get_id() for r in two_inputs)
            left_layers = list(set(region_id_fold[i][1] for i in left_region_ids))
            right_layers = list(set(region_id_fold[i][1] for i in right_region_ids))
            left_starts = torch.tensor([0] + [layer.fold_count for layer in left_layers]).cumsum(
                dim=0
            )
            right_starts = torch.tensor([0] + [layer.fold_count for layer in right_layers]).cumsum(
                dim=0
            )
            left_indices = torch.tensor(
                [  # type: ignore[misc]
                    region_id_fold[r.left.get_id()][0]
                    + left_starts[left_layers.index(region_id_fold[r.left.get_id()][1])]
                    for i, r in enumerate(two_inputs)
                ]
            )
            right_indices = torch.tensor(
                [  # type: ignore[misc]
                    region_id_fold[r.right.get_id()][0]
                    + right_starts[right_layers.index(region_id_fold[r.right.get_id()][1])]
                    for i, r in enumerate(two_inputs)
                ]
            )
            self.bookkeeping.append(((left_layers, left_indices), (right_layers, right_indices)))

            # when the SumProductLayer is followed by a MixingLayer, we produce a
            # dummy "node" which outputs 0 (-inf in log-domain) for zero-padding.
            dummy_idx: Optional[int] = None

            # the dictionary mixing_component_idx stores which nodes (axis 2 of the
            # log-density tensor) need to get mixed
            # in the following MixingLayer
            mixing_component_idx: Dict[RegionNode, List[int]] = defaultdict(list)

            for part_idx, partition in enumerate(partition_layer):
                # each product must have exactly 1 parent (sum node)
                assert len(partition.outputs) == 1
                out_region = partition.outputs[0]

                if len(out_region.inputs) == 1:
                    region_id_fold[out_region.get_id()] = (part_idx, inner_layer)
                else:  # case followed by MixingLayer
                    mixing_component_idx[out_region].append(part_idx)
                    dummy_idx = len(partition_layer)

            # The Mixing layer is only for regions which have multiple partitions as children.
            if multi_sums := [region for region in region_layer if len(region.inputs) > 1]:
                assert dummy_idx is not None
                max_components = max(len(region.inputs) for region in multi_sums)

                # The following code does some bookkeeping.
                # padded_idx indexes into the log-density tensor of the previous
                # SumProductLayer, padded with a dummy input which
                # outputs constantly 0 (-inf in the log-domain), see class SumProductLayer.
                padded_idx: List[List[int]] = []
                params_mask: Optional[Tensor] = None
                for reg_idx, region in enumerate(multi_sums):
                    num_components = len(mixing_component_idx[region])
                    this_idx = mixing_component_idx[region] + [dummy_idx] * (
                        max_components - num_components
                    )
                    padded_idx.append(this_idx)
                    if max_components > num_components:
                        if params_mask is None:
                            params_mask = torch.ones(num_outputs, len(multi_sums), max_components)
                        params_mask[:, reg_idx, num_components:] = 0.0
                mixing_layer = MixingLayer(
                    multi_sums, num_outputs, max_components, mask=params_mask
                )
                for reg_idx, region in enumerate(multi_sums):
                    region_id_fold[region.get_id()] = (reg_idx, mixing_layer)
                self.bookkeeping.append((inner_layers[-1], torch.tensor(padded_idx).T))
                inner_layers.append(mixing_layer)

        # TODO: can we annotate a list here?
        # TODO: actually we should not mix all the input/mix/ein different types in one list
        self.inner_layers: List[Layer] = nn.ModuleList(inner_layers)  # type: ignore[assignment]
        self.exp_reparam = False
        self.mixing_softmax = False

    # TODO: find a better way to do this. should be in Module? (what about multi device?)
    # TODO: maybe we should stick to some device agnostic impl rules
    def get_device(self) -> torch.device:
        """Get the device that params is on.

        Returns:
            torch.device: the device.
        """
        # TODO: ModuleList is not generic type
        return self.input_layer.params.device

    # TODO: this get/set is not good
    def set_marginalization_idx(self, idx: Tensor) -> None:
        """Set indicices of marginalized variables.

        Args:
            idx (Tensor): The indices.
        """
        self.input_layer.set_marginalization_idx(idx)

    def get_marginalization_idx(self) -> Optional[Tensor]:
        """Get indicices of marginalized variables.

        Returns:
            Tensor: The indices.
        """
        return self.input_layer.get_marginalization_idx()

    def partition_function(self, x: Optional[Tensor] = None) -> Tensor:
        """Do something that I don't know.

        Args:
            x (Optional[Tensor], optional): The input. Defaults to None.

        Returns:
            Tensor: The output.
        """
        old_marg_idx = self.get_marginalization_idx()
        # assert old_marg_idx is not None  # TODO: then why return None?
        self.set_marginalization_idx(torch.arange(self.num_vars))

        if x is not None:
            z = self(x)
        else:
            # TODO: check this, size=(1, self.args.num_var, self.args.num_dims) is appropriate
            # TODO: above is original, but size is not stated?
            # TODO: use tuple as shape because of line folding? or everywhere?
            fake_data = torch.ones((1, self.num_vars), device=self.get_device())
            z = self(fake_data)

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

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the TensorizedPC feed forward.

        Args:
            x (Tensor): the shape is (batch_size, self.num_var, self.num_dims)

        Returns:
            Tensor: Return value.
        """
        # TODO: can we have just a dictionary with integer keys instead?
        #  It would be much simpler and clean
        outputs: Dict[Layer, Tensor] = {self.input_layer: self.input_layer(x)}

        # TODO: use zip instead
        # TODO: Generalize if statements here, they should be layer agnostic
        for idx, inner_layer in enumerate(self.inner_layers):
            if isinstance(inner_layer, SumProductLayer):  # type: ignore[misc]
                left_addr, right_addr = self.bookkeeping[idx]
                assert isinstance(left_addr, tuple) and isinstance(right_addr, tuple)
                # TODO: we should use dim=2, check all code
                # TODO: duplicate code
                log_left_prob = torch.cat([outputs[layer] for layer in left_addr[0]], dim=0)
                log_left_prob = log_left_prob[left_addr[1]]
                log_right_prob = torch.cat([outputs[layer] for layer in right_addr[0]], dim=0)
                log_right_prob = log_right_prob[right_addr[1]]
                out = inner_layer(log_left_prob, log_right_prob)
            elif isinstance(inner_layer, MixingLayer):
                _, padded_idx = self.bookkeeping[idx]
                assert isinstance(padded_idx, Tensor)  # type: ignore[misc]
                # TODO: a better way to pad?
                # TODO: padding here breaks bookkeeping by changing the tensors shape.
                #  We need to find another way to implement it.
                # outputs[self.inner_layers[idx - 1]] = F.pad(
                #     outputs[self.inner_layers[idx - 1]], [0, 1], "constant", float("-inf")
                # )
                log_input_prob = outputs[self.inner_layers[idx - 1]][padded_idx]
                out = inner_layer(log_input_prob)
            else:
                assert False
            outputs[inner_layer] = out

        return outputs[self.inner_layers[-1]][0].T  # return shape (B, K)

    # TODO: and what's the meaning of this?
    # def backtrack(self, num_samples=1, class_idx=0, x=None, mode='sampling', **kwargs):
    # TODO: there's actually nothing to doc
    # pylint: disable-next=missing-param-doc
    def backtrack(self, *_: Any, **__: Any) -> None:  # type: ignore[misc]
        """Raise an error.

        Raises:
            NotImplementedError: Not implemented.
        """
        raise NotImplementedError

    # pylint: disable-next=missing-param-doc
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

    # pylint: disable-next=missing-param-doc
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
