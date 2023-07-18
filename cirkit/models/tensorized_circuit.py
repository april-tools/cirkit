from collections import defaultdict
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cirkit.atlas.integrate import IntegrationContext
from cirkit.layers.input import InputLayer
from cirkit.layers.input.constant import ConstantLayer
from cirkit.layers.input.exp_family import ExpFamilyLayer
from cirkit.layers.input.integral import IntegralLayer
from cirkit.layers.layer import Layer
from cirkit.layers.mixing import MixingLayer
from cirkit.layers.scope import ScopeLayer
from cirkit.layers.sum_product import SumProductLayer
from cirkit.region_graph import RegionGraph

# TODO: check all type casts. There should not be any without a good reason
# TODO: rework docstrings


class TensorizedPC(nn.Module):
    """Tensorized and folded PC implementation."""

    def __init__(  # type: ignore[misc]
        self,
        rg: RegionGraph,
        layer_cls: Type[SumProductLayer],
        efamily_cls: Type[ExpFamilyLayer],
        *,
        layer_kwargs: Optional[Dict[str, Any]] = None,
        efamily_kwargs: Optional[Dict[str, Any]] = None,
        num_inner_units: int = 2,
        num_input_units: int = 2,
        num_channels: int = 1,
        num_classes: int = 1,
    ) -> None:
        """Make an TensorizedPC.

        Args:
            rg (RegionGraph): The region rg.
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

        if layer_kwargs is None:  # type: ignore[misc]
            layer_kwargs = {}
        if efamily_kwargs is None:  # type: ignore[misc]
            efamily_kwargs = {}

        # TODO: check rg. but do we need it?
        self.rg = rg
        self.num_variables = rg.num_variables
        assert (
            len(list(rg.output_nodes)) == 1
        ), "Currently only circuits with single root region node are supported."

        # TODO: return a list instead?
        output_node = list(rg.output_nodes)[0]
        assert output_node.scope == set(
            range(self.num_variables)
        ), "The region rg should have been defined on the same number of variables"

        # Algorithm 1 in the paper -- organize the PC in layers  NOT BOTTOM UP !!!
        self.graph_layers = rg.topological_layers(bottom_up=False)

        # Initialize input layer
        self.input_layer: InputLayer = efamily_cls(
            self.graph_layers[0][1],
            num_channels,
            num_input_units,
            **efamily_kwargs,  # type: ignore[misc]
        )

        # Initialize scope layer
        self.scope_layer = ScopeLayer(self.graph_layers[0][1])

        # Book-keeping: for each layer keep track of the following information
        # (i) Whether the input tensor needs to be padded.
        #     This is necessary if we want to fold layers with different number of inputs.
        # (ii) The list of layers whose output tensors needs to be concatenated.
        #      This is necessary because the inputs of a layer might come from different layers.
        # (iii) The tensorized indices of shape (fold count, arity): (F, H),
        #       where arity is the number of inputs of the layer. When folding
        #       layers with different arity (e.g., the mixing layer) a padding will be
        #       added *and* the last dimension will correspond to the maximum arity.
        self.bookkeeping: List[Tuple[bool, List[int], Tensor]] = []

        # TODO: can we annotate a list here?
        self.inner_layers: List[Layer] = nn.ModuleList()  # type: ignore[assignment]

        # Build layers: this will populate the book-keeping data structure above
        self._build_layers(
            layer_cls,
            layer_kwargs,  # type: ignore[misc]
            num_inner_units,
            num_input_units,
            num_classes=num_classes,
        )

    def integrate(self, icontext: Optional[IntegrationContext] = None) -> "TensorizedPC":
        """Integrate the tensorized circuit encoding c(X_1, ..., X_d).

        Args:
            icontext: The integration context containing the variables to integrate.
             If this is None then all variables will be integrated.

        Returns:
            Another tensorized circuit computing the integral of c over some variables.
        """
        circuit = copy(self)  # Shallow copy
        if icontext is None:
            circuit.input_layer = ConstantLayer(
                circuit.input_layer.rg_nodes, value=circuit.input_layer.integrate
            )
        else:
            circuit.input_layer = IntegralLayer(
                circuit.input_layer.rg_nodes, circuit.input_layer, icontext
            )
        return circuit

    # pylint: disable-next=too-many-arguments,too-complex,too-many-locals,too-many-statements
    def _build_layers(  # type: ignore[misc]
        self,
        layer_cls: Type[SumProductLayer],
        layer_kwargs: Dict[str, Any],
        num_inner_units: int,
        num_input_units: int,
        num_classes: int = 1,
    ) -> None:
        """Build the layers of the network.

        Args:
            layer_cls (Type[SumProductNetwork]): The layer class.
            layer_kwargs (Dict[str, Any]): The layer arguments.
            num_inner_units (int): The number of units per inner layer.
            num_input_units (int): The number of units of the input layer.
            num_classes (int): The number of outputs of the network.
        """
        # A dictionary mapping each region node ID to
        #   (i) its index in the corresponding fold, and
        #   (ii) the id of the layer that computes such fold
        #        (0 for the input layer and > 0 for inner layers)
        region_id_fold: Dict[int, Tuple[int, int]] = {}
        for i, region in enumerate(self.graph_layers[0][1]):
            region_id_fold[region.get_id()] = (i, 0)

        # A list mapping layer ids to the number of folds in the output tensor
        num_folds = [len(self.graph_layers[0][1])]

        # Build inner layers
        for rg_layer_idx, (lpartitions, lregions) in enumerate(self.graph_layers[1:], start=1):
            # Gather the input regions of each partition
            input_regions = [sorted(p.inputs) for p in lpartitions]
            num_input_regions = list(len(ins) for ins in input_regions)
            max_num_input_regions = max(num_input_regions)

            # Retrieve which folds need to be concatenated
            input_regions_ids = [list(r.get_id() for r in ins) for ins in input_regions]
            input_layers_ids = [
                list(region_id_fold[i][1] for i in ids) for ids in input_regions_ids
            ]
            unique_layer_ids = list(set(i for ids in input_layers_ids for i in ids))
            cumulative_idx: List[int] = np.cumsum(  # type: ignore[misc]
                [0] + [num_folds[i] for i in unique_layer_ids]
            ).tolist()
            base_layer_idx = dict(zip(unique_layer_ids, cumulative_idx))

            # Build indices
            should_pad = False
            input_region_indices = []  # (F, H)
            for regions in input_regions:
                region_indices = []
                for r in regions:
                    fold_idx, layer_id = region_id_fold[r.get_id()]
                    region_indices.append(base_layer_idx[layer_id] + fold_idx)
                if len(regions) < max_num_input_regions:
                    should_pad = True
                    region_indices.extend(
                        [cumulative_idx[-1]] * (max_num_input_regions - len(regions))
                    )
                input_region_indices.append(region_indices)
            book_entry = (should_pad, unique_layer_ids, torch.tensor(input_region_indices))
            self.bookkeeping.append(book_entry)

            # Update dictionaries and number of folds
            region_mixing_indices: Dict[int, List[int]] = defaultdict(list)
            for i, p in enumerate(lpartitions):
                # Each partition must belong to exactly one region
                assert len(p.outputs) == 1
                out_region = p.outputs[0]
                if len(out_region.inputs) == 1:
                    region_id_fold[out_region.get_id()] = (i, len(self.inner_layers) + 1)
                else:
                    region_mixing_indices[out_region.get_id()].append(i)
            num_folds.append(len(lpartitions))

            # Build the actual layer
            num_outputs = (
                num_inner_units if rg_layer_idx < len(self.graph_layers) - 1 else num_classes
            )
            num_inputs = num_input_units if rg_layer_idx == 1 else num_inner_units
            layer = layer_cls(
                lpartitions, num_inputs, num_outputs, **layer_kwargs  # type: ignore[misc]
            )
            self.inner_layers.append(layer)

            # Fold mixing layers, if any
            if not (non_unary_regions := [r for r in lregions if len(r.inputs) > 1]):
                continue
            max_num_input_partitions = max(len(r.inputs) for r in non_unary_regions)

            # Same as above, construct indices and update dictionaries
            should_pad = False
            input_partition_indices = []  # (F, H)
            for i, region in enumerate(non_unary_regions):
                num_input_partitions = len(region.inputs)
                partition_indices = region_mixing_indices[region.get_id()]
                if max_num_input_partitions > num_input_partitions:
                    should_pad = True
                    partition_indices.extend(
                        [num_folds[-1]] * (max_num_input_partitions - num_input_partitions)
                    )
                input_partition_indices.append(partition_indices)
                region_id_fold[region.get_id()] = (i, len(self.inner_layers) + 1)
            num_folds.append(len(non_unary_regions))

            # Build the actual mixing layer
            mixing_layer = MixingLayer(non_unary_regions, num_outputs, max_num_input_partitions)
            self.bookkeeping.append(
                (should_pad, [len(self.inner_layers)], torch.tensor(input_partition_indices))
            )
            self.inner_layers.append(mixing_layer)

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
            x (Tensor): the shape is (batch_size, self.num_vars, self.num_channels)

        Returns:
            Tensor: Return value.
        """
        in_outputs = self.scope_layer(self.input_layer(x))
        layer_outputs: List[Tensor] = [in_outputs]

        for layer, (should_pad, in_layer_ids, fold_idx) in zip(self.inner_layers, self.bookkeeping):
            if len(in_layer_ids) == 1:
                # (F, K, B)
                (in_layer_id,) = in_layer_ids
                inputs = layer_outputs[in_layer_id]
            else:
                # (F_1 + ... + F_n, K, B)
                inputs = torch.cat([layer_outputs[i] for i in in_layer_ids], dim=0)
            if should_pad:
                # TODO: The padding value depends on the computation space.
                #  It should be the absorbing element (or annihilating element) of a group.
                #  For now computations are in log-space, thus -infinity is our pad value.
                inputs = F.pad(inputs, [0, 0, 0, 0, 0, 1], value=-float("inf"))
            inputs = inputs[fold_idx]  # inputs: (F, H, K, B)
            outputs = layer(inputs)  # outputs: (F, K, B)
            layer_outputs.append(outputs)

        return layer_outputs[-1][0].T  # (B, K)

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
