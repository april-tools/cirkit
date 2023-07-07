from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor, nn

from cirkit.layers.exp_family import ExpFamilyLayer
from cirkit.layers.layer import Layer
from cirkit.layers.mixing import MixingLayer
from cirkit.layers.sum_product import SumProductLayer
from cirkit.region_graph import RegionGraph, RegionNode

# TODO: check all type casts. There should not be any without a good reason
# TODO: rework docstrings


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
        #   (ii) the id of the layer that computes such fold (-1 for the input layer)
        region_id_fold: Dict[int, Tuple[int, int]] = {}
        for i, region in enumerate(self.graph_layers[0][1]):
            region_id_fold[region.get_id()] = (i, 0)

        # A dictionary mapping layer ids to the number of folds
        num_folds = [len(self.graph_layers[0][1])]

        # Book-keeping: for each layer
        self.bookkeeping: List[
            Tuple[bool, List[int], Tensor]
        ] = []

        # Build inner layers
        inner_layers: List[SumProductLayer] = []
        for layer_idx, (lpartitions, lregions) in enumerate(self.graph_layers[1:], start=1):
            # Gather the input regions of each partition
            input_regions = [sorted(p.inputs) for p in lpartitions]
            num_input_regions = list(len(ins) for ins in input_regions)
            max_num_input_regions = max(num_input_regions)

            input_regions_ids = [list(r.get_id() for r in ins) for ins in input_regions]
            input_layers_ids = [list(region_id_fold[i][1] for i in ids) for ids in input_regions_ids]
            unique_layer_ids = list(set(i for ids in input_layers_ids for i in ids))
            cumulative_idx = np.cumsum([0] + [num_folds[i] for i in unique_layer_ids]).tolist()
            base_layer_idx = {layer_id: idx for layer_id, idx in zip(unique_layer_ids, cumulative_idx)}

            should_pad = False
            input_region_indices = list()
            for regions in input_regions:
                region_indices = list()
                for r in regions:
                    fold_idx, layer_id = region_id_fold[r.get_id()]
                    region_indices.append(base_layer_idx[layer_id] + fold_idx)
                if len(regions) < max_num_input_regions:
                    should_pad = True
                    region_indices.extend([-1] * (max_num_input_regions - len(regions)))
                input_region_indices.append(region_indices)

            book_entry = (should_pad, unique_layer_ids, torch.tensor(input_region_indices))
            self.bookkeeping.append(book_entry)

            for i, p in enumerate(lpartitions):
                # Each partition must belong to exactly one region
                assert len(p.outputs) == 1
                out_region = p.outputs[0]
                region_id_fold[out_region.get_id()] = (i, layer_idx)
            num_folds.append(cumulative_idx[-1])

            num_outputs = num_output_units if layer_idx < len(self.graph_layers) - 1 else num_classes
            num_inputs = num_input_units if layer_idx == 1 else num_output_units
            layer = layer_cls(
                lpartitions, num_inputs, num_outputs, **layer_kwargs  # type: ignore[misc]
            )
            inner_layers.append(layer)

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
        in_outputs = self.input_layer(x)
        in_outputs = in_outputs.permute(2, 0, 1)
        outputs: List[Tensor] = [in_outputs]
        # (batch_size, num_units, num_regions)

        # TODO: Generalize if statements here, they should be layer agnostic
        for layer, (should_pad, in_layer_ids, fold_idx) in zip(self.inner_layers, self.bookkeeping):
            if isinstance(layer, SumProductLayer):  # type: ignore[misc]
                # (fold_1 + fold_2, batch_size, units)
                print(in_layer_ids)
                inputs = torch.cat([outputs[i] for i in in_layer_ids], dim=0)
                if should_pad:
                    # TODO: pad along dim 0
                    pass
                # (new_fold, arity, batch_size, units)
                inputs = inputs[fold_idx]
                print(inputs.shape)
                output = layer(inputs)
            elif isinstance(layer, MixingLayer):
                pass
            else:
                assert False
            outputs.append(output)

        return outputs[-1][:, :, 0]

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
