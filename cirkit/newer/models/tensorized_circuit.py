from typing import Dict, Optional, List, Tuple, Type, Iterable

import torch
from torch import Tensor, nn

from cirkit.newer.layers import InputLayer, Layer, DenseLayer, HadamardLayer, KroneckerLayer, CategoricalLayer, MixingLayer
from cirkit.newer.layers.input import ConstantLayer
from cirkit.newer.reparams import Reparameterization
from cirkit.newer.symbolic.layers.input.symb_constant import SymbConstantLayer
from cirkit.newer.symbolic.layers.input.symb_ef import SymbCategoricalLayer
from cirkit.newer.symbolic.symb_circuit import SymbCircuit, pipeline_topological_ordering
from cirkit.newer.symbolic.layers import (
    SymbLayer,
    SymbSumLayer, SymbInputLayer, SymbHadamardLayer, SymbKroneckerLayer, SymbMixingLayer, SymbProdLayer
)
from cirkit.utils import Scope


class TensorizedCircuit(nn.Module):
    """The tensorized circuit with concrete computational graph in PyTorch.

    This class is aimed for computation, and therefore does not include excessive strutural \
    properties. If those are really needed, use the properties of TensorizedCircuit.symb_circuit.
    """
    def __init__(
            self,
            symb_circuit: SymbCircuit,
            layers: List[Layer],
            bookkeeping: List[Tuple[List[int], Optional[Tensor]]]
    ) -> None:
        """Init class.

        Args:
            symb_circuit (SymbolicTensorizedCircuit): The symbolic version of the circuit.
        """
        super().__init__()
        self.symb_circuit = symb_circuit
        self.scope = symb_circuit.scope
        self.num_vars = symb_circuit.num_vars

        # Automatic nn.Module registry, also in publicly available children names.
        self.layers = nn.ModuleList(layers)
        self.bookkeeping = bookkeeping

    def __call__(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (*B, D, C).

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """  # TODO: single letter name?
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    # TODO: do we accept each variable separately?
    def forward(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input of the circuit, shape (*B, D, C).

        Returns:
            Tensor: The output of the circuit, shape (*B, num_out, num_cls).
        """
        outputs = []  # list of tensors of shape (K, *B)

        for (in_layer_indices, in_fold_idx), layer in zip(self.bookkeeping, self.layers):
            in_tensors = [
                outputs[i] for i in in_layer_indices
            ]  # list of tensors of shape (*B, K) or empty
            if in_tensors:
                inputs = torch.cat(in_tensors, dim=0)  # (H, *B, K)
            else:  # forward through input layers
                # in_fold_idx has shape (D') with D' <= D
                inputs = x[..., in_fold_idx, :]
                inputs = inputs.permute(0, 1, 2)
            outputs.append(layer(inputs))  # (*B, K)

        out_layer_indices, out_fold_idx = self.bookkeeping[-1]
        out_tensors = [outputs[i] for i in out_layer_indices]  # list of tensors of shape (*B, K)
        outputs = torch.cat(out_tensors, dim=0)
        if out_fold_idx is not None:
            outputs = outputs[out_fold_idx]  # (*B, K)
        return outputs


def materialize_pipeline(
        pipeline: SymbCircuit,
        reparam: Reparameterization,
        materialized_pipeline: Optional[Dict[SymbCircuit, Tuple[Dict[SymbLayer, int], TensorizedCircuit]]] = None
) -> Dict[SymbCircuit, Tuple[Dict[SymbLayer, int], TensorizedCircuit]]:
    if materialized_pipeline is None:
        materialized_pipeline: Dict[SymbCircuit, Tuple[Dict[SymbLayer, int], TensorizedCircuit]] = {}

    # Retrieve the topological ordering of the pipeline
    ordering = pipeline_topological_ordering(pipeline)

    # First, materialize the input symbolic circuits in the given circuit pipeline
    input_symb_circuits = filter(lambda sc: sc.operation is None, ordering)
    for symb_circuit in input_symb_circuits:
        # Check if the circuit in the pipeline has already been materialized
        if symb_circuit in materialized_pipeline:
            continue

        # Materialize the circuit
        materialized_pipeline[symb_circuit] = materialize(symb_circuit, reparam)

    # Then, materialize the inner symbolic circuit in the given pipeline
    # The parameters are currently saved in the input materialized circuits, and also shared
    inner_symb_circuits = filter(lambda sc: sc.operation is not None, ordering)
    for symb_circuit in inner_symb_circuits:
        pass

    return materialized_pipeline


def materialize(
        symb_circuit: SymbCircuit,
        reparam: Reparameterization,
        registry: Optional[Dict[Type[SymbLayer], Type[Layer]]] = None
) -> Tuple[Dict[SymbLayer, int], TensorizedCircuit]:

    # Registry mapping symbolic input layers to executable layers classes
    materialize_input_registry: Dict[Type[SymbInputLayer], Type[InputLayer]] = {
        SymbConstantLayer: ConstantLayer,
        SymbCategoricalLayer: CategoricalLayer
    }

    # Registry mapping symbolic inner layers to executable layer classes
    materialize_inner_registry: Dict[Type[SymbLayer], Type[Layer]] = {
        SymbSumLayer: DenseLayer,
        SymbMixingLayer: MixingLayer,
        SymbHadamardLayer: HadamardLayer,
        SymbKroneckerLayer: KroneckerLayer
    }
    if registry is not None:
        for slc, lc in registry.items():
            if issubclass(slc, SymbInputLayer) and issubclass(lc, InputLayer):
                materialize_input_registry[slc] = lc
            else:
                materialize_inner_registry[slc] = lc

    # The list of layers
    layers: List[Layer] = []

    # The bookkeeping data structure
    bookkeeping: List[Tuple[List[int], Optional[Tensor]]] = []

    # A useful map from symbolic layers to layer id (indices for the list of layers)
    symb_layer_map: Dict[SymbLayer, int] = {}

    # Construct the bookkeeping data structure while instantiating layers
    for sl in symb_circuit.layers:  # Assuming the layers are already sorted in topological ordering
        if isinstance(sl, SymbInputLayer):
            layer_cls = materialize_input_registry[type(sl)]
            layer = layer_cls(
                num_input_units=sl.num_channels,
                num_output_units=sl.num_units,
                arity=len(sl.scope),
                reparam=layer_cls.default_reparam()
                **sl.kwargs)
            bookkeeping_entry = ([], torch.tensor([list(sl.scope)]))
            bookkeeping.append(bookkeeping_entry)
        else:
            assert isinstance(sl, (SymbSumLayer, SymbProdLayer)) and len(sl.inputs[0]) > 0
            layer_cls = materialize_inner_registry[type(sl)]
            kwargs = sl.kwargs
            if isinstance(sl, SymbSumLayer):
                kwargs.update(reparam=reparam)
            layer = layer_cls(
                num_input_units=sl.inputs[0].num_units,
                num_output_units=sl.num_units,
                arity=len(sl.inputs),
                **kwargs
            )
            bookkeeping_entry = ([symb_layer_map[isl] for isl in sl.inputs], None)
            bookkeeping.append(bookkeeping_entry)
        layer_id = len(layers)
        symb_layer_map[sl] = layer_id
        layers.append(layer)

    # Append a last bookkeeping entry with the info to extract the (possibly multiple) outputs
    output_indices = [symb_layer_map[sl] for sl in symb_circuit.output_layers]
    bookkeeping_entry = (output_indices, None)
    bookkeeping.append(bookkeeping_entry)

    return symb_layer_map, TensorizedCircuit(symb_circuit, layers, bookkeeping)
