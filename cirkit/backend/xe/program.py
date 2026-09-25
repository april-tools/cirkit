import string
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Any, Callable, Literal, Union

import extended_einsum.interface as xe
import torch
from extended_einsum.backend_translation import (
    run_program,
    translate_to_backend_program,
)
from extended_einsum.backend_translation.backend import BackendProgram
from extended_einsum.backends.registry import (
    get_backend_compiler,
    get_backend_functions,
)
from extended_einsum.interface.tensor_expression import Parameter as XEParameter
from extended_einsum.interface.tensor_expression import TensorExpression
from extended_einsum.language.rich_program import RichProgram
from extended_einsum.language.types import StabilityMode
from extended_einsum.preprocess import (
    AnnotateShortSameIndexContractions,
    FoldSameShapedOperations,
    OptimizeContractionPaths,
)
from torch import Tensor

from cirkit.backend.xe.layers import (
    XEHadamardLayer,
    XEInputLayer,
    XEKroneckerLayer,
    XELayer,
    XESumLayer,
)

_EINSUM_SYMBOLS = string.ascii_lowercase + string.ascii_uppercase

# The type of the values flowing through the expression construction. Input layer
# outputs enter the expression graph as (wrapped) placeholder arrays, while the
# output of every inner layer is a tensor expression.
_XEValue = Union[TensorExpression, Any]


@dataclass(frozen=True)
class XECompilationConfig:
    """The configuration of the lowering of circuits to extended-einsum programs."""

    semiring: Literal["lse-sum", "sum-product"] = "lse-sum"
    """The semiring defining the output of compiled circuits: 'lse-sum' outputs
    log-values, while 'sum-product' outputs linear-space values."""
    stability: StabilityMode = "logspace_max"
    """The extended-einsum numerical stability mode used to evaluate programs."""
    fold: bool = False
    """Whether to fold (or vectorize) same-shaped operations in the program."""
    fold_depth: Literal["input", "output"] = "input"
    """The grouping strategy used when folding, i.e., whether to group operations
    by their depth with respect to the program inputs or the program output."""
    optimize: bool = False
    """Whether to optimize the contraction paths of einsum operations."""
    scale_interval: int = 3
    """How often to re-normalize intermediate results when a scaled stability
    mode is used."""
    jit: bool = False
    """Whether to just-in-time compile the lowered program with ```torch.compile```."""


@dataclass(frozen=True)
class _LeafSource:
    """Identifies the input-layer output that supplies an XE program input."""

    index: int


@dataclass(frozen=True)
class _WeightSource:
    """Identifies the weight tensor that supplies an XE program input."""

    index: int
    shape: tuple[int, ...]


_Source = Union[_LeafSource, _WeightSource]


@dataclass(frozen=True)
class _SingleInput:
    """Describes one XE input read from a leaf or weight source."""

    source: _Source
    axis0_order: Tensor | None = None


@dataclass(frozen=True)
class _StackedInput:
    entries: tuple[_SingleInput, ...]


@dataclass(frozen=True)
class _ConstantInput:
    """Stores a constant input required by the generated XE program."""

    value: Tensor


_RuntimeInput = Union[_SingleInput, _StackedInput, _ConstantInput]


@dataclass
class _ExpressionBuild:
    """Tracks placeholders and their sources while building an XE expression."""

    batch_size: int
    # Map from the id of a placeholder array wrapper to the source feeding it.
    sources: dict[int, _Source] = field(default_factory=dict)
    # Keep the placeholder wrappers alive so that their ids remain unique until
    # the sources of the extracted program inputs have been resolved.
    wrappers: list[Any] = field(default_factory=list)
    leaf_wrappers: dict[int, Any] = field(default_factory=dict)
    weight_wrappers: dict[tuple[int, tuple[int, ...]], XEParameter] = field(
        default_factory=dict
    )

    def leaf_placeholder(
        self, num_units: int, *, index: int, fold_idx: int, num_folds: int
    ) -> TensorExpression:
        """Create an XE placeholder for an input layer and select its fold."""
        wrapper = self.leaf_wrappers.get(index)
        if wrapper is None:
            wrapper = xe.array(
                torch.empty((num_folds, self.batch_size, num_units), device="meta")
            )
            self.sources[id(wrapper)] = _LeafSource(index)
            self.wrappers.append(wrapper)
            self.leaf_wrappers[index] = wrapper
        return xe.select(wrapper, fold_idx, axis=0)

    def weight_placeholder(
        self,
        shape: tuple[int, ...],
        *,
        index: int,
        fold_idx: int,
        num_folds: int,
    ) -> _XEValue:
        """Create an XE placeholder for a weight tensor and select its fold."""
        key = index, shape
        parameter = self.weight_wrappers.get(key)
        if parameter is None:
            parameter_shape = shape if num_folds == 1 else (num_folds, *shape)
            parameter = XEParameter(
                xe.array(torch.empty(parameter_shape, device="meta"))
            )
            self.sources[id(parameter.array)] = _WeightSource(index, parameter_shape)
            self.wrappers.append(parameter.array)
            self.weight_wrappers[key] = parameter
        if num_folds == 1:
            return parameter
        return xe.select(parameter, fold_idx, axis=0)


def _einsum_symbols(count: int) -> str:
    """Return the requested number of distinct einsum axis labels."""
    if count > len(_EINSUM_SYMBOLS):
        raise NotImplementedError(
            f"Cannot construct an einsum operation over {count} axes, "
            f"as at most {len(_EINSUM_SYMBOLS)} unique axes are supported"
        )
    return _EINSUM_SYMBOLS[:count]


def _build_input_layer(
    build: _ExpressionBuild, xl: XEInputLayer, *, module_idx: int
) -> tuple[_XEValue, tuple[int, ...]]:
    """Add a torch input-layer output to the XE expression."""
    return build.leaf_placeholder(
        xl.num_output_units,
        index=module_idx,
        fold_idx=xl.fold_idx,
        num_folds=xl.module.num_folds,
    ), (xl.num_output_units,)


def _build_sum_layer(
    build: _ExpressionBuild,
    xl: XESumLayer,
    children: Sequence[tuple[_XEValue, tuple[int, ...]]],
    *,
    module_idx: int,
) -> tuple[_XEValue, tuple[int, ...]]:
    """Add a Cirkit sum layer to the XE expression as a contraction."""

    def activate_weight(weight: _XEValue, shape: tuple[int, ...]) -> _XEValue:
        """Apply the weight softmax over the matching XE axes, when needed."""
        if xl.weight_softmax_dim is None:
            return weight
        dim = xl.weight_softmax_dim
        axis: int | tuple[int, ...] = dim
        if dim == len(xl.weight.shape) - 1 and len(shape) > len(xl.weight.shape):
            axis = tuple(range(dim, len(shape)))
        return xe.softmax(weight, axis=axis)

    units = children[0][1]
    if any(units_i != units for _, units_i in children[1:]):
        raise NotImplementedError(
            "Sum layers over layers having different unit factorizations "
            "are not supported by the extended-einsum backend"
        )
    if xl.num_input_units != _units_size(units):
        raise ValueError(
            f"Expected {xl.num_input_units} units in the layers that are input "
            f"to a sum layer, but found {_units_size(units)}"
        )
    if xl.arity == 1:
        # value: (B, *U), weight: (K_o, *U) -> (B, K_o)
        syms = _einsum_symbols(len(units) + 2)
        batch_sym, out_sym, unit_syms = syms[0], syms[1], syms[2:]
        weight = build.weight_placeholder(
            (xl.num_output_units, *units),
            index=module_idx,
            fold_idx=xl.fold_idx,
            num_folds=xl.weight.num_folds,
        )
        weight = activate_weight(weight, (xl.num_output_units, *units))
        format_string = (
            f"{batch_sym}{unit_syms},{out_sym}{unit_syms}->{batch_sym}{out_sym}"
        )
        value = xe.einsum(format_string, children[0][0], weight)
    else:
        # values: (B, H, *U), weight: (K_o, H, *U) -> (B, K_o)
        syms = _einsum_symbols(len(units) + 3)
        batch_sym, arity_sym, out_sym, unit_syms = syms[0], syms[1], syms[2], syms[3:]
        stacked = xe.stack([value for value, _ in children], axis=1)
        weight = build.weight_placeholder(
            (xl.num_output_units, xl.arity, *units),
            index=module_idx,
            fold_idx=xl.fold_idx,
            num_folds=xl.weight.num_folds,
        )
        weight = activate_weight(weight, (xl.num_output_units, xl.arity, *units))
        format_string = (
            f"{batch_sym}{arity_sym}{unit_syms},"
            f"{out_sym}{arity_sym}{unit_syms}->{batch_sym}{out_sym}"
        )
        value = xe.einsum(format_string, stacked, weight)
    return value, (xl.num_output_units,)


def _build_hadamard_layer(
    xl: XEHadamardLayer,
    children: Sequence[tuple[_XEValue, tuple[int, ...]]],
) -> tuple[_XEValue, tuple[int, ...]]:
    """Add a Hadamard layer as an elementwise XE contraction."""
    units = children[0][1]
    if any(units_i != units for _, units_i in children[1:]):
        raise NotImplementedError(
            "Hadamard layers over layers having different unit factorizations "
            "are not supported by the extended-einsum backend"
        )
    syms = _einsum_symbols(len(units) + 1)
    operand_syms = syms[0] + syms[1:]
    format_string = ",".join([operand_syms] * xl.arity) + "->" + operand_syms
    return xe.einsum(format_string, *(value for value, _ in children)), units


def _build_kronecker_layer(
    xl: XEKroneckerLayer,
    children: Sequence[tuple[_XEValue, tuple[int, ...]]],
) -> tuple[_XEValue, tuple[int, ...]]:
    """Add a Kronecker layer while keeping its unit axes separate."""
    if len(children) != xl.arity:
        raise ValueError(
            f"Expected {xl.arity} layers as input to a Kronecker layer, "
            f"but found {len(children)}"
        )
    units = tuple(chain.from_iterable(units_i for _, units_i in children))
    syms = _einsum_symbols(len(units) + 1)
    batch_sym, unit_syms = syms[0], syms[1:]
    operand_syms = []
    offset = 0
    for _, units_i in children:
        operand_syms.append(batch_sym + unit_syms[offset : offset + len(units_i)])
        offset += len(units_i)
    format_string = ",".join(operand_syms) + "->" + batch_sym + unit_syms
    return xe.einsum(format_string, *(value for value, _ in children)), units


def _units_size(units: tuple[int, ...]) -> int:
    """Return the flat number of units represented by several unit axes."""
    size = 1
    for dim in units:
        size *= dim
    return size


def _build_expression(
    layers: Sequence[XELayer],
    in_layers: Sequence[Sequence[int]],
    outputs: Sequence[int],
    *,
    batch_size: int,
    semiring: str,
) -> tuple[TensorExpression, _ExpressionBuild, int, int]:
    """Build one XE expression from the compiled circuit layers."""
    build = _ExpressionBuild(batch_size)
    input_module_ids = {
        module: idx
        for idx, module in enumerate(
            dict.fromkeys(xl.module for xl in layers if isinstance(xl, XEInputLayer))
        )
    }
    weight_module_ids = {
        module: idx
        for idx, module in enumerate(
            dict.fromkeys(xl.weight for xl in layers if isinstance(xl, XESumLayer))
        )
    }
    values: list[tuple[_XEValue, tuple[int, ...]]] = []
    for xl, in_idx in zip(layers, in_layers):
        children = [values[i] for i in in_idx]
        if isinstance(xl, XEInputLayer):
            value = _build_input_layer(
                build, xl, module_idx=input_module_ids[xl.module]
            )
        elif isinstance(xl, XESumLayer):
            value = _build_sum_layer(
                build, xl, children, module_idx=weight_module_ids[xl.weight]
            )
        elif isinstance(xl, XEHadamardLayer):
            value = _build_hadamard_layer(xl, children)
        elif isinstance(xl, XEKroneckerLayer):
            value = _build_kronecker_layer(xl, children)
        else:
            raise NotImplementedError(f"Unknown compiled layer of type {type(xl)}")
        values.append(value)

    output_values = [values[i] for i in outputs]
    output_units = output_values[0][1]
    if any(units_i != output_units for _, units_i in output_values[1:]):
        raise NotImplementedError(
            "Circuits having output layers with different numbers of units "
            "are not supported by the extended-einsum backend"
        )
    if len(output_values) == 1:
        root = output_values[0][0]
    else:
        root = xe.stack([value for value, _ in output_values], axis=0)
    if semiring == "lse-sum":
        root = xe.log(root)
    elif not isinstance(root, TensorExpression):
        # The program must have at least one operation, so we introduce an einsum
        # operation encoding the identity over a placeholder input.
        syms = _einsum_symbols(len(output_units) + 1)
        operand_syms = syms[0] + syms[1:]
        root = xe.einsum(f"{operand_syms}->{operand_syms}", root)
    return root, build, len(output_values), _units_size(output_units)


class XEProgramExecutor:
    """Runs an XE program for a specific batch size and device."""

    def __init__(
        self,
        backend_program: BackendProgram,
        runtime_inputs: Sequence[_RuntimeInput],
        *,
        num_outputs: int,
        num_output_units: int,
        jit: bool,
    ) -> None:
        """Initialize the executor with a program and its runtime input layout."""
        self._backend_program = backend_program
        self._runtime_inputs = runtime_inputs
        self.num_outputs = num_outputs
        self.num_output_units = num_output_units
        self._jit = jit
        self._run: Callable[[Sequence[Tensor]], Tensor] | None = None

    def _resolve_source(
        self, source: _Source, leaves: Sequence[Tensor], weights: Sequence[Tensor]
    ) -> Tensor:
        """Retrieve the tensor identified by a leaf or weight source."""
        if isinstance(source, _LeafSource):
            return leaves[source.index]
        return weights[source.index].reshape(source.shape)

    def _resolve_single(
        self, entry: _SingleInput, leaves: Sequence[Tensor], weights: Sequence[Tensor]
    ) -> Tensor:
        """Build one XE input and apply its requested axis order."""
        x = self._resolve_source(entry.source, leaves, weights)
        if entry.axis0_order is not None:
            x = x.index_select(0, entry.axis0_order)
        return x

    def __call__(self, leaves: Sequence[Tensor], weights: Sequence[Tensor]) -> Tensor:
        """Assemble the runtime inputs and execute the XE program."""
        inputs: list[Tensor] = []
        for entry in self._runtime_inputs:
            if isinstance(entry, _SingleInput):
                inputs.append(self._resolve_single(entry, leaves, weights))
            elif isinstance(entry, _StackedInput):
                inputs.append(
                    torch.stack(
                        [
                            self._resolve_single(e, leaves, weights)
                            for e in entry.entries
                        ],
                        dim=0,
                    )
                )
            else:
                inputs.append(entry.value)
        if self._run is None:
            if self._jit:
                self._run = get_backend_compiler("torch").compile(
                    self._backend_program, inputs
                )
            else:
                self._run = partial(run_program, self._backend_program)
        return self._run(inputs)


def _fold_program(
    program: RichProgram,
    sources: Sequence[_Source],
    *,
    fold_depth: str,
    device: torch.device,
) -> tuple[RichProgram, list[_RuntimeInput]]:
    """Fold an XE program and describe the inputs expected by the result."""
    if fold_depth == "input":
        folded = FoldSameShapedOperations.apply_with_input_depth_metadata(program)
    elif fold_depth == "output":
        folded = FoldSameShapedOperations.apply_with_metadata(program)
    else:
        raise ValueError(f"Unknown folding depth '{fold_depth}'")

    # The folded program has a different input convention: (i) unused inputs are
    # dropped, (ii) same-shaped parameter inputs are packed into stacked inputs, and
    # (iii) gather indices are introduced as new inputs. Moreover, the folding pass
    # may permute inputs along their first axis, as to optimize the memory layout.
    used_input_ids = {
        argument
        for instruction in program.instructions
        for argument in instruction.argument_ssa_ids
        if argument < program.n_inputs
    }
    packed_input_ids = {
        input_id
        for stack_order in folded.parameter_stack_orders
        for input_id in stack_order
    }
    retained_input_ids = [
        input_id
        for input_id in range(program.n_inputs)
        if input_id in used_input_ids and input_id not in packed_input_ids
    ]

    def single_input(input_id: int) -> _SingleInput:
        """Describe one retained input of the folded program."""
        axis0_order = folded.input_axis0_orders.get(input_id)
        return _SingleInput(
            sources[input_id],
            (
                None
                if axis0_order is None
                else torch.tensor(axis0_order, dtype=torch.int64, device=device)
            ),
        )

    runtime_inputs: list[_RuntimeInput] = [single_input(i) for i in retained_input_ids]
    runtime_inputs.extend(
        _StackedInput(tuple(single_input(i) for i in stack_order))
        for stack_order in folded.parameter_stack_orders
    )
    runtime_inputs.extend(
        _ConstantInput(torch.tensor(list(indices), dtype=torch.int64, device=device))
        for indices in folded.gather_index_orders
    )
    return folded.program, runtime_inputs


def build_source_axis0_orders(
    layers: Sequence[XELayer],
    in_layers: Sequence[Sequence[int]],
    outputs: Sequence[int],
    *,
    config: XECompilationConfig,
) -> tuple[
    dict[int, tuple[int, ...]],
    dict[int, tuple[int, ...]],
    list[list[int]],
]:
    """Find the input and weight orders preferred by XE folding.

    A trial lowering lets Cirkit arrange its folded torch tensors in the same order.
    """
    root, build, _, _ = _build_expression(
        layers, in_layers, outputs, batch_size=1, semiring=config.semiring
    )
    program, program_inputs = xe.extract_program(root, config.stability)
    sources = [build.sources[id(wrapper)] for wrapper in program_inputs]
    if config.fold_depth == "input":
        folded = FoldSameShapedOperations.apply_with_input_depth_metadata(program)
    else:
        folded = FoldSameShapedOperations.apply_with_metadata(program)

    leaf_orders: dict[int, tuple[int, ...]] = {}
    weight_orders: dict[int, tuple[int, ...]] = {}
    for input_id, order in folded.input_axis0_orders.items():
        source = sources[input_id]
        orders = leaf_orders if isinstance(source, _LeafSource) else weight_orders
        previous = orders.setdefault(source.index, order)
        if previous != order:
            raise ValueError(
                "XE requested incompatible orders for the same folded source"
            )
    weight_stacks: list[list[int]] = []
    for stack_order in folded.parameter_stack_orders:
        stack_sources = [sources[input_id] for input_id in stack_order]
        if not all(isinstance(source, _WeightSource) for source in stack_sources):
            raise ValueError("XE attempted to stack a non-parameter frontend input")
        weight_stacks.append([source.index for source in stack_sources])
    return leaf_orders, weight_orders, weight_stacks


def build_executor(
    layers: Sequence[XELayer],
    in_layers: Sequence[Sequence[int]],
    outputs: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    config: XECompilationConfig,
) -> XEProgramExecutor:
    """Build an executable XE program for a batch size and device."""
    root, build, num_outputs, num_output_units = _build_expression(
        layers, in_layers, outputs, batch_size=batch_size, semiring=config.semiring
    )
    program, program_inputs = xe.extract_program(root, config.stability)

    # Resolve the source of each program input, i.e., the output of which input layer
    # or of which parameter computational graph feeds each program input.
    sources = [build.sources[id(wrapper)] for wrapper in program_inputs]
    del build

    runtime_inputs: list[_RuntimeInput]
    if config.fold:
        program, runtime_inputs = _fold_program(
            program, sources, fold_depth=config.fold_depth, device=device
        )
    else:
        runtime_inputs = [_SingleInput(source) for source in sources]
    if config.optimize:
        program = OptimizeContractionPaths.apply(program)
        program = AnnotateShortSameIndexContractions.apply(program)

    backend_program = translate_to_backend_program(
        program,
        get_backend_functions("torch"),
        scale_interval=config.scale_interval,
    )
    return XEProgramExecutor(
        backend_program,
        runtime_inputs,
        num_outputs=num_outputs,
        num_output_units=num_output_units,
        jit=config.jit,
    )
