from dataclasses import replace
from typing import TYPE_CHECKING, cast

from cirkit.backend.compiler import (
    AbstractCompiler,
    CompilerInitializerRegistry,
    CompilerLayerRegistry,
    CompilerParameterRegistry,
    InitializerCompilationFunc,
    InitializerCompilationSign,
    LayerCompilationFunc,
    LayerCompilationSign,
    ParameterCompilationFunc,
    ParameterCompilationSign,
)
from cirkit.backend.registry import CompilationRuleNotFound
from cirkit.backend.torch.compiler import (
    TorchCompiler,
    fold_layers_group,
    fold_parameters,
)
from cirkit.backend.torch.graph.folding import group_foldable_modules
from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
from cirkit.backend.torch.parameters.nodes import (
    TorchSoftmaxParameter,
    TorchTensorParameter,
)
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.xe.circuits import XETorchCircuit
from cirkit.backend.xe.layers import (
    XEHadamardLayer,
    XEInputLayer,
    XEKroneckerLayer,
    XELayer,
    XESumLayer,
)
from cirkit.backend.xe.program import XECompilationConfig, build_source_axis0_orders
from cirkit.symbolic.circuit import Circuit, pipeline_topological_ordering
from cirkit.symbolic.layers import (
    HadamardLayer,
    InputLayer,
    KroneckerLayer,
    Layer,
    SumLayer,
)

if TYPE_CHECKING:
    from extended_einsum.language.types import StabilityMode

_DEFAULT_STABILITY_MODES: dict[str, "StabilityMode"] = {
    # In the lse-sum semiring the circuit computations are over non-negative values
    # (with the exception of the sum weights, which the stability translations keep
    # in linear space), so by default we evaluate programs in log-space with
    # max-normalized shifts.
    "lse-sum": "logspace_max",
    # In the sum-product semiring the circuit may compute over negative values,
    # e.g., when sum layers have negative weights, so by default we evaluate
    # programs without any numerical stability translation.
    "sum-product": "unstable",
}


class XETorchCompiler(AbstractCompiler[XETorchCircuit]):
    """The extended-einsum compiler, which lowers symbolic circuits into
    extended-einsum programs that are executed with torch. It is registered
    under the backend name 'xe-torch'.

    The compiler embeds a [TorchCompiler][cirkit.backend.torch.compiler.TorchCompiler]
    (set to the sum-product semiring, without folding nor optimizations), which is
    used to compile the input layers, the symbolic parameter computational graphs,
    and the initializers. This means that (i) all the input layers, parameter
    nodes and initializers supported by the torch backend are supported, and that
    (ii) tensor parameters that are shared across circuits, e.g., as the result of
    symbolic circuit operations such as integration and multiplication, are compiled
    into shared torch parameters, exactly as in the torch backend. The computational
    graph over the inner layers is instead lowered into an extended-einsum program,
    which folds, optimizes and stabilizes the computation.
    """

    def __init__(
        self,
        semiring: str = "lse-sum",
        fold: bool = False,
        optimize: bool = False,
        *,
        stability: "StabilityMode | None" = None,
        fold_depth: str = "input",
        scale_interval: int = 3,
        jit: bool = False,
    ) -> None:
        """Initializes an extended-einsum compiler.

        The compiler needs one configuration object shared by the dry layout pass
        and every lazily built executor; validating and fixing defaults here prevents
        those stages from silently using different lowering semantics.

        Args:
            semiring: The semiring defining the output of compiled circuits:
                'lse-sum' (the default) outputs log-values, while 'sum-product'
                outputs linear-space values. Note that, differently from the torch
                backend, the semiring does __not__ determine how the computations are
                evaluated, which is instead specified by the stability mode.
            fold: Whether to fold (or vectorize) same-shaped operations in the
                lowered extended-einsum programs.
            optimize: Whether to optimize the contraction paths of the einsum
                operations in the lowered extended-einsum programs.
            stability: The extended-einsum numerical stability mode used to evaluate
                programs, one of 'unstable', 'logspace_min', 'logspace_max',
                'scaled_min', 'scaled_max', or 'scaled_sum'. If it is None, then it
                defaults to 'logspace_max' for the lse-sum semiring and to 'unstable'
                for the sum-product semiring.
            fold_depth: The grouping strategy used when folding, either 'input' or
                'output', i.e., whether operations are grouped by their depth with
                respect to the program inputs or the program output.
            scale_interval: How often to re-normalize intermediate results when a
                scaled stability mode is used.
            jit: Whether to just-in-time compile the lowered programs with
                ```torch.compile```.

        Raises:
            ValueError: If the given semiring, stability mode, folding depth or
                scaling interval is invalid.
        """
        if semiring not in _DEFAULT_STABILITY_MODES:
            raise ValueError(
                f"The extended-einsum backend supports the semirings "
                f"{sorted(_DEFAULT_STABILITY_MODES)}, but found '{semiring}'"
            )
        if stability is None:
            stability = _DEFAULT_STABILITY_MODES[semiring]
        if fold_depth not in ("input", "output"):
            raise ValueError(f"Unknown folding depth '{fold_depth}'")
        if scale_interval <= 0:
            raise ValueError("The scale interval must be positive")
        # The embedded torch compiler used to compile input layers, parameters and
        # initializers. It uses the sum-product semiring such that the compiled torch
        # modules evaluate in linear space, which is the space the extended-einsum
        # stability translations expect the program inputs to be in.
        self._torch_compiler = TorchCompiler(
            semiring="sum-product", fold=False, optimize=False
        )
        # The rule registries are unused: input layers, parameters and initializers
        # are compiled by the embedded torch compiler (see the rule methods below),
        # while the closed set of inner layers is lowered directly by compile_layer
        super().__init__(
            CompilerLayerRegistry({}),
            CompilerParameterRegistry({}),
            CompilerInitializerRegistry({}),
            fold=fold,
            optimize=optimize,
        )
        self._config = XECompilationConfig(
            semiring=semiring,  # type: ignore[arg-type]
            stability=stability,
            fold=fold,
            fold_depth=fold_depth,  # type: ignore[arg-type]
            optimize=optimize,
            scale_interval=scale_interval,
            jit=jit,
        )

    @property
    def torch_compiler(self) -> TorchCompiler:
        """Retrieve the embedded torch compiler, which is used to compile input
        layers, symbolic parameter computational graphs, and initializers.

        Exposing it is necessary for callers that need to inspect or configure the
        torch-owned frontend while still compiling the inner graph with XE.

        Returns:
            The embedded torch compiler.
        """
        return self._torch_compiler

    @property
    def semiring(self) -> str:
        """Retrieve the name of the semiring defining the output of compiled circuits.

        Returns:
            The semiring name, either 'lse-sum' or 'sum-product'.
        """
        return self._config.semiring

    @property
    def is_fold_enabled(self) -> bool:
        """Retrieve whether the folding of extended-einsum programs is enabled.

        Circuit compilation uses this flag to decide whether the torch frontend must
        first be physically aligned with the folded XE program.

        Returns:
            True if folding is enabled, False otherwise.
        """
        return cast(bool, self._flags["fold"])

    @property
    def is_optimize_enabled(self) -> bool:
        """Retrieve whether the optimization of extended-einsum programs is enabled.

        The property preserves the generic compiler contract even though actual XE
        path optimization occurs later, when a shape-specialized executor is built.

        Returns:
            True if optimizations are enabled, False otherwise.
        """
        return cast(bool, self._flags["optimize"])

    def add_layer_rule(self, func: LayerCompilationFunc) -> None:
        self._torch_compiler.add_layer_rule(func)

    def add_parameter_rule(self, func: ParameterCompilationFunc) -> None:
        self._torch_compiler.add_parameter_rule(func)

    def add_initializer_rule(self, func: InitializerCompilationFunc) -> None:
        self._torch_compiler.add_initializer_rule(func)

    def retrieve_layer_rule(
        self, signature: LayerCompilationSign
    ) -> LayerCompilationFunc:
        return self._torch_compiler.retrieve_layer_rule(signature)

    def retrieve_parameter_rule(
        self, signature: ParameterCompilationSign
    ) -> ParameterCompilationFunc:
        return self._torch_compiler.retrieve_parameter_rule(signature)

    def retrieve_initializer_rule(
        self, signature: InitializerCompilationSign
    ) -> InitializerCompilationFunc:
        return self._torch_compiler.retrieve_initializer_rule(signature)

    def compile_layer(self, sl: Layer) -> XELayer:
        """Compile a symbolic layer. The computations of the inner layers, i.e., sum,
        Hadamard product and Kronecker product layers, are the ones lowered into the
        extended-einsum program (with the symbolic parameter computational graph of
        sum layer weights being compiled by the embedded torch compiler). Input layers
        are compiled by delegating to the embedded torch compiler, i.e., any (possibly
        custom) input layer having a torch compilation rule is supported.

        This split representation is necessary because XE needs lightweight semantic
        descriptors to construct a shape-specialized expression later, whereas input
        layers and weight graphs need real torch modules immediately for parameter
        ownership, initialization, and sharing.

        Args:
            sl: The symbolic layer to compile.

        Returns:
            The compiled layer representation.

        Raises:
            CompilationRuleNotFound: If the given layer is neither an inner layer nor
                an input layer.
        """
        if isinstance(sl, SumLayer):
            weight = self._torch_compiler.compile_parameter(sl.weight)
            return XESumLayer(
                weight=weight,
                num_input_units=sl.num_input_units,
                num_output_units=sl.num_output_units,
                arity=sl.arity,
            )
        if isinstance(sl, HadamardLayer):
            return XEHadamardLayer(num_input_units=sl.num_input_units, arity=sl.arity)
        if isinstance(sl, KroneckerLayer):
            return XEKroneckerLayer(num_input_units=sl.num_input_units, arity=sl.arity)
        if isinstance(sl, InputLayer):
            module = self._torch_compiler.compile_layer(sl)
            if not isinstance(module, TorchInputLayer):
                raise ValueError(
                    f"Expected the compilation of {type(sl).__name__} to result in an "
                    f"input layer torch module, but found {type(module).__name__}"
                )
            return XEInputLayer(module=module, num_output_units=sl.num_output_units)
        raise CompilationRuleNotFound(type(sl))

    def compile_pipeline(self, sc: Circuit) -> XETorchCircuit:
        """Compile a derived circuit and all operand circuits in its pipeline.

        A circuit produced by a symbolic operation can refer to the circuits from
        which it was derived through `Circuit.operation.operands`. Traversing those
        operands in topological order ensures that every such circuit is translated
        to its own `XETorchCircuit` before its dependents. Each circuit's layer graph
        is lowered independently to an XE program, while the embedded torch compiler
        reuses shared parameter tensors across those programs.

        Args:
            sc: The output circuit whose complete operation pipeline to compile.

        Returns:
            The compiled XE circuit corresponding to ``sc``.
        """
        # Compile the circuits following the topological ordering of the pipeline.
        for sci in pipeline_topological_ordering([sc]):
            # Check if the circuit in the pipeline has already been compiled
            if self.is_compiled(sci):
                continue

            # Compile the circuit
            self._compile_circuit(sci)

        # Return the compiled circuit (i.e., the output of the circuit pipeline)
        return self.get_compiled_circuit(sc)

    def _compile_circuit(self, sc: Circuit) -> XETorchCircuit:
        """Compile and register one symbolic circuit in topological layer order.

        This method is the boundary that turns symbolic edges into integer program
        references, optionally aligns the torch frontend with XE's fold layout,
        initializes newly owned tensors, and records the symbolic/compiled mapping.
        """
        # A map from symbolic layers to their index in the compiled layers list
        layer_ids: dict[Layer, int] = {}
        layers: list[XELayer] = []
        in_layers: list[list[int]] = []

        # Compile layers by following the topological ordering
        for sl in sc.topological_ordering():
            xl = self.compile_layer(sl)
            in_layers.append([layer_ids[sli] for sli in sc.layer_inputs(sl)])
            layer_ids[sl] = len(layers)
            layers.append(xl)

        if self.is_fold_enabled:
            outputs = [layer_ids[sl] for sl in sc.outputs]
            layers = self._fold_torch_frontend(layers, in_layers, outputs)

        # Construct the compiled circuit
        cc = XETorchCircuit(
            sc.scope,
            layers,
            in_layers,
            [layer_ids[sl] for sl in sc.outputs],
            properties=sc.properties,
            config=self._config,
        )

        # Allocate & initialize the parameters. Note that parameters that are shared
        # with previously-compiled circuits are compiled into pointers to the already
        # allocated (and possibly trained) torch parameters, and are not re-initialized.
        cc.reset_parameters()

        # Register the compiled circuit
        self.register_compiled_circuit(sc, cc)

        # Signal the end of the circuit compilation to the embedded torch compiler
        self._torch_compiler.state.finish_compilation()
        return cc

    def _fold_torch_frontend(
        self,
        layers: list[XELayer],
        in_layers: list[list[int]],
        outputs: list[int],
    ) -> list[XELayer]:
        """Fold torch producers in the physical order requested by XE.

        A dry XE lowering first discovers the fold pass's preferred leading-axis
        permutations and parameter stack order. Rebuilding the torch folds in that
        order is necessary to produce XE-ready tensors directly, avoiding an
        ``index_select`` or ``stack`` on every circuit evaluation.
        """
        folded_inputs = self._fold_input_layers(layers)
        input_orders, _, weight_stacks = build_source_axis0_orders(
            folded_inputs, in_layers, outputs, config=self._config
        )
        folded_inputs = self._fold_input_layers(layers, input_orders=input_orders)
        return self._fold_weight_layers(folded_inputs, weight_stacks)

    def _fold_input_layers(
        self,
        layers: list[XELayer],
        *,
        input_orders: dict[int, tuple[int, ...]] | None = None,
    ) -> list[XELayer]:
        """Replace compatible input modules with shared folded torch modules.

        ``input_orders`` maps each foldable group to XE's requested physical order.
        Assigning ``fold_idx`` after applying that order keeps every symbolic input
        layer attached to the correct slice of the newly folded module. This pass
        also records simple tensor-plus-softmax weight graphs that XE can evaluate.
        """
        input_orders = input_orders or {}
        input_layers = [layer for layer in layers if isinstance(layer, XEInputLayer)]
        folded_layers: dict[XELayer, XELayer] = {}
        input_modules: list[TorchLayer] = [layer.module for layer in input_layers]
        for group_idx, input_group in enumerate(group_foldable_modules(input_modules)):
            order = input_orders.get(group_idx)
            if order is not None:
                input_group = [input_group[idx] for idx in order]
            module = fold_layers_group(input_group, compiler=self._torch_compiler)
            assert isinstance(module, TorchInputLayer)
            for fold_idx, unfolded in enumerate(input_group):
                input_xl = next(
                    layer for layer in input_layers if layer.module is unfolded
                )
                folded_layers[input_xl] = replace(
                    input_xl, module=module, fold_idx=fold_idx
                )
        result = [folded_layers.get(xl, xl) for xl in layers]
        for layer_idx, layer in enumerate(result):
            if not isinstance(layer, XESumLayer) or layer.weight_input is not None:
                continue
            weight_input, weight_softmax_dim = self._lower_weight_parameter(
                layer.weight
            )
            if weight_input is not None:
                result[layer_idx] = replace(
                    layer,
                    weight_input=weight_input,
                    weight_softmax_dim=weight_softmax_dim,
                )
        return result

    def _fold_weight_layers(
        self, layers: list[XELayer], weight_stacks: list[list[int]]
    ) -> list[XELayer]:
        """Fold weight graphs according to XE's requested parameter stack order.

        XE reports stacks in terms of the original distinct weight sources. Folding
        those sources in the reported order and updating each layer's ``fold_idx``
        makes the leading axis emitted by torch already match the folded XE program.
        """
        sum_layers = [layer for layer in layers if isinstance(layer, XESumLayer)]
        weight_modules = list(dict.fromkeys(layer.weight for layer in sum_layers))
        folded_layers: dict[XELayer, XELayer] = {}
        for stack in weight_stacks:
            unfolded_weights = [weight_modules[idx] for idx in stack]
            weight = fold_parameters(self._torch_compiler, unfolded_weights)
            weight_input, weight_softmax_dim = self._lower_weight_parameter(weight)
            fold_indices = {
                unfolded: idx for idx, unfolded in enumerate(unfolded_weights)
            }
            for sum_layer in sum_layers:
                fold_idx = fold_indices.get(sum_layer.weight)
                if fold_idx is not None:
                    folded_layers[sum_layer] = replace(
                        sum_layer,
                        weight=weight,
                        fold_idx=fold_idx,
                        weight_input=weight_input,
                        weight_softmax_dim=weight_softmax_dim,
                    )

        return [folded_layers.get(xl, xl) for xl in layers]

    @staticmethod
    def _lower_weight_parameter(
        weight: TorchParameter,
    ) -> tuple[TorchTensorParameter | None, int | None]:
        """Recognize the weight-graph fragment that can be evaluated inside XE.

        A tensor followed by one softmax can be split safely: torch supplies the raw
        tensor and XE performs the softmax with the correct possibly-factorized axis.
        Returning ``None`` for any richer graph is necessary to preserve correctness
        by evaluating that complete graph with the general torch fallback.
        """
        if len(weight.nodes) != 2:
            return None, None
        parameter_input, activation = weight.nodes
        if not isinstance(parameter_input, TorchTensorParameter) or not isinstance(
            activation, TorchSoftmaxParameter
        ):
            return None, None
        if weight.node_inputs(activation) != [parameter_input] or weight.outputs != [
            activation
        ]:
            return None, None
        return parameter_input, activation.dim
