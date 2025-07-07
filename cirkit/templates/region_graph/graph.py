import itertools
import json
from abc import ABC
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import cached_property
from typing import TypeAlias, TypedDict, cast, final

import numpy as np
from numpy.typing import NDArray

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, KroneckerLayer, Layer, SumLayer
from cirkit.symbolic.parameters import ParameterFactory
from cirkit.templates.utils import InputLayerFactory, ProductLayerFactory, SumLayerFactory
from cirkit.utils.algorithms import DiAcyclicGraph
from cirkit.utils.scope import Scope

RGNodeMetadata: TypeAlias = dict[str, int | float | str | bool]


class RegionDict(TypedDict):
    """The structure of a region node in the json file."""

    scope: list[int]  # The scope of this region node, specified by id of variable.


class PartitionDict(TypedDict):
    """The structure of a partition node in the json file."""

    inputs: list[int]  # The inputs of this partition node, specified by id of region node.
    output: int  # The output of this partition node, specified by id of region node.


class RegionGraphJson(TypedDict):
    """The structure of the region graph json file."""

    # The regions of RG represented by a mapping from id in str to either a dict or only the scope.
    regions: dict[str, RegionDict | list[int]]

    # The list of region node roots str ids in the RG
    roots: list[str]

    # The graph of RG represented by a list of partitions.
    graph: list[PartitionDict]


class RegionGraphNode(ABC):
    """The abstract base class for nodes in region graphs."""

    def __init__(self, scope: Iterable[int] | Scope) -> None:
        """Init class.

        Args:
            scope (Scope): The scope of this node.
        """
        if not scope:
            raise ValueError("The scope of a region graph node must not be empty.")
        super().__init__()
        self.scope = Scope(scope)

    def __repr__(self) -> str:
        """Generate the repr string of the node.

        Returns:
            str: The str representation of the node.
        """
        return f"{type(self).__name__}@0x{id(self):x}({self.scope})"


class RegionNode(RegionGraphNode):
    """The region node in the region graph."""


class PartitionNode(RegionGraphNode):
    """The partition node in the region graph."""


# We mark RG as final to hint that RG algorithms should not be its subclasses but factories, so that
# constructed RGs and loaded RGs are all of type RegionGraph.
@final
class RegionGraph(DiAcyclicGraph[RegionGraphNode]):
    def __init__(
        self,
        nodes: Sequence[RegionGraphNode],
        in_nodes: Mapping[RegionGraphNode, Sequence[RegionGraphNode]],
        outputs: Sequence[RegionGraphNode],
    ) -> None:
        super().__init__(nodes, in_nodes, outputs)
        self._check_structure()

    def _check_structure(self):
        for node, node_children in self.nodes_inputs.items():
            if isinstance(node, RegionNode):
                for ptn in node_children:
                    if not isinstance(ptn, PartitionNode):
                        raise ValueError(
                            f"Expected partition node as children of '{node}', "
                            f"but found '{ptn}'"
                        )
                    if ptn.scope != node.scope:
                        raise ValueError(
                            f"Expected partition node with scope '{node.scope}', "
                            f"but found '{ptn.scope}"
                        )
                continue
            if not isinstance(node, PartitionNode):
                raise ValueError(
                    f"Region graph nodes must be either partition nodes or region nodes, "
                    f"found '{type(node)}'"
                )
            scopes = []
            for rgn in node_children:
                if not isinstance(rgn, RegionNode):
                    raise ValueError(
                        f"Expected region node as children of '{node}', but found '{rgn}'"
                    )
                scopes.append(rgn.scope)
            scope = Scope.union(*scopes)
            if scope != node.scope or sum(len(sc) for sc in scopes) != len(scope):
                raise ValueError(
                    f"Expecte partitioning of scope '{node.scope}', but found '{scopes}'"
                )
        for ptn in self.partition_nodes:
            rgn_outs = self.node_outputs(ptn)
            if len(rgn_outs) != 1:
                raise ValueError(
                    f"Expected each partition node to have exactly one parent region node,"
                    f" but found {len(rgn_outs)} parent nodes"
                )

    def region_inputs(self, rgn: RegionNode) -> Sequence[PartitionNode]:
        return [cast(PartitionNode, node) for node in self.node_inputs(rgn)]

    def partition_inputs(self, ptn: PartitionNode) -> Sequence[RegionNode]:
        return [cast(RegionNode, node) for node in self.node_inputs(ptn)]

    def region_outputs(self, rgn: RegionNode) -> Sequence[PartitionNode]:
        return [cast(PartitionNode, node) for node in self.node_outputs(rgn)]

    def partition_outputs(self, ptn: PartitionNode) -> Sequence[RegionNode]:
        return [cast(RegionNode, node) for node in self.node_outputs(ptn)]

    @property
    def inputs(self) -> Iterator[RegionNode]:
        return (cast(RegionNode, node) for node in super().inputs)

    @property
    def outputs(self) -> Iterator[RegionNode]:
        return (cast(RegionNode, node) for node in super().outputs)

    @property
    def region_nodes(self) -> Iterator[RegionNode]:
        """Region nodes in the graph."""
        return (node for node in self.nodes if isinstance(node, RegionNode))

    @property
    def partition_nodes(self) -> Iterator[PartitionNode]:
        """Partition nodes in the graph, which are always inner nodes."""
        return (node for node in self.nodes if isinstance(node, PartitionNode))

    @property
    def inner_nodes(self) -> Iterator[RegionGraphNode]:
        """Inner (non-input) nodes in the graph."""
        return (node for node in self.nodes if self.node_inputs(node))

    @property
    def inner_region_nodes(self) -> Iterator[RegionNode]:
        """Inner region nodes in the graph."""
        return (
            node for node in self.region_nodes if self.node_inputs(node) and self.node_outputs(node)
        )

    @cached_property
    def scope(self) -> Scope:
        return Scope.union(*(node.scope for node in self.outputs))

    @cached_property
    def num_variables(self) -> int:
        return len(self.scope)

    @cached_property
    def is_structured_decomposable(self) -> bool:
        is_structured_decomposable = True
        decompositions: dict[Scope, tuple[Scope, ...]] = {}
        for partition in self.partition_nodes:
            # The scopes are sorted by _sort_nodes(), so the tuple has a deterministic order.
            decomp = tuple(region.scope for region in self.node_inputs(partition))
            if partition.scope not in decompositions:
                decompositions[partition.scope] = decomp
            is_structured_decomposable &= decomp == decompositions[partition.scope]
        return is_structured_decomposable

    @cached_property
    def is_omni_compatible(self) -> bool:
        return all(
            len(region.scope) == 1
            for partition in self.partition_nodes
            for region in self.node_inputs(partition)
        )

    def is_compatible(self, other: "RegionGraph", /, *, scope: Iterable[int] | None = None) -> bool:
        """Test compatibility with another region graph over the given scope.

        Args:
            other (RegionGraph): The other region graph to compare with.
            scope (Optional[Iterable[int]], optional): The scope over which to check. If None, \
                will use the intersection of the scopes of the two RG. Defaults to None.

        Returns:
            bool: Whether self is compatible with other.
        """
        # _is_frozen is implicitly tested because is_smooth is set in freeze().
        scope = Scope(scope) if scope is not None else self.scope & other.scope

        # TODO: is this correct for more-than-2 partition?
        for partition1, partition2 in itertools.product(
            self.partition_nodes, other.partition_nodes
        ):
            if partition1.scope & scope != partition2.scope & scope:
                continue  # Only check partitions with the same scope.

            partition1_inputs = self.node_inputs(partition1)
            partition2_inputs = self.node_inputs(partition2)

            if any(partition1.scope <= input.scope for input in partition2_inputs) or any(
                partition2.scope <= input.scope for input in partition1_inputs
            ):
                continue  # Only check partitions not within another partition.

            adj_mat = np.zeros((len(partition1_inputs), len(partition2_inputs)), dtype=np.bool_)
            for (i, region1), (j, region2) in itertools.product(
                enumerate(partition1_inputs), enumerate(partition2_inputs)
            ):
                # I.e., if scopes intersect over the scope to test.
                adj_mat[i, j] = bool(region1.scope & region2.scope & scope)
            adj_mat = adj_mat @ adj_mat.T
            # Now we have adjencency from inputs1 (of self) to inputs1. An edge means the two
            # regions must be partitioned together.

            # The number of zero eigen values for the laplacian matrix is the number of connected
            # components in a graph.
            # ANNOTATE: Numpy has typing issues.
            # CAST: Numpy has typing issues.
            deg_mat = np.diag(cast(NDArray[np.int64], adj_mat.sum(axis=1)))
            laplacian: NDArray[np.int64] = deg_mat - adj_mat
            eigen_values = np.linalg.eigvals(laplacian)
            num_connected: int = cast(NDArray[np.int64], np.isclose(eigen_values, 0).sum()).item()
            if num_connected == 1:  # All connected, meaning there's no valid partitioning.
                return False

        return True

    ####################################    (De)Serialization    ###################################
    # The RG can be dumped and loaded from json files, which can be useful when we want to save and
    # share it. The load() is another way to construct a RG other than the RG algorithms.

    def dump(self, filename: str) -> None:
        """Dump the region graph to the json file.

        The file will be opened with mode="w" and encoding="utf-8".

        Args:
            filename (str): The file name for dumping.
        """
        # NOTE: Below we don't assume the existence of RGNode.metadata["sort_key"], and try to
        #       preserve the ordering by file structure. However, the sort_key will be saved when
        #       available and with_meta enabled.

        # ANNOTATE: Specify content for empty container.
        rg_json: RegionGraphJson = {}
        region_idx: dict[RegionNode, int] = {
            node: idx for idx, node in enumerate(self.region_nodes)
        }

        # Store the region nodes information
        rg_json["regions"] = {str(idx): list(node.scope) for node, idx in region_idx.items()}

        # Store the roots information
        rg_json["roots"] = [str(region_idx[rgn]) for rgn in self.outputs]

        # Store the partition nodes information
        rg_json["graph"] = []
        for partition in self.partition_nodes:
            input_idxs = [region_idx[cast(RegionNode, rgn)] for rgn in self.node_inputs(partition)]
            # partition.outputs is guaranteed to have len==1 by _validate().
            output_idx = region_idx[cast(RegionNode, self.node_outputs(partition)[0])]
            rg_json["graph"].append({"output": output_idx, "inputs": input_idxs})

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(rg_json, f, indent=4)

    @staticmethod
    def load(filename: str) -> "RegionGraph":
        """Load the region graph from the json file.

        The file will be opened with mode="r" and encoding="utf-8".

        Metadata is always loaded when available.

        Args:
            filename (str): The file name for loading.

        Returns:
            RegionGraph: The loaded region graph.
        """
        # NOTE: Below we don't assume the existence of RGNode.metadata["sort_key"], and try to
        #       recover the ordering from file structure. However, the sort_key will be used when
        #       available.

        with open(filename, encoding="utf-8") as f:
            # ANNOTATE: json.load gives Any.
            rg_json: RegionGraphJson = json.load(f)

        nodes: list[RegionGraphNode] = []
        in_nodes: dict[RegionGraphNode, list[RegionGraphNode]] = defaultdict(list)
        outputs = []
        region_idx: dict[int, RegionNode] = {}

        # Load the region nodes
        for idx, rgn_scope in rg_json["regions"].items():
            rgn = RegionNode(rgn_scope)
            nodes.append(rgn)
            region_idx[int(idx)] = rgn

        # Load the root region nodes
        for idx in rg_json["roots"]:
            outputs.append(region_idx[int(idx)])

        # Load the partition nodes
        for partitioning in rg_json["graph"]:
            in_rgns = [region_idx[int(idx)] for idx in partitioning["inputs"]]
            out_rgn = region_idx[partitioning["output"]]
            ptn = PartitionNode(out_rgn.scope)
            nodes.append(ptn)
            in_nodes[out_rgn].append(ptn)
            in_nodes[ptn] = in_rgns

        return RegionGraph(nodes, in_nodes, outputs=outputs)

    # TODO: refactor the following method as to simplify its structure (e.g., remove inline methods)
    def build_circuit(
        self,
        *,
        input_factory: InputLayerFactory,
        sum_product: str | None = None,
        sum_weight_factory: ParameterFactory | None = None,
        nary_sum_weight_factory: ParameterFactory | None = None,
        sum_factory: SumLayerFactory | None = None,
        prod_factory: ProductLayerFactory | None = None,
        num_input_units: int = 1,
        num_sum_units: int = 1,
        num_classes: int = 1,
        factorize_multivariate: bool = True,
    ) -> Circuit:
        """Construct a symbolic circuit from a region graph.
            There are two ways to use this method. The first one is to specify a sum-product layer
            abstraction, which can be either 'cp' (the CP layer), 'cp-t' (the CP-transposed layer),
            or 'tucker' (the Tucker layer). The second one is to manually specify the factories to
            build distinct um and product layers. If the first way is chosen, then one can possibly
            use a factory that builds the symbolic parameters of the sum-product layer abstractions.
            The factory that constructs the input factory must always be specified.

        Args:
            input_factory: A factory that builds an input layer.
            sum_product: The sum-product layer to use. It can be None, 'cp', 'cp-t', or 'tucker'.
            sum_weight_factory: The factory to construct the weights of the sum layers.
                It can be None, or a parameter factory, i.e., a map
                from a shape to a symbolic parameter. If it is None, then the default
                weight factory of the sum layer is used instead.
            nary_sum_weight_factory: The factory to construct the weight of sum layers havity arity
                greater than one. If it is None, then it will have the same value and semantics of
                the given sum_weight_factory.
            sum_factory: A factory that builds a sum layer. It can be None.
            prod_factory: A factory that builds a product layer. It can be None.
            num_input_units: The number of input units.
            num_sum_units: The number of sum units per sum layer.
            num_classes: The number of output classes.
            factorize_multivariate: Whether to fully factorize input layers, when they depend on
                more than one variable.

        Returns:
            Circuit: A symbolic circuit.

        Raises:
            NotImplementedError: If an unknown 'sum_product' is given.
            ValueError: If both 'sum_product' and layer factories are specified, or none of them.
            ValueError: If 'sum_product' is specified, but 'weight_factory' is not.
            ValueError: The given region graph is malformed.
        """
        if (sum_factory is None and prod_factory is not None) or (
            sum_factory is not None and prod_factory is None
        ):
            raise ValueError(
                "Both 'sum_factory' and 'prod_factory' must be specified or none of them"
            )
        if sum_product is None and (sum_factory is None or prod_factory is None):
            raise ValueError(
                "If 'sum_product' is not given, then both 'sum_factory' and 'prod_factory'"
                " must be specified"
            )
        if sum_product is not None and (sum_factory is not None or prod_factory is not None):
            raise ValueError(
                "At most one between 'sum_product' and the pair 'sum_factory' and 'prod_factory'"
                " must be specified"
            )
        if nary_sum_weight_factory is None:
            nary_sum_weight_factory = sum_weight_factory

        layers: list[Layer] = []
        in_layers: dict[Layer, list[Layer]] = {}
        node_to_layer: dict[RegionGraphNode, Layer] = {}

        def build_cp_(
            rgn: RegionNode, rgn_partitioning: Sequence[RegionNode]
        ) -> HadamardLayer | SumLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            denses = [
                SumLayer(
                    node_to_layer[rgn_in].num_output_units,
                    num_sum_units,
                    weight_factory=sum_weight_factory,
                )
                for rgn_in in rgn_partitioning
            ]
            hadamard = HadamardLayer(num_sum_units, arity=len(rgn_partitioning))
            layers.extend(denses)
            layers.append(hadamard)
            in_layers[hadamard] = denses
            for d, li in zip(denses, layer_ins):
                in_layers[d] = [li]
            # If the region is not a root region of the region graph,
            # then make Hadamard the last layer
            if self.region_outputs(rgn):
                node_to_layer[rgn] = hadamard
                return hadamard
            # Otherwise, introduce an additional sum layer to ensure the output layer is a sum
            output_dense = SumLayer(
                hadamard.num_output_units, num_classes, weight_factory=sum_weight_factory
            )
            layers.append(output_dense)
            in_layers[output_dense] = [hadamard]
            node_to_layer[rgn] = output_dense
            return output_dense

        def build_cp_transposed_(
            rgn: RegionNode, rgn_partitioning: Sequence[RegionNode]
        ) -> SumLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            num_in_units = list({li.num_output_units for li in layer_ins})
            if len(num_in_units) > 1:
                raise ValueError(
                    "Cannot build a CP transposed layer, as the inputs would have different units"
                )
            num_units = num_sum_units if self.region_outputs(rgn) else num_classes
            hadamard = HadamardLayer(num_in_units[0], arity=len(rgn_partitioning))
            dense = SumLayer(num_in_units[0], num_units, weight_factory=sum_weight_factory)
            layers.append(hadamard)
            layers.append(dense)
            in_layers[hadamard] = layer_ins
            in_layers[dense] = [hadamard]
            node_to_layer[rgn] = dense
            return dense

        def build_tucker_(rgn: RegionNode, rgn_partitioning: Sequence[RegionNode]) -> SumLayer:
            layer_ins = [node_to_layer[rgn_in] for rgn_in in rgn_partitioning]
            num_in_units = list({li.num_output_units for li in layer_ins})
            if len(num_in_units) > 1:
                raise ValueError(
                    "Cannot build a Tucker layer, as the inputs would have different units"
                )
            num_units = num_sum_units if self.region_outputs(rgn) else num_classes
            kronecker = KroneckerLayer(num_in_units[0], arity=len(rgn_partitioning))
            dense = SumLayer(
                kronecker.num_output_units,
                num_units,
                weight_factory=sum_weight_factory,
            )
            layers.append(kronecker)
            layers.append(dense)
            in_layers[kronecker] = layer_ins
            in_layers[dense] = [kronecker]
            node_to_layer[rgn] = dense
            return dense

        # Set the sum-product layer builder, if necessary
        sum_prod_builder_: Callable[[RegionNode, Sequence[RegionNode]], Layer] | None
        if sum_product is None:
            sum_prod_builder_ = None
        elif sum_product == "cp":
            sum_prod_builder_ = build_cp_
        elif sum_product == "cp-t":
            sum_prod_builder_ = build_cp_transposed_
        elif sum_product == "tucker":
            sum_prod_builder_ = build_tucker_
        else:
            raise NotImplementedError(f"Unknown sum-product layer abstraction called {sum_product}")

        # Loop through the region graph nodes, which are already sorted in a topological ordering
        for node in self.topological_ordering():
            if isinstance(node, PartitionNode):  # Partition node
                # If a sum-product layer abstraction has been specified,
                # then just skip partition nodes
                if sum_prod_builder_ is not None:
                    continue
                assert prod_factory is not None
                partition_inputs = self.partition_inputs(node)
                prod_inputs = [node_to_layer[rgn] for rgn in partition_inputs]
                prod_sl = prod_factory(num_sum_units, len(prod_inputs))
                layers.append(prod_sl)
                in_layers[prod_sl] = prod_inputs
                node_to_layer[node] = prod_sl
            assert isinstance(
                node, RegionNode
            ), "Region graph nodes must be either region or partition nodes"
            region_inputs = self.region_inputs(node)
            region_outputs = self.region_outputs(node)
            if not region_inputs:
                # Input region node
                if factorize_multivariate and len(node.scope) > 1:
                    factorized_input_sls = [
                        input_factory(Scope([sc]), num_input_units) for sc in node.scope
                    ]
                    input_sl = HadamardLayer(num_input_units, arity=len(factorized_input_sls))
                    layers.extend(factorized_input_sls)
                    in_layers[input_sl] = factorized_input_sls
                else:
                    input_sl = input_factory(node.scope, num_input_units)
                num_units = num_sum_units if self.region_outputs(node) else num_classes
                if sum_factory is None:
                    layers.append(input_sl)
                    node_to_layer[node] = input_sl
                    continue
                sum_sl = sum_factory(num_input_units, num_units)
                layers.append(input_sl)
                layers.append(sum_sl)
                in_layers[sum_sl] = [input_sl]
                node_to_layer[node] = sum_sl
            elif len(region_inputs) == 1:
                # Region node that is partitioned into one and only one way
                (ptn,) = region_inputs
                if sum_prod_builder_ is not None:
                    sum_prod_builder_(node, self.partition_inputs(ptn))
                    continue
                num_units = num_sum_units if region_outputs else num_classes
                sum_input = node_to_layer[ptn]
                sum_sl = sum_factory(sum_input.num_output_units, num_units)
                layers.append(sum_sl)
                in_layers[sum_sl] = [sum_input]
                node_to_layer[node] = sum_sl
            else:  # len(node_inputs) > 1:
                # Region node with multiple partitionings
                num_units = num_sum_units if region_outputs else num_classes
                if sum_prod_builder_ is None:
                    sum_ins = [node_to_layer[ptn] for ptn in region_inputs]
                    mix_ins = [sum_factory(sli.num_output_units, num_units) for sli in sum_ins]
                    layers.extend(mix_ins)
                    for mix_sl, sli in zip(mix_ins, sum_ins):
                        in_layers[mix_sl] = [sli]
                else:
                    mix_ins = [
                        sum_prod_builder_(node, self.partition_inputs(ptn)) for ptn in region_inputs
                    ]
                mix_sl = SumLayer(
                    num_units,
                    num_units,
                    arity=len(mix_ins),
                    weight_factory=nary_sum_weight_factory,
                )
                layers.append(mix_sl)
                in_layers[mix_sl] = mix_ins
                node_to_layer[node] = mix_sl

        outputs = [node_to_layer[rgn] for rgn in self.outputs]
        return Circuit(layers, in_layers, outputs)
