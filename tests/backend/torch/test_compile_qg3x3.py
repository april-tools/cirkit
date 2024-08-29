from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import DirichletInitializer
from cirkit.symbolic.layers import CategoricalLayer, MixingLayer, DenseLayer, HadamardLayer
from cirkit.symbolic.parameters import Parameter, TensorParameter
from cirkit.templates.region_graph import QuadGraph
from cirkit.utils.scope import Scope
from cirkit.pipeline import PipelineContext
from cirkit.backend.torch.layers import TorchCategoricalLayer, TorchDenseLayer, TorchHadamardLayer, TorchMixingLayer
from collections import Counter


def categorical_layer_factory(
    scope: Scope, num_units: int, num_channels: int, *, num_categories: int = 2
) -> CategoricalLayer:
    return CategoricalLayer(
        scope,
        num_units,
        num_channels,
        num_categories=num_categories,
        probs=Parameter.from_leaf(
            TensorParameter(
                num_units, num_channels, num_categories, initializer=DirichletInitializer()
            )
        ),
    )


def mixing_layer_factory(scope: Scope, num_units: int, arity: int) -> MixingLayer:
    return MixingLayer(
        scope,
        num_units,
        arity,
        weight=Parameter.from_leaf(
            TensorParameter(num_units, arity, initializer=DirichletInitializer())
        ),
    )


def test_build_circuit_qg_cp():
    rg = QuadGraph((3, 3))
    sc = Circuit.from_region_graph(
        rg,
        num_input_units=8,
        num_sum_units=8,
        sum_product="cp",
        input_factory=categorical_layer_factory,
        mixing_factory=mixing_layer_factory,
    )
    assert sc.is_smooth
    assert sc.is_decomposable
    ctx = PipelineContext(
        backend='torch',
        optimize=False,
        fold=False,
        semiring='lse-sum'
    )

    circuit = ctx.compile(sc)
    circuit_pf = ctx.integrate(circuit)
    nodes_sc = list(sc.topological_ordering())
    nodes_c = list(circuit.topological_ordering())

    # check numbers of nodes by type
    assert len(nodes_sc) == len(nodes_c) == 53
    for n1, n2 in zip(nodes_sc[:9], nodes_c[:9]):
        assert isinstance(n1, CategoricalLayer) and isinstance(n2, TorchCategoricalLayer) or \
               isinstance(n1, DenseLayer) and isinstance(n2, TorchDenseLayer) or \
               isinstance(n1, HadamardLayer) and isinstance(n2, TorchHadamardLayer) or \
               isinstance(n1, MixingLayer) and isinstance(n2, TorchMixingLayer)

    assert sum([1 for n1 in nodes_sc if isinstance(n1, CategoricalLayer)]) == \
           sum([1 for n2 in nodes_c if isinstance(n2, TorchCategoricalLayer)]) == 9
    assert sum([1 for n1 in nodes_sc if isinstance(n1, DenseLayer)]) == \
           sum([1 for n2 in nodes_c if isinstance(n2, TorchDenseLayer)]) == 28
    assert sum([1 for n1 in nodes_sc if isinstance(n1, HadamardLayer)]) == \
           sum([1 for n2 in nodes_c if isinstance(n2, TorchHadamardLayer)]) == 14
    assert sum([1 for n1 in nodes_sc if isinstance(n1, MixingLayer)]) == \
           sum([1 for n2 in nodes_c if isinstance(n2, TorchMixingLayer)]) == 2

    # check all input layers
    input_scopes = set([(i, ) for i in range(9)])
    scopes = set()
    for n1, n2 in zip(nodes_sc[:9], nodes_c[:9]):
        assert isinstance(n1, CategoricalLayer)
        assert isinstance(n2, TorchCategoricalLayer) and n2.probs._nodes[0].shape == (8, 1, 2)
        scopes.add(tuple(n1.scope))
    assert input_scopes == scopes

    # after the 9 input layers, 14 dense layers must follow, whose scopes are defined below
    dense_scopes = Counter([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8])
    scopes = Counter()
    for n1, n2 in zip(nodes_sc[9:], nodes_c[9:]):
        if not isinstance(n1, DenseLayer):
            break
        assert n2.weight._nodes[0]._ptensor.shape == (1, 8, 8)
        assert len(n1.scope) == 1
        scopes[tuple(n1.scope)[0]] += 1
    assert dense_scopes == scopes

    # after the first 9+14=23 layers, 6 hadamard layers must follow, whose scopes are defined below
    hadamard_scopes = Counter([(0, 1), (3, 4), (0, 3), (1, 4), (6, 7), (2, 5)])
    scopes = Counter()
    for n1, n2 in zip(nodes_sc[23:], nodes_c[23:]):
        if not isinstance(n1, HadamardLayer):
            break
        scopes[tuple(sorted(tuple(n1.scope)))] += 1
    assert hadamard_scopes == scopes

    # after the first 23+6=29 layers, 8 dense layers must follow, whose scopes are defined below
    dense_scopes = Counter([(0, 1), (3, 4), (0, 3), (1, 4), (2, 5), (2, 5), (6, 7), (6, 7)])
    scopes = Counter()
    for n1, n2 in zip(nodes_sc[29:], nodes_c[29:]):
        if not isinstance(n1, DenseLayer):
            break
        assert n2.weight._nodes[0]._ptensor.shape == (1, 8, 8)
        scopes[tuple(sorted(tuple(n1.scope)))] += 1
    assert dense_scopes == scopes

    # after the first 29+8=37 layers, 4 hadamard layers must follow, whose scopes are defined below
    hadamard_scopes = Counter([(0, 1, 3, 4), (0, 1, 3, 4), (6, 7, 8), (2, 5, 8)])
    scopes = Counter()
    for n1, n2 in zip(nodes_sc[37:], nodes_c[37:]):
        if not isinstance(n1, HadamardLayer):
            break
        scopes[tuple(sorted(tuple(n1.scope)))] += 1
    assert hadamard_scopes == scopes

    # after the first 37+4=41 layers, 1 mixing layer must follow, whose scope is defined below
    mixing_scope = (0, 1, 3, 4)
    assert mixing_scope == tuple(sorted(tuple(nodes_sc[41].scope)))
    assert nodes_c[41].weight._nodes[0]._ptensor.shape == (1, 8, 2)

    # after the first 41+1=42 layers, 4 dense layers must follow, whose scopes are defined below
    # this time though, 2 layers are KxK and 2 1xK
    dense_scopes = Counter([(0, 1, 3, 4), (0, 1, 3, 4), (6, 7, 8), (2, 5, 8)])
    scopes = Counter()
    for n1, n2 in zip(nodes_sc[42:], nodes_c[42:]):
        if not isinstance(n1, DenseLayer):
            break
        if tuple(sorted(tuple(n1.scope))) == (0, 1, 3, 4):
            assert n2.weight._nodes[0]._ptensor.shape == (1, 8, 8)
        elif tuple(sorted(tuple(n1.scope))) in [(6, 7, 8), (2, 5, 8)]:
            assert n2.weight._nodes[0]._ptensor.shape == (1, 1, 8)
        scopes[tuple(sorted(tuple(n1.scope)))] += 1
    assert dense_scopes == scopes

    # after the first 42+4=46 layers, 2 hadamard layers must follow, whose scopes are defined below
    hadamard_scopes = Counter([(0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 6, 7)])
    scopes = Counter()
    for n1, n2 in zip(nodes_sc[46:], nodes_c[46:]):
        if not isinstance(n1, HadamardLayer):
            break
        scopes[tuple(sorted(tuple(n1.scope)))] += 1
    assert hadamard_scopes == scopes

    # after the first 46+2=48 layers, 2 dense layers must follow, whose scopes are defined below
    dense_scopes = Counter([(0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 6, 7)])
    scopes = Counter()
    for n1, n2 in zip(nodes_sc[48:], nodes_c[48:]):
        if not isinstance(n1, DenseLayer):
            break
        assert n2.weight._nodes[0]._ptensor.shape == (1, 1, 8)
        scopes[tuple(sorted(tuple(n1.scope)))] += 1
    assert dense_scopes == scopes

    # after the first 48+2=50 layers, 2 hadamard layers must follow, whose scopes are defined below
    hadamard_scopes = Counter([(0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4, 5, 6, 7, 8)])
    scopes = Counter()
    for n1, n2 in zip(nodes_sc[50:], nodes_c[50:]):
        if not isinstance(n1, HadamardLayer):
            break
        scopes[tuple(sorted(tuple(n1.scope)))] += 1
    assert hadamard_scopes == scopes

    # finally, the circuit ends with a mixing layer
    assert tuple(nodes_sc[-1].scope) == tuple(range(9))
    assert nodes_c[-1].weight._nodes[0]._ptensor.shape == (1, 1, 2)
