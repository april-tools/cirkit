from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import DirichletInitializer
from cirkit.symbolic.layers import CategoricalLayer, MixingLayer, DenseLayer, HadamardLayer
from cirkit.symbolic.parameters import Parameter, TensorParameter
from cirkit.templates.region_graph import QuadGraph
from cirkit.utils.scope import Scope
from cirkit.pipeline import PipelineContext
from cirkit.backend.torch.layers import TorchCategoricalLayer, TorchDenseLayer, TorchHadamardLayer, TorchMixingLayer


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
    for n1, n2 in zip(nodes_sc[:9], nodes_c[:9]):
        assert isinstance(n1, CategoricalLayer)
        assert isinstance(n2, TorchCategoricalLayer) and n2.probs._nodes[0].shape == (8, 1, 2)
    # after the 9 input layers no more than 12 dense layers must follow
    counter = 0
    for n1, n2 in zip(nodes_sc[9:], nodes_c[9:]):
        if not isinstance(n1, DenseLayer):
            break
        counter += 1
        assert n2.weight._nodes[0]._ptensor.shape == (1, 8, 8)
    assert counter <= 12
