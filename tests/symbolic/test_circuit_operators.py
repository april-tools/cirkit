# import cirkit.symbolic.functional as SF
# from cirkit.symbolic.layers import DenseLayer, HadamardLayer, LogPartitionLayer
# from cirkit.symbolic.parameters import ConjugateParameter, KroneckerParameter, ReferenceParameter
# from tests.symbolic.test_utils import build_simple_circuit
#
#
# def test_integrate_circuit():
#     sc = build_simple_circuit(12, 4, 2, num_repetitions=3)
#     int_sc = SF.integrate(sc)
#     assert len(list(int_sc.inputs)) == len(list(sc.inputs))
#     assert all(isinstance(isl, LogPartitionLayer) for isl in int_sc.inputs)
#     assert len(list(int_sc.inner_layers)) == len(list(sc.inner_layers))
#     sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), int_sc.inner_layers))
#     assert all(isinstance(l.weight.output, ReferenceParameter) for l in sum_layers)
#
#
# def test_multiply_circuits():
#     sc1 = build_simple_circuit(12, 4, 2)
#     sc2 = build_simple_circuit(12, 3, 5)
#     prod_sc = SF.multiply(sc1, sc2)
#     assert len(list(prod_sc.inputs)) == len(list(sc1.inputs))
#     assert len(list(prod_sc.inputs)) == len(list(sc2.inputs))
#     assert len(list(prod_sc.inner_layers)) == len(list(sc1.inner_layers))
#     assert len(list(prod_sc.inner_layers)) == len(list(sc2.inner_layers))
#     sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), prod_sc.inner_layers))
#     assert all(isinstance(l.weight.output, KroneckerParameter) for l in sum_layers)
#     prod_layers = list(filter(lambda l: not isinstance(l, DenseLayer), prod_sc.inner_layers))
#     assert all(isinstance(l, HadamardLayer) for l in prod_layers)
#
#
# def test_multiply_integrate_circuits():
#     sc1 = build_simple_circuit(12, 4, 2)
#     sc2 = build_simple_circuit(12, 3, 5)
#     prod_sc = SF.multiply(sc1, sc2)
#     int_sc = SF.integrate(prod_sc)
#     assert len(list(int_sc.inputs)) == len(list(prod_sc.inputs))
#     assert all(isinstance(isl, LogPartitionLayer) for isl in int_sc.inputs)
#     assert len(list(int_sc.inner_layers)) == len(list(prod_sc.inner_layers))
#     sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), int_sc.inner_layers))
#     assert all(isinstance(l.weight.output, KroneckerParameter) for l in sum_layers)
#
#
# def test_conjugate_circuit():
#     sc = build_simple_circuit(12, 4, 2, num_repetitions=3)
#     conj_sc = SF.conjugate(sc)
#     assert len(list(conj_sc.inputs)) == len(list(sc.inputs))
#     assert len(list(conj_sc.inner_layers)) == len(list(sc.inner_layers))
#     assert list(map(type, conj_sc.topological_ordering())) == list(
#         map(type, sc.topological_ordering())
#     )
#     sum_layers = list(filter(lambda l: isinstance(l, DenseLayer), conj_sc.inner_layers))
#     assert all(isinstance(l.weight.output, ConjugateParameter) for l in sum_layers)
