import itertools

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import (
    ConstantTensorInitializer,
    DirichletInitializer,
    NormalInitializer,
    UniformInitializer,
)
from cirkit.symbolic.layers import (
    CategoricalLayer,
    EmbeddingLayer,
    GaussianLayer,
    HadamardLayer,
    Layer,
    PolynomialLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import (
    ExpParameter,
    LogSoftmaxParameter,
    Parameter,
    SoftmaxParameter,
    TensorParameter,
)
from cirkit.utils.scope import Scope


def build_bivariate_monotonic_structured_cpt_pc(
    *,
    num_units: int = 2,
    input_layer: str = "bernoulli",
    parameterize: bool = True,
    normalized: bool = True,
):
    # Build input layers
    if input_layer == "bernoulli":
        if parameterize:
            if normalized:
                logits_factory = lambda shape: Parameter.from_unary(
                    LogSoftmaxParameter(shape),
                    TensorParameter(*shape, initializer=NormalInitializer()),
                )
                probs_factory = None
            else:
                logits_factory = lambda shape: Parameter.from_input(
                    TensorParameter(*shape, initializer=NormalInitializer())
                )
                probs_factory = None
        else:
            probs_factory = lambda shape: Parameter.from_input(
                TensorParameter(*shape, initializer=DirichletInitializer())
            )
            logits_factory = None
        input_layers = {
            (vid,): CategoricalLayer(
                Scope([vid]),
                num_output_units=num_units,
                num_channels=1,
                num_categories=2,
                logits_factory=logits_factory,
                probs_factory=probs_factory,
            )
            for vid in range(2)
        }
    elif input_layer == "gaussian":
        if parameterize:
            stddev_factory = None
        else:
            stddev_factory = lambda shape: Parameter.from_input(
                TensorParameter(*shape, initializer=UniformInitializer())
            )
        input_layers = {
            (vid,): GaussianLayer(
                Scope([vid]),
                num_output_units=num_units,
                num_channels=1,
                stddev_factory=stddev_factory,
            )
            for vid in range(2)
        }
    else:
        raise NotImplementedError()

    # Build dense layers
    if parameterize:
        if normalized:
            dense_weight_factory = lambda shape: Parameter.from_unary(
                SoftmaxParameter(shape, axis=1),
                TensorParameter(*shape, initializer=NormalInitializer()),
            )
        else:
            dense_weight_factory = lambda shape: Parameter.from_unary(
                ExpParameter(shape), TensorParameter(*shape, initializer=NormalInitializer())
            )
    else:
        dense_weight_factory = lambda shape: Parameter.from_input(
            TensorParameter(*shape, initializer=DirichletInitializer())
        )
    dense_layers = {
        scope: SumLayer(
            num_input_units=num_units,
            num_output_units=1 if len(scope) == 2 else num_units,
            weight_factory=dense_weight_factory,
        )
        for scope in [(0,), (1,), (0, 1)]
    }

    # Build hadamard product layer
    product_layer = HadamardLayer(num_input_units=num_units, arity=2)

    # Set the connections between layers
    in_layers: dict[Layer, list[Layer]] = {
        dense_layers[(0,)]: [input_layers[(0,)]],
        dense_layers[(1,)]: [input_layers[(1,)]],
        product_layer: [dense_layers[(0,)], dense_layers[(1,)]],
        dense_layers[(0, 1)]: [product_layer],
    }

    # Build the symbolic circuit
    circuit = Circuit(
        num_channels=1,
        layers=list(itertools.chain(input_layers.values(), [product_layer], dense_layers.values())),
        in_layers=in_layers,
        outputs=[dense_layers[(0, 1)]],
    )

    assert circuit.is_smooth
    assert circuit.is_decomposable
    assert circuit.is_structured_decomposable
    assert circuit.is_omni_compatible
    return circuit


def build_multivariate_monotonic_structured_cpt_pc(
    *,
    num_units: int = 2,
    input_layer: str = "bernoulli",
    parameterize: bool = True,
    normalized: bool = True,
):
    # Build input layers
    if input_layer == "bernoulli":
        if parameterize:
            if normalized:
                logits_factory = lambda shape: Parameter.from_unary(
                    LogSoftmaxParameter(shape),
                    TensorParameter(*shape, initializer=NormalInitializer()),
                )
                probs_factory = None
            else:
                logits_factory = lambda shape: Parameter.from_input(
                    TensorParameter(*shape, initializer=NormalInitializer())
                )
                probs_factory = None
        else:
            probs_factory = lambda shape: Parameter.from_input(
                TensorParameter(*shape, initializer=DirichletInitializer())
            )
            logits_factory = None
        input_layers = {
            (vid,): CategoricalLayer(
                Scope([vid]),
                num_output_units=num_units,
                num_channels=1,
                num_categories=2,
                logits_factory=logits_factory,
                probs_factory=probs_factory,
            )
            for vid in range(5)
        }
    elif input_layer == "embedding":
        input_layers = {
            (vid,): EmbeddingLayer(
                Scope([vid]),
                num_output_units=num_units,
                num_channels=1,
                num_states=2,
            )
            for vid in range(5)
        }
    elif input_layer == "gaussian":
        if parameterize:
            stddev_factory = None
        else:
            stddev_factory = lambda shape: Parameter.from_input(
                TensorParameter(*shape, initializer=UniformInitializer())
            )
        input_layers = {
            (vid,): GaussianLayer(
                Scope([vid]),
                num_output_units=num_units,
                num_channels=1,
                stddev_factory=stddev_factory,
            )
            for vid in range(5)
        }
    elif input_layer == "polynomial":
        if parameterize:
            coeff_factory = None
        else:
            coeff_factory = lambda shape: Parameter.from_input(
                TensorParameter(*shape, initializer=UniformInitializer())
            )
        input_layers = {
            (vid,): PolynomialLayer(
                Scope([vid]),
                num_output_units=num_units,
                num_channels=1,
                degree=2,  # TODO: currently hard-coded
                coeff_factory=coeff_factory,
            )
            for vid in range(5)
        }
    else:
        raise NotImplementedError()

    # Build dense layers
    if parameterize:
        if normalized:
            dense_weight_factory = lambda shape: Parameter.from_unary(
                SoftmaxParameter(shape, axis=1),
                TensorParameter(*shape, initializer=NormalInitializer()),
            )
        else:
            dense_weight_factory = lambda shape: Parameter.from_unary(
                ExpParameter(shape), TensorParameter(*shape, initializer=NormalInitializer())
            )
    else:
        dense_weight_factory = lambda shape: Parameter.from_input(
            TensorParameter(*shape, initializer=DirichletInitializer())
        )
    dense_layers = {
        scope: SumLayer(
            num_input_units=num_units,
            num_output_units=1 if len(scope) == 5 else num_units,
            weight_factory=dense_weight_factory,
        )
        for scope in [(0, 2), (1, 3), (0, 1, 2, 3), (0, 1, 2, 3, 4)]
    }

    # Build hadamard product layers
    product_layers = {
        scope: HadamardLayer(num_input_units=num_units, arity=2) for scope in dense_layers
    }

    # Set the connections between layers
    in_layers: dict[Layer, list[Layer]] = {
        dense_layer: [product_layers[scope]] for scope, dense_layer in dense_layers.items()
    }
    in_layers.update(
        {
            product_layers[(0, 2)]: [input_layers[(0,)], input_layers[(2,)]],
            product_layers[(1, 3)]: [input_layers[(1,)], input_layers[(3,)]],
            product_layers[(0, 1, 2, 3)]: [dense_layers[(0, 2)], dense_layers[(1, 3)]],
            product_layers[(0, 1, 2, 3, 4)]: [dense_layers[(0, 1, 2, 3)], input_layers[(4,)]],
        }
    )

    # Build the symbolic circuit
    circuit = Circuit(
        num_channels=1,
        layers=list(
            itertools.chain(input_layers.values(), product_layers.values(), dense_layers.values())
        ),
        in_layers=in_layers,
        outputs=[dense_layers[(0, 1, 2, 3, 4)]],
    )

    assert circuit.is_smooth
    assert circuit.is_decomposable
    assert circuit.is_structured_decomposable
    assert not circuit.is_omni_compatible
    return circuit


def build_monotonic_structured_categorical_cpt_pc(
    return_ground_truth: bool = False,
) -> Circuit | tuple[Circuit, dict[str, dict[tuple[int, ...], float]], float]:
    # The probabilities of Bernoulli layers
    bernoulli_probs: dict[tuple[int, ...], np.ndarray] = {
        (0,): np.array([[[0.5, 0.5]], [[0.4, 0.6]]]),
        (1,): np.array([[[0.2, 0.8]], [[0.3, 0.7]]]),
        (2,): np.array([[[0.3, 0.7]], [[0.1, 0.9]]]),
        (3,): np.array([[[0.5, 0.5]], [[0.5, 0.5]]]),
        (4,): np.array([[[0.1, 0.9]], [[0.8, 0.2]]]),
    }

    # The parameters of dense weights
    dense_weights: dict[tuple[int, ...], np.ndarray] = {
        (0, 2): np.array([[1.0, 1.0], [3.0, 2.0]]),
        (1, 3): np.array([[1.0, 2.0], [2.0, 1.0]]),
        (0, 1, 2, 3): np.array([[4.0, 3.0], [1.0, 1.0]]),
        (0, 1, 2, 3, 4): np.array([[4.0, 2.0]]),
    }

    # Build the symbolic circuit
    circuit = build_multivariate_monotonic_structured_cpt_pc(
        num_units=2, input_layer="bernoulli", parameterize=False
    )

    for sl in circuit.inputs:
        assert isinstance(sl, CategoricalLayer)
        next(sl.probs.inputs).initializer = ConstantTensorInitializer(
            bernoulli_probs[tuple(sl.scope)]
        )
    for sl in circuit.sum_layers:
        assert isinstance(sl, SumLayer)
        next(sl.weight.inputs).initializer = ConstantTensorInitializer(
            dense_weights[tuple(circuit.layer_scope(sl))]
        )

    if not return_ground_truth:
        return circuit

    # Input: (0, 0, 0, 0, 0)
    # Outputs:
    # - Categorical layers:
    #   - 0: [0.5, 0.4]
    #   - 1: [0.2, 0.3]
    #   - 2: [0.3, 0.1]
    #   - 3: [0.5, 0.5]
    #   - 4: [0.1, 0.8]
    # - Product layers:
    #   - (0, 2): [0.15, 0.04]
    #   - (1, 3): [0.10, 0.15]
    #   - (0, 1, 2, 3): [0.19 * 0.40, 0.53 * 0.35] = [0.0760, 0.1855]
    #   - (0, 1, 2, 3, 4): [0.8605 * 0.1, 0.2615 * 0.8] = [0.08605, 0.20920]
    # - Dense layers:
    #   - (0, 2): [0.15 + 0.04, 0.15 * 3 + 0.04 * 2] = [0.19, 0.53]
    #   - (1, 3): [0.10 + 0.15 * 2, 0.10 * 2 + 0.15] = [0.40, 0.35]
    #   - (0, 1, 2, 3): [0.0760 * 4 + 0.1855 * 3, 0.0760 + 0.1855] = [0.8605, 0.2615]
    #   - (0, 1, 2, 3, 4): [0.08605 * 4 + 0.20920 * 2] = 0.7626
    #
    # Input: (1, 0, 1, 1, 0)
    # Outputs:
    # - Categorical layers:
    #   - 0: [0.5, 0.6]
    #   - 1: [0.2, 0.3]
    #   - 2: [0.7, 0.9]
    #   - 3: [0.5, 0.5]
    #   - 4: [0.1, 0.8]
    # - Product layers:
    #   - (0, 2): [0.35, 0.54]
    #   - (1, 3): [0.10, 0.15]
    #   - (0, 1, 2, 3): [0.89 * 0.40, 2.13 * 0.35] = [0.3560, 0.7455]
    #   - (0, 1, 2, 3, 4): [3.6605 * 0.1, 1.1015 * 0.8] = [0.36605, 0.88120]
    # - Dense layers:
    #   - (0, 2): [0.35 + 0.54, 1.05 + 1.08] = [0.89, 2.13]
    #   - (1, 3): [0.10 + 0.30, 0.20 + 0.15] = [0.40, 0.35]
    #   - (0, 1, 2, 3): [0.3560 * 4 + 0.7455 * 3, 0.3560 + 0.7455] = [3.6605, 1.1015]
    #   - (0, 1, 2, 3, 4): [0.36605 * 4 + 0.88120 * 2] = 3.2266
    #
    # Input: (1, 0, 1, 1, -1)
    # Outputs:
    # - Categorical layers:
    #   - 0: [0.5, 0.6]
    #   - 1: [0.2, 0.3]
    #   - 2: [0.7, 0.9]
    #   - 3: [0.5, 0.5]
    #   - 4: [1.0, 1.0]
    # - Product layers:
    #   - (0, 2): [0.35, 0.54]
    #   - (1, 3): [0.10, 0.15]
    #   - (0, 1, 2, 3): [0.89 * 0.40, 2.13 * 0.35] = [0.3560, 0.7455]
    #   - (0, 1, 2, 3, 4): [3.6605, 1.1015] = [3.6605, 1.1015]
    # - Dense layers:
    #   - (0, 2): [0.35 + 0.54, 1.05 + 1.08] = [0.89, 2.13]
    #   - (1, 3): [0.10 + 0.30, 0.20 + 0.15] = [0.40, 0.35]
    #   - (0, 1, 2, 3): [0.3560 * 4 + 0.7455 * 3, 0.3560 + 0.7455] = [3.6605, 1.1015]
    #   - (0, 1, 2, 3, 4): [3.6605 * 4 + 1.1015 * 2] = 16.845
    #
    # Partition function
    # Outputs:
    # - Categorical layers:
    #   - 0: [1.0, 1.0]
    #   - 1: [1.0, 1.0]
    #   - 2: [1.0, 1.0]
    #   - 3: [1.0, 1.0]
    #   - 4: [1.0, 1.0]
    # - Product layers:
    #   - (0, 2): [1.0, 1.0]
    #   - (1, 3): [1.0, 1.0]
    #   - (0, 1, 2, 3): [6, 15]
    #   - (0, 1, 2, 3, 4): [69, 21]
    # - Dense layers:
    #   - (0, 2): [2, 5]
    #   - (1, 3): [3, 3]
    #   - (0, 1, 2, 3): [6 * 4 + 15 * 3, 6 + 15] = [69, 21]
    #   - (0, 1, 2, 3, 4): [69 * 4 + 21 * 2] = 318.0

    # The ground truth outputs we expect and the partition function
    gt_outputs: dict[str, dict[tuple[int, ...] : float]] = {
        "evi": {(0, 0, 0, 0, 0): 0.7626, (1, 0, 1, 1, 0): 3.2266},
        "mar": {(1, 0, 1, 1, None): 16.845},
    }
    gt_partition_func = 318.0
    return circuit, gt_outputs, gt_partition_func


def build_monotonic_bivariate_gaussian_hadamard_dense_pc(
    return_ground_truth: bool = False,
) -> Circuit | tuple[Circuit, dict[str, dict[tuple[int, ...], float]], float]:
    # The mean and standard deviations of Gaussian layers
    gaussian_mean_stddev: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {
        (0,): (np.array([[0.0], [0.5]]), np.array([[1.0], [0.5]])),
        (1,): (np.array([[2.0], [-1.0]]), np.array([[1.5], [2.0]])),
    }

    # The parameters of dense weights
    dense_weights: dict[tuple[int, ...], np.ndarray] = {
        (0,): np.array([[1.0, 1.0], [3.0, 2.0]]),
        (1,): np.array([[4.0, 3.0], [1.0, 2.0]]),
        (0, 1): np.array([[1.0, 2.0]]),
    }

    # Build the symbolic circuit
    circuit = build_bivariate_monotonic_structured_cpt_pc(
        num_units=2, input_layer="gaussian", parameterize=False
    )

    for sl in circuit.inputs:
        assert isinstance(sl, GaussianLayer)
        next(sl.mean.inputs).initializer = ConstantTensorInitializer(
            gaussian_mean_stddev[tuple(sl.scope)][0]
        )
        next(sl.stddev.inputs).initializer = ConstantTensorInitializer(
            gaussian_mean_stddev[tuple(sl.scope)][1]
        )
    for sl in circuit.sum_layers:
        assert isinstance(sl, SumLayer)
        next(sl.weight.inputs).initializer = ConstantTensorInitializer(
            dense_weights[tuple(circuit.layer_scope(sl))]
        )

    if not return_ground_truth:
        return circuit

    # Input: (0.3, 1.2)
    # Outputs:
    # - Gaussian layers (log probs):
    #   - 0: [-0.96393853, -0.30579135]
    #   - 1: [-1.46662586, -2.21708571]
    # - Product layers:
    #   - (0, 1): [1.3969502663871303, 1.1739772980345813]
    # - Dense layers:
    #   - (0,): [exp(-0.96393853) + exp(-0.30579135), 3 * exp(-0.96393853) + 2 * exp(-0.30579135)] = [1.1179280992373424, 2.6172440151574317]
    #   - (1,): [4 * exp(-1.46662586) + 3 * exp(-2.21708571), exp(-1.46662586) + 2 * exp(-2.21708571)] = [1.2495886518463384, 0.44855477411951006]
    #   - (0,1): [1 * 1.3969502663871303 + 2 * 1.1739772980345813] = [3.744904862456293]
    #
    # Input: (0.3, None)
    # Outputs:
    # - Gaussian layers (log probs):
    #   - 0: [-0.96393853, -0.30579135]
    #   - 1: [1.0, 1.0]
    # - Product layers:
    #   - (0, 1): [7.825496694661396, 7.8517320454722945]
    # - Dense layers:
    #   - (0,): [exp(-0.96393853) + exp(-0.30579135), 3 * exp(-0.96393853) + 2 * exp(-0.30579135)] = [1.1179280992373424, 2.6172440151574317]
    #   - (1,): [4 * 1 + 3 * 1, 1 + 2 * 1] = [7, 3]
    #   - (0, 1): [7.825496694661396 + 2 * 7.8517320454722945] = [23.528960785605985]
    #
    # Partition function
    # Outputs:
    # - Gaussian layers (log probs):
    #   - 0: [0.0, 0.0]
    #   - 1: [0.0, 0.0]
    # - Product layers:
    #   - (0, 1): [14, 15]
    # - Dense layers:
    #   - (0,): [2, 5]
    #   - (1,): [7, 3]
    #   - (0, 1): [1 * 14 + 2 * 15] = [44]

    # The ground truth outputs we expect and the partition function
    gt_outputs: dict[str, dict[tuple[int, ...] : float]] = {
        "evi": {(0.3, 1.2): 3.744904862456293},
        "mar": {(0.3, None): 23.528960785605985},
    }
    gt_partition_func = 44.0
    return circuit, gt_outputs, gt_partition_func
