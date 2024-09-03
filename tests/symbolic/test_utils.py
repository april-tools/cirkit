import itertools
from typing import Dict, List, Tuple, Union

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import (
    ConstantTensorInitializer,
    DirichletInitializer,
    NormalInitializer,
)
from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, GaussianLayer, HadamardLayer, Layer
from cirkit.symbolic.parameters import (
    ExpParameter,
    LogSoftmaxParameter,
    Parameter,
    SoftmaxParameter,
    TensorParameter,
)
from cirkit.utils.scope import Scope


def build_structured_monotonic_cpt_pc(
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
                logits_factory = lambda shape: Parameter.from_leaf(
                    TensorParameter(*shape, initializer=NormalInitializer())
                )
                probs_factory = None
        else:
            probs_factory = lambda shape: Parameter.from_leaf(
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
    elif input_layer == "gaussian":
        input_layers = {
            (vid,): GaussianLayer(Scope([vid]), num_output_units=num_units, num_channels=1)
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
        dense_weight_factory = lambda shape: Parameter.from_leaf(
            TensorParameter(*shape, initializer=DirichletInitializer())
        )
    dense_layers = {
        scope: DenseLayer(
            Scope(scope),
            num_input_units=num_units,
            num_output_units=1 if len(scope) == 5 else num_units,
            weight_factory=dense_weight_factory,
        )
        for scope in [(0, 2), (1, 3), (0, 1, 2, 3), (0, 1, 2, 3, 4)]
    }

    # Build hadamard product layers
    product_layers = {
        scope: HadamardLayer(Scope(scope), num_input_units=num_units, arity=2)
        for scope in dense_layers
    }

    # Set the connections between layers
    in_layers: Dict[Layer, List[Layer]] = {
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
        scope=Scope([0, 1, 2, 3, 4]),
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
) -> Union[Circuit, Tuple[Circuit, Dict[str, Dict[Tuple[int, ...], float]], float]]:
    # The probabilities of Bernoulli layers
    bernoulli_probs: Dict[Tuple[int, ...], np.ndarray] = {
        (0,): np.array([[[0.5, 0.5]], [[0.4, 0.6]]]),
        (1,): np.array([[[0.2, 0.8]], [[0.3, 0.7]]]),
        (2,): np.array([[[0.3, 0.7]], [[0.1, 0.9]]]),
        (3,): np.array([[[0.5, 0.5]], [[0.5, 0.5]]]),
        (4,): np.array([[[0.1, 0.9]], [[0.8, 0.2]]]),
    }

    # The parameters of dense weights
    dense_weights: Dict[Tuple[int, ...], np.ndarray] = {
        (0, 2): np.array([[1.0, 1.0], [3.0, 2.0]]),
        (1, 3): np.array([[1.0, 2.0], [2.0, 1.0]]),
        (0, 1, 2, 3): np.array([[4.0, 3.0], [1.0, 1.0]]),
        (0, 1, 2, 3, 4): np.array([[4.0, 2.0]]),
    }

    # Build the symbolic circuit
    circuit = build_structured_monotonic_cpt_pc(
        num_units=2, input_layer="bernoulli", parameterize=False
    )

    for sl in circuit.inputs:
        assert isinstance(sl, CategoricalLayer)
        next(sl.probs.inputs).initializer = ConstantTensorInitializer(
            bernoulli_probs[tuple(sl.scope)]
        )
    for sl in circuit.sum_layers:
        assert isinstance(sl, DenseLayer)
        next(sl.weight.inputs).initializer = ConstantTensorInitializer(
            dense_weights[tuple(sl.scope)]
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
    # - Sum layers:
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
    # - Sum layers:
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
    # - Sum layers:
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
    # - Sum layers:
    #   - (0, 2): [2, 5]
    #   - (1, 3): [3, 3]
    #   - (0, 1, 2, 3): [6 * 4 + 15 * 3, 6 + 15] = [69, 21]
    #   - (0, 1, 2, 3, 4): [69 * 4 + 21 * 2] = 318.0

    # The ground truth outputs we expect and the partition function
    gt_outputs: Dict[str, Dict[Tuple[int, ...] : float]] = {
        "evi": {(0, 0, 0, 0, 0): 0.7626, (1, 0, 1, 1, 0): 3.2266},
        "mar": {(1, 0, 1, 1, -1): 16.845},
    }
    gt_partition_func = 318.0
    return circuit, gt_outputs, gt_partition_func
