import itertools
from typing import Dict, List, Tuple

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import ConstantTensorInitializer
from cirkit.symbolic.layers import CategoricalLayer, DenseLayer, HadamardLayer, Layer
from cirkit.symbolic.parameters import Parameter, TensorParameter
from cirkit.utils.scope import Scope


def build_monotonic_structured_categorical_pc() -> (
    Tuple[Circuit, Dict[Tuple[int, ...], float], float]
):
    categorical_probs: Dict[Tuple[int, ...], np.ndarray] = {
        (0,): np.array([[[0.5, 0.5]], [[0.4, 0.6]]]),
        (1,): np.array([[[0.2, 0.8]], [[0.3, 0.7]]]),
        (2,): np.array([[[0.3, 0.7]], [[0.1, 0.9]]]),
        (3,): np.array([[[0.5, 0.5]], [[0.5, 0.5]]]),
        (4,): np.array([[[0.1, 0.9]], [[0.8, 0.2]]]),
    }

    dense_weights: Dict[Tuple[int, ...], np.ndarray] = {
        (0, 2): np.array([[1.0, 1.0], [3.0, 2.0]]),
        (1, 3): np.array([[1.0, 2.0], [2.0, 1.0]]),
        (0, 1, 2, 3): np.array([[4.0, 3.0], [1.0, 1.0]]),
        (0, 1, 2, 3, 4): np.array([[4.0, 2.0]]),
    }

    scope_vtree: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], ...]] = {
        (0, 2): ((0,), (2,)),
        (1, 3): ((1,), (3,)),
        (0, 1, 2, 3): ((0, 2), (1, 3)),
        (0, 1, 2, 3, 4): ((0, 1, 2, 3), (4,)),
    }

    categorical_layers = {
        vids: CategoricalLayer(
            Scope(vids),
            num_output_units=2,
            num_channels=1,
            num_categories=2,
            probs=Parameter.from_leaf(
                TensorParameter(2, 1, 2, initializer=ConstantTensorInitializer(probs)),
            ),
        )
        for vids, probs in categorical_probs.items()
    }

    product_layers = {
        scope: HadamardLayer(Scope(scope), num_input_units=2, arity=2) for scope in dense_weights
    }

    dense_layers = {
        scope: DenseLayer(
            Scope(scope),
            num_input_units=2,
            num_output_units=1 if len(scope) == 5 else 2,
            weight=Parameter.from_leaf(
                TensorParameter(
                    1 if len(scope) == 5 else 2, 2, initializer=ConstantTensorInitializer(weight)
                )
            ),
        )
        for scope, weight in dense_weights.items()
    }

    layers = list(
        itertools.chain(categorical_layers.values(), product_layers.values(), dense_layers.values())
    )
    in_layers: Dict[Layer, List[Layer]] = {
        dense_layer: [product_layers[scope_factorization]]
        for scope_factorization, dense_layer in dense_layers.items()
    }
    in_layers.update(
        {
            product_layer: [
                categorical_layers[scope_vtree[scope][0]]
                if len(scope_vtree[scope][0]) == 1
                else dense_layers[scope_vtree[scope][0]],
                categorical_layers[scope_vtree[scope][1]]
                if len(scope_vtree[scope][1]) == 1
                else dense_layers[scope_vtree[scope][1]],
            ]
            for scope, product_layer in product_layers.items()
        }
    )

    circuit = Circuit(
        scope=Scope([0, 1, 2, 3, 4]),
        num_channels=1,
        layers=layers,
        in_layers=in_layers,
        outputs=[dense_layers[(0, 1, 2, 3, 4)]],
    )

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

    gt_outputs = {
        (0, 0, 0, 0, 0): 0.7626,
        (1, 0, 1, 1, 0): 3.2266,
    }
    gt_partition_func = 318.0
    return circuit, gt_outputs, gt_partition_func


def test_build_monotonic_categorical_pc() -> Tuple[Circuit, Dict[Tuple[int, ...], float], float]:
    categorical_probs: Dict[Tuple[int, ...], np.ndarray] = {
        (0,): np.array([[[0.5, 0.5]], [[0.4, 0.6]]]),
        (1,): np.array([[[0.2, 0.8]], [[0.3, 0.7]]]),
        (2,): np.array([[[0.3, 0.7]], [[0.1, 0.9]]]),
        (3,): np.array([[[0.5, 0.5]], [[0.5, 0.5]]]),
        (4,): np.array([[[0.1, 0.9]], [[0.8, 0.2]]]),
        (5,): np.array([[[0.2, 0.8]], [[0.3, 0.7]]]),
        (6,): np.array([[[0.5, 0.5]], [[0.3, 0.7]]]),
        (7,): np.array([[[0.9, 0.1]], [[0.3, 0.7]]]),
        (8,): np.array([[[0.2, 0.8]], [[0.1, 0.9]]]),
    }
