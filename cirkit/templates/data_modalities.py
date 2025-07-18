import functools
from typing import Any

import numpy as np

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.parameters import ParameterFactory, mixing_weight_factory
from cirkit.templates.region_graph import PoonDomingos, QuadGraph, QuadTree, RandomBinaryTree
from cirkit.templates.utils import (
    Parameterization,
    name_to_input_layer_factory,
    parameterization_to_factory,
)


def image_data(
    image_shape: tuple[int, int, int],
    region_graph: str = "quad-graph",
    *,
    input_layer: str,
    num_input_units: int,
    sum_product_layer: str,
    num_sum_units: int,
    num_classes: int = 1,
    input_params: dict[str, Parameterization] | None = None,
    sum_weight_param: Parameterization | None = None,
    use_mixing_weights: bool = True,
) -> Circuit:
    """Constructs a symbolic circuit whose structure is tailored for image data sets.

    Args:
        image_shape: The image shape (C, H, W), where C is the number of channels, H is the height
            of the images, and W is their width.
        region_graph: The name of the region graph to use. It can be one of the following:
            'quad-tree-2' (the Quad-Tree with two splits per region node),
            'quad-tree-4' (the Quad-Tree with four splits per region node),
            'quad-graph'  (the Quad-Graph region graph),
            'random-binary-tree' (the random binary tree on flattened image pixels),
            'poon-domingos' (the Poon-Domingos architecture).
        input_layer: The name of the input layer. It can be one of the following:
            'categorical' (encoding a Categorical distribution over pixel channel values),
            'binomial' (encoding a Binomial distribution over pixel channel values),
            'embedding' (encoding an Embedding vector over pixel channel values),
            'gaussian' (encoding a Gaussian distribution over pixel channel values).
        num_input_units: The number of input units per input layer.
        sum_product_layer: The name of the sum-product inner layer. It can be one of the following:
            'cp' (the canonical decomposition layer, consisting of dense layers followed by a
            hadamard product layer), 'cpt' (the transposed canonical decomposition layer, consisting
            of a hadamard product layer followed by a single dense layer), 'tucker' (the Tucker
            decomposition layer, consisting of a kronecker product layer followed by a single dense
            layer).
        num_classes: The number of output classes (default=1).
        num_sum_units: The number of sum units in each sum layer, i.e., either dense or mixing
            layer.
        input_params: A dictionary mapping each name of a parameter of the input layer to
            its parameterization. If it is None, then the default parameterization of the chosen
            input layer will be chosen.
        sum_weight_param: The parameterization to use for sum layers parameters. If it None,
            then a softmax parameterization of the sum weights will be used.
        use_mixing_weights: Whether to parameterize sum layers having arity > 1 in a way such
            that they compute a linear combinations of the input vectors, instead of computing
            a matrix-vector product where the vector is the concatenation of input vectors.
            Sum layers having this semantics are also sometimes referred to as "mixing" layers.
            Defaults to True.

    Returns:
        Circuit: A symbolic circuit.

    Raises:
        ValueError: If one of the arguments is not one of the specified allowed ones.
    """
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 3
        or any(d <= 0 for d in image_shape)
    ):
        raise ValueError(
            "Expected the image shape to be a tuple of three positive integers,"
            f" but found {image_shape}"
        )
    if region_graph not in [
        "quad-tree-2",
        "quad-tree-4",
        "quad-graph",
        "random-binary-tree",
        "poon-domingos",
    ]:
        raise ValueError(f"Unknown region graph called {region_graph}")
    if input_layer not in ["categorical", "binomial", "embedding", "gaussian"]:
        raise ValueError(f"Unknown input layer called {input_layer}")

    # Construct the image-tailored region graph
    match region_graph:
        case "quad-tree-2":
            rg = QuadTree(image_shape, num_patch_splits=2)
        case "quad-tree-4":
            rg = QuadTree(image_shape, num_patch_splits=4)
        case "quad-graph":
            rg = QuadGraph(image_shape)
        case "random-binary-tree":
            rg = RandomBinaryTree(image_shape[0] * image_shape[1] * image_shape[2])
        case "poon-domingos":
            delta = int(max(np.ceil(image_shape[1] / 8), np.ceil(image_shape[2] / 8)))
            rg = PoonDomingos(image_shape, delta=delta)
        case _:
            raise ValueError(f"Unknown region graph called {region_graph}")

    # Get the input layer factory
    input_kwargs: dict[str, Any]
    match input_layer:
        case "categorical":
            input_kwargs = {"num_categories": 256}
        case "binomial":
            input_kwargs = {"total_count": 255}
        case "embedding":
            input_kwargs = {"num_states": 256}
        case "gaussian":
            input_kwargs = {}
        case _:
            assert False
    if input_params is not None:
        input_kwargs.update(
            (name + "_factory", parameterization_to_factory(param))
            for name, param in input_params.items()
        )
    input_factory = name_to_input_layer_factory(input_layer, **input_kwargs)

    # Set the sum weight factory
    if sum_weight_param is None:
        sum_weight_param = Parameterization(activation="softmax", initialization="normal")
    sum_weight_factory = parameterization_to_factory(sum_weight_param)

    # Set the nary sum weight factory
    nary_sum_weight_factory: ParameterFactory
    if use_mixing_weights:
        nary_sum_weight_factory = functools.partial(
            mixing_weight_factory, param_factory=sum_weight_factory
        )
    else:
        nary_sum_weight_factory = sum_weight_factory

    # Build and return the symbolic circuit
    return rg.build_circuit(
        input_factory=input_factory,
        sum_product=sum_product_layer,
        sum_weight_factory=sum_weight_factory,
        nary_sum_weight_factory=nary_sum_weight_factory,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        num_classes=num_classes,
        factorize_multivariate=True,
    )
