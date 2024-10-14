from typing import Any

from cirkit.symbolic.circuit import Circuit
from cirkit.templates.circuit_templates.utils import (
    Parameterization,
    build_image_region_graph,
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
    input_params: dict[str, Parameterization] | None = None,
    sum_weight_param: Parameterization | None = None,
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
            'embedding' (encoding an Embedding vector over pixel channel values).
        num_input_units: The number of input units per input layer.
        sum_product_layer: The name of the sum-product inner layer. It can be one of the following:
            'cp' (the canonical decomposition layer, consisting of dense layers followed by a
            hadamard product layer), 'cpt' (the transposed canonical decomposition layer, consisting
            of a hadamard product layer followed by a single dense layer), 'tucker' (the Tucker
            decomposition layer, consisting of a kronecker product layer followed by a single dense
            layer).
        num_sum_units: The number of sum units in each sum layer, i.e., either dense or mixing
            layer.
        input_params: A dictionary mapping each name of a parameter of the input layer to
            its parameterization. If it is None, then the default parameterization of the chosen
            input layer will be chosen.
        sum_weight_param: The parameterization to use for sum layers parameters. If it None,
            then a softmax parameterization of the sum weights will be used.

    Returns:
        Circuit: A symbolic circuit.

    Raises:
        ValueError: If one of the arguments is not one of the specified allowed ones.
    """
    if region_graph not in [
        "quad-tree-2",
        "quad-tree-4",
        "quad-graph",
        "random-binary-tree",
        "poon-domingos",
    ]:
        raise ValueError(f"Unknown region graph called {region_graph}")
    if input_layer not in ["categorical", "binomial", "embedding"]:
        raise ValueError(f"Unknown input layer called {input_layer}")

    # Construct the image-tailored region graph
    rg = build_image_region_graph(region_graph, (image_shape[1], image_shape[2]))

    # Get the input layer factory
    input_kwargs: dict[str, Any]
    match input_layer:
        case "categorical":
            input_kwargs = {"num_categories": 256}
        case "binomial":
            input_kwargs = {"total_count": 255}
        case "embedding":
            input_kwargs = {"num_states": 256}
        case _:
            assert False
    if input_params is not None:
        input_kwargs.update(
            (name + "_factory", parameterization_to_factory(param))
            for name, param in input_params.items()
        )
    input_factory = name_to_input_layer_factory(input_layer, **input_kwargs)

    # Get the sum weight factory
    if sum_weight_param is None:
        sum_weight_param = Parameterization(activation="softmax", initialization="normal")
    sum_weight_factory = parameterization_to_factory(sum_weight_param)

    # Build and return the symbolic circuit
    return Circuit.from_region_graph(
        rg,
        input_factory=input_factory,
        sum_product=sum_product_layer,
        sum_weight_factory=sum_weight_factory,
        num_channels=image_shape[0],
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        num_classes=1,
        factorize_multivariate=True,
    )
