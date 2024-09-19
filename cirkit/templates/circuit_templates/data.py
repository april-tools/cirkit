import functools
from typing import Tuple

from cirkit.symbolic.circuit import Circuit
from cirkit.templates.circuit_templates._factories import (
    build_image_region_graph,
    mixing_layer_factory,
    name_to_initializer,
    name_to_input_layer_factory,
    name_to_parameter_factory,
)


def image_data(
    image_shape: Tuple[int, int, int],
    *,
    region_graph: str = "quad-graph",
    input_layer: str,
    num_input_units: int,
    sum_product_layer: str,
    num_sum_units: int,
    sum_weight_param: str,
) -> Circuit:
    """Constructs a symbolic circuit whose structure is tailored for image data sets.

    Args:
        image_shape: The image shape (C, H, W), where C is the number of channels, H is the height
            of the images, and W is their width.
        region_graph: The name of the region graph to use. It can be one of the following:
            'quad-tree-2' (the Quad-Tree with two splits per region node),
            'quad-tree-4' (the Quad-Tree with four splits per region node),
            'quad-graph'  (the Quad-Graph region graph),
            'poon-domingos' (the Poon-Domingos architecture).
        input_layer: The name of the input layer. It can be one of the following: 'categorical'.
        num_input_units: The number of input units per input layer.
        sum_product_layer: The name of the sum-product inner layer. It can be one of the following:
            'cp' (the canonical decomposition layer, consisting of dense layers followed by a
            hadamard product layer), 'cpt' (the transposed canonical decomposition layer, consisting
            of a hadamard product layer followed by a single dense layer), 'tucker' (the Tucker
            decomposition layer, consisting of a kronecker product layer followed by a single dense
            layer).
        num_sum_units: The number of sum units in each sum layer, i.e., either dense or mixing
            layer.
        sum_weight_param: The method to use to parameterize the weights of sum layers. It can be
            one of the following: 'id' (identity, i.e., no parameterization), 'softmax',
            'positive-clamp' (equivalent to max(., 1e-18)).

    Returns:
        Circuit: A symbolic circuit.

    Raises:
        ValueError: If one of the arguments is not one of the specified allowed ones.
    """
    if region_graph not in ["quad-tree-2", "quad-tree-4", "quad-graph", "poon-domingos"]:
        raise ValueError(f"Unknown region graph called {region_graph}")
    if input_layer not in ["categorical"]:
        raise ValueError(f"Unknown input layer called {input_layer}")
    if sum_weight_param not in ["id", "softmax", "positive-clamp"]:
        raise ValueError(f"Unknown sum weight parameterization called {sum_weight_param}")

    # Construct the image-tailored region graph
    rg = build_image_region_graph(region_graph, (image_shape[1], image_shape[2]))

    # Get the input layer factory
    input_factory = name_to_input_layer_factory(input_layer, num_categories=256)

    # Get the dense and mixing layers parameterization factory
    sum_weight_init = "normal" if sum_weight_param == "softmax" else "uniform"
    initializer_kwargs = {"axis": -1} if sum_weight_init in {"dirichlet"} else {}
    initializer = name_to_initializer(sum_weight_init, **initializer_kwargs)
    sum_weight_factory = name_to_parameter_factory(sum_weight_param, initializer=initializer)

    # Get the mixing layer factory (this might not be needed, but we pass it anyway)
    mixing_factory = functools.partial(mixing_layer_factory, weight_factory=sum_weight_factory)

    # Build and return the symbolic circuit
    return Circuit.from_region_graph(
        rg,
        input_factory=input_factory,
        sum_product=sum_product_layer,
        sum_weight_factory=sum_weight_factory,
        mixing_factory=mixing_factory,
        num_channels=image_shape[0],
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        num_classes=1,
        factorize_multivariate=True,
    )
