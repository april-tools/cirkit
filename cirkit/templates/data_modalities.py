import functools
from typing import Any, Mapping

import numpy as np
from torch import Tensor

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.parameters import ParameterFactory, mixing_weight_factory
from cirkit.templates.region_graph import (
    ChowLiuTree,
    LinearTree,
    PoonDomingos,
    QuadGraph,
    QuadTree,
    RandomBinaryTree,
    RegionGraph,
)
from cirkit.templates.utils import (
    InputLayerFactory,
    Parameterization,
    name_to_input_layer_factory,
    parameterization_to_factory,
)
from cirkit.utils.scope import Scope


def _flatten_image_data(data: Tensor, image_shape: tuple[int, int, int]) -> Tensor:
    """Flattens image samples into the tabular form required for region-graph learning.

    Accepts data already in tabular form (num_samples, C * H * W), as a full image tensor
    (num_samples, C, H, W) matching image_shape, or as a single-channel image without the
    channel axis (num_samples, H, W) when C == 1.

    Args:
        data (Tensor): The image data to flatten, whose leading axis indexes the samples.
            Excluding that axis, its shape must be one of (C * H * W,), (C, H, W) matching
            image_shape, or (H, W) for single-channel images (i.e. when C == 1).
        image_shape (tuple[int, int, int]): The image shape, given as (C, H, W).

    Returns:
        The data reshaped into tabular form, i.e. a matrix of shape (num_samples, C * H * W).

    Raises:
        ValueError: If the shape of data is not compatible with image_shape.
    """
    num_variables = image_shape[0] * image_shape[1] * image_shape[2]
    valid_feature_shapes = [(num_variables,), tuple(image_shape)]
    if image_shape[0] == 1:
        valid_feature_shapes.append(tuple(image_shape[1:]))
    if tuple(data.shape[1:]) not in valid_feature_shapes:
        raise ValueError(
            f"Expected image data of shape (num_samples, {image_shape[0]}, {image_shape[1]},"
            f" {image_shape[2]}) or (num_samples, {num_variables}),"
            f" but found a tensor of shape {tuple(data.shape)}"
        )
    return data.reshape(data.shape[0], num_variables)


def image_data(  # pylint: disable=too-many-arguments
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
    data: Tensor | None = None,
    num_bins: int | None = None,
    mi_chunk_size: int | None = None,
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
            'poon-domingos' (the Poon-Domingos architecture),
            'chow-liu-tree' (the Chow-Liu tree / HCLT, learned from data; requires the 'data'
            argument),
            'linear-tree' (the linear chain over the flattened image
            pixels, i.e. the HMM region graph).
        input_layer: The name of the input layer. It can be one of the following:
            'categorical' (encoding a Categorical distribution over pixel channel values),
            'binomial' (encoding a Binomial distribution over pixel channel values),
            'embedding' (encoding an Embedding vector over pixel channel values),
            'gaussian' (encoding a Gaussian distribution over pixel channel values).
        num_input_units: The number of input units per input layer.
        sum_product_layer: The name of the sum-product inner layer. It can be one of the following:
            'cp' (the canonical decomposition layer, consisting of dense layers followed by a
            hadamard product layer), 'cp-t' (the transposed canonical decomposition layer,
            consisting of a hadamard product layer followed by a single dense layer), 'tucker'
            (the Tucker decomposition layer, consisting of a kronecker product layer followed by
            a single dense layer).
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
        data: The data tensor from which the 'chow-liu-tree' region graph is learned. It is
            required when region_graph is 'chow-liu-tree' and ignored otherwise. It can be of shape
            (num_samples, C, H, W) matching image_shape, flattened (num_samples, C * H * W),
            or (num_samples, H, W) for single-channel images (C == 1). When input_layer is
            'gaussian' it must be floating-point; otherwise it is treated as discrete
            pixel values.
        num_bins: Used by the categorical, binomial, and embedding 'chow-liu-tree' region graph.
            If given, the 256 pixel levels are binned into num_bins bins before learning the tree.
            Must be in [1, 256]. If None, the raw pixel values are used.
        mi_chunk_size: Only used by the 'chow-liu-tree' region graph. The number of samples
            per chunk passed to the chunked mutual-information estimators. If None, all samples
            are processed at once.

    Returns:
        Circuit: A symbolic circuit.

    Raises:
        ValueError:
            - If one of the arguments is not one of the specified allowed ones.
            - If region_graph is 'chow-liu-tree' but data is None, num_bins is outside [1, 256], or
                input_layer is 'gaussian' with non-floating-point data.
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
        "chow-liu-tree",
        "linear-tree",
    ]:
        raise ValueError(f"Unknown region graph called {region_graph}")
    if input_layer not in [
        "categorical",
        "binomial",
        "embedding",
        "gaussian",
        "discretized_logistic",
    ]:
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
        case "chow-liu-tree":
            if data is None:
                raise ValueError("A data tensor is required when region_graph is 'chow-liu-tree'")
            flattened_data = _flatten_image_data(data, image_shape)
            match input_layer:
                case "gaussian":
                    if not flattened_data.is_floating_point():
                        raise ValueError(
                            "A floating-point data tensor is required when input_layer is "
                            f"'gaussian', but found a tensor of dtype {flattened_data.dtype}"
                        )
                    rg_result = ChowLiuTree(
                        data=flattened_data,
                        input_type="gaussian",
                        chunk_size=mi_chunk_size,
                        as_region_graph=True,
                    )

                case "categorical" | "binomial" | "embedding":
                    clt_data = flattened_data.long()
                    num_categories = 256

                    if num_bins is not None:
                        if not 1 <= num_bins <= 256:
                            raise ValueError(
                                f"Expected num_bins to be in [1, 256], but found {num_bins}"
                            )

                        # Maps values in {0, ..., 255} to {0, ..., num_bins - 1}
                        clt_data = (clt_data * num_bins).div(256, rounding_mode="floor")
                        clt_data = clt_data.clamp_max(num_bins - 1)
                        num_categories = num_bins
                    # Categorical, binomial, and embedding image inputs are discrete variables.
                    # Their Chow-Liu edge weights are computed with the categorical mutual
                    # information estimator.
                    rg_result = ChowLiuTree(
                        data=clt_data,
                        input_type="categorical",
                        num_categories=num_categories,
                        chunk_size=mi_chunk_size,
                        as_region_graph=True,
                    )

                case _:
                    raise ValueError(f"Unsupported input layer called {input_layer}")

            if not isinstance(rg_result, RegionGraph):
                raise ValueError(f"Expected a RegionGraph, but got {type(rg_result).__name__}")
            rg = rg_result
        case "linear-tree":
            # The linear chain (HMM region graph) over the flattened image pixels,
            # in their natural row-major order.
            rg = LinearTree(image_shape[0] * image_shape[1] * image_shape[2])
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
        case "discretized_logistic":
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


def tabular_data(
    region_graph: str = "random-binary-tree",
    *,
    num_features: int | None = None,
    data: Tensor | None = None,
    input_layers: dict | list[dict],
    num_input_units: int,
    sum_product_layer: str,
    num_sum_units: int,
    num_classes: int = 1,
    sum_weight_param: Parameterization | None = None,
    use_mixing_weights: bool = True,
) -> Circuit:
    """
    Constructs a symbolic circuit whose structure is tailored for tabular data sets,
    supporting either a fixed random-binary-tree or a learned Chow–Liu tree.

    Args:
        region_graph:
            Which region graph to use.
            - `"random-binary-tree"`: build a random binary tree over the feature indices.
            - `"chow-liu-tree"`: learn a Chow–Liu tree from data.
        num_features:
            Number of features (columns) in the dataset.
            **Required** if `region_graph="random-binary-tree"`.
        data:
            A Torch tensor of shape `(n_samples, n_features)`.
            **Required** if `region_graph="chow-liu-tree"`, since the tree structure is
                learned from these samples.
        input_layers:
            Which per-feature distribution to use.
            The provided dictionaries should be of the following form:
            {
                'name': <name: str>,
                'args': <dictionary of arguments: dict>
            }
            for example: {'name': 'categorical', 'args': {'num_categories': 27}}
                or {'name': 'gaussian', 'args': {}}
            If a dict is provided, the same input layer is used for all features.
                If a list of dictionaries is provided,
            each feature will have its own input layer (input_layers[i] corresponds
                to feature i of the data).
        num_input_units:
            Number of parallel input units (e.g. mixtures/components) per feature.
        sum_product_layer:
            Which inner sum/product decomposition to use. E.g. `"cp"`, `"cp-t"`, or `"tucker"`.
        num_sum_units:
            Number of sum (or mixing) units in each sum layer.
        num_classes:
            Number of output classes (or root-layer mixtures). Often 1 for pure density estimation.
        sum_weight_param:
            If provided, a `Parameterization` object specifying activation & initialization
            for sum-layer weights.  Defaults internally to a softmax + Normal init.
        use_mixing_weights:
            Whether to use “mixing” sum layers (i.e. learn a linear combination of child outputs)
            for nodes of arity >1.  If False, falls back to a matrix-vector product.

    Returns:
        Circuit
            A fully-specified sum-product circuit over the given region graph with the chosen
            input distributions and inner decomposition layer.

    Raises:
        ValueError:
            - If one of the names of the input layers is not known, or the related arguments.
            - If the number of input layers (the length of the list) does not match the number
                of features (`num_features` or inferred from `data`).
            - If `region_graph="random-binary-tree"` but `num_features` is `None` and `data`
                is None.
            - If `region_graph="chow-liu-tree"` but `data` is `None`.
            - If `region_graph` is not one of the supported strings.
    """

    match region_graph:
        case "random-binary-tree":
            if num_features is None:
                if data is not None:
                    num_features = data.shape[1]
                else:
                    raise ValueError(
                        f"You must pass `num_features=` if you ask for {region_graph}."
                    )
            rg = RandomBinaryTree(num_features)
        case "chow-liu-tree":
            if data is None:
                raise ValueError("You must pass `data=` if you ask for `chow-liu-tree`.")
            rg_result = ChowLiuTree(
                data=data,
                input_type=(
                    input_layers["name"]
                    if isinstance(input_layers, dict)
                    else [input_layers["name"] for input_layers in input_layers]
                ),
                num_categories=(
                    input_layers["args"]["num_categories"]
                    if isinstance(input_layers, dict) and input_layers["name"] == "categorical"
                    else None
                ),
                as_region_graph=True,
            )
            if not isinstance(rg_result, RegionGraph):
                raise ValueError(f"Expected a RegionGraph, but got {type(rg_result).__name__}.")
            rg = rg_result
        case _:
            raise ValueError(f"Unknown region graph called {region_graph}")

    if sum_weight_param is None:
        sum_weight_param = Parameterization(activation="softmax", initialization="normal")
    sum_weight_factory = parameterization_to_factory(sum_weight_param)

    nary_sum_weight_factory: ParameterFactory
    if use_mixing_weights:
        nary_sum_weight_factory = functools.partial(
            mixing_weight_factory, param_factory=sum_weight_factory
        )
    else:
        nary_sum_weight_factory = sum_weight_factory

    input_factories: InputLayerFactory | Mapping[Scope, InputLayerFactory]
    if isinstance(input_layers, dict):
        input_factories = name_to_input_layer_factory(input_layers["name"], **input_layers["args"])
    else:
        if len(input_layers) != len(rg.scope):
            raise ValueError(
                f"Number of provided input layers ({len(input_layers)}) does \
                not match the number of features ({len(rg.scope)})."
            )
        input_factories = {
            Scope([i]): name_to_input_layer_factory(input_layer["name"], **input_layer["args"])
            for i, input_layer in enumerate(input_layers)
        }

    return rg.build_circuit(
        input_factory=input_factories,
        sum_product=sum_product_layer,
        sum_weight_factory=sum_weight_factory,
        nary_sum_weight_factory=nary_sum_weight_factory,
        num_input_units=num_input_units,
        num_sum_units=num_sum_units,
        num_classes=num_classes,
        factorize_multivariate=True,
    )
